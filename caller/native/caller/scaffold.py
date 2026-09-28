"""Truth-free SNP scaffold from BAM likelihoods and a family-excluded panel."""
from reference_cache import CachedReference
import argparse,collections,ctypes,hashlib,json,pathlib,subprocess,sys,time
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'base'))
import caller
from panel_fast import load_panel_fast
from read_index import ReadIndex
from artifact import site_likelihoods
from state_selection import select_haplotypes
from genetic_map import transitions
from native_build import build_library
from prepare import sha

def kernel():
    binary=build_library(HERE/'scaffold_neutral.cpp')
    lib=ctypes.CDLL(str(binary));p=ctypes.POINTER(ctypes.c_double)
    lib.scaffold_gibbs.argtypes=[ctypes.c_int,ctypes.c_int,p,p,p,ctypes.c_double,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_uint64,p]
    lib.scaffold_gibbs.restype=ctypes.c_int;return lib

def impute(hap,gl,refresh,*,mutation=.001,burn=8,iterations=24,chains=2,seed=17):
    h=np.ascontiguousarray(hap,dtype=np.float64);g=np.ascontiguousarray(gl,dtype=np.float64);r=np.ascontiguousarray(refresh,dtype=np.float64)
    if h.ndim!=2 or not 0<=mutation<.5 or burn<0 or iterations<1 or chains<1:raise ValueError('Invalid sampler configuration')
    m,k=h.shape
    if g.shape!=(m,3) or r.shape!=(m,) or not np.isfinite(h).all() or not np.isfinite(g).all() or not np.isfinite(r).all():raise ValueError('Invalid scaffold dimensions/values')
    if np.any((h<0)|(h>1)) or np.any((r<0)|(r>1)) or k<1 or m<1:raise ValueError('Invalid HMM probabilities')
    out=np.zeros((m,3),dtype=np.float64);p=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    code=kernel().scaffold_gibbs(m,k,p(h),p(g),p(r),mutation,burn,iterations,chains,seed,p(out))
    if code or not np.isfinite(out).all() or not np.allclose(out.sum(1),1):
        if code not in (0,2,3,4):raise RuntimeError('Conditional scaffold failed: '+str(code))
        original_code=code
        stable=ctypes.CDLL(str(build_library(HERE/'scaffold_runs.cpp')))
        stable.scaffold_gibbs.argtypes=kernel().scaffold_gibbs.argtypes
        stable.scaffold_gibbs.restype=ctypes.c_int
        out.fill(0.)
        code=stable.scaffold_gibbs(m,k,p(h),p(g),p(r),mutation,burn,iterations,chains,seed,p(out))
        print('VARDEPTH_SCAFFOLD_NUMERICAL_FALLBACK '+json.dumps(dict(
            original_code=original_code,fallback_code=code,markers=m,donors=k,
            positive_refresh_edges=int(np.count_nonzero(r[1:])),same_model_parameters=True,
            same_seed=seed,algorithm='log-likelihood accumulation within exactly zero-refresh runs')),
            file=sys.stderr,flush=True)
        if code or not np.isfinite(out).all() or not np.allclose(out.sum(1),1):
            raise RuntimeError('Stable conditional scaffold failed: '+str(code))
    return out

def run(args):
    import pysam
    from base_adapter import parse_region
    begun=time.monotonic();chrom,start,end=parse_region(args.region);fa=CachedReference(args.reference)
    if chrom not in fa.references or end>fa.get_reference_length(chrom):raise ValueError('Region outside matching FASTA')
    fetch=lambda a,b:fa.fetch(chrom,max(0,a),min(b,fa.get_reference_length(chrom)))
    bams=collections.OrderedDict()
    for item in args.bam:
        sample,path=item.split('=',1);bams.setdefault(sample,[]).append(path)
    excluded=set(bams)|set(filter(None,getattr(args,'exclude_samples','').split(',')));raw={};seen=set();seen_sms=set()
    for sample,paths in bams.items():
        if any(str(pathlib.Path(p).resolve()) in seen for p in paths):raise ValueError('BAM reused between samples')
        seen.update(str(pathlib.Path(p).resolve()) for p in paths)
        reads,sms,headers=caller.collect_reads(paths,args.reference,chrom,start,end,10)
        if seen_sms & sms:raise ValueError('BAM SM assigned to multiple individuals')
        seen_sms |= sms
        platforms={str(r['PL']).upper() for h in headers for r in h.get('RG',[]) if r.get('PL')}
        if platforms-{'ILLUMINA'}:raise ValueError('Illumina only')
        excluded|=sms;raw[sample]=reads
    for family in caller.FAMILIES:
        if family&excluded:excluded|=family
    panel,names,skips=load_panel_fast(args.panel,chrom,start,end,fetch,excluded,50,reference_identity=args.reference)
    from evidence_trace import enabled, ScaffoldTrace
    trace=ScaffoldTrace(args,names,{p:caller.file_identity(p)['sha256'] for p in
        [args.reference,args.panel,args.genetic_map]+[p for paths in bams.values() for p in paths]}) if enabled() else None
    snvs=[v for v in panel if len(v[1])==len(v[2])==1];counts=collections.Counter(v[0] for v in snvs)
    variants=sorted(v for v in snvs if counts[v[0]]==1)
    if not variants:
        dest=pathlib.Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,positions=np.zeros(0,np.int64),ref=np.array([],dtype='U1'),alt=np.array([],dtype='U1'),samples=np.array(list(bams)),gp=np.zeros((len(bams),0,3)))
        dest.with_suffix('.json').write_text(json.dumps(dict(arguments=vars(args),sites=0,reason='No unique biallelic SNP scaffold sites; neural calling remains available'))+'\n')
        if trace is not None:trace.finish()
        return
    h=np.stack([panel[v][0] for v in variants]);positions=np.array([v[0] for v in variants],np.int64)
    refresh=transitions(positions,reference_haplotypes=max(2,2*len(names)),path=args.genetic_map,ne=args.ne,chrom=chrom)
    posterior=[];depth=[];selected={};timing={};read_gl=[]
    for sample in bams:
        t=time.monotonic();obs=caller.likelihoods(variants,ReadIndex(raw[sample]),fetch,20,caller.core(),compute_gl=False)
        gl=site_likelihoods(variants,obs[4],.01)
        keep=select_haplotypes(h,gl,min(args.states,h.shape[1]),positions=positions,mode='paired',windows=16)
        selected[sample]=keep.tolist();seed=(args.seed+int.from_bytes(hashlib.sha256(sample.encode()).digest()[:4],'little'))%(2**64)
        if trace is not None:trace.add(sample,variants,obs[4],keep,raw[sample])
        gp=impute(h[:,keep],gl,refresh,mutation=args.mutation,burn=args.burn,iterations=args.iterations,chains=args.chains,seed=seed)
        posterior.append(gp);depth.append(obs[1]);read_gl.append(gl);timing[sample]=time.monotonic()-t
        print(sample,'SNP scaffold',len(variants),'sites',len(keep),'states',timing[sample],'seconds',flush=True)
    dest=pathlib.Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,positions=positions,ref=np.array([v[1] for v in variants]),alt=np.array([v[2] for v in variants]),
        samples=np.array(list(bams)),gp=np.array(posterior),depth=np.array(depth),gl=np.array(read_gl))
    manifest=dict(version='0.3-experimental-scaffold',arguments=vars(args),samples=list(bams),sites=len(variants),panel_samples=len(names),
        input_sha256={p:caller.file_identity(p)['sha256'] for p in [args.reference,args.panel,args.genetic_map]+[p for paths in bams.values() for p in paths]},
        excluded=sorted(excluded),panel_skips=skips,selected_external_haplotypes=selected,seconds=time.monotonic()-begun,sample_seconds=timing,
        method='Alternating conditional haploid copying Gibbs sampler; forward-backward Rao-Blackwell genotype probabilities; SNP scaffold only; no truth',
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'scaffold.py',HERE/'scaffold.cpp']})
    dest.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if trace is not None:trace.finish()
    return manifest

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--bam',action='append',required=True);p.add_argument('--reference',required=True);p.add_argument('--panel',required=True)
    p.add_argument('--genetic-map',required=True);p.add_argument('--region',required=True);p.add_argument('--output',required=True)
    p.add_argument('--states',type=int,default=1024);p.add_argument('--ne',type=float,default=100000);p.add_argument('--mutation',type=float,default=.001)
    p.add_argument('--exclude-samples',default='')
    p.add_argument('--burn',type=int,default=12);p.add_argument('--iterations',type=int,default=48);p.add_argument('--chains',type=int,default=2);p.add_argument('--seed',type=int,default=17)
    args=p.parse_args(argv)
    if args.states<2 or args.burn<0 or args.iterations<1 or args.chains<1 or not 0<=args.mutation<.5:p.error('Invalid sampler settings')
    run(args)


if __name__=='__main__':main()
