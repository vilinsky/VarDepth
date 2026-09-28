#!/usr/bin/env python3
"""Research caller: read candidates and likelihoods + joint diploid copying HMM.
No competitor caller is invoked. See DESIGN.md for explicit approximations.
"""
import argparse,collections,ctypes,datetime,fcntl,hashlib,json,math,os,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import pysam
import pysam.bcftools

VERSION='0.2.0-feature-instrumented-base'
HERE=Path(__file__).resolve().parent
FAMILIES=[{'HG001'}|{'NA'+str(n) for n in range(12877,12894)}, {'HG002','HG003','HG004','NA24385','NA24149','NA24143'}, {'HG005','HG006','HG007','NA24631','NA24694','NA24695'}]

def nonoverlapping_candidates(variants):
    """Conservative biallelic scope: reject entire overlapping REF-span clusters.

    A proper multiallelic/assembly model is required to retain these. Rejected
    truth variants must still contribute FN in external evaluation.
    """
    kept=[];rejected=[];group=[];end=-1
    for v in sorted(variants):
        if group and v[0]>=end:
            (kept if len(group)==1 else rejected).extend(group);group=[]
        if not group:end=-1
        group.append(v);end=max(end,v[0]+len(v[1]))
    if group:(kept if len(group)==1 else rejected).extend(group)
    return kept,rejected

def core():
    suffix='dylib' if sys.platform=='darwin' else 'so'
    p=HERE/'build'/('libcaller.'+suffix)
    p.parent.mkdir(exist_ok=True);fingerprint=hashlib.sha256((HERE/'core/core.cpp').read_bytes()+(HERE/'build.sh').read_bytes()).hexdigest();marker=p.parent/'core.source.sha256'
    with (p.parent/'core.build.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if not p.exists() or not marker.exists() or marker.read_text().strip()!=fingerprint:
            subprocess.run(['bash',str(HERE/'build.sh')],check=True);marker.write_text(fingerprint+'\n')
    lib=ctypes.CDLL(str(p));ptr=ctypes.POINTER(ctypes.c_double)
    lib.copying_hmm.argtypes=[ctypes.c_int,ctypes.c_int]+[ptr]*6;lib.copying_hmm.restype=ctypes.c_double
    lib.read_haplotype.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_ubyte),ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_double,ctypes.c_double];lib.read_haplotype.restype=ctypes.c_double
    lib.read_haplotype_batch.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_ubyte),ctypes.c_int,ctypes.POINTER(ctypes.c_char_p),ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_double,ctypes.c_double,ptr]
    lib.read_haplotype_batch.restype=None
    return lib

def ptr(a):return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
def normalize_variant(pos,ref,alt,fetch):
    ref,alt=ref.upper(),alt.upper()
    while len(ref)>1 and len(alt)>1 and ref[-1]==alt[-1]:ref,alt=ref[:-1],alt[:-1]
    if len(ref)!=len(alt):
        while pos>0 and ref[-1]==alt[-1]:
            prev=fetch(pos-1,pos).upper();ref=prev+ref[:-1];alt=prev+alt[:-1];pos-=1
    while len(ref)>1 and len(alt)>1 and ref[0]==alt[0]:ref,alt=ref[1:],alt[1:];pos+=1
    return pos,ref,alt

def valid_read(r,mapq):return not(r.is_unmapped or r.is_secondary or r.is_supplementary or r.is_duplicate or r.is_qcfail) and mapq<=r.mapping_quality<255 and r.query_sequence is not None

from input_identity import file_identity

def _collect_reads_uncached(paths,reference,chrom,start,end,mapq):
    reads=[];seen=set();sms=set();headers=[]
    fasta=pysam.FastaFile(reference)
    for path in paths:
        with pysam.AlignmentFile(path,'rc' if str(path).endswith('.cram') else 'rb',reference_filename=reference) as bam:
            header=bam.header.to_dict();headers.append(header)
            if chrom not in bam.references or bam.get_reference_length(chrom)!=fasta.get_reference_length(chrom):raise ValueError('BAM/FASTA contig name or length mismatch: '+path)
            groups={r['ID']:r for r in header.get('RG',[])}
            sms.update(r['SM'] for r in header.get('RG',[]) if 'SM' in r)
            for r in bam.fetch(chrom,max(0,start-250),end+250):
                if not valid_read(r,mapq):continue
                group=groups.get(r.get_tag('RG') if r.has_tag('RG') else '',{})
                # Missing library metadata must not merge independent read groups
                # with coincident QNAME. Known LB remains shared across files.
                lib=group.get('LB') or (str(Path(path).resolve())+'::RG='+group.get('ID','no-RG'))
                key=(lib,r.query_name,r.is_read1,r.is_read2,r.reference_start,r.cigarstring)
                if key in seen:continue
                seen.add(key);reads.append((lib,r))
    reads.sort(key=lambda x:x[1].reference_start)
    return reads,sms,headers

def collect_reads(paths,reference,chrom,start,end,mapq):
    from read_collection_cache import collect
    return collect(_collect_reads_uncached,paths,reference,chrom,start,end,mapq)


def discover(reads,fetch,start,end,min_bq,max_indel):
    support=collections.defaultdict(set)
    for lib,r in reads:
        seq=r.query_sequence.upper();qual=r.query_qualities
        if qual is None:continue
        q=0;p=r.reference_start;fragment=(lib,r.query_name)
        for op,n in r.cigartuples or []:
            if op in (0,7,8):
                ref=fetch(p,p+n).upper()
                if ref==seq[q:q+n]:
                    q+=n;p+=n
                    continue
                for j,(a,b) in enumerate(zip(ref,seq[q:q+n])):
                    if start<=p+j<end and a!=b and a in 'ACGT' and b in 'ACGT' and qual[q+j]>=min_bq:
                        support[(p+j,a,b)].add(fragment)
                q+=n;p+=n
            elif op==1:
                if 0<n<=max_indel and p>0 and q>0 and min(qual[max(0,q-1):min(len(qual),q+n+1)])>=min_bq:
                    anchor=fetch(p-1,p).upper();v=normalize_variant(p-1,anchor,anchor+seq[q:q+n],fetch)
                    if start<=v[0]<end:support[v].add(fragment)
                q+=n
            elif op==2:
                if n<=max_indel and p>0 and q>0 and min(qual[max(0,q-1):min(len(qual),q+1)])>=min_bq:
                    ref=fetch(p-1,p+n).upper();v=normalize_variant(p-1,ref,ref[0],fetch)
                    if start<=v[0]<end:support[v].add(fragment)
                p+=n
            elif op==3:p+=n
            elif op==4:q+=n
    return support

def load_panel(path,chrom,start,end,fetch,excluded,max_indel):
    panel={};names=[];skips=collections.Counter()
    if not path:return panel,names,dict(skips)
    with pysam.VariantFile(path) as f:
        names=[s for s in f.header.samples if s not in excluded]
        if not names:raise ValueError('All panel samples excluded')
        for r in f.fetch(chrom,start,end):
            if not r.alts or len(r.alts)!=1 or max(len(r.ref),len(r.alts[0]))>max_indel+1 or any(x not in 'ACGT' for x in r.ref+r.alts[0]):skips['unsupported']+=1;continue
            if fetch(r.start,r.stop).upper()!=r.ref:skips['ref_mismatch']+=1;continue
            v=normalize_variant(r.start,r.ref,r.alts[0],fetch)
            if v[0]<start:continue
            h=[]
            for name in names:
                s=r.samples[name];g=s.get('GT')
                if g is None or len(g)!=2 or any(x not in (0,1) for x in g) or (g[0]!=g[1] and not s.phased):
                    h.extend((np.nan,np.nan));skips['unusable_sample_genotypes']+=1
                else:h.extend(g)
            h=np.array(h,dtype=np.float64)
            af=float(np.nanmean(h)) if np.isfinite(h).any() else 0
            h=np.nan_to_num(h,nan=af)
            panel[v]=(h,af)
    return panel,names,dict(skips)

def pair_score(lib,seq,quals,hap):
    qs=(ctypes.c_ubyte*len(quals))(*quals)
    return lib.read_haplotype(seq.encode(),qs,len(seq),hap.encode(),len(hap),.001,.1)

def pair_scores(lib,seq,quals,haps):
    qs=(ctypes.c_ubyte*len(quals))(*quals)
    encoded=[h.encode() for h in haps]
    hp=(ctypes.c_char_p*len(encoded))(*encoded)
    lengths=(ctypes.c_int*len(encoded))(*(len(h) for h in haps))
    out=np.empty(len(encoded),np.float64)
    lib.read_haplotype_batch(seq.encode(),qs,len(seq),hp,lengths,len(encoded),.001,.1,ptr(out))
    return out

def pair_scores_many(lib,sequences,qualities,haps):
    """Batch the same base-model emissions without changing molecule selection."""
    import os
    choice=os.environ.get('VARDEPTH_PAIRHMM_BACKEND','cpu')
    cells=sum(map(len,sequences))*sum(map(len,haps))
    if choice=='cuda' or (choice=='auto' and cells>=2_000_000):
        from accelerator import score_many
        return score_many(sequences,qualities,haps)
    from cpu_batch import many
    return many(sequences,qualities,haps,mode=0)

from functools import lru_cache

@lru_cache(maxsize=20000)
def snp_emission(bq,mq,ref_match,alt_match):
    e=10**(-bq/10);me=10**(-mq/10)
    lr=(1-e if ref_match else e/3)*(1-me)+me/4
    la=(1-e if alt_match else e/3)*(1-me)+me/4
    return tuple(map(float,np.log([lr,(lr+la)/2,la])))

def likelihoods(variants,reads,fetch,min_bq,lib,*,alignments=None,alignment_ranks=None,compute_gl=True):
    m=len(variants);gl=np.zeros((m,3)) if compute_gl else None;dp=np.zeros(m,dtype=int);ad=np.zeros((m,2),dtype=int);ambiguous=np.zeros(m,dtype=int)
    snps=collections.defaultdict(list)
    for i,v in enumerate(variants):
        if len(v[1])==len(v[2])==1:snps[v[0]].append(i)
    from snp_observations import collect as collect_observations
    observations=collect_observations(reads,snps,min_bq,alignments,alignment_ranks)
    import os
    from likelihood_batch import indel_inputs,finish_indel,indel_batches
    batched_indels=indel_batches(variants,reads,fetch,min_bq) if os.environ.get('VARDEPTH_PAIRHMM_BACKEND','cpu')!='cpu' else None
    for i,v in enumerate(variants):
        pos,ref,alt=v
        if len(ref)==len(alt)==1:
            g0=g1=g2=0.;ar=aa=0
            for base,bq,mq in observations[i].values():
                if compute_gl:
                    v0,v1,v2=snp_emission(min(bq,60),min(mq,60),base==ref,base==alt)
                    g0+=v0;g1+=v1;g2+=v2
                if base==ref:ar+=1
                elif base==alt:aa+=1
            if compute_gl:gl[i]=g0,g1,g2
            dp[i]=len(observations[i]);ad[i]=ar,aa
            continue
        if batched_indels is not None:
            per_fragment=batched_indels[i]
        else:
            pending,haps=indel_inputs(v,reads,fetch,min_bq)
            values=pair_scores_many(lib,[x[2] for x in pending],[x[3] for x in pending],haps)
            per_fragment=finish_indel(pending,values,compute_gl=compute_gl)
        for vals,certainty,lr,la in per_fragment.values():
            if compute_gl:gl[i]+=vals
            dp[i]+=1
            if certainty>=math.log(3):ad[i,int(la>lr)]+=1
            else:ambiguous[i]+=1
        observations[i]={key:(val[2],val[3],None) for key,val in per_fragment.items()}
    if compute_gl:gl-=gl.max(axis=1)[:,None]
    return gl,dp,ad,ambiguous,observations

def infer(lib,variants,panel,gl,k,cohort=None,no_ld=False,fragment_observations=None,artifact_prior=None,state_selection='marginal',local_blocks=None,sequence_data=None,genetic_map=None,ne=100000,chrom=None,indel_artifact_prior=None,factor_cache=None,selected_states=None):
    m=len(variants);nh=len(next(iter(panel.values()))[0]) if panel else 0
    if nh:
        if selected_states is None:
            h=np.vstack([panel[v][0] if v in panel else np.zeros(nh) for v in variants])
            from state_selection import select_haplotypes
            chosen=select_haplotypes(h,gl,k,positions=np.array([v[0] for v in variants]),mode=state_selection,windows=4);h=h[:,chosen]
        else:
            # Internal LD-proposal rerun: its panel, seed catalog and read GL
            # are unchanged; only local compatible sequence factors differ.
            chosen=np.asarray(selected_states,dtype=np.int64)
            if chosen.ndim!=1 or len(chosen)!=min(k,nh) or len(set(chosen))!=len(chosen) or np.any((chosen<0)|(chosen>=nh)):raise ValueError('Invalid fixed external states')
            h=np.stack([panel[v][0][chosen] if v in panel else np.zeros(len(chosen)) for v in variants])
    else:h=np.zeros((m,2));chosen=np.array([],dtype=int)
    if cohort: h=np.column_stack([h]+cohort)
    if h.shape[1]>64:raise ValueError('More than 64 states; use fewer batch donors or smaller --states')
    mu=np.array([1e-4 if len(v[1])!=len(v[2]) else .001 for v in variants],dtype=float)
    pos=np.array([v[0] for v in variants]);dist=np.diff(pos,prepend=pos[0]);r=1-np.exp(-dist*1e-6)
    if no_ld:r[:]=1
    r=np.clip(r,1e-8,.5) if not no_ld else np.ones(m)
    recombination=None
    if genetic_map:
        from genetic_map import transitions
        recombination={'reference_haplotypes':max(2,nh),'path':genetic_map,'ne':ne,'chrom':chrom};r=transitions(pos,no_ld=no_ld,**recombination)
    h=np.ascontiguousarray(h,dtype=np.float64);gl=np.ascontiguousarray(gl,dtype=np.float64)
    gp=np.empty((m,3));phase=np.empty((m,2));ll=None
    block_details={}
    if local_blocks is not None:
        from local_alleles import local_infer
        gp,phase,block_details=local_infer(lib,variants,h,gl,mu,fragment_observations or {},local_blocks,sequence_data or {},no_ld,artifact_prior,recombination,indel_artifact_prior,factor_cache=factor_cache)
    elif fragment_observations is not None:
        from blocks import block_infer
        gp,phase,block_details=block_infer(lib,variants,h,gl,mu,fragment_observations,no_ld,artifact_prior,recombination,indel_artifact_prior)
    else:ll=lib.copying_hmm(m,h.shape[1],ptr(h),ptr(gl),ptr(mu),ptr(r),ptr(gp),ptr(phase))
    if not np.isfinite(gp).all() or not np.allclose(gp.sum(axis=1),1):raise RuntimeError('Invalid HMM posterior')
    return gp,phase,{'selected_external_haplotypes':chosen.tolist(),'states':h.shape[1],'log_marginal_likelihood':ll,**block_details}

def emit(args,variants,panel,read_sources,results,observations,contig_length):
    header=pysam.VariantHeader();header.add_meta('source',value='VarDepthResearch-'+VERSION)
    header.add_meta('QUAL_semantics',value='Approximate Phred probability all samples are reference, product of per-sample marginal GP0; not an exact cohort joint probability')
    header.contigs.add(args.chrom,length=contig_length)
    for name,num,typ,desc in [('GT',1,'String','Unphased diploid genotype; missing when max GP below threshold'),('GQ',1,'Integer','Phred scaled posterior genotype error probability'),('DP',1,'Integer','Number of distinct observed fragments used at this candidate'),('AD','R','Integer','Distinct fragments supporting each local allele; ambiguous fragments excluded'),('PL','G','Integer','Read-only local genotype likelihoods normalized in phred units'),('GP','G','Float','Copying-HMM posterior genotype probabilities in VCF genotype order'),('DS','A','Float','Expected alternate allele dosage')]:header.formats.add(name,num,typ,desc)
    header.info.add('SOURCE',1,'String','Candidate source PANEL, READS or BOTH');header.info.add('PANEL_AF','A','Float','External reference-panel alternate allele frequencies; missing for novel alleles');header.info.add('READ_SUPPORT',1,'Integer','Sum of candidate-discovery support across represented alternatives and samples')
    header.info.add('LOCAL_ALLELES',1,'Integer','Number of compatible local sequence alleles in a joint block')
    header.filters.add('LowQual',None,None,'No non-reference sample genotype reaches minimum GQ and GP')
    for name in results:header.add_sample(name)
    mode='wb' if args.output.endswith('.bcf') else 'wz' if args.output.endswith('.gz') else 'w'
    joint=next(iter(results.values()))[2].get('joint_records',{}) if results else {};skip=set()
    with pysam.VariantFile(args.output,mode,header=header) as out:
        for i,v in enumerate(variants):
            if i in skip:continue
            payload={}
            if i in joint:
                block=joint[i];ids=block['ids'];skip.update(ids[1:]);pos=block['start'];alleles=tuple(block['alleles']);afs=[]
                # Normalization can coalesce equivalent allele seeds. Match each
                # padded local sequence to the original panel/read candidate.
                for alt in alleles[1:]:
                    matches=[]
                    for j in ids:
                        p,ref,a=variants[j];padded=alleles[0][:p-pos]+a+alleles[0][p-pos+len(ref):]
                        if padded==alt and variants[j] in panel:matches.append(panel[variants[j]][1])
                    afs.append(max(matches) if matches else None)
                if block.get('allele_af') is not None:afs=block['allele_af'][1:]
                for name in results:payload[name]=results[name][2]['joint_records'][i]
            else:
                ids=[i];pos=v[0];alleles=(v[1],v[2]);afs=[panel[v][1] if v in panel else None]
                for name,(gp,phase,details) in results.items():
                    gl,dp,ad,ambiguous,_=observations[name]
                    payload[name]={'gp':gp[i],'pl':gl[i],'dp':dp[i],'ad':ad[i],'genotypes':[(0,0),(0,1),(1,1)]}
            isvar=any(any(x['genotypes'][int(np.argmax(x['gp']))]) and max(x['gp'])>=args.min_gp for x in payload.values())
            if args.variants_only and not isvar:continue
            r=out.new_record(contig=args.chrom,start=pos,stop=pos+len(alleles[0]),alleles=alleles)
            in_panel=any(variants[j] in panel for j in ids);in_reads=any(variants[j] in read_sources for j in ids)
            r.info['SOURCE']='BOTH' if in_panel and in_reads else 'PANEL' if in_panel else 'READS'
            if any(x is not None for x in afs):r.info['PANEL_AF']=tuple(afs)
            if i in joint:r.info['LOCAL_ALLELES']=len(alleles)
            r.info['READ_SUPPORT']=sum(read_sources.get(variants[j],0) for j in ids)
            reference_probability=1.;passing=False
            for name,x in payload.items():
                probs=x['gp'];best=int(np.argmax(probs));gt=x['genotypes'][best];confidence=float(probs[best]);gq=min(99,int(round(-10*math.log10(max(1e-12,1-confidence)))))
                call=r.samples[name];call['GT']=tuple(gt) if confidence>=args.min_gp else (None,None)
                call['GQ']=gq;call['DP']=int(x['dp']);call['AD']=tuple(map(int,x['ad']));call['PL']=tuple(min(999,int(round(-10*y/math.log(10)))) for y in x['pl']);call['GP']=tuple(map(float,probs))
                call['DS']=tuple(float(sum(p*(int(g[0]==a)+int(g[1]==a)) for p,g in zip(probs,x['genotypes']))) for a in range(1,len(alleles)))
                reference_probability*=float(probs[0]);passing|=any(gt) and gq>=args.min_gq and confidence>=args.min_gp
            r.qual=min(999,-10*math.log10(max(1e-100,reference_probability)));r.filter.add('PASS' if passing else 'LowQual');out.write(r)
    if mode=='wz':pysam.tabix_index(args.output,preset='vcf',force=True)
    elif mode=='wb':pysam.bcftools.index('-f',args.output)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True);p.add_argument('--region',required=True);p.add_argument('--bam',action='append',required=True,help='SAMPLE=path; repeat for distinct samples or libraries')
    p.add_argument('--panel');p.add_argument('--exclude-samples',default='');p.add_argument('--output',required=True);p.add_argument('--states',type=int,default=16);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--panel-reader',choices=['fast','python'],default='fast',help='Equivalent HTSlib bulk reader or reference Python reader')
    p.add_argument('--state-selection',choices=['marginal','paired'],default='marginal')
    p.add_argument('--local-alleles',choices=['joint','conservative'],default='joint',help='Joint compatible local multiallelic model, or old overlap-cluster exclusion ablation')
    p.add_argument('--max-local-alleles',type=int,default=12)
    p.add_argument('--genetic-map',help='Supplied map with header pos chr cM, 1-based positions')
    p.add_argument('--ne',type=float,default=100000,help='Li-Stephens effective-size model parameter when a genetic map is supplied')
    p.add_argument('--no-read-index',action='store_true',help='Computational ablation: scan all regional reads per indel')
    p.add_argument('--read-model',choices=['iid','mixture'],default='mixture',help='IID read errors or local shared artifact plus clean-state allele-balance mixture')
    p.add_argument('--artifact-prior',type=float,default=.01,help='Shared artifact-state prior in mixture model; zero retains only the clean allele-balance mixture')
    p.add_argument('--indel-artifact-prior',type=float,help='Optional separate artifact prior for local blocks containing indels')
    p.add_argument('--cohort-iterations',type=int,choices=[1,2],default=1,help='1 external only; 2 adds one cross-fitted leave-self-out cohort pass, no cyclic feedback')
    p.add_argument('--max-cohort-donors',type=int,default=8,help='Maximum other individuals selected by read compatibility; also bounded by 64 total haplotype states')
    p.add_argument('--min-support',type=int,default=2);p.add_argument('--min-bq',type=int,default=20);p.add_argument('--min-mq',type=int,default=10);p.add_argument('--max-indel',type=int,default=50);p.add_argument('--min-gp',type=float,default=.9);p.add_argument('--min-gq',type=int,default=20);p.add_argument('--variants-only',action='store_true');p.add_argument('--no-ld',action='store_true')
    p.add_argument('--no-fragment-blocks',action='store_true',help='Ablation: factor likelihood independently per site')
    p.add_argument('--context-read-mode',choices=['enclosing','partial'],default='partial')
    p.add_argument('--context-gap',type=int,default=30,help='Merge nearby retained indel/SNV groups within this reference gap')
    p.add_argument('--context-span',type=int,default=100,help='Bound total jointly modeled reference span')
    args=p.parse_args();start_time=time.monotonic()
    source_files=[HERE/'caller.py',HERE/'blocks.py',HERE/'core/core.cpp',HERE/'build.sh',HERE/'state_selection.py',HERE/'input_identity.py',HERE/'read_index.py']
    if args.panel_reader=='fast':source_files += [HERE/'panel_fast.py',HERE/'core/panel_io.cpp',HERE/'build_panel.sh']
    if args.read_model=='mixture' or args.local_alleles=='joint':source_files += [HERE/'artifact.py']
    if args.local_alleles=='joint':source_files += [HERE/'local_alleles.py']
    if args.genetic_map:source_files += [HERE/'genetic_map.py']
    if args.local_alleles=='joint':source_files += [HERE/'context_blocks.py']
    source_hashes={str(x.relative_to(HERE)):hashlib.sha256(x.read_bytes()).hexdigest() for x in source_files}
    try:args.chrom,bounds=args.region.rsplit(':',1);start,end=map(int,bounds.split('-'));start-=1
    except ValueError:p.error('--region must be CHROM:START-END, 1-based inclusive')
    if not 1<=args.states<=64 or args.threads<1 or args.max_cohort_donors<0:p.error('states must be 1..64, threads >=1 and max-cohort-donors >=0')
    if not 0<=args.min_gp<=1 or args.min_gq<0 or args.min_support<1 or not 1<=args.max_indel<=50:p.error('Invalid probability, GQ, support or indel length threshold')
    if args.context_gap<0 or args.context_span<50:p.error('context-gap must be nonnegative; context-span must be at least50bp')
    if not 2<=args.max_local_alleles<=16:p.error('max-local-alleles must be 2..16')
    if not 0<=args.artifact_prior<1:p.error('Artifact prior must be in [0,1)')
    if args.indel_artifact_prior is not None and not 0<=args.indel_artifact_prior<1:p.error('Indel artifact prior must be in [0,1)')
    if not math.isfinite(args.ne) or args.ne<=0:p.error('Ne must be finite and positive')
    if not 0<=args.min_bq<=60 or not 0<=args.min_mq<=60:p.error('BQ and MQ thresholds must be 0..60')
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    fa=pysam.FastaFile(args.reference);fetch=lambda a,b:fa.fetch(args.chrom,max(0,a),b)
    if args.chrom not in fa.references:p.error('Region contig absent from FASTA: '+args.chrom)
    if start<0 or end<=start or end>fa.get_reference_length(args.chrom):p.error('Region outside FASTA bounds or empty')
    bams=collections.defaultdict(list)
    for val in args.bam:
        if '=' not in val:p.error('--bam requires SAMPLE=path')
        name,path=val.split('=',1)
        if not name or any(c.isspace() for c in name) or not path:p.error('Invalid BAM sample label/path')
        bams[name].append(path)
    input_identity={path:file_identity(path) for path in {args.reference,*([args.panel] if args.panel else []),*([args.genetic_map] if args.genetic_map else []),*(path for paths in bams.values() for path in paths)}}
    raw={};excluded=set(args.exclude_samples.split(','))|set(bams);headers={};sources=collections.Counter();fragment_seeds=collections.defaultdict(set);seen_sms=set();seen_paths=set()
    for name,paths in bams.items():
        current_paths={str(Path(x).resolve()) for x in paths}
        if seen_paths&current_paths:raise ValueError('Same BAM assigned to multiple sample labels; combine libraries under one name')
        seen_paths|=current_paths
        raw[name],sms,headers[name]=collect_reads(paths,args.reference,args.chrom,start,end,args.min_mq)
        if seen_sms&sms:raise ValueError('BAM SM identifiers overlap between different sample labels; possible duplicate individual')
        seen_sms|=sms;excluded|=sms
        sup=discover(raw[name],fetch,start,end,args.min_bq,args.max_indel)
        for v,s in sup.items():
            sources[v]+=len(s);fragment_seeds[v].update((name,*fragment) for fragment in s)
    for family in FAMILIES:
        if family&excluded:excluded|=family
    reader=load_panel
    if args.panel_reader=='fast':
        from panel_fast import load_panel_fast
        reader=load_panel_fast
    panel,panel_names,skips=reader(args.panel,args.chrom,start,end,fetch,excluded,args.max_indel)
    actual_panel_exclusions=[]
    if args.panel:
        with pysam.VariantFile(args.panel) as f:actual_panel_exclusions=sorted(set(f.header.samples)&excluded)
    read_sources={v:n for v,n in sources.items() if n>=args.min_support}
    candidates=sorted(set(panel)|set(read_sources));local_blocks=None
    if args.local_alleles=='joint':
        from local_alleles import supported_candidates
        from context_blocks import make_context_blocks
        variants,rejected=supported_candidates(candidates,read_sources,panel,args.max_local_alleles)
        local_blocks=make_context_blocks(variants,fetch,context_gap=args.context_gap,context_span=args.context_span,partial_reads=args.context_read_mode=='partial',max_distance=0 if args.no_fragment_blocks else 500,panel=panel,support=read_sources,fragment_seeds=fragment_seeds,max_alleles=args.max_local_alleles)
    else:variants,rejected=nonoverlapping_candidates(candidates)
    lib=core();observations={};initial={};sequence_data={}
    from read_index import ReadIndex
    indexed=raw if args.no_read_index else {name:ReadIndex(reads) for name,reads in raw.items()}
    for name in bams:
        if not variants:
            observations[name]=(np.zeros((0,3)),np.zeros(0,dtype=int),np.zeros((0,2),dtype=int),np.zeros(0,dtype=int),{})
            initial[name]=(np.zeros((0,3)),np.zeros((0,2)),{'states':0,'empty_interval':True})
            continue
        print(f'{name}: likelihoods for {len(variants)} candidates',flush=True)
        observations[name]=likelihoods(variants,indexed[name],fetch,args.min_bq,lib)
        sequence_data[name]={}
        if local_blocks is not None:
            from local_alleles import sequence_likelihoods_many
            indices=[t for t,block in enumerate(local_blocks) if block['kind']=='multiallelic']
            sequence_data[name]=dict(zip(indices,sequence_likelihoods_many([local_blocks[t] for t in indices],indexed[name],fetch,args.min_bq,lib)))
        if args.read_model=='mixture':
            from artifact import site_likelihoods
            obs=observations[name];observations[name]=(site_likelihoods(variants,obs[4],args.artifact_prior,args.indel_artifact_prior),*obs[1:])
        initial[name]=infer(lib,variants,panel,observations[name][0],args.states,no_ld=args.no_ld,fragment_observations=None if args.no_fragment_blocks and local_blocks is None else observations[name][4],artifact_prior=args.artifact_prior if args.read_model=='mixture' else None,state_selection=args.state_selection,local_blocks=local_blocks,sequence_data=sequence_data[name],genetic_map=args.genetic_map,ne=args.ne,chrom=args.chrom,indel_artifact_prior=args.indel_artifact_prior)
    results=initial.copy()
    selected_cohort={name:[] for name in bams}
    if variants and len(bams)>1 and args.cohort_iterations==2:
        # Cavity approximation: donor estimates use donor reads + external panel
        # ONLY. Never feed target results back into donors, or multiply posteriors.
        for name in bams:
            l=np.exp(observations[name][0]);rank=[]
            for other in bams:
                if other==name:continue
                h=initial[other][1]
                compatible=(1-h)*(l[:,0]+l[:,1])[:,None]/2+h*(l[:,1]+l[:,2])[:,None]/2
                rank.append((float(np.log(np.maximum(1e-300,compatible)).sum()),other))
            maximum=min(args.max_cohort_donors,(64-initial[name][2]['states'])//2)
            names=[other for _,other in sorted(rank,key=lambda x:(-x[0],x[1]))[:maximum]]
            selected_cohort[name]=names;donor=[initial[other][1] for other in names]
            results[name]=infer(lib,variants,panel,observations[name][0],args.states,cohort=donor,no_ld=args.no_ld,fragment_observations=None if args.no_fragment_blocks and local_blocks is None else observations[name][4],artifact_prior=args.artifact_prior if args.read_model=='mixture' else None,state_selection=args.state_selection,local_blocks=local_blocks,sequence_data=sequence_data[name],genetic_map=args.genetic_map,ne=args.ne,chrom=args.chrom,indel_artifact_prior=args.indel_artifact_prior)
    emit(args,variants,panel,read_sources,results,observations,fa.get_reference_length(args.chrom))
    warnings=[]
    for name,items in headers.items():
        if any(not h.get('RG') or any(not r.get('LB') for r in h.get('RG',[])) for h in items):warnings.append(name+': missing LB/RG metadata; fragment namespace falls back to input file plus RG ID; duplicate fragments across differently named files cannot be identified automatically')
    limitations=[
        'CIGAR seed discovery and bounded compatible edits, not de novo assembly from soft-clipped reads',
        'Fragment phase is integrated inside local blocks; fragment origin is factorized across blocks',
        'Joint blocks merge retained nearby SNV/indel; flanks are clipped before other retained candidates. Unknown or pruned variants and cross-block fragment factorization remain approximate',
        'Cross-fitted donor likelihoods exclude target reads conditional on a shared candidate catalog; catalog ascertainment uses all batch read support',
        'Donor phase uncertainty is approximated conditional on a modal block state; no cyclic genotype updates',
        'Local sequence enumeration is bounded; excluded true candidates/alleles remain false negatives in external evaluation',
        'Partial reads marginalize hidden retained allele contexts uniformly within visible-marker classes; this is a bounded approximate nuisance-context model, not an exact global read likelihood',
        'No impute-first remapping, pangenome placement integration or structural variant model',
        'Site QUAL is a product of sample marginal reference probabilities, not exact joint cohort probability',
        'Region implementation retains reads and panel in RAM; biobank-scale performance is not demonstrated']
    if args.local_alleles=='conservative':limitations.append('Conservative ablation excludes overlapping candidate clusters')
    if not args.genetic_map:limitations.append('Uniform physical-distance recombination approximation used; no map supplied')
    manifest={
        'version':VERSION,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arguments':vars(args),
        'input_identity':input_identity,'actual_panel_exclusions':actual_panel_exclusions,'metadata_warnings':warnings,
        'wall_seconds':time.monotonic()-start_time,'candidates':len(variants),'panel_candidates':len(panel),
        'read_candidates':len(read_sources),'panel_samples':len(panel_names),'excluded_panel_samples':sorted(excluded),
        'panel_skips':skips,'rejected_candidates':len(rejected),'candidate_exclusion_policy':args.local_alleles,
        'candidate_list_before_complexity_filter':len(variants)+len(rejected),
        'samples':{s:{'read_records':len(raw[s]),'inference':{k:v for k,v in results[s][2].items() if k!='joint_records'},'cohort_donors':selected_cohort[s]} for s in bams},
        'algorithm':'exact diploid block-state/genotype forward-backward conditional on selected haplotypes and shared catalog; approximate cross-fitted cohort donors and local read factorization',
        'limitations':limitations,'source_sha256':source_hashes,'core_binary':file_identity(lib._name),
        'software':{'python':sys.version,'numpy':np.__version__,'pysam':pysam.__version__,'pysam_samtools':pysam.__samtools_version__,'platform':platform.platform()}}
    Path(args.output+'.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'output':args.output,'candidates':len(variants),'seconds':manifest['wall_seconds']}))
if __name__=='__main__':main()
