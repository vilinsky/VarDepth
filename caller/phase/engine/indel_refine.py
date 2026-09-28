"""Read-derived local haplotypes and anchored indel evidence; no truth inputs.

This is a bounded spanning-read haplotype proposer, not full de novo assembly or
genome remapping. Original alleles are retained; at most 12 consensus proposals
are added. Each proposal requires >=2 distinct quality-filtered molecules.
"""
from reference_cache import CachedReference
import argparse,collections,ctypes,hashlib,json,pathlib,subprocess,sys,time
import numpy as np
import pysam
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'base'))
from local_alleles import likelihood_matrix
from local_alleles import minimal_edit
from native_build import build_library
from prepare import sha
from caller import normalize_variant, file_identity

def fragment_seeded_haplotypes(record,bams,fetch,reference=None):
    """Propose compatible edits linked by a physical fragment, including mates.

    Only bases/indel anchors passing BQ20 and primary MQ20 alignments seed a
    proposal. Two distinct molecules must independently encode a pattern.
    Unobserved positions are proposals on the reference background, not calls:
    the independent spanning-read likelihood and its support gates decide GT.
    """
    from caller import discover,valid_read
    from local_alleles import compatible_edits,apply_edits
    counts=collections.Counter();start=record['start0'];end=record['end0'];ref=fetch(start,end)
    for sample,paths in bams.items():
        reads=[]
        for path in paths:
            with pysam.AlignmentFile(path,reference_filename=reference) as bam:
                rg={r['ID']:r.get('LB',str(pathlib.Path(path).resolve())+'::'+r['ID']) for r in bam.header.to_dict().get('RG',[])}
                for read in bam.fetch(record['chrom'],max(0,start-3),end+3):
                    if not valid_read(read,20) or read.query_qualities is None:continue
                    fallback=str(pathlib.Path(path).resolve())+'::unknown'
                    library=rg.get(read.get_tag('RG'),fallback) if read.has_tag('RG') else fallback
                    reads.append((library,read))
        support=discover(reads,fetch,start,end,20,50);patterns=collections.defaultdict(list)
        for variant,molecules in support.items():
            edit=minimal_edit(variant)
            if len(molecules)<2 or edit[0]<start or edit[0]+len(edit[1])>end:continue
            for key in molecules:patterns[key].append(edit)
        local=collections.Counter()
        for edits in patterns.values():
            edits=sorted(set(edits))
            if not any(len(r)!=len(a) for _,r,a in edits) or not compatible_edits(edits):continue
            seq=apply_edits(ref,start,edits)
            if seq and seq!=ref and abs(len(seq)-len(ref))<=50:local[seq]+=1
        for seq,n in local.items():
            if n>=2:counts[seq]+=n
    return counts

def kernel():
    binary=build_library(HERE/'anchored.cpp')
    lib=ctypes.CDLL(str(binary));lib.anchored_pairhmm.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_ubyte),ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_double,ctypes.c_double]
    lib.anchored_pairhmm.restype=ctypes.c_double;return lib

def observations(bams,chrom,start,end,fetch,anchor=3,reference=None):
    from evidence_trace import enabled
    tracing=enabled()
    from alignment_queries import target_queries
    from bam_cache import open_cached
    left=fetch(start-anchor,start);right=fetch(end,end+anchor);result={}
    anchor_positions=list(range(start-anchor,start))+list(range(end,end+anchor))
    if len(left)!=anchor or len(right)!=anchor:return []
    for path in bams:
        bam=open_cached(path,reference);resolved=str(pathlib.Path(path).resolve());fallback=resolved+'::unknown'
        rg={r['ID']:r.get('LB',resolved+'::'+r['ID']) for r in bam.header.to_dict().get('RG',[])}
        for read in bam.fetch(chrom,max(0,start-anchor),end+anchor):
            if read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate or read.is_qcfail or not 10<=read.mapping_quality<255 or read.query_qualities is None:continue
            if read.reference_start>start-anchor or read.reference_end<end+anchor:continue
            aligned={p:q for q,p in target_queries(read,anchor_positions)}
            qs=[aligned.get(p) for p in anchor_positions]
            if any(q is None for q in qs) or qs[:anchor]!=list(range(qs[0],qs[0]+anchor)) or qs[anchor:]!=list(range(qs[anchor],qs[anchor]+anchor)):continue
            sequence=read.query_sequence.upper();q0=qs[0];q1=qs[-1]+1
            read_qualities=read.query_qualities
            # Anchor coordinates, not reference bases: requiring a reference
            # match here selectively removes haplotypes carrying flank SNPs.
            seq=sequence[q0:q1];qualities=bytes(read_qualities[q0:q1])
            if any(b not in 'ACGT' for b in seq) or not qualities or (sum(qualities)/len(qualities))<20:continue
            inner=seq[anchor:-anchor]
            if not inner:continue
            lib=rg.get(read.get_tag('RG'),fallback) if read.has_tag('RG') else fallback
            key=(lib,read.query_name);entry=dict(sequence=seq,inner=inner,qualities=qualities,mq=read.mapping_quality,reverse=read.is_reverse,proposal_quality=min(qualities)>=20)
            if tracing:entry['_molecule_key']=key
            if key not in result or (sum(qualities)/len(qualities))>(sum(result[key]['qualities'])/len(result[key]['qualities'])):result[key]=entry
    return [result[k] for k in sorted(result)]

def repeat_bounds(record,fetch,limit=512):
    """Move anchors outside every equivalent pure-indel placement.

    A coordinate anchor inside a repeat can trim away a right-shifted insertion
    and create false reference support. Bounds are derived from candidate and
    reference sequence only, independently of BAM CIGAR placement.
    """
    start=min(record['start0'],record['evidence_start0']);end=max(record['end0'],record['evidence_end0'])
    for alt in record['alleles'][1:]:
        p,ref,seq=minimal_edit((record['start0'],record['alleles'][0],alt))
        if bool(ref)==bool(seq):continue
        motif=ref or seq;left=p;rot=motif;steps=0
        while left>0 and steps<limit and fetch(left-1,left)==rot[-1]:
            left-=1;rot=rot[-1]+rot[:-1];steps+=1
        right=p;rot=motif;steps=0
        while steps<limit and fetch(right+len(ref),right+len(ref)+1)==rot[0]:
            right+=1;rot=rot[1:]+rot[0];steps+=1
        # Include a nonrepeat reference base on both sides of the edit.
        start=min(start,max(0,left-1));end=max(end,right+len(ref)+1)
    return start,end

def owns_haplotype(record,start,ref,alt,fetch):
    if alt==ref:return True
    p,r,a=normalize_variant(start,ref,alt,fetch)
    return record['start0']<=p and p+len(r)<=record['end0']

def _refine_record_steps(record,bams,fetch,lib,reference=None,fragment_proposals=False):
    start,end=repeat_bounds(record,fetch)
    expanded=[fetch(start,record['start0'])+a+fetch(record['end0'],end) for a in record['alleles']]
    if expanded[0]!=fetch(start,end):raise ValueError('Indel REF mismatch')
    observed={s:observations(paths,record['chrom'],start,end,fetch,reference=reference) for s,paths in bams.items()}
    counts={s:collections.Counter(o['inner'] for o in obs if o['proposal_quality']) for s,obs in observed.items()}
    candidates=collections.Counter()
    def owned(seq):
        return owns_haplotype(record,start,expanded[0],seq,fetch)
    # Evidence from each individual gets a quota before pooling. The caller
    # never knows which sequences are true and never uses a truth-driven locus list.
    for sample,c in counts.items():
        for seq,n in c.most_common(4):
            if n>=2 and seq not in expanded and abs(len(seq)-len(expanded[0]))<=100 and owned(seq):candidates[seq]+=n
    fragment_candidates={}
    if fragment_proposals:
        for seq,n in fragment_seeded_haplotypes(record,bams,fetch,reference).items():
            expanded_seq=fetch(start,record['start0'])+seq+fetch(record['end0'],end)
            if expanded_seq not in expanded and owned(expanded_seq):
                candidates[expanded_seq]+=n;fragment_candidates[expanded_seq]=n
    additions=[seq for seq,n in sorted(candidates.items(),key=lambda x:(-x[1],x[0]))[:12]]
    alleles=expanded+additions;n=len(alleles);order=[(a,b) for b in range(n) for a in range(b+1)]
    af=np.array([max(.001,float(a or 0)) for a in record['panel_af']]+[.001]*len(additions));af/=af.sum()
    prior=np.array([af[a]*af[b]*(1 if a==b else 2) for a,b in order]);prior/=prior.sum()
    samples={}
    for sample,obs in observed.items():
        ll=[];exact=np.zeros(n,int);left=fetch(start-3,start);right=fetch(end,end+3)
        from accelerator import backend
        gpu_likelihoods=None
        if backend()!='cpu' and obs:
            jobs=[(o['sequence'],o['qualities'],left+allele+right) for o in obs for allele in alleles]
            gpu_likelihoods=iter((yield jobs))
        elif obs:
            from cpu_batch import pairs
            jobs=[(o['sequence'],o['qualities'],left+allele+right) for o in obs for allele in alleles]
            gpu_likelihoods=iter(pairs(jobs,mode=1))
        batch_values=None
        if gpu_likelihoods is not None:
            from molecule_rows import normalized_rows
            likelihood_rows=np.array([next(gpu_likelihoods) for _ in range(len(obs)*n)]).reshape(len(obs),n)
            batch_values,_=normalized_rows(likelihood_rows,[o['mq'] for o in obs])
        for row_index,o in enumerate(obs):
            if batch_values is not None:v=batch_values[row_index]
            else:
                q=(ctypes.c_ubyte*len(o['qualities']))(*o['qualities']);seq=o['sequence'].encode();values=[]
                for allele in alleles:
                    hap=(left+allele+right).encode();values.append(lib.anchored_pairhmm(seq,q,len(seq),hap,len(hap),.001,.1))
                values=np.array(values);v=np.exp(values-values.max());error=10**(-min(o['mq'],60)/10);v=(1-error)*v+error*v.mean()
            ll.append(v)
            if o['inner'] in alleles:exact[alleles.index(o['inner'])]+=1
        matrix=np.array(ll) if ll else np.empty((0,n));logl=likelihood_matrix(matrix,.001)
        gl=np.array([logl[a,b] for a,b in order]);lp=gl+np.log(prior);post=np.exp(lp-lp.max());post/=post.sum()
        best=int(np.argmax(post));a,b=order[best];support=sum(exact[j] for j in set((a,b)))
        trusted=len(obs)>=6 and post[best]>=.9 and all(exact[j]>=2 and owned(alleles[j]) for j in set((a,b))) and support>=.7*len(obs)
        samples[sample]=dict(gp=post.tolist(),pl=(-10/np.log(10)*(gl-gl.max())).tolist(),dp=len(obs),exact_ad=exact.tolist(),
            proposed_sequences={k:int(v) for k,v in counts[sample].items() if v>=2},trusted=bool(trusted),support_fraction=float(support/max(1,len(obs))))
        from evidence_trace import enabled, molecules
        if enabled():
            samples[sample]['lineage_molecules']=molecules(sample,[o['_molecule_key'] for o in obs])
            samples[sample]['lineage_kind']='spanning_anchored'
    return dict(record_id=record['record_id'],chrom=record['chrom'],start0=start,end0=end,alleles=alleles,
        original_allele_count=len(expanded),added_alleles=len(additions),samples=samples,
        fragment_proposals=fragment_candidates if fragment_proposals else {})

def refine_record(record,bams,fetch,lib,reference=None,fragment_proposals=False):
    from accelerator import score_pairs
    from batch_coroutines import score_generators
    return score_generators([_refine_record_steps(record,bams,fetch,lib,reference,fragment_proposals)],
                            lambda jobs:score_pairs(jobs,anchored=True))[0]


def refinement_eligible(record,depth):
    # Compound SNP haplotypes need the same read-backed reconstruction as
    # indels; a pruned catalog may no longer retain the initiating indel seed.
    alleles=record['alleles']
    return depth>=20 and (len(alleles[0])>1 or any(len(a)!=len(alleles[0]) for a in alleles))

def run(args):
    begun=time.monotonic();fa=CachedReference(args.reference);bams=collections.OrderedDict()
    for item in args.bam:
        s,p=item.split('=',1);bams.setdefault(s,[]).append(p)
    identities={p:file_identity(p)['sha256'] for paths in bams.values() for p in paths};reference_sha=file_identity(args.reference)['sha256']
    output=pathlib.Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);rows=[];lib=kernel();seen=set();pending=[]
    ownership=json.loads(pathlib.Path(args.owned_regions).read_text()) if getattr(args,'owned_regions',None) else None
    for directory in args.features:
        directory=pathlib.Path(directory);meta=json.load(open(directory/'metadata.json'))
        if set(meta['samples'])!=set(bams):raise ValueError('Feature and BAM samples disagree')
        original=collections.defaultdict(list)
        for entry in meta['arguments']['bam']:
            s,p=entry.split('=',1);original[s].append(meta['input_identity'][p]['sha256'])
        if any(sorted(identities[p] for p in bams[s])!=sorted(original[s]) for s in bams):raise ValueError('Feature BAM content differs from refinement BAM')
        if meta['input_identity'][meta['arguments']['reference']]['sha256']!=reference_sha:raise ValueError('Feature reference content differs')
        maxdp=collections.defaultdict(float)
        for shard in meta['shards']:
            if sha(directory/shard['path'])!=shard['sha256']:raise ValueError('Corrupted feature shard')
            with np.load(directory/shard['path'],allow_pickle=False) as z:
                for r,d in zip(z['record_index'],z['depth']):maxdp[int(r)]=max(maxdp[int(r)],float(d))
        owned_core=ownership[str(directory.resolve())] if ownership else None
        for record in meta['records']:
            if ownership:
                core=owned_core
                if record['chrom']!=core['chrom'] or not core['start0']<=record['start0']<core['end0']:continue
            if record['record_id'] in seen:raise ValueError('Duplicate feature record')
            seen.add(record['record_id'])
            if not refinement_eligible(record,maxdp[record['record_index']]):continue
            pending.append(record)
        print(directory.name,'indel candidates selected',len(pending),flush=True)
    from accelerator import score_pairs
    from batch_coroutines import score_generators
    def steps(record):
        chrom=record['chrom']
        fetch=lambda a,b:fa.fetch(chrom,max(0,a),min(b,fa.get_reference_length(chrom))).upper()
        return _refine_record_steps(record,bams,fetch,lib,args.reference,getattr(args,'fragment_proposals',False))
    rows=score_generators((steps(record) for record in pending),lambda jobs:score_pairs(jobs,anchored=True))
    fa.close()
    output.write_text(json.dumps(rows,separators=(',',':'))+'\n')
    manifest=dict(version='0.4-read-haplotypes',arguments=vars(args),records=len(rows),with_new_alleles=sum(r['added_alleles']>0 for r in rows),
        trusted_sample_experts=sum(s['trusted'] for r in rows for s in r['samples'].values()),seconds=time.monotonic()-begun,
        policy='All indel or compound SNP blocks with max sample DP >=20; spanning-read proposals and optional fragment-linked edits require two quality-filtered molecules; original alleles retained; globally anchored PairHMM',
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'indel_refine.py',HERE/'anchored.cpp']})
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--features',action='append',required=True);p.add_argument('--bam',action='append',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--fragment-proposals',action='store_true');p.add_argument('--owned-regions');run(p.parse_args(argv))


if __name__=='__main__':main()
