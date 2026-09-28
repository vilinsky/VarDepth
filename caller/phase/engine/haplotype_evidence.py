"""Full-spanning molecular likelihoods for competing local sequence haplotypes.

Reference-coordinate endpoints are expanded past equivalent pure-indel shifts.
The original BAM CIGAR only extracts sequence; all competing haplotypes are
scored with the same global PairHMM. No competitor code or truth is used.
"""
import ctypes,math
import numpy as np
from indel_refine import repeat_bounds,kernel

def interval(record,fetch,anchor=10):
    descriptor=dict(start0=record.start,end0=record.stop,evidence_start0=record.start,evidence_end0=record.stop,alleles=list(record.alleles))
    lo,hi=repeat_bounds(descriptor,fetch,limit=256)
    return max(0,lo-anchor),hi+anchor

def segment(read,lo,hi):
    if read.is_unmapped or read.reference_end is None or read.query_sequence is None:return None
    if read.reference_start>lo or read.reference_end<hi or read.query_qualities is None:return None
    if read.mapping_quality<30 or read.mapping_quality==255 or read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate or read.is_qcfail:return None
    if read.is_paired and not read.is_proper_pair:return None
    # Require five aligned endpoint bases; interior gaps may take any placement.
    wanted=set(range(lo,lo+5))|set(range(hi-5,hi));q=0;r=read.reference_start;aligned={}
    for op,n in read.cigartuples or []:
        if op in (0,7,8):
            for x in wanted:
                if r<=x<r+n:aligned[x]=q+x-r
            q+=n;r+=n
        elif op in (1,4):q+=n
        elif op in (2,3):r+=n
    if len(aligned)!=len(wanted):return None
    left=[aligned[x] for x in range(lo,lo+5)];right=[aligned[x] for x in range(hi-5,hi)]
    if left!=list(range(left[0],left[0]+5)) or right!=list(range(right[0],right[0]+5)):return None
    a,b=left[0],right[-1]+1;seq=read.query_sequence[a:b].upper();quals=bytes(read.query_qualities[a:b])
    if not seq or set(seq)-set('ACGT') or sum(quals)/len(quals)<25:return None
    return seq,quals

class Scorer:
    def __init__(self):self.lib=kernel()
    def score(self,sequence,qualities,haplotypes):
        q=(ctypes.c_ubyte*len(qualities))(*qualities);seq=sequence.encode();values=[]
        for h in haplotypes:
            encoded=h.encode();values.append(self.lib.anchored_pairhmm(seq,q,len(seq),encoded,len(encoded),.001,.1))
        return np.asarray(values)

def observations(record,bam,reference,alleles=None,anchor=10,scorer=None,max_span=180):
    scorer=scorer or Scorer();fetch=lambda a,b:reference.fetch(record.contig,max(0,a),b).upper()
    lo,hi=interval(record,fetch,anchor)
    if hi-lo>max_span or hi-lo<10:return [],dict(reason='outside_spanning_scope',span=hi-lo)
    indices=list(range(len(record.alleles))) if alleles is None else list(alleles)
    haps=[fetch(lo,record.start)+record.alleles[i]+fetch(record.stop,hi) for i in indices]
    rg={r['ID']:r.get('LB',r['ID']) for r in bam.header.to_dict().get('RG',[])};molecules={}
    for read in bam.fetch(record.contig,lo,hi):
        observed=segment(read,lo,hi)
        if observed is None:continue
        sequence,qualities=observed;ll=scorer.score(sequence,qualities,haps)
        if not np.isfinite(ll).all():continue
        order=np.argsort(ll);best=int(order[-1]);gap=float(ll[best]-ll[order[-2]]) if len(ll)>1 else 0
        library=rg.get(read.get_tag('RG'),'unknown') if read.has_tag('RG') else 'unknown';key=(library,read.query_name)
        row=dict(ll=ll,reverse=bool(read.is_reverse),mq=read.mapping_quality,best=best,gap=gap,key=key)
        # Preserve conflicts rather than selecting the mate most compatible with a call.
        if key not in molecules:molecules[key]=row
        elif molecules[key] is not None:
            old=molecules[key]
            if old['best']!=best and min(old['gap'],gap)>=math.log(100):molecules[key]=None
            elif gap>old['gap']:molecules[key]=row
    return [r for r in molecules.values() if r is not None],dict(span=hi-lo,start=lo,end=hi,alleles=indices,discordant=sum(r is None for r in molecules.values()))
