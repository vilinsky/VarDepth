"""Candidate-sequence projections with repeat-safe common coordinate anchors."""
import ctypes
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
from native_build import build_library
_LIB=None

class SafeAnchors(np.ndarray):
    """Immutable per-locus coordinate index reused for every read."""
    pass


def project(reference,alternate):
    if reference==alternate:return np.arange(len(reference),dtype=np.int32),[]
    global _LIB
    if _LIB is None:
        _LIB=ctypes.CDLL(str(build_library(HERE/'project.cpp')))
        _LIB.candidate_projection.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p]
        _LIB.candidate_projection.restype=ctypes.c_int
    ops=ctypes.create_string_buffer(len(reference)+len(alternate)+1)
    n=_LIB.candidate_projection(reference.encode(),len(reference),alternate.encode(),len(alternate),ops)
    if n<0:raise ValueError('Candidate sequence outside bounded projection size')
    mapping=np.full(len(reference),-1,np.int32);events=[];i=j=k=0;trace=ops.raw[:n].decode()
    while k<n:
        op=trace[k];stop=k+1
        while stop<n and trace[stop]==op:stop+=1
        size=stop-k
        if op=='M':mapping[i:i+size]=np.arange(j,j+size);i+=size;j+=size
        elif op=='D':events.append((i,reference[i:i+size],''));i+=size
        else:events.append((i,'',alternate[j:j+size]));j+=size
        k=stop
    return mapping,events


def repeat_tract(position,deleted,inserted,fetch,limit=512):
    motif=deleted or inserted;left=position;right=position;rot=motif
    for step in range(limit):
        if left<=0 or fetch(left-1,left)!=rot[-1]:break
        left-=1;rot=rot[-1]+rot[:-1]
    rot=motif
    for step in range(limit):
        if fetch(right+len(deleted),right+len(deleted)+1)!=rot[0]:break
        right+=1;rot=rot[1:]+rot[0]
    return max(0,left-1),right+len(deleted)+1


def common_anchors(start,reference,alleles,fetch,anchor=3):
    maps=[];unsafe=np.zeros(len(reference),bool)
    for allele in alleles:
        mapping,events=project(reference,allele);maps.append(mapping)
        for p,r,a in events:
            left,right=repeat_tract(start+p,r,a,fetch)
            unsafe[max(0,left-start):min(len(reference),right-start)]=True
    maps=np.stack(maps);safe=np.zeros(len(reference),bool)
    count=len(reference)-anchor+1
    if count>0:
        first=maps[:,:count];valid=(first>=0).all(axis=0)
        for offset in range(anchor):
            valid &= ~unsafe[offset:offset+count]
            valid &= (maps[:,offset:offset+count]==first+offset).all(axis=0)
        safe[:count]=valid
    safe=safe.view(SafeAnchors);safe.positions=np.flatnonzero(safe).tolist()
    try:
        from pysam_observations import Positions
    except ModuleNotFoundError as error:
        if error.name!='pysam_observations':raise
    else:safe.native_positions=Positions(safe.positions)
    safe.flags.writeable=False
    return maps,safe


def expand_compound_context(start,alleles,fetch):
    """Find repeats inside compound SNP+indel alleles before choosing anchors.

    Looking only at the net minimal edit misses a pure indel accompanied by a
    nearby SNP. All added flanks are common to every candidate, preserving the
    ordered biological allele catalog after common-flank trimming.
    """
    left=start;right=start+len(alleles[0]);reference=alleles[0]
    for allele in alleles[1:]:
        mapping,events=project(reference,allele)
        for p,r,a in events:
            lo,hi=repeat_tract(start+p,r,a,fetch)
            left=min(left,lo);right=max(right,hi)
    if right-left>4000:raise ValueError('Compound context exceeds bounded projection window')
    prefix=fetch(left,start);suffix=fetch(start+len(reference),right)
    return left,[prefix+a+suffix for a in alleles]


def crop_observation(read,start,reference,alleles,maps,safe,anchor=3):
    """Both read and every candidate retain the same two genomic anchors."""
    from alignment_queries import outer_safe_anchors,unusual
    if unusual(read):
        aligned={p-start:q for q,p in read.get_aligned_pairs(matches_only=True) if start<=p<start+len(reference)}
        possible=[int(p) for p in np.flatnonzero(safe) if p in aligned and all(aligned.get(p+j)==aligned[p]+j for j in range(anchor))]
        if len(possible)<2:return None
        left=possible[0];right=possible[-1]+anchor;q0=aligned[left];q1=aligned[right-1]+1
    else:
        positions=safe.positions if isinstance(safe,SafeAnchors) else np.flatnonzero(safe).tolist()
        if hasattr(safe,'native_positions'):
            left,right,q0,q1=outer_safe_anchors(read,start,positions,anchor,safe.native_positions)
        else:left,right,q0,q1=outer_safe_anchors(read,start,positions,anchor)
        if left is None or right-anchor<=left:return None
    # Globally anchored short indels can have only one inner reference base.
    # Do not discard them using the minimum length of a semiglobal aligner.
    if right-left<2*anchor+1:return None
    if q1-q0<2*anchor+1:return None
    sequences=[a[m[left]:m[right-1]+1] for a,m in zip(alleles,maps)]
    if len(set(sequences))<2:return None
    return dict(left=start+left,right=start+right,sequence=read.query_sequence[q0:q1].upper(),
                qualities=bytes(read.query_qualities[q0:q1]),haplotypes=sequences,
                whole_context=left==0 and right==len(reference))
