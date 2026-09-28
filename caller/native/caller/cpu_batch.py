"""Batched exact-model CPU likelihoods, including anchored and whole-read modes."""
import ctypes
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
import numpy as np
from native_build import build_library
_CACHE=OrderedDict()
_CACHE_LIMIT=131072

@lru_cache(maxsize=1)
def library():
    lib=ctypes.CDLL(str(build_library(Path(__file__).with_name('batch_hmm.cpp'))))
    u=ctypes.POINTER(ctypes.c_ubyte);i=ctypes.POINTER(ctypes.c_int)
    lib.pairhmm_pairs.argtypes=[ctypes.POINTER(ctypes.c_char_p),ctypes.POINTER(u),i,
        ctypes.POINTER(ctypes.c_char_p),i,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double)]
    lib.pairhmm_pairs.restype=ctypes.c_int
    return lib

def pairs(jobs,mode=0,lanes=0,*,canonical=False):
    jobs=list(jobs);count=len(jobs)
    if not count:return np.empty(0,np.float64)
    # A likelihood is a pure function of sequence, quality, haplotype and mode.
    # Reusing it does not merge molecules or change downstream read weighting.
    quality={};unique={};missing=[];slots=[];result=np.empty(count,np.float64)
    for i,(s,q,h) in enumerate(jobs):
        if len(s)!=len(q):raise ValueError('Read and quality lengths differ')
        if id(q) not in quality:quality[id(q)]=q if isinstance(q,bytes) else np.ascontiguousarray(q,dtype=np.uint8).tobytes()
        key=(mode,lanes,s,quality[id(q)],h,canonical)
        if key in _CACHE:
            result[i]=_CACHE[key];_CACHE.move_to_end(key)
        else:
            if key not in unique:unique[key]=len(missing);missing.append((s,key[3],h))
            slots.append((i,unique[key]))
    if not missing:return result
    from trie_vector import score
    values=_uncached_pairs(missing,mode,lanes) if canonical else score(missing,mode,_uncached_pairs,lanes=lanes)
    for key,index in unique.items():
        _CACHE[key]=float(values[index])
        if len(_CACHE)>_CACHE_LIMIT:_CACHE.popitem(last=False)
    for i,j in slots:result[i]=values[j]
    return result

def _uncached_pairs(jobs,mode,lanes):
    count=len(jobs)
    # Equal-length groups minimize padding. Restore original read/allele order.
    order=sorted(range(count),key=lambda j:(len(jobs[j][0]),len(jobs[j][2])))
    encoded={s:s.encode('ascii') for j in jobs for s in (j[0],j[2])}
    quality={}
    for j in jobs:
        if len(j[0])!=len(j[1]):raise ValueError('Read and quality lengths differ')
        if id(j[1]) not in quality:quality[id(j[1])]=j[1] if isinstance(j[1],bytes) else np.ascontiguousarray(j[1],dtype=np.uint8).tobytes()
    reads=(ctypes.c_char_p*count)(*(encoded[jobs[i][0]] for i in order))
    haps=(ctypes.c_char_p*count)(*(encoded[jobs[i][2]] for i in order))
    u=ctypes.POINTER(ctypes.c_ubyte)
    qp=(ctypes.c_char_p*count)(*(quality[id(jobs[i][1])] for i in order))
    qs=ctypes.cast(qp,ctypes.POINTER(u))
    rn=np.array([len(jobs[i][0]) for i in order],np.int32);hn=np.array([len(jobs[i][2]) for i in order],np.int32)
    out=np.empty(count,np.float64);ptr=lambda a,t:a.ctypes.data_as(ctypes.POINTER(t))
    code=library().pairhmm_pairs(reads,qs,ptr(rn,ctypes.c_int),haps,ptr(hn,ctypes.c_int),count,mode,lanes,ptr(out,ctypes.c_double))
    if code:raise RuntimeError('CPU PairHMM failed: '+str(code))
    result=np.empty_like(out);result[order]=out;return result

def many(sequences,qualities,haplotypes,mode=0):
    if len(sequences)!=len(qualities):raise ValueError('Read batch lengths differ')
    return pairs([(s,q,h) for s,q in zip(sequences,qualities) for h in haplotypes],mode).reshape(len(sequences),len(haplotypes))
