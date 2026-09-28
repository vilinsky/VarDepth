"""Experimental bounded-loss PairHMM with per-read likelihood-vector fallback."""
import ctypes,time
from functools import lru_cache
from pathlib import Path
import numpy as np
from native_build import build_library

@lru_cache(maxsize=1)
def library():
    lib=ctypes.CDLL(str(build_library(Path(__file__).with_name('sparse_pairhmm.cpp'))))
    u=ctypes.POINTER(ctypes.c_ubyte);i=ctypes.POINTER(ctypes.c_int);d=ctypes.POINTER(ctypes.c_double)
    lib.sparse_pairs.argtypes=[ctypes.POINTER(ctypes.c_char_p),ctypes.POINTER(u),i,ctypes.POINTER(ctypes.c_char_p),i,ctypes.c_int,ctypes.c_int,ctypes.c_double,d,d,ctypes.POINTER(ctypes.c_uint64)]
    lib.sparse_pairs.restype=ctypes.c_int
    return lib

def approximate(jobs,mode=0,threshold=1e-20):
    count=len(jobs);values=np.empty(count);lost=np.empty(count)
    if not count:return values,lost,0
    encoded={s:s.encode('ascii') for s,q,h in jobs for s in (s,h)}
    qualities=[np.ascontiguousarray(q,dtype=np.uint8) for s,q,h in jobs]
    u=ctypes.POINTER(ctypes.c_ubyte);i=ctypes.POINTER(ctypes.c_int);d=ctypes.POINTER(ctypes.c_double)
    reads=(ctypes.c_char_p*count)(*(encoded[s] for s,q,h in jobs));haps=(ctypes.c_char_p*count)(*(encoded[h] for s,q,h in jobs))
    qs=(u*count)(*(q.ctypes.data_as(u) for q in qualities))
    rn=np.array([len(s) for s,q,h in jobs],np.int32);hn=np.array([len(h) for s,q,h in jobs],np.int32);cells=ctypes.c_uint64()
    code=library().sparse_pairs(reads,qs,rn.ctypes.data_as(i),haps,hn.ctypes.data_as(i),count,mode,threshold,values.ctypes.data_as(d),lost.ctypes.data_as(d),ctypes.byref(cells))
    if code:raise RuntimeError('Sparse PairHMM failed: '+str(code))
    return values,lost,cells.value

def score(jobs,mode,dense,tolerance=1e-10):
    values,lost,cells=approximate(jobs,mode)
    groups={}
    for i,(s,q,h) in enumerate(jobs):groups.setdefault((s,np.asarray(q,np.uint8).tobytes()),[]).append(i)
    accepted=np.zeros(len(jobs),bool)
    for indices in groups.values():
        best=float(values[indices].max())
        # Nonnegative arithmetic forward-error allowance (not fast-math).
        length=max(len(jobs[i][0])+len(jobs[i][2]) for i in indices)
        rounding=16*length*np.finfo(float).eps
        if best>0 and rounding<tolerance:
            accepted[indices]=lost[indices]/best+rounding<tolerance
    out=np.full(len(jobs),-np.inf)
    positive=accepted&(values>0);out[positive]=np.log(values[positive])
    fallback=np.flatnonzero(~accepted)
    if len(fallback):out[fallback]=dense([jobs[i] for i in fallback],mode,0)
    return out,dict(pairs=len(jobs),fallback=len(fallback),cells=cells,full_cells=sum(len(s)*len(h) for s,q,h in jobs))
