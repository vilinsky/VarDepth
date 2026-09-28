"""SIMD over reads and lossless sharing of haplotype prefixes."""
import ctypes
from functools import lru_cache
from pathlib import Path
import numpy as np
from native_build import build_library

@lru_cache(maxsize=1)
def library():
    lib=ctypes.CDLL(str(build_library(Path(__file__).with_name('trie_vector.cpp'))))
    p=ctypes.POINTER;i=ctypes.c_int
    lib.trie_many.argtypes=[p(ctypes.c_char_p),p(p(ctypes.c_ubyte)),p(i),i,p(ctypes.c_char_p),p(i),i,i,i,p(ctypes.c_double),p(ctypes.c_uint64)]
    lib.trie_many.restype=i
    return lib

def score(jobs,mode,dense,lanes=0):
    if not jobs:return np.empty(0)
    grouped={};qualities={};encoded={}
    for index,(s,q,h) in enumerate(jobs):
        if len(s)!=len(q):raise ValueError('Read and quality lengths differ')
        if id(q) not in qualities:qualities[id(q)]=q if isinstance(q,bytes) else np.ascontiguousarray(q,np.uint8).tobytes()
        key=(s,qualities[id(q)])
        grouped.setdefault(key,{}).setdefault(h,[]).append(index)
        for seq in (s,h):
            if seq not in encoded:encoded[seq]=seq.encode('ascii')
    batches={}
    for key,haps in grouped.items():batches.setdefault(tuple(sorted(haps)),[]).append(key)
    result=np.empty(len(jobs));p=ctypes.POINTER;i=ctypes.c_int
    for haps,reads in batches.items():
        reads.sort(key=lambda x:len(x[0]));nr=len(reads);nh=len(haps)
        rp=(ctypes.c_char_p*nr)(*(encoded[s] for s,q in reads));qp=(ctypes.c_char_p*nr)(*(q for s,q in reads))
        hp=(ctypes.c_char_p*nh)(*(encoded[h] for h in haps))
        rn=np.array([len(s) for s,q in reads],np.int32);hn=np.array([len(h) for h in haps],np.int32)
        out=np.empty((nr,nh));cells=ctypes.c_uint64()
        code=library().trie_many(rp,ctypes.cast(qp,p(p(ctypes.c_ubyte))),rn.ctypes.data_as(p(i)),nr,hp,hn.ctypes.data_as(p(i)),nh,mode,lanes,out.ctypes.data_as(p(ctypes.c_double)),ctypes.byref(cells))
        if code:raise RuntimeError('Vector prefix PairHMM failed: '+str(code))
        for ri,key in enumerate(reads):
            for hi,h in enumerate(haps):
                for j in grouped[key][h]:result[j]=out[ri,hi]
    fallback=np.flatnonzero(np.isnan(result))
    if len(fallback):result[fallback]=dense([jobs[j] for j in fallback],mode,0)
    return result
