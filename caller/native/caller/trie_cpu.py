"""Unpruned prefix-sharing PairHMM with normalization-safe fallback."""
import ctypes
from functools import lru_cache
from pathlib import Path
import numpy as np
from native_build import build_library

@lru_cache(maxsize=1)
def library():
    lib=ctypes.CDLL(str(build_library(Path(__file__).with_name('trie_hmm.cpp'))))
    p=ctypes.POINTER
    lib.trie_pairs.argtypes=[p(ctypes.c_char_p),p(p(ctypes.c_ubyte)),p(ctypes.c_int),p(ctypes.c_char_p),p(ctypes.c_int),p(ctypes.c_int),ctypes.c_int,ctypes.c_int,ctypes.c_int,p(ctypes.c_double),p(ctypes.c_uint64)]
    lib.trie_pairs.restype=ctypes.c_int
    return lib

def score(jobs,mode,dense):
    if not jobs:return np.empty(0)
    groups={};qualities={};encoded={}
    for index,(s,q,h) in enumerate(jobs):
        if len(s)!=len(q):raise ValueError('Read and quality lengths differ')
        if id(q) not in qualities:qualities[id(q)]=np.ascontiguousarray(q,np.uint8).tobytes()
        key=(s,qualities[id(q)])
        groups.setdefault(key,[]).append(index)
        for seq in (s,h):
            if seq not in encoded:encoded[seq]=seq.encode('ascii')
    order=[];starts=[0];qbytes=[]
    for (s,q),indices in groups.items():
        indices.sort(key=lambda i:jobs[i][2]);order.extend(indices);starts.append(len(order));qbytes.extend([q]*len(indices))
    count=len(jobs);p=ctypes.POINTER;i=ctypes.c_int
    reads=(ctypes.c_char_p*count)(*(encoded[jobs[j][0]] for j in order))
    haps=(ctypes.c_char_p*count)(*(encoded[jobs[j][2]] for j in order))
    # Byte strings are immutable and owned until the C call returns.
    qp=(ctypes.c_char_p*count)(*qbytes)
    rn=np.array([len(jobs[j][0]) for j in order],np.int32);hn=np.array([len(jobs[j][2]) for j in order],np.int32)
    offsets=np.array(starts,np.int32);out=np.empty(count);cells=ctypes.c_uint64()
    code=library().trie_pairs(reads,ctypes.cast(qp,p(p(ctypes.c_ubyte))),rn.ctypes.data_as(p(i)),haps,hn.ctypes.data_as(p(i)),offsets.ctypes.data_as(p(i)),len(groups),count,mode,out.ctypes.data_as(p(ctypes.c_double)),ctypes.byref(cells))
    if code:raise RuntimeError('Trie PairHMM failed: '+str(code))
    result=np.empty(count);result[order]=out
    fallback=np.flatnonzero(np.isnan(result))
    if len(fallback):result[fallback]=dense([jobs[j] for j in fallback],mode,0)
    return result
