"""Lossless canonical k-mer counting; proposal traversal remains unchanged."""
import ctypes,collections
from functools import lru_cache
from pathlib import Path
from native_build import build_library

@lru_cache(maxsize=1)
def library():
    lib=ctypes.CDLL(str(build_library(Path(__file__).with_name('kmer_graph.cpp'))))
    p=ctypes.POINTER;i=ctypes.c_int;v=ctypes.c_void_p
    lib.kmer_build.argtypes=[ctypes.c_char_p,i,p(ctypes.c_char_p),p(i),p(i),i,i,i,i];lib.kmer_build.restype=v
    lib.kmer_size.argtypes=[v,p(i),p(i)];lib.kmer_size.restype=i
    lib.kmer_copy.argtypes=[v,ctypes.c_char_p,p(i),p(i)];lib.kmer_copy.restype=None
    lib.kmer_free.argtypes=[v];lib.kmer_free.restype=None
    return lib
def build(reference,observations,k,minimum):
    if not 3<=k<=63 or any(c not in 'ACGT' for c in reference):return None
    ids={};seq=[];molecules=[]
    for molecule,sequence in observations:
        if molecule not in ids:ids[molecule]=len(ids)
        seq.append(sequence.upper().encode('ascii','replace'));molecules.append(ids[molecule])
    n=len(seq);p=ctypes.c_int;strings=(ctypes.c_char_p*n)(*seq);sizes=(p*n)(*map(len,seq));ms=(p*n)(*molecules)
    lib=library();handle=lib.kmer_build(reference.encode(),len(reference),strings,sizes,ms,n,len(ids),k,minimum)
    if not handle:raise RuntimeError('Native k-mer graph failed')
    try:
        observed=p();nm=p();count=lib.kmer_size(handle,ctypes.byref(observed),ctypes.byref(nm));words=ctypes.create_string_buffer(count*k);support=(p*count)();novel=(p*count)()
        lib.kmer_copy(handle,words,support,novel);flat=words.raw.decode('ascii');adjacency=collections.defaultdict(list)
        for j in range(count):word=flat[j*k:(j+1)*k];adjacency[word[:-1]].append((word[-1],support[j],bool(novel[j])))
        return adjacency,dict(nodes=len(adjacency),observed_edges=observed.value,molecules=nm.value,novel_retained_edges=sum(novel))
    finally:lib.kmer_free(handle)
