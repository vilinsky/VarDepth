"""Reuse binary donor emissions while retaining the marker summation order."""
import ctypes
from functools import lru_cache
from pathlib import Path
import numpy as np
from native_build import build_library
@lru_cache(maxsize=1)
def library():
    lib=ctypes.CDLL(str(build_library(Path(__file__).with_name('selection_sum.cpp'))))
    p=ctypes.POINTER(ctypes.c_double)
    lib.binary_sums.argtypes=[ctypes.c_int,ctypes.c_int,p,p,p];lib.binary_sums.restype=None
    lib.binary_sums_u8.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_uint8),p,p];lib.binary_sums_u8.restype=None
    return lib
def scores(h,ref,alt,mutation):
    if not np.all((h==0)|(h==1)):return None
    b=mutation+(1-2*mutation)*np.array([0.,1.])
    table=np.ascontiguousarray(np.log(np.maximum(ref[:,None]*(1-b)+alt[:,None]*b,1e-300)))
    h=np.ascontiguousarray(h);out=np.empty(h.shape[1]);p=lambda x:x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    library().binary_sums(*h.shape,p(h),p(table),p(out))
    return out

def scores_u8(h,ref,alt,mutation):
    """Caller supplies a validated contiguous binary panel, shared by anchors."""
    if h.dtype!=np.uint8 or not h.flags.c_contiguous:raise ValueError('Expected contiguous byte panel')
    b=mutation+(1-2*mutation)*np.array([0.,1.])
    table=np.ascontiguousarray(np.log(np.maximum(ref[:,None]*(1-b)+alt[:,None]*b,1e-300)))
    out=np.empty(h.shape[1]);p=lambda x:x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    library().binary_sums_u8(*h.shape,h.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),p(table),p(out))
    return out
