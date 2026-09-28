"""Same biallelic finite mixture, reusing equal molecular likelihood rows."""
import ctypes
from functools import lru_cache
from pathlib import Path
import numpy as np
from native_build import build_library
@lru_cache(maxsize=1)
def library():
    lib=ctypes.CDLL(str(build_library(Path(__file__).with_name('mixture_two.cpp'))))
    p=ctypes.POINTER(ctypes.c_double);lib.mixture_two.argtypes=[ctypes.c_int,p,ctypes.c_double,p];lib.mixture_two.restype=None
    return lib
def two(lh,prior):
    x=np.ascontiguousarray(lh,np.float64);out=np.empty((2,2));p=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    if not np.isfinite(x).all() or np.any(x<0) or not 0<=prior<1:raise ValueError('Invalid mixture likelihood/prior')
    library().mixture_two(len(x),p(x),prior,p(out));return out

@lru_cache(maxsize=1)
def general_library():
    lib=ctypes.CDLL(str(build_library(Path(__file__).with_name('mixture_general.cpp'))))
    p=ctypes.POINTER(ctypes.c_double);lib.mixture_general.argtypes=[ctypes.c_int,ctypes.c_int,p,ctypes.c_double,ctypes.c_int,p];lib.mixture_general.restype=None
    return lib

def general(lh,prior,triangular=False):
    x=np.ascontiguousarray(lh,np.float64);n=x.shape[1]
    if n<1 or not np.isfinite(x).all() or np.any(x<0) or (prior is not None and not 0<=prior<1):raise ValueError('Invalid mixture likelihood/prior')
    out=np.empty(n*(n+1)//2 if triangular else n*n);p=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    general_library().mixture_general(len(x),n,p(x),-1 if prior is None else prior,int(triangular),p(out))
    return out if triangular else out.reshape(n,n)
