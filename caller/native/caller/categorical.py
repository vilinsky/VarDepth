"""Categorical copying posterior for read GL and actual panel haplotypes."""
import ctypes
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
from native_build import build_library


def impute(haplotypes, genotype_log_likelihoods, transitions, *, mutation=.002,
           burn=24, iterations=96, chains=2, seed=17):
    raw = np.asarray(haplotypes)
    gl = [np.asarray(x, np.float64) for x in genotype_log_likelihoods]
    counts = np.array([int((np.sqrt(8*len(x)+1)-1)/2) for x in gl], np.int32)
    if raw.ndim != 2 or raw.shape[0] != len(gl) or raw.shape[1] < 1 or not len(gl):
        raise ValueError('Inconsistent categorical scaffold dimensions')
    if np.any(counts < 2) or np.any(counts > 24) or any(len(g) != n*(n+1)//2 for g,n in zip(gl,counts)):
        raise ValueError('Invalid genotype vector lengths')
    if not np.isfinite(raw).all() or np.any((raw != 255) & ((raw < 0) | (raw >= counts[:,None]) | (raw != np.floor(raw)))):
        raise ValueError('Unknown panel alleles must be explicitly encoded 255')
    h = np.ascontiguousarray(raw, np.uint8)
    offsets = np.concatenate(([0], np.cumsum([len(x) for x in gl]))).astype(np.int64)
    g = np.concatenate(gl); r = np.ascontiguousarray(transitions, np.float64)
    if r.shape != (len(gl),) or not np.isfinite(g).all() or not np.isfinite(r).all() or np.any((r<0)|(r>1)):
        raise ValueError('Invalid likelihoods or transitions')
    if not 0 <= mutation < 1 or burn < 0 or iterations < 1 or chains < 1:
        raise ValueError('Invalid sampling configuration')
    lib = ctypes.CDLL(str(build_library(HERE / 'categorical_neutral.cpp')))
    pd = ctypes.POINTER(ctypes.c_double)
    lib.categorical_gibbs.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64), pd, pd, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64, pd]
    lib.categorical_gibbs.restype = ctypes.c_int
    out = np.zeros_like(g)
    ptr = lambda a,t: a.ctypes.data_as(ctypes.POINTER(t))
    code = lib.categorical_gibbs(len(gl),h.shape[1],ptr(h,ctypes.c_ubyte),ptr(counts,ctypes.c_int32),
        ptr(offsets,ctypes.c_int64),ptr(g,ctypes.c_double),ptr(r,ctypes.c_double),mutation,burn,iterations,chains,seed,ptr(out,ctypes.c_double))
    if code or not np.isfinite(out).all():
        if code not in (0, 4, 5, 6, 7):
            raise RuntimeError('Categorical copying failed: ' + str(code))
        original_code = code
        stable = ctypes.CDLL(str(build_library(HERE / 'categorical_runs.cpp')))
        stable.categorical_gibbs.argtypes = lib.categorical_gibbs.argtypes
        stable.categorical_gibbs.restype = ctypes.c_int
        out.fill(0.)
        code = stable.categorical_gibbs(len(gl),h.shape[1],ptr(h,ctypes.c_ubyte),ptr(counts,ctypes.c_int32),
        ptr(offsets,ctypes.c_int64),ptr(g,ctypes.c_double),ptr(r,ctypes.c_double),mutation,burn,iterations,chains,seed,ptr(out,ctypes.c_double))
        import json
        print('VARDEPTH_NUMERICAL_FALLBACK ' + json.dumps(dict(
            original_code=original_code, fallback_code=code, markers=len(gl),
            donors=h.shape[1], positive_refresh_edges=int(np.count_nonzero(r[1:])),
            same_model_parameters=True, same_seed=seed,
            algorithm='log-likelihood accumulation within exactly zero-refresh runs')),
            file=sys.stderr, flush=True)
        if code or not np.isfinite(out).all():
            raise RuntimeError('Stable categorical copying failed: ' + str(code))
    return [out[offsets[t]:offsets[t+1]] for t in range(len(gl))]
