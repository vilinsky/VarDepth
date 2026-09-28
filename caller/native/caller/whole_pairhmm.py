"""Prepared haplotypes and architecture-dispatched exact double precision HMM."""
import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
_LIB = None


def library():
    global _LIB
    if _LIB is not None:
        return _LIB
    source = HERE / 'wide.cpp'
    suffix = 'dylib' if sys.platform == 'darwin' else 'so'
    binary = HERE / ('libwide.' + suffix)
    marker = HERE / 'build.sha256'
    fingerprint = hashlib.sha256(source.read_bytes() + b'-O3 -std=c++17 -ffp-contract=off').hexdigest()
    with (HERE / 'build.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not binary.exists() or not marker.exists() or marker.read_text() != fingerprint:
            temporary = binary.with_name(binary.name + '.' + str(os.getpid()))
            try:
                subprocess.run(['c++', '-O3', '-std=c++17', '-ffp-contract=off', '-fPIC',
                                '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                                str(source), '-o', str(temporary)], check=True)
                temporary.replace(binary)
                marker.write_text(fingerprint)
            finally:
                temporary.unlink(missing_ok=True)
    lib = ctypes.CDLL(str(binary))
    arguments = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
                 ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int),
                 ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
    lib.read_haplotype_batch_mode.argtypes = arguments + [ctypes.c_int]
    lib.read_haplotype_batch_mode.restype = None
    lib.read_haplotype_batch.argtypes = arguments
    lib.read_haplotype_batch.restype = None
    lib.wide_capability.argtypes = []
    lib.wide_capability.restype = ctypes.c_int
    _LIB = lib
    return lib


class Prepared:
    def __init__(self, haplotypes, mode=0):
        if mode not in [0, 2, 4, 8]:
            raise ValueError('Unsupported lane request')
        self.lib = library()
        self.encoded = [h.encode('ascii') for h in haplotypes]
        self.n = len(self.encoded)
        self.hp = (ctypes.c_char_p * self.n)(*self.encoded)
        self.lengths = (ctypes.c_int * self.n)(*(len(h) for h in self.encoded))
        self.mode = mode

    def many(self, sequences, qualities):
        if len(sequences)!=len(qualities): raise ValueError('Sequence/quality batch lengths differ')
        from accelerator import backend, score_many
        if backend()!='cpu':
            return score_many(sequences,qualities,[h.decode('ascii') for h in self.encoded])
        from cpu_batch import many
        return many(sequences,qualities,[h.decode('ascii') for h in self.encoded],mode=2)

    def __call__(self, sequence, qualities):
        if len(sequence) != len(qualities):
            raise ValueError('Read and quality lengths disagree')
        quality = np.frombuffer(qualities,np.uint8) if isinstance(qualities,bytes) else np.asarray(qualities, dtype=np.uint8)
        if not quality.flags.c_contiguous:
            quality = np.ascontiguousarray(quality)
        result = np.empty(self.n, np.float64)
        self.lib.read_haplotype_batch_mode(
            sequence.encode('ascii'), quality.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            len(sequence), self.hp, self.lengths, self.n, .001, .1,
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self.mode)
        return result
