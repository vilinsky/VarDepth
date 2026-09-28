"""Genetic distances and explicit Li-Stephens copying refresh probabilities.

Mathematical model: r=1-exp(-4 Ne delta_M / N_reference_haplotypes).
Ne is a model parameter, not an inferred demographic population size here.
Implementation is independent; genetic-map data are supplied by the user.
"""
import gzip
from pathlib import Path
from functools import lru_cache

import numpy as np


class GeneticMap:
    def __init__(self, positions, centimorgans):
        self.positions = np.asarray(positions, dtype=np.float64)
        self.cm = np.asarray(centimorgans, dtype=np.float64)
        if self.positions.ndim != 1 or self.positions.shape != self.cm.shape or len(self.cm) < 2:
            raise ValueError('At least two matched genetic-map coordinates are required')
        if not np.isfinite(self.positions).all() or not np.isfinite(self.cm).all():
            raise ValueError('Genetic-map values must be finite')
        if np.any(np.diff(self.positions) <= 0) or np.any(np.diff(self.cm) < 0):
            raise ValueError('Map positions must increase strictly and cM must not decrease')

    @classmethod
    def from_file(cls, path, chrom):
        opener = gzip.open if str(path).endswith('.gz') else open
        positions, distances = [], []
        expected = str(chrom).removeprefix('chr')
        with opener(path, 'rt') as stream:
            header = stream.readline().strip().split()
            if header != ['pos', 'chr', 'cM']:
                raise ValueError('Expected genetic-map header: pos chr cM (1-based bp, cM)')
            for line in stream:
                if not line.strip() or line.startswith('#'):
                    continue
                position, c, cm = line.split()
                if c.removeprefix('chr') == expected:
                    position, cm = int(position) - 1, float(cm)
                    if positions and position == positions[-1] and cm == distances[-1]:
                        continue
                    positions.append(position)
                    distances.append(cm)
        return cls(positions, distances)

    def interpolate(self, positions):
        positions = np.asarray(positions, dtype=float)
        if not np.isfinite(positions).all():
            raise ValueError('Positions must be finite')
        result = np.interp(positions, self.positions, self.cm)
        # Explicit linear tail extrapolation with the nearest map segment.
        left = positions < self.positions[0]
        right = positions > self.positions[-1]
        result[left] = self.cm[0] + (positions[left] - self.positions[0]) * (self.cm[1]-self.cm[0]) / (self.positions[1]-self.positions[0])
        result[right] = self.cm[-1] + (positions[right] - self.positions[-1]) * (self.cm[-1]-self.cm[-2]) / (self.positions[-1]-self.positions[-2])
        return result

    def refresh_probabilities(self, positions, reference_haplotypes, ne=100000):
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 1 or np.any(np.diff(positions) < 0):
            raise ValueError('Inference positions must be a sorted vector')
        if reference_haplotypes < 1 or ne <= 0:
            raise ValueError('Positive reference haplotype count and Ne are required')
        if not len(positions):
            return np.empty(0, dtype=float)
        cm = self.interpolate(positions)
        distances = np.diff(cm, prepend=cm[0]) / 100.
        return -np.expm1(-4 * ne * np.maximum(0., distances) / reference_haplotypes)


@lru_cache(maxsize=8)
def _cached_map(path,chrom,size,mtime,ctime):
    return GeneticMap.from_file(path,chrom)

def transitions(positions, reference_haplotypes, path=None, chrom=None, ne=100000, physical_rate=1e-6, no_ld=False):
    positions = np.asarray(positions)
    if no_ld:
        return np.ones(len(positions), dtype=float)
    if path:
        path=Path(path).resolve();stat=path.stat()
        return _cached_map(str(path),chrom,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns).refresh_probabilities(positions, reference_haplotypes, ne)
    if not len(positions):
        return np.empty(0, dtype=float)
    # Preserve the initial prototype for a controlled ablation.
    return np.clip(-np.expm1(-np.diff(positions, prepend=positions[0]) * physical_rate), 1e-8, .5)
