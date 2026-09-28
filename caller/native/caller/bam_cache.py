"""Bounded, process-private interval reads; immutable alignment objects only.

The exact BAM iterator order and interval overlap rule are retained. Large
queries use HTSlib directly. Cached reads must never be modified by callers.
"""
import atexit
from bisect import bisect_left,bisect_right
from collections import OrderedDict
from pathlib import Path
import pysam

class RegionReader:
    def __init__(self,bam,tile_bp=16384,capacity=4):
        self.bam=bam;self.tile_bp=tile_bp;self.capacity=capacity;self.tiles=OrderedDict()
    def __getattr__(self,name):return getattr(self.bam,name)
    def close(self):self.tiles.clear();self.bam.close()
    def fetch(self,chrom,start,end):
        if end<=start or end-start>self.tile_bp:return self.bam.fetch(chrom,start,end)
        left=start//self.tile_bp*self.tile_bp
        right=min(left+2*self.tile_bp,self.bam.get_reference_length(chrom))
        key=(chrom,left)
        if key not in self.tiles:
            reads=list(self.bam.fetch(chrom,left,right))
            high=-1;prefix_ends=[]
            for read in reads:
                high=max(high,read.reference_end if read.reference_end is not None else -1);prefix_ends.append(high)
            self.tiles[key]=(reads,[r.reference_start for r in reads],prefix_ends)
            if len(self.tiles)>self.capacity:self.tiles.popitem(last=False)
        self.tiles.move_to_end(key);reads,positions,prefix_ends=self.tiles[key]
        begin=bisect_right(prefix_ends,start);stop=bisect_left(positions,end)
        return (read for read in reads[begin:stop] if read.reference_end is not None and read.reference_end>start)

_OPEN=OrderedDict()
def open_cached(path,reference=None):
    # Each worker owns these handles; no cross-process or cross-run cache.
    key=(str(Path(path).resolve()),str(reference) if reference else None)
    if key not in _OPEN:
        _OPEN[key]=RegionReader(pysam.AlignmentFile(path,reference_filename=reference))
        if len(_OPEN)>8:_OPEN.popitem(last=False)[1].close()
    _OPEN.move_to_end(key);return _OPEN[key]

def clear():
    for reader in _OPEN.values():reader.close()
    _OPEN.clear()
atexit.register(clear)
