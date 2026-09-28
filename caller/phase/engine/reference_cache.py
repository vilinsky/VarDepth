"""Bounded immutable FASTA tile cache, preserving sequence case and coordinates."""
from collections import OrderedDict
import pysam

class CachedReference:
    def __init__(self,path,tile_bp=1048576,capacity=4):
        if tile_bp<1 or capacity<1:raise ValueError('Positive cache dimensions required')
        self._fasta=pysam.FastaFile(path);self._tile_bp=tile_bp;self._capacity=capacity
        self._tiles=OrderedDict();self._lengths=dict(zip(self._fasta.references,self._fasta.lengths))
    def __getattr__(self,name):return getattr(self._fasta,name)
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):
        self._tiles.clear();self._fasta.close()
    def fetch(self,reference=None,start=None,end=None,**kwargs):
        length=self._lengths.get(reference)
        if (kwargs or type(start) is not int or type(end) is not int or length is None
                or start<0 or end<start or start>length or end>length or end-start>self._tile_bp):
            return self._fasta.fetch(reference,start,end,**kwargs)
        left=start//self._tile_bp*self._tile_bp
        key=(reference,left)
        if key not in self._tiles:
            text=self._fasta.fetch(reference,left,min(length,left+2*self._tile_bp))
            self._tiles[key]=text
            if len(self._tiles)>self._capacity:self._tiles.popitem(last=False)
        self._tiles.move_to_end(key)
        return self._tiles[key][start-left:end-left]

