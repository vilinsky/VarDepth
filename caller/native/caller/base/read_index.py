"""Exact point-overlap index for bounded regional read likelihood evaluation.

Avoids scanning all region reads for every indel. Long alignments are retained
in a separate list; indexing never imposes a read-span filter. Original read
order is preserved to minimize floating-point summation differences.
"""
import collections

class ReadIndex:
    def __init__(self,reads,bin_size=1000,max_bins=100):
        if bin_size<1 or max_bins<1:raise ValueError('Positive index dimensions required')
        self.reads=list(reads);self.bin_size=bin_size;self.bins=collections.defaultdict(list);self.long=[]
        for i,(_,r) in enumerate(self.reads):
            if r.reference_end is None or r.reference_end<=r.reference_start:continue
            first=r.reference_start//bin_size;last=(r.reference_end-1)//bin_size
            if last-first+1>max_bins:self.long.append(i)
            else:
                for b in range(first,last+1):self.bins[b].append(i)
    def __iter__(self):return iter(self.reads)
    def __len__(self):return len(self.reads)
    def at(self,position):
        ids=self.bins.get(position//self.bin_size,[])
        if self.long:ids=sorted([*ids,*self.long])
        return [self.reads[i] for i in ids if self.reads[i][1].reference_start<=position<self.reads[i][1].reference_end]
    def overlap(self,start,end):
        if end<=start:return []
        ids=set(self.long)
        for bucket in range(start//self.bin_size,(end-1)//self.bin_size+1):ids.update(self.bins.get(bucket,[]))
        return [self.reads[i] for i in sorted(ids) if self.reads[i][1].reference_start<end and self.reads[i][1].reference_end>start]
