"""Reuse decoded immutable read sets without changing requested scientific windows."""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import os
import threading
_CACHE=OrderedDict()
_LOCK=threading.RLock()
_MAX_READS=500000
_TOTAL=0
STATS=dict(hits=0,misses=0,reused_reads=0)

def _stamp(path):
    p=Path(path).resolve()
    if not p.exists():return (str(p),None)
    s=p.stat()
    return (str(p),s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def collect(uncached,paths,reference,chrom,start,end,mapq):
    if os.environ.get('VARDEPTH_INPROCESS_CPU_STAGES')!='1':
        return uncached(paths,reference,chrom,start,end,mapq)
    files=[reference,str(reference)+'.fai',str(reference)+'.gzi']
    for p in paths:files.extend([p,str(p)+'.bai',str(p)+'.crai',str(Path(p).with_suffix('.bai'))])
    key=(tuple(_stamp(p) for p in files),chrom,mapq)
    left=max(0,start-250);right=end+250
    global _TOTAL
    with _LOCK:
        saved=_CACHE.get(key)
        if saved is not None and saved[0]<=left and right<=saved[1]:
            _CACHE.move_to_end(key)
            _,_,reads,sms,headers=saved
            # All saved reads have a defined positive reference span.
            result=[(library,r) for library,r in reads if r.reference_start<right and r.reference_end>left]
            STATS['hits']+=1;STATS['reused_reads']+=len(result)
            return result,set(sms),deepcopy(headers)
    reads,sms,headers=uncached(paths,reference,chrom,start,end,mapq)
    with _LOCK:
        STATS['misses']+=1
        if len(reads)<=_MAX_READS and all(r.reference_end is not None and r.reference_end>r.reference_start for _,r in reads):
            previous=_CACHE.pop(key,None)
            if previous is not None:_TOTAL-=len(previous[2])
            _CACHE[key]=(left,right,list(reads),set(sms),deepcopy(headers));_TOTAL+=len(reads)
            while _TOTAL>_MAX_READS:
                _,entry=_CACHE.popitem(last=False);_TOTAL-=len(entry[2])
    return reads,sms,headers

