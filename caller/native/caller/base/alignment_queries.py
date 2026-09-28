"""Exact CIGAR coordinate queries without materializing every aligned base."""
from bisect import bisect_left,bisect_right

def match_runs(read):
    q=0;p=read.reference_start;pending=None
    for op,n in read.cigartuples or []:
        if op in (0,7,8):
            if pending is not None and pending[0]+pending[2]==q and pending[1]+pending[2]==p:
                pending=(pending[0],pending[1],pending[2]+n)
            else:
                if pending is not None:yield pending
                pending=(q,p,n)
            q+=n;p+=n
        else:
            if pending is not None:yield pending;pending=None
            if op in (1,4):q+=n
            elif op in (2,3):p+=n
    if pending is not None:yield pending

def unusual(read):return any(op in (6,9) for op,n in (read.cigartuples or []))

def target_queries(read,positions):
    if unusual(read):
        selected=set(positions)
        for q,p in read.get_aligned_pairs(matches_only=True):
            if p in selected:yield q,p
        return
    index=bisect_left(positions,read.reference_start)
    for q,p,n in match_runs(read):
        index=bisect_left(positions,p,index)
        while index<len(positions) and positions[index]<p+n:
            target=positions[index];yield q+target-p,target;index+=1

def point_query(read,position):
    if unusual(read):
        return next((q for q,p in read.get_aligned_pairs(matches_only=True) if p==position),None)
    for q,p,n in match_runs(read):
        if p<=position<p+n:return q+position-p
        if p>position:return None
    return None

def window_pairs(read,start,end):
    if unusual(read):
        return [(q,p) for q,p in read.get_aligned_pairs(matches_only=True) if start<=p<end]
    return [(q+x-p,x) for q,p,n in match_runs(read) for x in range(max(p,start),min(p+n,end))]

def window_bounds(read,start,end):
    """First and last aligned query base in a half-open reference interval."""
    if unusual(read):
        pairs=window_pairs(read,start,end)
        return (pairs[0][0],pairs[-1][0]+1) if pairs else None
    first=last=None
    for q,p,n in match_runs(read):
        lo=max(p,start);hi=min(p+n,end)
        if lo>=hi:continue
        if first is None:first=q+lo-p
        last=q+hi-p
    return None if first is None else (first,last)

def outer_safe_anchors(read,start,safe_positions,anchor):
    """First/last safe anchors lying inside uninterrupted aligned runs."""
    left=right=q0=q1=None
    for q,p,n in match_runs(read):
        lo=bisect_left(safe_positions,p-start);hi=bisect_right(safe_positions,p+n-start-anchor)
        if lo>=hi:continue
        if left is None:left=safe_positions[lo];q0=q+start+left-p
        right=safe_positions[hi-1]+anchor;q1=q+start+right-p
    return left,right,q0,q1


def cached_point_query(read,position,cache):
    """Cache scoped to a fixed, immutable read collection, owned by preparation."""
    key=id(read)
    if key not in cache:
        cache[key]=(read,None if unusual(read) else tuple(match_runs(read)))
    runs=cache[key][1]
    if runs is None:return point_query(read,position)
    for q,p,n in runs:
        if p<=position<p+n:return q+position-p
        if p>position:return None
    return None


_window_bounds_python=window_bounds
_outer_safe_anchors_python=outer_safe_anchors
try:
    from pysam_observations import window_bounds,outer_bounds
except ModuleNotFoundError as error:
    if error.name!='pysam_observations':raise
else:
    def outer_safe_anchors(read,start,safe_positions,anchor,prepared=None):
        return outer_bounds(read,start,safe_positions,anchor,prepared)
