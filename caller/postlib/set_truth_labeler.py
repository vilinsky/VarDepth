import bisect,gzip
from pathlib import Path

class ConfidentBed:
    def __init__(self,path):
        self.intervals={};op=gzip.open if str(path).endswith('.gz') else open
        with op(path,'rt') as f:
            for line in f:
                if not line.strip() or line.startswith(('#','track','browser')):continue
                c,s,e,*_=line.split();self.intervals.setdefault(c,[]).append((int(s),int(e)))
        self.starts={}
        for c,rows in self.intervals.items():
            merged=[]
            for s,e in sorted(rows):
                if merged and s<=merged[-1][1]:merged[-1]=(merged[-1][0],max(e,merged[-1][1]))
                else:merged.append((s,e))
            self.intervals[c]=merged;self.starts[c]=[s for s,e in merged]
    def contains(self,c,s,e):
        i=bisect.bisect_right(self.starts.get(c,[]),s)-1
        return i>=0 and self.intervals[c][i][1]>=e

def normalize_biallelic(start, ref, alt, reference, limit=100000):
    """Canonical bounds for search only; original truth edits remain unchanged."""
    ref, alt = ref.upper(), alt.upper()
    if not ref or not alt or any(b not in 'ACGT' for b in ref + alt):
        raise ValueError('noncanonical_normalization')
    if reference[start:start + len(ref)] != ref:
        raise ValueError('normalization_ref_mismatch')
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    shifts = 0
    if len(ref) != len(alt):
        while start > 0 and ref[-1] == alt[-1]:
            if shifts >= limit:
                raise ValueError('normalization_limit')
            previous = reference[start - 1]
            if previous not in 'ACGT':
                raise ValueError('noncanonical_normalization')
            ref, alt = previous + ref[:-1], previous + alt[:-1]
            start -= 1
            shifts += 1
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt = ref[1:], alt[1:]
        start += 1
    return start, ref, alt
