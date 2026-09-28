"""Bounded graph proposals after r05 partial evidence; no truth inputs."""
import itertools,re
from functools import lru_cache
from graph_assemble import assemble

_LOW_QUALITY=re.compile(rb"[\x00-\x13]")
_NONCANONICAL=re.compile("[^ACGT]")

def quality_segments(sequence,qualities,k):
    return [segment for segment in _quality_runs(sequence,bytes(qualities)) if len(segment)>=k]

@lru_cache(maxsize=4096)
def _quality_runs(sequence,qualities):
    cuts=sorted({m.start() for m in _LOW_QUALITY.finditer(qualities)}|{m.start() for m in _NONCANONICAL.finditer(sequence)})
    begin=0;segments=[]
    for end in [*cuts,len(sequence)]:
        if end>begin:segments.append(sequence[begin:end])
        begin=end+1
    return tuple(segments)

def extend(row,bams,fetch,maximum=48,coverage=None):
    if len(row['alleles'])>=maximum or len(row['alleles'][0])<20 or max(coverage.values() if coverage is not None else (s['dp'] for s in row['samples'].values()))<20:
        return row,dict(added=0,reason='outside graph proposal eligibility')
    if not any(len(a)!=len(row['alleles'][0]) for a in row['alleles'][1:]):return row,dict(added=0,reason='no indel')
    start=row['start0'];end=row['end0'];old=row['alleles'];groups=[];diagnostics=[]
    for k in [31,41,51]:
        left=start-k-32;right=end+k+32
        if left<0:continue
        prefix=fetch(left,start);suffix=fetch(end,right);reference=fetch(left,right)
        if len(prefix)!=k+32 or len(suffix)!=k+32:continue
        observations=[]
        for sample,inputs in bams.items():
            for bam,rg,fallback in inputs:
                for read in bam.fetch(row['chrom'],left,right):
                    if read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate or read.is_qcfail or not 20<=read.mapping_quality<255 or read.query_qualities is None:continue
                    library=rg.get(read.get_tag('RG'),fallback) if read.has_tag('RG') else fallback
                    molecule=(sample,library,read.query_name);sequence=read.query_sequence.upper()
                    observations.extend((molecule,segment) for segment in quality_segments(sequence,read.query_qualities,k))
        candidates,stats=assemble(reference,observations,k=k,beam=128,max_delta=50)
        retained=[]
        for sequence in candidates:
            # Graph connectivity outside this block is useful for assembly,
            # but outside variants are owned by their own output blocks.
            if not sequence.startswith(prefix) or not sequence.endswith(suffix):continue
            inner=sequence[len(prefix):-len(suffix)]
            if inner and inner not in old and abs(len(inner)-len(old[0]))<=50:retained.append(inner)
        groups.append(retained);diagnostics.append(dict(k=k,eligible=len(retained),graph=stats))
    seen=set(old);added=[]
    # Fixed interleaving preserves diversity across the three graph resolutions.
    for level in itertools.zip_longest(*groups):
        for sequence in level:
            if sequence is not None and sequence not in seen and len(added)<min(12,maximum-len(old)):
                seen.add(sequence);added.append(sequence)
    summary=dict(added=len(added),resolutions=diagnostics,policy='Fixed k31/41/51 interleaving, at most 12 extra alleles; no calls or read likelihood assigned')
    return dict(row,alleles=old+added,added_alleles=row['added_alleles']+len(added),graph_proposals=summary),summary
