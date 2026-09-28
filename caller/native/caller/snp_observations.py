"""Reference molecular observation collector and optional native dispatch."""
import collections

def collect_python(reads,snps,min_bq,alignments=None,alignment_ranks=None):
    observations=collections.defaultdict(dict)
    from alignment_queries import target_queries
    snp_positions=sorted(snps)
    for library,r in reads:
        qv=r.query_qualities
        if qv is None:continue
        key=(library,r.query_name)
        for q,p in target_queries(r,snp_positions):
            indices=snps.get(p)
            if not indices or qv[q]<min_bq:continue
            for i in indices:
                old=observations[i].get(key);obs=(r.query_sequence[q].upper(),int(qv[q]),r.mapping_quality)
                if old is None or obs[1]>old[1]:
                    observations[i][key]=obs
                    if alignments is not None:alignments.setdefault(i,{})[key]=(r,q)
                elif alignments is not None and obs==old:
                    previous=alignments[i][key][0]
                    if alignment_ranks is not None and alignment_ranks[id(r)]<alignment_ranks[id(previous)]:alignments[i][key]=(r,q)
    return observations

def collect(reads,snps,min_bq,alignments=None,alignment_ranks=None):
    try:
        from pysam_observations import collect as native
    except ModuleNotFoundError as error:
        if error.name!="pysam_observations":raise
        return collect_python(reads,snps,min_bq,alignments,alignment_ranks)
    return native(reads,snps,min_bq,alignments,alignment_ranks)
