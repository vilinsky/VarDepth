"""Process-isolated, ordered local evidence jobs with private BAM/FASTA handles."""
from reference_cache import CachedReference
import atexit,collections
import pysam

_BAMS=None
_FA=None
_LIB=None

def initialize(arguments,reference):
    global _BAMS,_FA,_LIB
    from indel_refine import kernel
    from bam_cache import RegionReader
    _FA=CachedReference(reference);_BAMS=collections.defaultdict(list);_LIB=kernel()
    for argument in arguments:
        sample,path=argument.split('=',1);bam=RegionReader(pysam.AlignmentFile(path,reference_filename=reference),tile_bp=4096,capacity=16)
        rg={r['ID']:r.get('LB',path+'::'+r['ID']) for r in bam.header.to_dict().get('RG',[])}
        _BAMS[sample].append((bam,rg,path+'::unknown'))
    def close():
        for inputs in _BAMS.values():
            for bam,rg,fallback in inputs:bam.close()
        _FA.close()
    atexit.register(close)

def fetch_for(chrom):
    return lambda x,y:_FA.fetch(chrom,max(0,x),min(y,_FA.get_reference_length(chrom))).upper()

def proposal(task):
    from proposals import propose
    row,original=task
    return propose(row,original,_BAMS,fetch_for(row['chrom']))[0]

def graph(task):
    from graph_extend import extend
    row,depths=task
    return extend(row,_BAMS,fetch_for(row['chrom']),maximum=48,coverage=depths)

def full_cpu(task):
    row,original,depths=task
    return local_cpu(row,original,depths,_BAMS,fetch_for(row['chrom']),_LIB)

def local_cpu(row,original,depths,bams,fetch,lib):
    """Complete one locus before advancing, retaining identical per-locus order."""
    from partial_anchor import refine
    from whole_read import score
    from proposals import propose
    from graph_extend import extend
    row=propose(row,original,bams,fetch)[0]
    updated,detail=refine(row,original,bams,fetch,lib)
    eligible=max(depths.values())>=20 and (len(updated['alleles'][0])>=12 or len(updated['alleles'])>=4)
    whole=None
    if eligible:
        extended,summary=extend(updated,bams,fetch,maximum=48,coverage=depths)
        if summary['added']:
            extended=dict(extended,samples={name:dict(s,dp=s['original_spanning_dp']) for name,s in updated['samples'].items()})
            updated,detail=refine(extended,original,bams,fetch,lib)
        else:updated=extended
        whole=dict(updated,samples=score(updated,bams,fetch),whole_read_expert=True)
    return updated,detail,whole

def full_cpu_batch(tasks):
    return [(index,full_cpu(task)) for index,task in tasks]

def locality_batches(rows,original,coverage,workers,tile_bp=8192):
    """Keep nearby BAM work together, then balance bins by estimated DP cost."""
    groups=collections.defaultdict(list)
    for i,row in enumerate(rows):
        cost=max(coverage[row['record_id']].values())*len(row['alleles'])*max(map(len,row['alleles']))
        groups[(row['chrom'],row['start0']//tile_bp)].append((i,cost))
    # Heavy bins can be split without changing any locus inputs or thresholds.
    target=max(1,sum(c for group in groups.values() for i,c in group)/max(1,workers*4))
    batches=[]
    for group in groups.values():
        batch=[];cost=0
        for index,weight in sorted(group,key=lambda x:-x[1]):
            if batch and cost>=target:batches.append((cost,batch));batch=[];cost=0
            row=rows[index];batch.append((index,(row,original[row['record_id']],coverage[row['record_id']])));cost+=weight
        if batch:batches.append((cost,batch))
    return [batch for cost,batch in sorted(batches,key=lambda x:-x[0])]
