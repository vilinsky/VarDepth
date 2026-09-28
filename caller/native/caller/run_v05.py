#!/usr/bin/env python3
"""Regional Illumina calling with joint categorical LD and partial-read indels."""
from reference_cache import CachedReference
import argparse,collections,json,os,sys,time
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import pysam
from prepare import add_inputs,sha
from base_adapter import parse_region
from genome import Chunk,filter_owned_core,merge_owned_vcfs
from run_v04 import execute
import joint_ld
from partial_anchor import refine,refine_rows
from proposals import propose
from graph_extend import extend as graph_extend
from whole_read import score as whole_score,score_rows as whole_score_rows
from indel_refine import kernel

HERE=Path(__file__).resolve().parent

def partial_evidence(component,bam_arguments,reference):
    """Add partial-read proposals, then score common observed windows."""
    original={}
    for directory in component['features']:
        for row in json.loads((Path(directory)/'metadata.json').read_text())['records']:
            original[row['record_id']]=row
    rows=json.loads(Path(component['refinements']).read_text());result=[];diagnostics=[];whole=[]
    spanning={row['record_id']:row for row in rows};coverage={}
    existing={row['record_id'] for row in rows}
    with pysam.VariantFile(component['neural']) as neural:
        for record in neural:
            rid=f'{record.chrom}:{record.pos}:{record.ref}:{",".join(record.alts or [])}'
            coverage[rid]={name:int(record.samples[name].get('DP') or 0) for name in record.samples}
            if rid in existing or not any(len(a)!=len(record.ref) for a in record.alts):continue
            if max(int(record.samples[name].get('DP',0) or 0) for name in record.samples)<6:continue
            metadata=original[rid];n=len(metadata['alleles']);ng=n*(n+1)//2
            rows.append(dict(record_id=rid,chrom=record.chrom,start0=metadata['start0'],end0=metadata['end0'],
                alleles=metadata['alleles'],added_alleles=0,supplemented_low_depth=True,
                samples={name:dict(dp=0,gp=[1/ng]*ng,pl=[0]*ng,exact_ad=[0]*n,trusted=False,support_fraction=0.,proposed_sequences={}) for name in record.samples}))
    rows.sort(key=lambda row:(row['start0'],row['record_id']))
    with CachedReference(reference) as fa:
        bams=collections.defaultdict(list);opened=[];pool=None
        try:
            for argument in bam_arguments:
                sample,path=argument.split('=',1)
                bam=pysam.AlignmentFile(path,reference_filename=reference)
                if os.environ.get('VARDEPTH_PAIRHMM_BACKEND','cpu')=='cpu':
                    from bam_cache import RegionReader
                    bam=RegionReader(bam,tile_bp=4096,capacity=16)
                opened.append(bam)
                rg={r['ID']:r.get('LB',path+'::'+r['ID']) for r in bam.header.to_dict().get('RG',[])}
                bams[sample].append((bam,rg,path+'::unknown'))
            lib=kernel()
            from evidence_workers import initialize,proposal,graph,full_cpu,full_cpu_batch,locality_batches
            jobs=int(os.environ.get('VARDEPTH_EVIDENCE_JOBS','1'))
            if jobs<1:raise ValueError('Evidence jobs must be positive')
            actor_mode=jobs>1 and os.environ.get('VARDEPTH_PAIRHMM_BACKEND','cpu')!='cpu' and os.environ.get('VARDEPTH_GPU_EVIDENCE_PIPELINE','0')=='1'
            if jobs>1 and not actor_mode:
                pool=ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize,initargs=(bam_arguments,reference))
            if actor_mode:
                from gpu_evidence_pipeline import run_tasks
                tasks=((row,original[row['record_id']],coverage[row['record_id']]) for row in rows)
                for updated,detail,full in run_tasks(tasks,bam_arguments,reference,workers=jobs):
                    rid=updated['record_id'];result.append(updated);diagnostics.append(dict(record_id=rid,samples=detail))
                    if full is not None:whole.append(full)
                    elif rid in spanning:whole.append(spanning[rid])
            elif pool is not None and os.environ.get('VARDEPTH_PAIRHMM_BACKEND','cpu')=='cpu':
                # Long loci start first, avoiding a final tail of expensive indels.
                # Restore the genomic order before writing any evidence or VCF.
                tasks=locality_batches(rows,original,coverage,jobs)
                computed=[None]*len(rows)
                for batch in pool.map(full_cpu_batch,tasks,chunksize=1):
                    for index,value in batch:computed[index]=value
                for updated,detail,full in computed:
                    rid=updated['record_id'];result.append(updated);diagnostics.append(dict(record_id=rid,samples=detail))
                    if full is not None:whole.append(full)
                    elif rid in spanning:whole.append(spanning[rid])
            elif pool is None and os.environ.get('VARDEPTH_PAIRHMM_BACKEND','cpu')=='cpu':
                from evidence_workers import local_cpu
                for row in rows:
                    chrom=row['chrom']
                    fetch=lambda x,y:fa.fetch(chrom,max(0,x),min(y,fa.get_reference_length(chrom))).upper()
                    updated,detail,full=local_cpu(row,original[row['record_id']],coverage[row['record_id']],bams,fetch,lib)
                    rid=updated['record_id'];result.append(updated);diagnostics.append(dict(record_id=rid,samples=detail))
                    if full is not None:whole.append(full)
                    elif rid in spanning:whole.append(spanning[rid])
            else:
                def fetch_for_chrom(chrom):
                    return lambda x,y:fa.fetch(chrom,max(0,x),min(y,fa.get_reference_length(chrom))).upper()
                if pool is None:
                    proposed=[propose(row,original[row['record_id']],bams,fetch_for_chrom(row['chrom']))[0] for row in rows]
                else:proposed=list(pool.map(proposal,((row,original[row['record_id']]) for row in rows),chunksize=8))
                first=refine_rows(proposed,original,bams,fetch_for_chrom,lib)
                eligible=[];second_rows=[];second_indices=[]
                graph_indices=[i for i,(row,detail) in enumerate(first) if max(coverage[row['record_id']].values())>=20 and (len(row['alleles'][0])>=12 or len(row['alleles'])>=4)]
                graph_results={}
                if pool is not None:
                    tasks=((first[i][0],coverage[first[i][0]['record_id']]) for i in graph_indices)
                    graph_results=dict(zip(graph_indices,pool.map(graph,tasks,chunksize=8)))
                for i,(updated,detail) in enumerate(first):
                    rid=updated['record_id'];depths=coverage[rid]
                    use_whole=max(depths.values())>=20 and (len(updated['alleles'][0])>=12 or len(updated['alleles'])>=4)
                    eligible.append(use_whole)
                    if use_whole:
                        extended,summary=graph_results[i] if pool is not None else graph_extend(updated,bams,fetch_for_chrom(updated['chrom']),maximum=48,coverage=depths)
                        if summary['added']:
                            extended=dict(extended,samples={name:dict(s,dp=s['original_spanning_dp']) for name,s in updated['samples'].items()})
                            second_rows.append(extended);second_indices.append(i)
                        else:first[i]=(extended,detail)
                for i,pair in zip(second_indices,refine_rows(second_rows,original,bams,fetch_for_chrom,lib)):
                    first[i]=pair
                whole_rows=[updated for use_whole,(updated,detail) in zip(eligible,first) if use_whole]
                full_results=iter(whole_score_rows(whole_rows,bams,fetch_for_chrom))
                for use_whole,(updated,detail) in zip(eligible,first):
                    rid=updated['record_id']
                    if use_whole:whole.append(dict(updated,samples=next(full_results),whole_read_expert=True))
                    elif rid in spanning:whole.append(spanning[rid])
                    result.append(updated);diagnostics.append(dict(record_id=rid,samples=detail))
        finally:
            if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
            for bam in opened:bam.close()
    out=Path(component['refinements']).with_name('partial-refinements.json')
    out.write_text(json.dumps(result,separators=(',',':'))+'\n')
    out.with_name('whole-refinements.json').write_text(json.dumps(whole,separators=(',',':'))+'\n')
    out.with_suffix('.diagnostics.json').write_text(json.dumps(diagnostics,separators=(',',':'))+'\n')
    return out

def finalize_component(item):
    component,chunk,bam_arguments,reference,trust_checkpoint,joint,samples,chrom,length,*prepared=item
    directory=Path(component['directory']);partial=Path(prepared[0]) if prepared else partial_evidence(component,bam_arguments,reference)
    raw=directory/'fused.vcf.gz';normalized=directory/'normalized.vcf.gz';owned=directory/'owned.vcf.gz'
    cmd=[sys.executable,HERE/'fuse_trust.py','--neural',component['neural'],'--refinements',partial,
        '--spanning-refinements',partial.with_name('whole-refinements.json'),'--trust-checkpoint',trust_checkpoint,'--output',raw]
    if component['scaffold']:cmd+=['--scaffold',component['scaffold']]
    for sample,path in joint.items():cmd+=['--joint-ld',sample+'='+path]
    execute(cmd,directory/'fusion.log')
    execute(['bcftools','norm','-f',reference,'-c','e','-Oz','-o',normalized,raw],directory/'normalize.log')
    filter_owned_core(normalized,owned,chunk,samples,{chrom:length});return owned

def run(a):
    started=time.monotonic();chrom,padded_start,padded_end=parse_region(a.region)
    core_chrom,start,end=parse_region(a.core_region)
    if chrom!=core_chrom or not padded_start<=start<end<=padded_end:raise ValueError('Core must be inside its LD window')
    if end-start>4_000_000:raise ValueError('Joint LD cores are bounded to 4 Mb')
    work=Path(a.work_dir);work.mkdir(parents=True,exist_ok=True)
    with CachedReference(a.reference) as fa:length=fa.get_reference_length(chrom)
    samples=list(dict.fromkeys(value.split('=',1)[0] for value in a.bam))
    components=[];chunks=[]
    specs=list(enumerate(range(start,end,1_000_000)))
    workers=getattr(a,'component_jobs',4)
    if workers<1:raise ValueError('component-jobs must be positive')
    def prepare_one(spec):
        i,s=spec
        e=min(s+1_000_000,end);left=max(padded_start,s-200_000);right=min(padded_end,e+200_000)
        directory=work/f'component-{i:05d}';directory.mkdir(exist_ok=True)
        cmd=[sys.executable,HERE/'run_v04.py','--components-only','--region',f'{chrom}:{left+1}-{right}',
            '--checkpoint',a.checkpoint,'--trust-checkpoint',a.trust_checkpoint,'--config',a.config,
            '--reference',a.reference,'--output',directory/'unused.vcf.gz','--work-dir',directory]
        for value in a.bam:cmd+=['--bam',value]
        for name in ['panel','genetic_map','exclude_samples','panel_reader','states','ne','min_support','min_bq','min_mq','max_indel',
                     'artifact_prior','indel_artifact_prior','context_gap','context_span','shard_rows','device']:
            value=getattr(a,name)
            if value is not None:cmd+=['--'+name.replace('_','-'),str(value)]
        for name in ['no_cohort','no_ld_proposals']:
            if getattr(a,name):cmd+=['--'+name.replace('_','-')]
        execute(cmd,directory/'components.log')
        component=json.loads((directory/'COMPONENTS.json').read_text())
        owned_left=padded_start if s==start else s;owned_right=padded_end if e==end else e
        component.update(start0=owned_left,end0=owned_right,directory=str(directory))
        return component,Chunk(str(i),chrom,owned_left,owned_right,left,right)
    if workers==1:
        for component,chunk in map(prepare_one,specs):
            components.append(component);chunks.append(chunk)
    else:
        with ThreadPoolExecutor(max_workers=min(workers,len(specs))) as pool:
            for component,chunk in pool.map(prepare_one,specs):
                components.append(component);chunks.append(chunk)
    joint={};prepared_partials=None
    local_budget=int(os.environ.get('VARDEPTH_COMPONENT_CPU_BUDGET','1'))
    evidence_jobs=int(os.environ.get('VARDEPTH_EVIDENCE_JOBS','1'))
    overlap_joint=bool(a.panel) and local_budget>evidence_jobs
    partial_pool=None
    try:
        if overlap_joint:
            partial_pool=ProcessPoolExecutor(max_workers=min(workers,len(components)),mp_context=multiprocessing.get_context('spawn'))
            futures=[partial_pool.submit(partial_evidence,c,a.bam,a.reference) for c in components]
        if a.panel:
            joint=joint_ld.run(components,a.reference,a.panel,a.genetic_map,chrom,padded_start,padded_end,work/'joint-ld',
                exclude_samples=[x for x in a.exclude_samples.split(',') if x])
        if partial_pool is not None:prepared_partials=[f.result() for f in futures]
    finally:
        if partial_pool is not None:partial_pool.shutdown(wait=True,cancel_futures=True)
    # Ordered map keeps deterministic ownership and merge order. Each worker
    # opens its own BAM/FASTA handles and writes its own component directory.
    tasks=[(component,chunk,a.bam,a.reference,a.trust_checkpoint,joint,samples,chrom,length)
           for component,chunk in zip(components,chunks)]
    if prepared_partials is not None:tasks=[(*task,path) for task,path in zip(tasks,prepared_partials)]
    if workers==1:
        paths=[finalize_component(task) for task in tasks]
    else:
        # Spawn workers receive paths and metadata; no shared pysam handles or
        # inherited native-thread state. Ordered map preserves all output order.
        with ProcessPoolExecutor(max_workers=min(workers,len(tasks)),
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            paths=list(pool.map(finalize_component,tasks))
    merged=merge_owned_vcfs(chunks,paths,a.output,samples,{chrom:length})
    report=dict(version='0.5.11-graph-whole-high-residual-process-finalization',arguments=vars(a),**merged,seconds=time.monotonic()-started,
        truth_used=False,cohort='automatic' if not a.no_cohort else 'explicit ablation',
        cpu_execution=dict(joint_overlapped_with_reads=overlap_joint,evidence_order='locality bins balanced by estimated cost; genomic output order restored'),
        joint_ld=dict(scope='4 Mb core plus external padding, 1 Mb read-component cores',outputs=joint),
        components=components,checkpoint_sha256=sha(a.checkpoint),trust_checkpoint_sha256=sha(a.trust_checkpoint))
    for suffix in ['.run.json','.pipeline.json']:
        Path(a.output+suffix).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);add_inputs(p)
    for key in ['checkpoint','trust-checkpoint','output','work-dir','config','core-region']:p.add_argument('--'+key,required=True)
    p.add_argument('--component-jobs',type=int,default=4)
    p.add_argument('--device',default='cpu');p.add_argument('--no-cohort',action='store_true')
    run(p.parse_args())
