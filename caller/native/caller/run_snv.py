#!/usr/bin/env python3
"""Regional SNP-scaffold integration; indel refinement disabled by default.

New components are evaluated on <=1 Mb intervals. Whole-genome scalability and
cross-window phasing have not been established by this experiment.
"""
import argparse,json,pathlib,subprocess,sys,time
from base_adapter import parse_region
from prepare import sha
HERE=pathlib.Path(__file__).resolve().parent

def command(args,log):
    with pathlib.Path(log).open('w') as out:subprocess.run(list(map(str,args)),check=True,stdout=out,stderr=subprocess.STDOUT)

def run(a):
    started=time.monotonic();chrom,start,end=parse_region(a.region)
    if end-start>1_000_000:raise ValueError('This experimental runner is validated for regional intervals <=1 Mb')
    if not 1<=a.chunk_size<=1_000_000:raise ValueError('Invalid chunk size')
    if a.panel and not a.genetic_map:raise ValueError('A matching genetic map is required with the panel')
    config=json.loads(pathlib.Path(a.config).read_text());work=pathlib.Path(a.work_dir);work.mkdir(parents=True,exist_ok=True)
    if pathlib.Path(a.output).exists():raise ValueError('Output exists; choose a new destination')
    common=['--reference',a.reference]
    for bam in a.bam:common+=['--bam',bam]
    panel=['--panel',a.panel,'--genetic-map',a.genetic_map] if a.panel else []
    calls=[];features=[];runs=[]
    for i,s in enumerate(range(start,end,a.chunk_size)):
        part=work/f'part-{i:03d}';part.mkdir(exist_ok=True);vcf=part/'neural.vcf.gz';calls.append(vcf)
        cmd=[sys.executable,HERE/'pipeline.py',*common,*panel,'--region',f'{chrom}:{s+1}-{min(end,s+a.chunk_size)}',
             '--checkpoint',a.checkpoint,'--output',vcf,'--work-dir',part/'cache','--device',a.device]
        command(cmd,part/'pipeline.log');metadata=json.loads(pathlib.Path(str(vcf)+'.pipeline.json').read_text())
        features.append(metadata['feature_directory']);runs.append(metadata)
    concatenated=work/'neural-concatenated.vcf.gz';merged=work/'neural.vcf.gz'
    command(['bcftools','concat','-a','-Oz','-o',concatenated,*calls],work/'concat.log')
    command(['bcftools','sort','-Oz','-o',merged,concatenated],work/'sort.log')
    command(['bcftools','index','-f','-t',merged],work/'index.log')
    scaffold=work/'scaffold.npz'
    if a.panel:
        cmd=[sys.executable,HERE/'scaffold.py',*common,*panel,'--region',a.region,'--output',scaffold]
        for key,default in [('states',512),('burn',8),('iterations',24),('chains',2)]:cmd+=['--'+key,str(config.get('scaffold_'+key,default))]
        command(cmd,work/'scaffold.log')
    refinement=None
    if config.get('anchored_weight',0)>0:
        refinement=work/'refinements.json';cmd=[sys.executable,HERE/'indel_refine.py',*common,'--output',refinement]
        for path in features:cmd+=['--features',path]
        command(cmd,work/'refine.log')
    cmd=[sys.executable,HERE/'fuse.py','--neural',merged,'--config',a.config,'--output',a.output]
    if refinement:cmd+=['--refinements',refinement]
    if a.panel:cmd+=['--scaffold',scaffold]
    command(cmd,work/'fuse.log')
    report=dict(version='0.3-snv-integration',arguments=vars(a),config=config,cohort='automatic; singleton supported',
                baseline_runs=runs,output_sha256=sha(a.output),checkpoint_sha256=sha(a.checkpoint),
                seconds=time.monotonic()-started,truth_used=False)
    pathlib.Path(a.output+'.run.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bam',action='append',required=True)
    for k in ['reference','region','checkpoint','output','work-dir']:p.add_argument('--'+k,required=True)
    p.add_argument('--panel');p.add_argument('--genetic-map');p.add_argument('--device',default='cpu')
    p.add_argument('--config',default=str(HERE.parent/'snv_only_config.json'));p.add_argument('--chunk-size',type=int,default=500000)
    run(p.parse_args())
