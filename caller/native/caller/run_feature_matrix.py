#!/usr/bin/env python3
"""Bounded checkpointed train/validation preparation; no truth or test inputs."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from prepare import source_identity

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='/home/researcher/neural-data')
    p.add_argument('--out',default='/home/researcher/neural-features')
    p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--freeze',default=str(HERE/'feature.freeze.v020.json'))
    a=p.parse_args()
    if not 1<=a.jobs<=8:p.error('Use 1..8 independent CPU workers')
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    data=Path(a.data);freeze_path=Path(a.freeze)
    rows=[]
    # Validation appears early to allow a bounded initial training/checkpoint
    # smoke; the full matrix remains predefined and immutable.
    for depth in ['1','6','30','0.1','mixed','100']:
        for split,start,chunks in [('validation',29000000,2),('train',24000000,8)]:
            for chunk in range(chunks):
                region=f'chr22:{start+chunk*500000+1}-{start+(chunk+1)*500000}'
                output=root/split/f'{chunk:03d}.d{depth}'
                command=[sys.executable,str(HERE/'prepare.py'),'--reference',str(data/'reference/chr22.fa'),
                         '--panel',str(data/'panel/panel.clean.bcf'),'--genetic-map',str(data/'reference/chr22.b38.gmap.gz'),
                         '--region',region,'--out',str(output),'--states','16','--shard-rows','512']
                bams=[]
                for sample in ['HG001','HG005','HG006']:
                    bam=data/split/f'{sample}.d{depth}.s17.bam'
                    if not bam.exists() or not Path(str(bam)+'.bai').exists():raise ValueError('Missing complete public input: '+str(bam))
                    command+=['--bam',sample+'='+str(bam)];bams.append(str(bam))
                rows.append({'split':split,'depth':depth,'chunk':chunk,'region':region,'output':str(output),'bams':bams,'command':command})
    manifest={'version':'0.2.0-feature-freeze','source_sha256':source_identity(),
              'samples':['HG001','HG005','HG006'],'splits':{'train':'chr22:24000001-28000000','validation':'chr22:29000001-30000000'},
              'depths':['0.1','1','6','30','100','mixed'],'ld_proposal_stage':True,'cohort_context':True,
              'truth_access':False,'jobs':rows,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if freeze_path.exists():
        if json.loads(freeze_path.read_text())!=manifest:raise RuntimeError('Feature freeze/configuration changed')
    else:freeze_path.write_text(json.dumps(manifest,indent=2)+'\n')
    lock=threading.Lock();results=[]
    def save():
        with lock:
            tmp=root/'matrix.json.tmp'
            tmp.write_text(json.dumps({'freeze_sha256':hashlib.sha256(freeze_path.read_bytes()).hexdigest(),'jobs':results},indent=2)+'\n')
            tmp.replace(root/'matrix.json')
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     VARDEPTH_HASH_CACHE_DIR=str(root/'input-hashes'))
    def run(row):
        if source_identity()!=manifest['source_sha256']:raise RuntimeError('Frozen feature source changed before job')
        output=Path(row['output']);started=time.monotonic();done=output/'metadata.json'
        if done.exists():
            metadata=json.loads(done.read_text())
            if metadata['source_sha256']!=manifest['source_sha256'] or metadata['arguments']['region']!=row['region']:
                raise RuntimeError('Existing feature metadata uses a different source or region')
            result={**row,'reused':True,'seconds':0.,'exit_code':0,'rows':len(metadata['rows'])}
        else:
            if output.exists() and any(output.iterdir()):
                stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
                output.rename(output.parent/(output.name+'.interrupted-'+stamp))
            output.parent.mkdir(parents=True,exist_ok=True)
            log=output.parent/(output.name+'.log')
            with log.open('w') as stream:
                call=subprocess.run(row['command'],env=environment,stdout=stream,stderr=subprocess.STDOUT)
            result={**row,'reused':False,'seconds':time.monotonic()-started,'exit_code':call.returncode}
            if call.returncode==0:result['rows']=len(json.loads(done.read_text())['rows'])
        with lock:results.append(result)
        save();print(json.dumps({k:result[k] for k in ['split','depth','chunk','exit_code','seconds']}),flush=True)
        if result['exit_code']:raise RuntimeError('Feature preparation failed: '+str(output))
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(run,rows))
    (root/'FEATURE_MATRIX_COMPLETE.json').write_text(json.dumps({'jobs':len(rows),'rows':sum(x['rows'] for x in results),
                'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_sha256':hashlib.sha256(freeze_path.read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':main()
