#!/usr/bin/env python3
"""Append ten predefined heterogeneous-depth batches; frozen feature code unchanged."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from prepare import source_identity


def main():
    here=Path(__file__).resolve().parent;data=Path('/home/researcher/neural-data');root=Path('/home/researcher/neural-features')
    original=here/'feature.freeze.v020.json';base=json.loads(original.read_text())
    if source_identity()!=base['source_sha256']:raise RuntimeError('Feature source differs from original60-job freeze')
    rows=[];samples=['HG001','HG005','HG006'];depths=['0.1','6','30']
    for split,start,chunks in [('validation',29000000,2),('train',24000000,8)]:
        for chunk in range(chunks):
            region=f'chr22:{start+chunk*500000+1}-{start+(chunk+1)*500000}'
            output=root/split/f'{chunk:03d}.dheterogeneous';mapping={s:depths[(i+chunk)%3] for i,s in enumerate(samples)}
            command=[sys.executable,str(here/'prepare.py'),'--reference',str(data/'reference/chr22.fa'),
                     '--panel',str(data/'panel/panel.clean.bcf'),'--genetic-map',str(data/'reference/chr22.b38.gmap.gz'),
                     '--region',region,'--out',str(output),'--states','16','--shard-rows','512']
            bams=[]
            for sample in samples:
                bam=data/split/f'{sample}.d{mapping[sample]}.s17.bam'
                if not bam.exists() or not Path(str(bam)+'.bai').exists():raise ValueError('Missing complete input '+str(bam))
                bams.append(str(bam));command+=['--bam',sample+'='+str(bam)]
            rows.append(dict(split=split,depth='heterogeneous',chunk=chunk,region=region,output=str(output),bams=bams,sample_depths=mapping,command=command))
    freeze={**base,'version':'0.2.0-feature-freeze-heterogeneous-extension','original_plan_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
            'jobs':base['jobs']+rows,'depths':base['depths']+['heterogeneous'],'additional_jobs':10,
            'extension_policy':'Rotate nominal0.1/6/30 across HG001/HG005/HG006 by chunk index; same frozen feature code',
            'append_runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path=here/'feature.freeze.v020-heterogeneous.json'
    if path.exists():
        if json.loads(path.read_text())!=freeze:raise RuntimeError('Extension plan changed')
    else:path.write_text(json.dumps(freeze,indent=2)+'\n')
    lock=threading.Lock();results=[];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VARDEPTH_HASH_CACHE_DIR=str(root/'input-hashes'))
    def run(row):
        if source_identity()!=freeze['source_sha256']:raise RuntimeError('Frozen feature source changed')
        output=Path(row['output']);started=time.monotonic();done=output/'metadata.json';reused=done.exists()
        if reused:
            if json.loads(done.read_text())['source_sha256']!=freeze['source_sha256']:raise RuntimeError('Existing feature source mismatch')
            code=0
        else:
            output.parent.mkdir(exist_ok=True,parents=True)
            with (output.parent/(output.name+'.log')).open('w') as log:code=subprocess.run(row['command'],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        result={**row,'reused':reused,'exit_code':code,'seconds':time.monotonic()-started}
        if not code:result['rows']=len(json.loads(done.read_text())['rows'])
        with lock:
            results.append(result);tmp=root/'matrix.heterogeneous.json.tmp'
            tmp.write_text(json.dumps({'freeze_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'jobs':results},indent=2)+'\n')
            tmp.replace(root/'matrix.heterogeneous.json')
        print(json.dumps({k:result[k] for k in ['split','chunk','exit_code','seconds']}),flush=True)
        if code:raise RuntimeError('Heterogeneous feature preparation failed')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,rows))
    (root/'HETEROGENEOUS_COMPLETE.json').write_text(json.dumps({'jobs':10,'rows':sum(x['rows'] for x in results),'freeze_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':main()
