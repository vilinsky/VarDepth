#!/usr/bin/env python3
"""Fitted public-smoke model: one-command inference, reuse, and CRAM equality."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pysam


def body(path):
    with pysam.VariantFile(str(path)) as stream:return [str(record) for record in stream]


def main():
    here=Path(__file__).resolve().parent;out=here/'results/pipeline-functional';out.mkdir(parents=True,exist_ok=True)
    data=Path('/home/researcher/neural-data');checkpoint=Path('/home/researcher/neural-results/pilot/model/best.pt')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,str(here/'pipeline.py'),'--bam','HG005='+str(data/'train/HG005.d1.s17.bam'),
             '--reference',str(data/'reference/chr22.fa'),'--panel',str(data/'panel/panel.clean.bcf'),
             '--genetic-map',str(data/'reference/chr22.b38.gmap.gz'),'--region','chr22:24000001-24100000',
             '--checkpoint',str(checkpoint),'--device','cuda','--work-dir',str(out/'cache')]
    for label in ['first','reused']:
        with (out/(label+'.log')).open('w') as log:
            subprocess.run(command+['--output',str(out/(label+'.vcf.gz'))],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    a=json.loads((out/'first.vcf.gz.pipeline.json').read_text());b=json.loads((out/'reused.vcf.gz.pipeline.json').read_text())
    if a['features_reused'] or not b['features_reused']:raise AssertionError('Feature reuse did not match expected fresh/reused runs')
    if a['feature_cache_key']!=b['feature_cache_key']:raise AssertionError('Feature cache identity changed')
    if body(out/'first.vcf.gz')!=body(out/'reused.vcf.gz'):raise AssertionError('Repeated inference records differ')
    cram_command=[sys.executable,str(here/'call.py'),'--features',str(here/'results/public-bam-cram/cram'),
                  '--checkpoint',str(checkpoint),'--device','cuda','--output',str(out/'cram.vcf.gz')]
    with (out/'cram.log').open('w') as log:
        subprocess.run(cram_command,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    if body(out/'first.vcf.gz')!=body(out/'cram.vcf.gz'):raise AssertionError('Neural BAM/CRAM VCF records differ')
    summary={'status':'PASS','public_sample':'HG005','region':'chr22:24000001-24100000',
              'records':len(body(out/'first.vcf.gz')),'first_prepares_then_reuses':True,
              'repeated_vcf_records_exact':True,'bam_cram_neural_vcf_records_exact':True,
              'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'feature_cache_key':a['feature_cache_key'],
              'scope':'Functional integration using a fitted public training-smoke checkpoint, not independent accuracy validation'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
