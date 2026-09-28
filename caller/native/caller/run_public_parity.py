#!/usr/bin/env python3
"""Public GIAB BAM/CRAM feature identity under the frozen preparation code."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare import source_identity
from call import load_feature_directory
import numpy as np


def main():
    here=Path(__file__).resolve().parent;out=here/'results/public-bam-cram'
    out.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((here/'feature.freeze.v020.json').read_text())
    if source_identity()!=frozen['source_sha256']:raise RuntimeError('Feature source changed after freeze')
    data=Path('/home/researcher/neural-data');bam=data/'train/HG005.d1.s17.bam'
    reference=data/'reference/chr22.fa';cram=out/'HG005.d1.regional.cram'
    # The public BAM retains a whole-reference SQ header but only chr22 FASTA
    # is downloaded. Use lossless reference-free CRAM for this conversion,
    # avoiding spurious attempts to fetch absent unused contigs from a network.
    subprocess.run(['samtools','view','-C','--output-fmt-option','no_ref=1','-o',str(cram),str(bam),'chr22:23999501-24100500'],check=True)
    subprocess.run(['samtools','index',str(cram)],check=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VARDEPTH_HASH_CACHE_DIR=str(out/'input-hashes'))
    for label,path in [('bam',bam),('cram',cram)]:
        command=[sys.executable,str(here/'prepare.py'),'--bam','HG005='+str(path),'--reference',str(reference),
                 '--panel',str(data/'panel/panel.clean.bcf'),'--genetic-map',str(data/'reference/chr22.b38.gmap.gz'),
                 '--region','chr22:24000001-24100000','--out',str(out/label)]
        with (out/(label+'.log')).open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    a,ma=load_feature_directory(out/'bam');b,mb=load_feature_directory(out/'cram')
    if ma['records']!=mb['records']:raise AssertionError('Record catalogs differ')
    for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    report={'source_freeze_sha256':hashlib.sha256((here/'feature.freeze.v020.json').read_bytes()).hexdigest(),
            'public_sample':'HG005','region':'chr22:24000001-24100000','records':len(ma['records']),
            'arrays_exactly_identical':list(a),'all_identical':True,
            'ld_stage':ma['base']['ld_proposal_stage'],
            'scope':'Functional feature equality only; no accuracy assessment or truth access'}
    report['cram_encoding']='lossless no_ref=1; whole-BAM SQ header preserved, regional chr22 reads'
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
