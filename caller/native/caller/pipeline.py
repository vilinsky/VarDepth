#!/usr/bin/env python3
"""One-command BAM/CRAM -> cohort-aware neural VCF with verified feature reuse."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from prepare import add_inputs, validate, source_identity, sha, prepare
from base_adapter import base_modules
from call import load_feature_directory

HERE = Path(__file__).resolve().parent
FEATURE_OPTIONS = ('bam','reference','region','panel','genetic_map','exclude_samples','panel_reader',
                   'states','ne','min_support','min_bq','min_mq','max_indel','artifact_prior',
                   'indel_artifact_prior','context_gap','context_span','shard_rows','no_cohort','no_ld_proposals')


def feature_fingerprint(arguments, input_identity, source_sha256):
    payload = {'arguments':arguments,'input_identity':input_identity,'source_sha256':source_sha256}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest(),payload


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);add_inputs(parser)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--work-dir',required=True)
    parser.add_argument('--device',default='cpu');parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--min-gp',type=float,default=0.);parser.add_argument('--min-gq',type=int,default=20)
    parser.add_argument('--variants-only',action='store_true');parser.add_argument('--no-cohort',action='store_true')
    args=parser.parse_args(argv);validate(args);started=time.monotonic()
    if args.batch_size<1 or not 0<=args.min_gp<=1 or args.min_gq<0:parser.error('Invalid neural batch or quality threshold')
    work=Path(args.work_dir);work.mkdir(parents=True,exist_ok=True)
    # Full file hashes are cached by source stat identity with locks and atomic
    # updates; changing an input or feature algorithm cannot reuse stale data.
    import os
    os.environ.setdefault('VARDEPTH_HASH_CACHE_DIR',str(work/'input-hashes'))
    files={args.reference,*([args.panel] if args.panel else []),*([args.genetic_map] if args.genetic_map else []),
           *(x.split('=',1)[1] for x in args.bam)}
    caller=base_modules()
    identities={p:caller.file_identity(p) for p in sorted(files)}
    feature_args={k:getattr(args,k) for k in FEATURE_OPTIONS}
    # Base likelihoods can now execute on CUDA. Keep accelerator identity in
    # feature-cache semantics, even when the mathematical model is unchanged.
    from accelerator import backend
    acceleration={'pairhmm_backend':backend()}
    if backend()!='cpu':
        from accelerator import CudaPairHMM
        try:acceleration['runtime']=CudaPairHMM(int(os.environ.get('VARDEPTH_CUDA_DEVICE','0')),int(os.environ.get('VARDEPTH_CUDA_MEMORY_MB','256'))).metadata
        except RuntimeError:
            if backend()=='cuda':raise
            acceleration['runtime']={'fallback':'cpu','cuda_available':False}
    from evidence_trace import enabled as trace_enabled, VERSION as TRACE_VERSION
    feature_args=dict(feature_args,acceleration_identity=acceleration,
                      evidence_trace_version=TRACE_VERSION if trace_enabled() else None)
    key,identity=feature_fingerprint(feature_args,identities,source_identity())
    directory=work/('features-'+key[:24]);lockpath=work/(directory.name+'.lock');reused=False
    with lockpath.open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if (directory/'metadata.json').exists():
            stored=json.loads((directory/'pipeline-cache.json').read_text())
            if stored!=identity:raise RuntimeError('Feature cache identity differs; cache may have been altered')
            load_feature_directory(directory,verify=True);reused=True
        else:
            if directory.exists() and any(directory.iterdir()):
                stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
                directory.rename(work/(directory.name+'.interrupted-'+stamp))
            feature_namespace=argparse.Namespace(**{k:v for k,v in feature_args.items() if k not in ('acceleration_identity','evidence_trace_version')},out=str(directory))
            prepare(feature_namespace)
            (directory/'pipeline-cache.json').write_text(json.dumps(identity,indent=2)+'\n')
    command=[sys.executable,str(HERE/'call.py'),'--features',str(directory),'--checkpoint',args.checkpoint,
             '--output',args.output,'--device',args.device,'--batch-size',str(args.batch_size),
             '--min-gp',str(args.min_gp),'--min-gq',str(args.min_gq)]
    if args.no_cohort:command.append('--no-cohort')
    if args.variants_only:command.append('--variants-only')
    from call import main as call_main
    call_main(command[2:])
    report=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),feature_cache_key=key,
                feature_directory=str(directory),features_reused=reused,arguments=vars(args),
                input_identity=identities,checkpoint_sha256=sha(args.checkpoint),
                pipeline_sha256=sha(HERE/'pipeline.py'),call_command=command,
                output_sha256=sha(args.output),wall_seconds=time.monotonic()-started)
    Path(args.output+'.pipeline.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':args.output,'features_reused':reused,'feature_cache_key':key,'seconds':report['wall_seconds']}))


if __name__=='__main__':main()
