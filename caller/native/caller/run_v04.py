#!/usr/bin/env python3
"""BAM/CRAM -> called and imputed short variants, with automatic cohort input.

The outer genome scheduler supplies overlapping LD windows. Within each window,
bounded feature chunks use read/context padding and normalized-start ownership.
All tools invoked here are this project's own scientific code plus HTSlib I/O.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import pysam
from prepare import add_inputs, sha
from base_adapter import parse_region
from genome import Chunk, filter_owned_core

HERE = Path(__file__).resolve().parent


def execute(cmd, log):
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    command=list(map(str,cmd))
    allowed={'pipeline.py','scaffold.py','indel_refine.py','fuse_trust.py','run_v04.py'}
    inprocess=(os.environ.get('VARDEPTH_INPROCESS_CPU_STAGES')=='1' and len(command)>1
               and command[0]==sys.executable and Path(command[1]).resolve().parent==HERE
               and Path(command[1]).name in allowed)
    with log.open('w') as f:
        if inprocess:
            import contextlib,importlib
            with contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
                module=importlib.import_module(Path(command[1]).stem)
                if Path(module.__file__).resolve().parent!=HERE:raise RuntimeError('Wrong in-process stage module')
                module.main(command[2:])
        else:subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT)


def run(a):
    started = time.monotonic(); chrom, start, end = parse_region(a.region)
    work = Path(a.work_dir); work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    config = json.loads(Path(a.config).read_text())
    if a.panel and not a.genetic_map:
        raise ValueError('A genetic map is required with an external panel')
    if not 1 <= a.chunk_size <= 1_000_000 or not 0 <= a.feature_padding <= 100_000:
        raise ValueError('Invalid feature chunk/padding')
    with pysam.FastaFile(a.reference) as fa:
        length = fa.get_reference_length(chrom)
    if end > length:
        raise ValueError('Region outside reference')
    common = ['--reference', a.reference]
    for bam in a.bam:
        common += ['--bam', bam]
    panel = ['--panel', a.panel, '--genetic-map', a.genetic_map] if a.panel else []
    feature_options = []
    for key in ['states', 'ne', 'min_support', 'min_bq', 'min_mq', 'max_indel', 'artifact_prior',
                'indel_artifact_prior', 'context_gap', 'context_span', 'shard_rows', 'panel_reader', 'exclude_samples']:
        feature_options += ['--' + key.replace('_', '-'), str(getattr(a, key))]
    for key in ['no_cohort', 'no_ld_proposals']:
        if getattr(a, key):
            feature_options.append('--' + key.replace('_', '-'))
    features = []; calls = []; runs = []; ownership = {}
    samples = list(dict.fromkeys(b.split('=', 1)[0] for b in a.bam))
    scaffold = work / 'scaffold.npz'
    scaffold_command=None
    if a.panel:
        scaffold_command=[sys.executable,HERE/'scaffold.py',*common,*panel,'--region',a.region,'--output',scaffold]
        if a.exclude_samples:scaffold_command+=['--exclude-samples',a.exclude_samples]
        for key,default in [('states',1024),('burn',12),('iterations',48),('chains',2)]:
            scaffold_command+=['--'+key,str(config.get('scaffold_'+key,default))]
    specs=list(enumerate(range(start,end,a.chunk_size)))
    budget=int(os.environ.get('VARDEPTH_COMPONENT_CPU_BUDGET','1'))
    neural_threads=int(os.environ.get('VARDEPTH_NEURAL_CPU_THREADS','1'))
    overlap=scaffold_command is not None and budget>=neural_threads+1
    feature_jobs=min(len(specs),max(1,(budget-int(overlap))//neural_threads))
    def prepare_feature(spec):
        i,s=spec
        e = min(end, s + a.chunk_size)
        left, right = max(0, s - a.feature_padding), min(length, e + a.feature_padding)
        part = work / f'part-{i:05d}'; part.mkdir(exist_ok=True)
        vcf = part / 'padded-neural.vcf.gz'
        cmd = [sys.executable, HERE / 'pipeline.py', *common, *panel, *feature_options,
               '--region', f'{chrom}:{left + 1}-{right}', '--checkpoint', a.checkpoint,
               '--output', vcf, '--work-dir', part / 'cache', '--device', a.device]
        execute(cmd, part / 'pipeline.log')
        metadata = json.loads(Path(str(vcf) + '.pipeline.json').read_text())
        core = part / 'core-neural.vcf.gz'
        filter_owned_core(vcf, core, Chunk(str(i), chrom, s, e, left, right), samples, {chrom: length})
        return metadata,core,dict(chrom=chrom,start0=s,end0=e)
    if feature_jobs==1 and not overlap:
        for metadata,core,owned_core in map(prepare_feature,specs):
            features.append(metadata['feature_directory']);runs.append(metadata);calls.append(core)
            ownership[str(Path(metadata['feature_directory']).resolve())]=owned_core
    else:
        with ThreadPoolExecutor(max_workers=feature_jobs) as pool, ThreadPoolExecutor(max_workers=1) as scaffold_pool:
            scaffold_future=scaffold_pool.submit(execute,scaffold_command,work/'scaffold.log') if overlap else None
            for metadata,core,owned_core in pool.map(prepare_feature,specs):
                features.append(metadata['feature_directory']);runs.append(metadata);calls.append(core)
                ownership[str(Path(metadata['feature_directory']).resolve())]=owned_core
            if scaffold_future is not None:scaffold_future.result()
    owned = work / 'feature-ownership.json'; owned.write_text(json.dumps(ownership, indent=2) + '\n')
    merged = work / 'neural.vcf.gz'
    execute(['bcftools', 'concat', '-a', '-Oz', '-o', merged, *calls], work / 'concat.log')
    execute(['bcftools', 'index', '-f', '-t', merged], work / 'index.log')
    if scaffold_command is not None and not overlap:execute(scaffold_command,work/'scaffold.log')
    refinement = work / 'refinements.json'
    cmd = [sys.executable, HERE / 'indel_refine.py', *common, '--output', refinement, '--owned-regions', owned]
    if config.get('fragment_proposals', False):
        cmd.append('--fragment-proposals')
    for f in features:
        cmd += ['--features', f]
    execute(cmd, work / 'refine.log')
    if getattr(a, 'components_only', False):
        report=dict(neural=str(merged),scaffold=str(scaffold) if a.panel else None,
                    refinements=str(refinement),features=features,ownership=ownership,
                    samples=samples,arguments=vars(a),truth_used=False)
        (work/'COMPONENTS.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    temporary = work / 'unnormalized-final.vcf.gz'
    cmd = [sys.executable, HERE / 'fuse_trust.py', '--neural', merged, '--refinements', refinement,
           '--trust-checkpoint', a.trust_checkpoint, '--output', temporary]
    if a.panel:
        cmd += ['--scaffold', scaffold]
    for source in a.disable_expert:
        cmd += ['--disable-expert', source]
    execute(cmd, work / 'trust.log')
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    execute(['bcftools', 'norm', '-f', a.reference, '-c', 'e', '-Oz', '-o', a.output, temporary], work / 'normalize.log')
    execute(['bcftools', 'index', '-f', '-t', a.output], work / 'final-index.log')
    report = dict(version='0.4-regional-workflow', arguments=vars(a), config=config,
                  seconds=time.monotonic() - started, output_sha256=sha(a.output),
                  checkpoint_sha256=sha(a.checkpoint), trust_checkpoint_sha256=sha(a.trust_checkpoint),
                  baseline_runs=runs, feature_ownership=ownership, truth_used=False,
                  cohort='automatic' if not a.no_cohort else 'explicit ablation',
                  limitations=['Illumina diploid short variants; chromosome window LD is approximate',
                               'Whole-genome quality and speed require independent measurement'])
    for suffix in ['.run.json', '.pipeline.json']:
        Path(a.output + suffix).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['baseline_runs', 'feature_ownership']}), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); add_inputs(p)
    for key in ['checkpoint', 'trust-checkpoint', 'output', 'work-dir', 'config']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--device', default='cpu'); p.add_argument('--no-cohort', action='store_true')
    p.add_argument('--chunk-size', type=int, default=500000); p.add_argument('--feature-padding', type=int, default=2000)
    p.add_argument('--disable-expert', action='append', default=[])
    p.add_argument('--components-only', action='store_true')
    run(p.parse_args(argv))


if __name__=='__main__':main()
