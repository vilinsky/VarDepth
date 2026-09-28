#!/usr/bin/env python3
"""Prepare truth-free molecular and external-LD features from BAM/CRAM."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np

from features import SCHEMA_VERSION, FRAGMENT_METADATA, ALLELE_METADATA, build_row, empty_arrays
from base_adapter import compute, fragment_lookup, initial_cohort_context

HERE = Path(__file__).resolve().parent


def add_inputs(parser, required=True):
    parser.add_argument('--bam', action='append', required=required, help='SAMPLE=path, repeated across individuals/libraries')
    parser.add_argument('--reference', required=required)
    parser.add_argument('--region', required=required, help='CHROM:START-END, 1-based inclusive; bounded regional preparation')
    parser.add_argument('--panel')
    parser.add_argument('--genetic-map')
    parser.add_argument('--exclude-samples', default='')
    parser.add_argument('--panel-reader', choices=['fast', 'python'], default='fast')
    parser.add_argument('--states', type=int, default=16)
    parser.add_argument('--ne', type=float, default=100000.)
    parser.add_argument('--min-support', type=int, default=2)
    parser.add_argument('--min-bq', type=int, default=20)
    parser.add_argument('--min-mq', type=int, default=10)
    parser.add_argument('--max-indel', type=int, default=50)
    parser.add_argument('--artifact-prior', type=float, default=.01)
    parser.add_argument('--indel-artifact-prior', type=float, default=.001)
    parser.add_argument('--context-gap', type=int, default=30)
    parser.add_argument('--context-span', type=int, default=100)
    parser.add_argument('--no-ld-proposals', action='store_true', help='Ablate the bounded LD-posterior selected local sequence proposal/realignment stage')
    parser.add_argument('--shard-rows', type=int, default=512)


def validate(args):
    if not args.bam or not args.reference or not args.region:
        raise ValueError('--bam, --reference, and --region are required')
    if not 1 <= args.states <= 64 or args.shard_rows < 1:
        raise ValueError('states must be 1..64 and shard-rows positive')
    if not 1 <= args.max_indel <= 50 or args.min_support < 1:
        raise ValueError('max-indel must be 1..50 and support positive')
    if not 0 <= args.min_bq <= 60 or not 0 <= args.min_mq <= 60:
        raise ValueError('BQ/MQ must be 0..60')
    if not all(math.isfinite(x) and 0 <= x < 1 for x in [args.artifact_prior, args.indel_artifact_prior]):
        raise ValueError('Artifact priors must be finite and in [0,1)')
    if not math.isfinite(args.ne) or args.ne <= 0 or args.context_gap < 0 or not 50 <= args.context_span <= 100:
        raise ValueError('Invalid Ne or bounded context parameters')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def source_identity():
    paths = [HERE/'features.py', HERE/'prepare.py', HERE/'base_adapter.py',HERE/'ld_proposals.py']
    paths += [HERE/'accelerator.py',HERE/'pairhmm_cuda.cu',HERE/'pairhmm_warp.cu',HERE/'whole_pairhmm.py',HERE/'wide.cpp']
    paths += list(HERE.glob('*.pyx')) + list(HERE.glob('*.py')) + list(HERE.glob('*.cpp')) + list(HERE.glob('*.hpp'))
    if (HERE/'cohort_context.py').exists():
        paths.append(HERE/'cohort_context.py')
    paths += list((HERE/'base').glob('*.py')) + list((HERE/'base').glob('*.sh')) + list((HERE/'base/core').glob('*.cpp'))
    return {str(p.relative_to(HERE)): sha(p) for p in sorted(paths)}


def prepare(args):
    validate(args)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    if (output/'metadata.json').exists() or (output/'PREPARING.json').exists():
        raise ValueError('Output already contains a completed or interrupted preparation; use a new directory')
    began = time.monotonic()
    (output/'PREPARING.json').write_text(json.dumps({'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arguments':vars(args)}, indent=2)+'\n')
    base = compute(args)
    from base_adapter import base_modules
    caller = base_modules()
    import pysam
    identities = {p:caller.file_identity(p) for p in sorted({args.reference, *([args.panel] if args.panel else []),
                   *([args.genetic_map] if args.genetic_map else []), *(p for paths in base['bams'].values() for p in paths)})}
    fragment_maps = {s:None for s in base['samples']} if 'observation_alignments' in base else {s:fragment_lookup(base['raw'][s]) for s in base['samples']}
    cohort, strength, cohort_provenance = initial_cohort_context(base, enabled=not getattr(args,'no_cohort',False))
    from evidence_trace import enabled as trace_enabled, BaseTrace
    lineage = BaseTrace(base,args,identities) if trace_enabled() else None
    rows, shard_rows, shards, notes = [], [], [], []

    def flush():
        nonlocal shard_rows
        if not shard_rows:
            return
        name = f'features.{len(shards):05d}.npz'
        arrays = {k:np.stack([r[k] for r in shard_rows]) for k in shard_rows[0]}
        np.savez_compressed(output/name, **arrays)
        shards.append(dict(path=name, rows=len(shard_rows), sha256=sha(output/name)))
        shard_rows = []

    for record in base['records']:
        for sample_index, sample in enumerate(base['samples']):
            payload, lh, keys, metadata = __import__('base_adapter').record_evidence(base, record, sample, fragment_maps[sample])
            row, detail = build_row(record, payload, lh, keys, metadata, base['fetch'], sample)
            if lineage is not None:
                lineage.add(record,sample,keys,detail['lineage_neural_molecules'])
            row.update(record_index=np.int32(record['record_index']), sample_index=np.int32(sample_index), positions=np.int64(record['start0']))
            row['cohort_context'] = np.asarray(cohort[sample_index,record['record_index']],np.float32)
            row['context_strength'] = np.float32(strength[sample_index,record['record_index']])
            identity = dict(row_index=len(rows), record_index=record['record_index'], sample_index=sample_index,
                            sample=sample, record_id=record['record_id'], shard=f'features.{len(shards):05d}.npz', shard_row=len(shard_rows))
            rows.append(identity); notes.append({**identity, **detail}); shard_rows.append(row)
            if len(shard_rows) == args.shard_rows:
                flush()
    flush()
    if not shards:
        name = 'features.00000.npz'; np.savez_compressed(output/name, **empty_arrays())
        shards.append(dict(path=name, rows=0, sha256=sha(output/name)))
    with (output/'molecular_audit.jsonl').open('w') as stream:
        for row in notes:
            stream.write(json.dumps(row, separators=(',', ':'))+'\n')
    public_records = [{k:v for k,v in r.items() if k != 'first_seed'} for r in base['records']]
    if lineage is not None:
        lineage.finish(output)
    metadata = dict(schema_version=SCHEMA_VERSION, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    samples=base['samples'], records=public_records, rows=rows, shards=shards,
                    dimensions=dict(alleles=12, genotypes=78, fragments=64, sequence_length=256),
                    feature_names=dict(fragment_metadata=list(FRAGMENT_METADATA), allele_metadata=list(ALLELE_METADATA)),
                    row_order='record-major, then input sample order; explicit indices authoritative',
                    genotype_order='for b in range(A): for a in range(b+1): (a,b)',
                    base_posterior='External-only copying HMM GP; independent across sample read sets conditional on shared catalog',
                    read_pl_semantics='Phred-normalized read-only local shared-artifact/allele-balance likelihood; not posterior',
                    arguments=vars(args), input_identity=identities, source_sha256=source_identity(),
                    base=base['manifest'], cohort=cohort_provenance, wall_seconds=time.monotonic()-began,
                    software=dict(python=sys.version, numpy=np.__version__, pysam=pysam.__version__, platform=platform.platform()),
                    limitations=['Bounded CIGAR-seeded allele enumeration, not de novo assembly',
                                 'PairHMM contexts and partial-read hidden-allele marginalization remain approximate',
                                 '64 molecular features are hash-sampled; base GP, read PL, AD, DP use all retained fragments',
                                 'Shared candidate ascertainment uses all batch read support; per-sample initial inference uses only that sample reads',
                                 'Missing NM is encoded as zero; missing read-group library identity is recorded in metadata warnings',
                                 'LD-first local proposals use initial posterior only to choose candidates; final likelihoods recomputed and original external prior reused once',
                                 'No full-genome impute-first remapping or pangenome placement'])
    temporary = output/'metadata.json.tmp'
    temporary.write_text(json.dumps(metadata,separators=(',',':'))+'\n'); temporary.replace(output/'metadata.json')
    (output/'PREPARING.json').unlink()
    base['fasta'].close()
    print(json.dumps({'out':str(output), 'records':len(public_records), 'rows':len(rows), 'shards':len(shards), 'seconds':metadata['wall_seconds']}), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inputs(parser)
    parser.add_argument('--out', required=True)
    parser.add_argument('--no-cohort', action='store_true', help='Explicit cohort ablation; default includes leave-self-out context for multiple people')
    args = parser.parse_args()
    try:
        prepare(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
