#!/usr/bin/env python3
"""Neural multiallelic calling with automatic leave-self-out cohort context."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

from features import SCHEMA_VERSION, genotypes
from prepare import sha, add_inputs

HERE = Path(__file__).resolve().parent


def load_feature_directory(path, verify=True):
    directory = Path(path)
    metadata = json.loads((directory/'metadata.json').read_text())
    if metadata['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported feature schema')
    pieces = {}
    for shard in metadata['shards']:
        p = directory/shard['path']
        if verify and sha(p) != shard['sha256']:
            raise ValueError('Feature shard checksum mismatch: '+str(p))
        with np.load(p, allow_pickle=False) as arrays:
            for name in arrays.files:
                pieces.setdefault(name, []).append(arrays[name])
    arrays = {k:np.concatenate(v, axis=0) for k,v in pieces.items()}
    if any(len(x) != len(metadata['rows']) for x in arrays.values()):
        raise ValueError('Feature/metadata row count mismatch')
    seen = set()
    for i, row in enumerate(metadata['rows']):
        key = row['sample_index'], row['record_index']
        if key in seen or row['row_index'] != i or arrays['sample_index'][i] != key[0] or arrays['record_index'][i] != key[1]:
            raise ValueError('Duplicate, inconsistent, or noncanonical row identity')
        seen.add(key)
    if len(seen) != len(metadata['samples'])*len(metadata['records']):
        raise ValueError('Expected complete sample-by-record catalog')
    return arrays, metadata


def add_cohort_context(arrays, metadata, enabled=True):
    """Reshape explicit row indices; never assume samples are contiguous."""
    n_samples, n_records = len(metadata['samples']), len(metadata['records'])
    rows = len(metadata['rows'])
    arrays = dict(arrays)
    if not enabled or n_records == 0:
        arrays['cohort_context'] = np.zeros((rows, 12, 6), np.float32)
        arrays['context_strength'] = np.zeros(rows, np.float32)
        return arrays, {'enabled':bool(enabled), 'reason':'ablation' if not enabled else 'empty catalog', 'samples':n_samples}
    from cohort_context import build_cohort_context
    initial = np.zeros((n_samples, n_records, 78), np.float32)
    read_pl = np.zeros_like(initial)
    depth = np.zeros((n_samples, n_records), np.float32)
    sample = arrays['sample_index']; record = arrays['record_index']
    initial[sample, record] = arrays['base_gp']
    read_pl[sample, record] = arrays['read_pl']
    depth[sample, record] = arrays['depth']
    result = build_cohort_context(initial, np.array([len(r['alleles']) for r in metadata['records']]),
                                  np.array([r['start0'] for r in metadata['records']]), metadata['samples'],
                                  read_depth=depth, contigs=np.array([r['chrom'] for r in metadata['records']]),
                                  read_pl=read_pl)
    arrays['cohort_context'] = np.asarray(result.allele_context[sample, record], dtype=np.float32)
    arrays['context_strength'] = np.asarray(result.context_strength[sample, record], dtype=np.float32)
    return arrays, {'enabled':True, **result.provenance}


def check_probabilities(probabilities, allele_counts):
    gp = np.asarray(probabilities)
    if gp.shape != (len(allele_counts), 78) or not np.isfinite(gp).all() or np.any(gp < 0):
        raise ValueError('Invalid neural posterior output')
    counts=np.asarray(allele_counts)
    for count in np.unique(counts):
        n=int(count)*(int(count)+1)//2
        rows=gp[counts==count]
        if not np.isclose(rows[:,:n].sum(axis=1),1.,atol=1e-5).all() or np.any(rows[:,n:]>1e-8):
            raise ValueError('Genotype mask/posterior normalization mismatch')
    return gp


def decode_probabilities(probabilities, count):
    count = int(count)
    if count==2:
        gp=np.asarray(probabilities)[:3];best=int(np.argmax(gp));confidence=float(gp[best])
        gq=min(99,int(round(-10*math.log10(max(1e-12,1-confidence)))))
        return ((0,0),(0,1),(1,1))[best],gq,(float(0+gp[0]*0+gp[1]*1+gp[2]*2),)
    order = genotypes(count); gp = np.asarray(probabilities)[:len(order)]
    best = int(np.argmax(gp)); confidence = float(gp[best])
    gq = min(99, int(round(-10*math.log10(max(1e-12, 1-confidence)))))
    dosage = tuple(float(sum(p*(int(a==j)+int(b==j)) for p,(a,b) in zip(gp, order))) for j in range(1,count))
    return order[best], gq, dosage


def emit_vcf(output, arrays, metadata, probabilities, min_gp=0., min_gq=20, variants_only=False):
    import pysam
    import pysam.bcftools
    header = pysam.VariantHeader()
    header.add_meta('source', value='VarDepthNeuralResearch-0.2.0')
    header.add_meta('QUAL_semantics', value='Approximate Phred product of neural marginal reference probabilities; not exact joint cohort probability')
    header.add_meta('GQ_semantics', value='Derived from neural GP; calibration depends on checkpoint and heldout validation')
    header.contigs.add(metadata['base']['chrom'], length=metadata['base']['contig_length'])
    for name, number, typ, desc in [
        ('GT',1,'String','Unphased diploid genotype in local allele numbering'),
        ('GQ',1,'Integer','Phred posterior genotype error from neural probabilities'),
        ('DP',1,'Integer','All distinct local evidence fragments before neural sampling'),
        ('AD','R','Integer','All confident local allele-supporting fragments before neural sampling'),
        ('PL','G','Integer','Read-only normalized local likelihood in Phred units; not neural posterior'),
        ('GP','G','Float','Neural posterior in VCF triangular genotype order'),
        ('BGP','G','Float','Initial external-only copying-HMM posterior'),
        ('DS','A','Float','Expected alternate allele dosage from neural posterior'),
        ('CS',1,'Float','Available leave-self-out cohort context strength')]:
        header.formats.add(name,number,typ,desc)
    header.info.add('SOURCE',1,'String','Shared catalog source PANEL, READS, or BOTH')
    header.info.add('PANEL_AF','A','Float','External panel local sequence allele frequency, missing if unknown')
    header.info.add('READ_SUPPORT',1,'Integer','Shared candidate seed support; not another likelihood')
    header.filters.add('LowQual',None,None,'No nonreference sample meets requested GP and GQ thresholds')
    for sample in metadata['samples']:
        header.add_sample(sample)
    row_index = {(int(s),int(r)):i for i,(s,r) in enumerate(zip(arrays['sample_index'],arrays['record_index']))}
    mode = 'wb' if str(output).endswith('.bcf') else 'wz' if str(output).endswith('.gz') else 'w'
    counts = {'records':0, 'nonreference_sample_genotypes':0}
    with pysam.VariantFile(str(output), mode, header=header) as stream:
        for record in metadata['records']:
            count = len(record['alleles']); ng = count*(count+1)//2
            indices = [row_index[(s, record['record_index'])] for s in range(len(metadata['samples']))]
            decoded = [decode_probabilities(probabilities[i], count) for i in indices]
            if variants_only and not any(any(x[0]) for x in decoded):
                continue
            item = stream.new_record(contig=record['chrom'],start=record['start0'],stop=record['end0'],alleles=record['alleles'])
            item.info['SOURCE'] = record['source']; item.info['READ_SUPPORT'] = record['read_support']
            if any(x is not None for x in record['panel_af'][1:]):
                item.info['PANEL_AF'] = tuple(record['panel_af'][1:])
            passing = False; log_reference_probability = 0.
            for name, row, (gt,gq,ds) in zip(metadata['samples'],indices,decoded):
                gp = probabilities[row,:ng]; confident = float(gp.max()) >= min_gp
                sample = item.samples[name]
                sample['GT'] = gt if confident else (None,None)
                sample['GQ'] = gq; sample['DP'] = int(arrays['depth'][row])
                sample['AD'] = tuple(map(int, arrays['allele_depth'][row,:count]))
                sample['PL'] = tuple(map(int, np.rint(arrays['read_pl'][row,:ng])))
                sample['GP'] = tuple(map(float, gp)); sample['BGP'] = tuple(map(float, arrays['base_gp'][row,:ng]))
                sample['DS'] = ds; sample['CS'] = float(arrays['context_strength'][row])
                log_reference_probability += math.log(max(1e-100,float(gp[0])))
                passing |= any(gt) and confident and gq >= min_gq
                counts['nonreference_sample_genotypes'] += int(any(gt) and confident)
            item.qual = min(999., -10/math.log(10)*log_reference_probability)
            item.filter.add('PASS' if passing else 'LowQual'); stream.write(item); counts['records'] += 1
    if mode == 'wz':
        pysam.tabix_index(str(output), preset='vcf', force=True)
    elif mode == 'wb':
        pysam.bcftools.index('-f',str(output))
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', help='Completed prepare output directory')
    parser.add_argument('--work-dir', help='Feature directory when directly supplying BAM/CRAM inputs')
    add_inputs(parser, required=False)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--no-cohort', action='store_true', help='Explicit ablation; default automatically uses other individuals')
    parser.add_argument('--min-gp', type=float, default=0.)
    parser.add_argument('--min-gq', type=int, default=20)
    parser.add_argument('--variants-only', action='store_true')
    args = parser.parse_args(argv); started = time.monotonic()
    if args.batch_size < 1 or not 0 <= args.min_gp <= 1 or args.min_gq < 0:
        parser.error('Invalid batch size or genotype confidence threshold')
    if args.features and args.bam:
        parser.error('Use prepared features or BAM/CRAM inputs, not both')
    if not args.features:
        if not args.work_dir:
            parser.error('--work-dir is required when preparing directly from BAM/CRAM')
        from prepare import prepare
        args.out = args.work_dir
        prepare(args); args.features = args.work_dir
    arrays, metadata = load_feature_directory(args.features)
    arrays, provenance = add_cohort_context(arrays, metadata, enabled=not args.no_cohort)
    from model import load_model, predict
    import torch
    model = load_model(args.checkpoint, args.device)
    probabilities = np.zeros((len(metadata['rows']),78),np.float32)
    grouped=bool(getattr(model,'compact_inference',False)) and str(args.device)=='cpu'
    if grouped:
        # Same records, features and checkpoint; avoid padding small biallelic
        # records to the width of unrelated multiallelic records in a batch.
        # Restore the explicit original row order before any cohort/VCF output.
        for count in np.unique(arrays['n_alleles']):
            indices=np.flatnonzero(arrays['n_alleles']==count)
            for start in range(0,len(indices),args.batch_size):
                selected=indices[start:start+args.batch_size]
                probabilities[selected]=predict(model,{k:v[selected] for k,v in arrays.items()})
    else:
        for start in range(0,len(probabilities),args.batch_size):
            stop = min(len(probabilities),start+args.batch_size)
            probabilities[start:stop] = predict(model,{k:v[start:stop] for k,v in arrays.items()})
    check_probabilities(probabilities, arrays['n_alleles'])
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    counts = emit_vcf(args.output, arrays, metadata, probabilities,args.min_gp,args.min_gq,args.variants_only)
    np.savez_compressed(args.output+'.posterior.npz', gp=probabilities,record_index=arrays['record_index'],sample_index=arrays['sample_index'])
    manifest = dict(execution=dict(compact_inference=bool(getattr(model,'compact_inference',False)),neural_batch_schedule='allele_count' if grouped else 'record_order',neural_device=str(args.device),torch_cpu_threads=torch.get_num_threads()),version='0.2.0-neural-cohort', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    arguments=vars(args),feature_metadata_sha256=sha(Path(args.features)/'metadata.json'),
                    checkpoint_sha256=sha(args.checkpoint),checkpoint_metadata=getattr(model,'checkpoint_metadata',{}),
                    source_sha256={p.name:sha(p) for p in [HERE/'call.py',HERE/'features.py',HERE/'model.py',HERE/'cohort_context.py'] if p.exists()},
                    cohort=provenance,wall_seconds=time.monotonic()-started,counts=counts,
                    output_sha256=sha(args.output),limitations=metadata['limitations']+
                    ['Neural GP calibration and accuracy require independent validation; a fitted checkpoint alone does not establish improvement'])
    Path(args.output+'.manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')
    print(json.dumps({'output':args.output,**counts,'seconds':manifest['wall_seconds']}))


if __name__ == '__main__':
    main()
