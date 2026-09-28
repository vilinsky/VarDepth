"""Apply the trained dependent-expert gate to a truth-free candidate catalog."""
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import pysam
from trust import Predictor, EXPERTS, VERSION, bundle, pairs, scaffold_data, probability, joint_ld_data


@lru_cache(maxsize=24)
def dosage_terms(n):
    # Keep VCF genotype summation order. Dropped terms are exactly zero.
    return tuple(tuple((i, int(a == j) + int(b == j)) for i, (a, b) in enumerate(pairs(n)) if a == j or b == j) for j in range(1, n))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def run(args):
    import torch
    torch.set_num_threads(args.threads)
    begun = time.monotonic()
    refined = {r['record_id']: r for r in json.loads(Path(args.refinements).read_text())} if args.refinements else {}
    spanning = {r['record_id']: r for r in json.loads(Path(args.spanning_refinements).read_text())} if args.spanning_refinements else {}
    scaffold, indices = scaffold_data(args.scaffold)
    joint = {}
    for entry in args.joint_ld:
        name, path = entry.split('=', 1)
        if name in joint:
            raise ValueError('Duplicate joint LD sample')
        joint[name] = joint_ld_data(path, name)
    predictor = Predictor(args.trust_checkpoint, args.disable_expert)
    from evidence_trace import enabled as trace_enabled
    from lineage_features import Exporter
    trace = Exporter(args) if trace_enabled() else None
    source = pysam.VariantFile(args.neural)
    if indices and set(indices) != set(source.header.samples):
        raise ValueError('Scaffold/neural sample identity mismatch')
    header = pysam.VariantHeader()
    header.add_meta('source', value='VarDepthResearch-' + VERSION)
    header.add_meta('GP_semantics', value='Learned genotype posterior with a frozen dependent-expert parent; read sources are not independent')
    header.add_meta('trust_checkpoint_sha256', value=sha(args.trust_checkpoint))
    for c in source.header.contigs:
        header.contigs.add(c, length=source.header.contigs[c].length)
    for s in source.header.samples:
        header.add_sample(s)
    for key, number, kind, desc in [
        ('GT', 1, 'String', 'Unphased diploid genotype'),
        ('GQ', 1, 'Integer', 'Phred uncertainty derived from learned GP'),
        ('GP', 'G', 'Float', 'Trained mixture genotype probabilities'),
        ('DS', 'A', 'Float', 'Alternate allele posterior dosage'),
        ('DP', 1, 'Integer', 'Original local distinct molecule depth'),
        ('AD', 'R', 'Integer', 'Original allele support, omitted for expanded alleles'),
        ('PL', 'G', 'Integer', 'Read likelihood, anchored at expanded records'),
        ('NGP', 'G', 'Float', 'Original neural GP, zero for newly added alleles'),
        ('BGP', 'G', 'Float', 'Original external HMM GP, zero for newly added alleles'),
        ('CS', 1, 'Float', 'Automatic leave-self-out cohort context strength'),
        ('TW', len(EXPERTS), 'Float', 'Frozen parent expert weights before genotype residual: ' + ','.join(EXPERTS)),
        ('ADP', 1, 'Integer', 'Anchored molecule depth'),
        ('AAD', 'R', 'Integer', 'Exact anchored haplotype support')]:
        header.formats.add(key, number, kind, desc)
    header.info.add('SOURCE', 1, 'String', 'Catalog origin PANEL,READS,BOTH')
    header.info.add('REFINED', 0, 'Flag', 'Local anchored haplotypes evaluated')
    header.info.add('NEW_ALLELES', 1, 'Integer', 'Number of additional read-derived sequences')
    header.info.add('PANEL_AF', 'A', 'Float', 'Original external frequency, missing for new alleles')
    header.filters.add('LowQual', None, None, 'No nonreference sample reaches GQ20')
    records = []; pending = []; stats = dict(records=0, changed_genotypes=0, refined_records=0)
    def flush():
        if not pending:
            return
        rows = [b for _, _, _, entries in pending for b in entries]
        probabilities, weights = predictor(rows); offset = 0
        if trace is not None:
            trace.write(pending, list(header.samples), probabilities, spanning)
        for original, refinement, item, entries in pending:
            ng = entries[0]['n_genotypes']; n = len(item.alleles); order = pairs(n)
            passing = False; logref = 0.
            for s, data in zip(header.samples, entries):
                c = original.samples[s]; p = probabilities[offset, :ng]; w = weights[offset]; offset += 1
                gt = order[int(p.argmax())]
                gq = min(99, int(round(-10 * math.log10(max(1e-12, 1 - float(p.max()))))))
                out = item.samples[s]; out['GT'] = gt; out['GQ'] = gq; out['GP'] = tuple(map(float, p))
                out['DP'] = int(c.get('DP', 0) or 0); out['CS'] = float(c.get('CS', 0) or 0)
                out['NGP'] = tuple(probability(c['GP'], ng)); out['BGP'] = tuple(probability(c.get('BGP', c['GP']), ng))
                out['TW'] = tuple(map(float, w))
                out['DS'] = tuple(float(sum(p[i] * count for i, count in terms)) for terms in dosage_terms(n))
                if refinement:
                    expert = refinement['samples'][s]
                    out['ADP'] = expert['dp']; out['AAD'] = tuple(expert['exact_ad'])
                    out['PL'] = tuple(min(999, int(round(v))) for v in expert['pl'])
                else:
                    for field in ['AD', 'PL']:
                        if c.get(field) is not None:
                            out[field] = c[field]
                passing |= any(gt) and gq >= 20
                logref += math.log(max(1e-100, float(p[0])))
                stats['changed_genotypes'] += int(gt != tuple(c['GT']))
            item.qual = min(999., -10 / math.log(10) * logref)
            item.filter.add('PASS' if passing else 'LowQual'); records.append(item)
        pending.clear()
    for original in source:
        rid = f'{original.chrom}:{original.pos}:{original.ref}:{",".join(original.alts or [])}'
        refinement = refined.get(rid)
        if refinement and set(refinement['samples']) != set(header.samples):
            raise ValueError('Refinement sample identity mismatch')
        alleles = refinement['alleles'] if refinement else list(original.alleles)
        position = refinement['start0'] if refinement else original.start
        item = header.new_record(contig=original.chrom, start=position, stop=position + len(alleles[0]), alleles=alleles)
        item.info['SOURCE'] = original.info.get('SOURCE', 'UNKNOWN')
        if 'PANEL_AF' in original.info:
            af = list(original.info['PANEL_AF']) + [None] * (len(alleles) - len(original.alleles))
            item.info['PANEL_AF'] = tuple(af)
        if refinement:
            item.info['REFINED'] = True; item.info['NEW_ALLELES'] = refinement['added_alleles']
            stats['refined_records'] += 1
        key = (original.chrom, original.start, original.ref, original.alts[0]) if len(original.alleles) == 2 else None
        entries = [bundle(original, s, refinement, scaffold[key][indices[s]] if key in scaffold else None,
                          joint.get(s, {}).get(rid), spanning.get(rid), compact=True) for s in header.samples]
        pending.append((original, refinement, item, entries)); stats['records'] += 1
        if len(pending) >= 128:
            flush()
    flush(); source.close()
    if trace is not None:
        trace.finish()
    contigs = {c: i for i, c in enumerate(header.contigs)}
    records.sort(key=lambda r: (contigs[r.chrom], r.start, r.alleles))
    dest = Path(args.output); dest.parent.mkdir(parents=True, exist_ok=True)
    with pysam.VariantFile(dest, 'wz', header=header) as out:
        for r in records:
            out.write(r)
    pysam.tabix_index(str(dest), preset='vcf', force=True)
    inputs = [args.neural, args.trust_checkpoint] + [p for p in [args.refinements, args.scaffold] if p]
    if args.spanning_refinements:
        inputs.append(args.spanning_refinements)
    for entry in args.joint_ld:
        _, directory = entry.split('=', 1)
        inputs.extend([str(Path(directory) / 'posterior.npz'), str(Path(directory).parent / 'PREPARED.json')])
    report = dict(version=VERSION, arguments=vars(args), stats=stats, seconds=time.monotonic() - begun,
                  input_sha256={p: sha(p) for p in inputs}, output_sha256=sha(dest), truth_used=False)
    Path(str(dest) + '.trust.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['neural', 'trust-checkpoint', 'output']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--scaffold'); p.add_argument('--refinements')
    p.add_argument('--spanning-refinements')
    p.add_argument('--joint-ld', action='append', default=[], help='Sample=directory containing joint LD posterior.npz')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--disable-expert', action='append', choices=EXPERTS, default=[])
    run(p.parse_args(argv))


if __name__=='__main__':main()
