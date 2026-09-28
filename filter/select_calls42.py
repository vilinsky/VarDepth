"""Explicit research operating point, independent of sample, assay or truth.

Sensitivity accepts complete diploid nonreference GTs currently marked only
LowQual. This lowers the acceptance threshold; it does not improve or recalibrate
their posterior. GQ, QUAL, alleles and phase are unchanged. Other filters remain.
"""
import argparse
import collections
import gzip
import hashlib
import io
import json
from pathlib import Path
import time

VERSION = '0.42.0-dev1'
ORIGINAL_FILTER = 'V42OF'


def complete_nonref(fields):
    if len(fields) != 10:
        raise ValueError('Exactly one sample is required')
    keys = fields[8].split(':');values = fields[9].split(':')
    if 'GT' not in keys or keys.index('GT') >= len(values):
        return False
    gt = values[keys.index('GT')].replace('|', '/').split('/')
    if len(gt) != 2 or any(not x.isdigit() for x in gt):
        return False
    gt = list(map(int, gt));alleles = 1 + len(fields[4].split(','))
    if any(x >= alleles for x in gt):
        raise ValueError('GT allele outside the VCF catalog')
    return any(gt)


def policy_line(line, profile):
    fields = line.rstrip('\r\n').split('\t')
    called = complete_nonref(fields)
    filters = set(fields[6].split(';'))
    if any(x.split('=', 1)[0] == ORIGINAL_FILTER for x in fields[7].split(';')):
        raise ValueError('Already contains v42 selection provenance')
    promoted = profile == 'sensitivity' and called and filters == {'LowQual'}
    if promoted:
        fields[7] = ('' if fields[7] == '.' else fields[7] + ';') + ORIGINAL_FILTER + '=LowQual'
        fields[6] = 'PASS'
    return '\t'.join(fields) + '\n', called, promoted


def identity(line):
    fields = line.rstrip('\r\n').split('\t')
    fields[6] = '.'
    fields[7] = ';'.join(x for x in fields[7].split(';')
                         if x.split('=', 1)[0] != ORIGINAL_FILTER) or '.'
    return ('\t'.join(fields) + '\n').encode()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):h.update(block)
    return h.hexdigest()


def transform(source, target, profile):
    if profile not in ('sensitivity', 'conservative'):raise ValueError('Unknown profile')
    source, target = Path(source), Path(target)
    if target.exists() or source.resolve() == target.resolve():raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic();cpu = time.process_time()
    before = hashlib.sha256();after = hashlib.sha256()
    all_nonref = hashlib.sha256();selected_nonref = hashlib.sha256()
    counts = collections.Counter();filters = collections.Counter()
    tmp = target.with_name(target.name + '.partial')
    if tmp.exists():raise FileExistsError(tmp)
    reader = gzip.open(source, 'rt') if source.suffix == '.gz' else source.open()
    compressed = target.suffix == '.gz'
    if compressed:
        import pysam
        writer = io.TextIOWrapper(pysam.BGZFile(str(tmp), 'w'), newline='')
    else:writer = tmp.open('w', newline='')
    samples_checked = False
    try:
        with reader, writer:
            for line in reader:
                if line.startswith('##INFO=<ID=' + ORIGINAL_FILTER + ','):
                    raise ValueError('Input already processed by v42 policy')
                if line.startswith('#CHROM'):
                    if len(line.rstrip().split('\t')) != 10:raise ValueError('Expected one sample')
                    samples_checked = True
                    writer.write('##VarDepth_selection="' + VERSION + ';profile=' + profile +
                                 ';LowQual acceptance is NOT GQ recalibration;research only"\n')
                    writer.write('##INFO=<ID=V42OF,Number=1,Type=String,Description="Original FILTER before v42 sensitivity selection; GQ and QUAL retained">\n')
                if line.startswith('#'):
                    writer.write(line);continue
                if not samples_checked:raise ValueError('VCF header missing')
                new, called, promoted = policy_line(line, profile)
                original_id = identity(line);new_id = identity(new)
                before.update(original_id);after.update(new_id)
                counts['records'] += 1;counts['promoted'] += promoted
                if called:
                    counts['complete_nonref'] += 1;all_nonref.update(original_id)
                    filters[line.split('\t')[6]] += 1
                    if new.split('\t')[6] in ['PASS', '.']:
                        counts['selected_nonref'] += 1;selected_nonref.update(new_id)
                writer.write(new)
        if not samples_checked or before.digest() != after.digest():
            raise AssertionError('Genotypes, quality, evidence or phase changed')
        tmp.replace(target)
        if compressed:pysam.tabix_index(str(target), preset='vcf', force=False)
    except BaseException:
        # Preserve any partial output as evidence; never label it complete.
        raise
    report = dict(version=VERSION, profile=profile, success=True, source=str(source), output=str(target),
                  source_sha256=sha(source), output_sha256=sha(target), counts=dict(counts),
                  original_nonref_filters=dict(filters), genotype_quality_phase_unchanged=True,
                  invariant_sha256=before.hexdigest(), source_ALL_nonref_sha256=all_nonref.hexdigest(),
                  output_PASS_nonref_sha256=selected_nonref.hexdigest(),
                  output_PASS_exactly_source_ALL=(all_nonref.digest() == selected_nonref.digest()),
                  cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-begin,
                  quality_recalibrated=False, truth_used=False, inference_recomputed=False,
                  acceptance_policy='complete diploid nonref; promote only LowQual; preserve hard filters')
    Path(str(target)+'.selection.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True);p.add_argument('--output', required=True)
    p.add_argument('--profile', choices=['sensitivity','conservative'], default='sensitivity')
    a = p.parse_args();print(json.dumps(transform(a.source,a.output,a.profile)))


if __name__ == '__main__':main()
