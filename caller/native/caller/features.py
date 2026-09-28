"""Versioned, truth-free molecular features for the bounded neural caller.

This module has no Torch dependency. Genotype likelihood and posterior are
deliberately separate arrays; fragments are sampled without consulting either.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np

SCHEMA_VERSION = 'vardepth-neural-features-0.2.0'
MAX_ALLELES, MAX_FRAGMENTS, SEQUENCE_LENGTH = 12, 64, 256
MAX_GENOTYPES = MAX_ALLELES * (MAX_ALLELES + 1) // 2
DNA = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 5}
FRAGMENT_METADATA = ('mean_bq_over_60', 'mean_mq_over_60', 'reverse_fraction',
                     'absolute_insert_over_1000', 'nm_per_query_base',
                     'clipped_query_fraction', 'mean_query_length_over_300',
                     'observed_marker_fraction')
ALLELE_METADATA = ('ad_fraction', 'panel_af', 'length_delta_over_50',
                   'allele_length_over_150', 'seed_support_fraction',
                   'panel_present', 'is_reference', 'length_differs_from_ref')


def genotypes(count):
    if not 1 <= count <= MAX_ALLELES:
        raise ValueError('Allele count must be 1..12')
    return [(a, b) for b in range(count) for a in range(b + 1)]


def genotype_index(a, b):
    if a < 0 or b < 0:
        raise ValueError('Negative allele index')
    a, b = sorted((int(a), int(b)))
    return b * (b + 1) // 2 + a


def normalize_record(start, alleles, fetch):
    """Joint left normalization preserving the original allele numbering."""
    alleles = [a.upper() for a in alleles]
    if len(alleles) < 2 or len(alleles) > MAX_ALLELES:
        raise ValueError('Expected 2..12 non-symbolic local alleles')
    if any(not a or any(b not in 'ACGTN' for b in a) for a in alleles):
        raise ValueError('Empty or symbolic allele')
    while min(map(len, alleles)) > 1 and len({a[-1] for a in alleles}) == 1:
        alleles = [a[:-1] for a in alleles]
    if len(set(map(len, alleles))) > 1:
        while start > 0 and len({a[-1] for a in alleles}) == 1:
            previous = fetch(start - 1, start).upper()
            if len(previous) != 1 or previous not in 'ACGTN':
                break
            alleles = [previous + a[:-1] for a in alleles]
            start -= 1
    while min(map(len, alleles)) > 1 and len({a[0] for a in alleles}) == 1:
        alleles = [a[1:] for a in alleles]
        start += 1
    if len(set(alleles)) != len(alleles):
        raise ValueError('Equivalent local sequences were not coalesced upstream')
    if fetch(start, start + len(alleles[0])).upper() != alleles[0]:
        raise ValueError('Normalized REF does not match FASTA')
    return start, tuple(alleles)


_DNA_BYTES=bytes(DNA.get(chr(i),5) for i in range(256))

def encode_alleles(start, alleles, fetch):
    """Context tokens: variant anchor has a shared coordinate across alleles.

    Include up to 64 reference bases before the event, then each local allele
    and the reference suffix. Right truncation is reported, never silent.
    """
    prefix = fetch(max(0, start - 64), start).upper()
    suffix = fetch(start + len(alleles[0]), start + len(alleles[0]) + 192).upper()
    tokens = np.zeros((MAX_ALLELES, SEQUENCE_LENGTH), dtype=np.int8)
    truncated = []
    for i, allele in enumerate(alleles):
        context = prefix + allele + suffix
        tokens[i, :min(len(context), SEQUENCE_LENGTH)] = np.frombuffer(context[:SEQUENCE_LENGTH].encode('ascii','replace').translate(_DNA_BYTES),np.int8)
        truncated.append(max(0, len(prefix) + len(allele) - SEQUENCE_LENGTH))
    return tokens, len(prefix), truncated


def alignment_metadata(read, q0=0, q1=None, observed_fraction=1., cache=None):
    if cache is not None:
        key=id(read)
        if key not in cache:cache[key]=(read,alignment_metadata(read,0,0,1.))
        result=cache[key][1].copy();qualities=read.query_qualities
        q=qualities[q0:q1] if qualities is not None else []
        result[0]=min(60.,sum(q)/len(q) if len(q) else 0.)/60.
        result[7]=min(1.,max(0.,observed_fraction))
        return result
    q1 = len(read.query_sequence) if q1 is None else q1
    qualities = read.query_qualities
    q = qualities[q0:q1] if qualities is not None else []
    qlen = max(1, len(read.query_sequence))
    clip = sum(n for op, n in (read.cigartuples or []) if op in (4, 5))
    hard = sum(n for op, n in (read.cigartuples or []) if op == 5)
    nm = read.get_tag('NM') if read.has_tag('NM') else 0
    return np.array([
        min(60., sum(q)/len(q) if len(q) else 0.) / 60.,
        min(60, read.mapping_quality) / 60., float(read.is_reverse),
        min(2., abs(read.template_length) / 1000.),
        min(1., float(nm) / qlen), min(1., clip / (qlen + hard)),
        min(2., qlen / 300.), min(1., max(0., observed_fraction)),
    ], dtype=np.float32)


def combine_alignment_metadata(alignments,cache=None):
    """Summarize only alignments actually retained in a molecular likelihood."""
    if not alignments:
        return np.zeros(8, dtype=np.float32)
    if len(alignments)==1:
        read,q0,q1,fraction=alignments[0]
        return alignment_metadata(read,q0,q1,fraction,cache=cache)
    rows = np.stack([alignment_metadata(r, q0, q1, fraction,cache=cache)
                     for r, q0, q1, fraction in alignments])
    weights = np.array([max(1, q1 - q0) for _, q0, q1, _ in alignments])
    result = np.average(rows, axis=0, weights=weights).astype(np.float32)
    result[3] = rows[:, 3].max()
    result[7] = min(1., rows[:, 7].sum())
    return result


class DeferredMetadata:
    """Compute neural metadata only for the unchanged hash-selected molecules."""
    def __init__(self, count, row):
        self.shape=(count,8);self.row=row
    def selected_rows(self, indices):
        return np.stack([self.row(int(i)) for i in indices]) if len(indices) else np.empty((0,8),np.float32)
    def __array__(self, dtype=None, copy=None):
        result=self.selected_rows(range(self.shape[0]))
        return result.astype(dtype,copy=False) if dtype is not None else result

def pack_fragments(likelihoods, keys, metadata, record_id, sample):
    lh = np.asarray(likelihoods, dtype=np.float64)
    if lh.ndim != 2 or not 1 <= lh.shape[1] <= MAX_ALLELES:
        raise ValueError('Invalid fragment likelihood matrix')
    if len(keys) != len(lh) or (metadata.shape if isinstance(metadata,DeferredMetadata) else np.asarray(metadata).shape) != (len(lh), 8):
        raise ValueError('Molecular likelihood/metadata rows must align exactly')
    if not np.isfinite(lh).all() or np.any(lh < 0):
        raise ValueError('Non-finite or negative molecular likelihood')
    # Hash downsampling is independent of allele evidence, posterior and truth.
    ranks = [(hashlib.sha256((sample + '\0' + record_id + '\0' + '\0'.join(map(str, key))).encode()).digest(), i)
             for i, key in enumerate(keys)]
    selected = [i for _, i in sorted(ranks)[:MAX_FRAGMENTS]]
    features = np.zeros((MAX_FRAGMENTS, MAX_ALLELES + 8), dtype=np.float32)
    mask = np.zeros(MAX_FRAGMENTS, dtype=np.bool_)
    if selected:
        logs = np.log(np.maximum(lh[selected], 1e-300))
        logs -= logs.max(axis=1, keepdims=True)
        features[:len(selected), :lh.shape[1]] = np.clip(logs, -20., 0.)
        features[:len(selected), MAX_ALLELES:] = metadata.selected_rows(selected) if isinstance(metadata,DeferredMetadata) else np.asarray(metadata)[selected]
        mask[:len(selected)] = True
    return features, mask, selected


def build_row(record, payload, likelihoods, keys, read_metadata, fetch, sample):
    alleles = record['alleles']; count = len(alleles); n_genotypes = count * (count + 1) // 2
    gp = np.asarray(payload['gp'], dtype=np.float64)
    log_likelihood = np.asarray(payload['pl'], dtype=np.float64)
    if gp.shape != (n_genotypes,) or log_likelihood.shape != (n_genotypes,):
        raise ValueError('GP/read PL dimension does not match multiallelic record')
    if not np.isfinite(gp).all() or np.any(gp < 0) or not np.isclose(gp.sum(), 1., atol=1e-5):
        raise ValueError('Invalid external-only posterior')
    if not np.isfinite(log_likelihood).all():
        raise ValueError('Nonfinite read likelihood')
    base_gp = np.zeros(MAX_GENOTYPES, dtype=np.float32); base_gp[:n_genotypes] = gp
    read_pl = np.zeros(MAX_GENOTYPES, dtype=np.float32)
    read_pl[:n_genotypes] = np.minimum(999., -10. / math.log(10) * (log_likelihood - log_likelihood.max()))
    genotype_mask = np.arange(MAX_GENOTYPES) < n_genotypes
    allele_mask = np.arange(MAX_ALLELES) < count
    tokens, anchor, truncated = encode_alleles(record['start0'], alleles, fetch)
    fragment_features, fragment_mask, selected = pack_fragments(likelihoods, keys, read_metadata, record['record_id'], sample)
    dp = int(payload['dp']); ad = np.asarray(payload['ad']); afs = record['panel_af']
    allele_features = np.zeros((MAX_ALLELES, 8), dtype=np.float32)
    for i, allele in enumerate(alleles):
        af = afs[i] if i < len(afs) else None
        allele_features[i] = (ad[i] / max(1, dp), float(af or 0),
                             (len(allele) - len(alleles[0])) / 50., len(allele) / 150.,
                             min(2., record['read_support'] / max(1, dp)) if i else 0.,
                             float(af is not None), float(i == 0), float(len(allele) != len(alleles[0])))
    row = dict(allele_sequences=tokens, fragment_features=fragment_features,
               allele_mask=allele_mask, fragment_mask=fragment_mask, genotype_mask=genotype_mask,
               base_gp=base_gp, read_pl=read_pl, depth=np.float32(dp), allele_features=allele_features,
               n_alleles=np.int16(count), total_fragments=np.int32(len(likelihoods)),
               kept_fragments=np.int16(len(selected)), sequence_anchor=np.int16(anchor))
    row['allele_depth'] = np.pad(np.asarray(ad, dtype=np.int32), (0, MAX_ALLELES-count))
    notes = dict(allele_sequence_truncated_bases=truncated, total_fragments=len(likelihoods),
                 retained_fragments=len(selected), sampled_fragment_id_sha256=[
                     hashlib.sha256('\0'.join(map(str, keys[i])).encode()).hexdigest() for i in selected])
    from evidence_trace import enabled, molecules
    if enabled():
        notes['lineage_neural_molecules'] = molecules(sample, [keys[i] for i in selected])
    return row, notes


def empty_arrays():
    return dict(allele_sequences=np.zeros((0, 12, 256), np.int8),
                fragment_features=np.zeros((0, 64, 20), np.float32),
                allele_features=np.zeros((0, 12, 8), np.float32),
                allele_mask=np.zeros((0, 12), bool), fragment_mask=np.zeros((0, 64), bool),
                genotype_mask=np.zeros((0, 78), bool), base_gp=np.zeros((0, 78), np.float32),
                read_pl=np.zeros((0, 78), np.float32), depth=np.zeros(0, np.float32),
                n_alleles=np.zeros(0, np.int16), total_fragments=np.zeros(0, np.int32),
                kept_fragments=np.zeros(0, np.int16), sequence_anchor=np.zeros(0, np.int16),
                allele_depth=np.zeros((0, 12), np.int32),
                cohort_context=np.zeros((0, 12, 6), np.float32), context_strength=np.zeros(0,np.float32),
                record_index=np.zeros(0, np.int32), sample_index=np.zeros(0, np.int32),
                positions=np.zeros(0, np.int64))
