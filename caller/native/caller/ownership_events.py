"""Canonical event marginals from one joint haplotype posterior.

No likelihoods from overlapping blocks are multiplied. A source is selected
explicitly; the complete original joint VCF remains an input artifact.
"""
import math
from pathlib import Path
import sys

import numpy as np

from projection import project
from base_adapter import base_modules
normalize_variant = base_modules().normalize_variant


def minimal(start, ref, alt):
    while ref and alt and ref[0] == alt[0]:
        start += 1; ref = ref[1:]; alt = alt[1:]
    while ref and alt and ref[-1] == alt[-1]:
        ref = ref[:-1]; alt = alt[:-1]
    return start, ref, alt


def apply(reference, start, events):
    edits = sorted(minimal(*e) for e in events)
    for i, (p, ref, alt) in enumerate(edits):
        if p < start or p + len(ref) > start + len(reference) or reference[p-start:p-start+len(ref)] != ref:
            raise ValueError('event_reference_or_boundary')
        for q, r, a in edits[i+1:]:
            if not ref and not r and p == q:raise ValueError('overlapping_insertions')
            if ref and r and max(p, q) < min(p+len(ref), q+len(r)):raise ValueError('overlapping_edits')
            if (not ref and q < p < q+len(r)) or (not r and p < q < p+len(ref)):
                raise ValueError('insertion_inside_deletion')
    result = reference
    for p, ref, alt in sorted(edits, key=lambda e:(e[0], len(e[1])), reverse=True):
        j = p-start; result = result[:j] + alt + result[j+len(ref):]
    return result


def canonical_event(start, deleted, inserted, fetch):
    if not deleted or not inserted:
        if start > 0:
            anchor = fetch(start-1, start)
            if len(anchor) != 1:raise ValueError('missing_left_anchor')
            start -= 1; deleted = anchor + deleted; inserted = anchor + inserted
        else:
            anchor = fetch(start+len(deleted), start+len(deleted)+1)
            if len(anchor) != 1:raise ValueError('missing_right_anchor')
            deleted += anchor; inserted += anchor
    return normalize_variant(start, deleted, inserted, fetch)


def catalog_events(start, alleles, fetch):
    """Require exact reconstruction of every allele after canonicalization."""
    ref = alleles[0]
    if not ref or fetch(start, start+len(ref)) != ref:raise ValueError('reference_mismatch')
    if any(not a or any(b not in 'ACGT' for b in a) for a in alleles):raise ValueError('noncanonical_allele')
    result = []
    for alt in alleles:
        if alt == ref:
            result.append(frozenset()); continue
        if len(ref) == len(alt) == 1:
            result.append(frozenset([(start, ref, alt)])); continue
        mapping, indels = project(ref, alt)
        edits = [(start+p, deleted, inserted) for p, deleted, inserted in indels]
        edits += [(start+i, ref[i], alt[j]) for i, j in enumerate(mapping) if j >= 0 and ref[i] != alt[j]]
        events = [canonical_event(p, r, a, fetch) for p, r, a in edits]
        if len(events) != len(set(events)):raise ValueError('aliased_events_within_haplotype')
        left = min([start] + [p for p, r, a in events])
        right = max([start+len(ref)] + [p+len(r) for p, r, a in events])
        expected = fetch(left, start) + alt + fetch(start+len(ref), right)
        if apply(fetch(left, right), left, events) != expected:raise ValueError('canonicalization_changes_haplotype')
        result.append(frozenset(events))
    return result


def marginals(catalog, gp, genotype):
    n = len(catalog)
    order = [(a, b) for b in range(n) for a in range(b+1)]
    p = np.asarray(gp, np.float64)
    if p.shape != (len(order),) or not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
        raise ValueError('invalid_joint_gp')
    if len(genotype) != 2 or any(g is None or not 0 <= g < n for g in genotype):raise ValueError('invalid_diploid_gt')
    p = p/p.sum()
    union = sorted(set().union(*catalog))
    result = {}
    for event in union:
        present = [event in events for events in catalog]
        q = np.zeros(3, np.float64)
        for mass, (a, b) in zip(p, order):q[int(present[a])+int(present[b])] += mass
        gt = tuple(sorted(int(present[a]) for a in genotype))
        copies = sum(gt)
        result[event] = dict(gp=q.tolist(), gt=gt, dosage=float(q[1]+2*q[2]),
                             gq=min(99, int(round(-10*math.log10(max(1e-12, 1-float(q[copies])))))))
    return result
