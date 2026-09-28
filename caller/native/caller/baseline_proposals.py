"""Candidate proposals from partial observations; unobserved sequence is a proposal.

No genotype is assigned here. The downstream partial-window likelihood scores
all retained hypotheses and does not count filled-in flanks as read evidence.
"""
import collections
import numpy as np
from projection import common_anchors, crop_observation, expand_compound_context

def splice_haplotype(template, mapping, left, right, observed):
    if left < 0 or right > len(mapping) or left >= right: raise ValueError('Invalid observation bounds')
    a, b = int(mapping[left]), int(mapping[right - 1]) + 1
    if a < 0 or b <= a or b > len(template): raise ValueError('Unaligned splice anchors')
    return template[:a] + observed + template[b:]

def propose(row, original, bams, fetch, maximum=24):
    start, old = expand_compound_context(row['start0'], row['alleles'], fetch)
    end = start + len(old[0]); prefix = fetch(start - 3, start); suffix = fetch(end, end + 3)
    if len(prefix) != 3 or len(suffix) != 3 or len(old) >= maximum: return row, dict(added=0)
    reference = prefix + old[0] + suffix; alleles = [prefix + a + suffix for a in old]
    maps, safe = common_anchors(start - 3, reference, alleles, fetch)
    support = collections.defaultdict(dict)
    # Changes outside the original candidate block are not imported from reads.
    left_flank = original['start0'] - start; right_flank = end - original['end0']
    for sample, inputs in bams.items():
        for bam, rg, fallback in inputs:
            for read in bam.fetch(row['chrom'], max(0, start - 3), end + 3):
                if read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate or read.is_qcfail: continue
                if not 20 <= read.mapping_quality < 255 or read.query_qualities is None: continue
                obs = crop_observation(read, start - 3, reference, alleles, maps, safe)
                if obs is None or (sum(obs['qualities'])/len(obs['qualities'])) < 25 or any(b not in 'ACGT' for b in obs['sequence']): continue
                if min(obs['qualities'][:3] + obs['qualities'][-3:]) < 20: continue
                library = rg.get(read.get_tag('RG'), fallback) if read.has_tag('RG') else fallback
                molecule = (sample, library, read.query_name)
                l = obs['left'] - (start - 3); r = obs['right'] - (start - 3)
                for template, mapping in zip(alleles, maps):
                    sequence = splice_haplotype(template, mapping, l, r, obs['sequence'])
                    if not sequence.startswith(prefix) or not sequence.endswith(suffix): continue
                    inner = sequence[3:-3]
                    if inner in old or not inner or abs(len(inner) - len(old[0])) > 50: continue
                    if left_flank and inner[:left_flank] != old[0][:left_flank]: continue
                    if right_flank and inner[-right_flank:] != old[0][-right_flank:]: continue
                    # A mate or a second template cannot inflate molecule support.
                    support[inner][molecule] = max(support[inner].get(molecule, 0), r - l)
    ordered = sorted((sequence for sequence, molecules in support.items() if len(molecules) >= 2),
                     key=lambda sequence: (-len(support[sequence]), -sum(support[sequence].values()), sequence))
    added = ordered[:max(0, min(12, maximum - len(old)))]
    updated = dict(row, start0=start, end0=end, alleles=old + added,
                   added_alleles=row['added_alleles'] + len(added))
    summary = dict(added=len(added), eligible_proposals=len(ordered),
                   proposals={sequence: dict(distinct_molecules=len(support[sequence]),
                              summed_observed_span=sum(support[sequence].values())) for sequence in added})
    updated['partial_proposals'] = summary
    return updated, summary
