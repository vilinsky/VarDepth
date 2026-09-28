"""Mechanistic checks for self-excluded, non-recycled cohort feature messages."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cohort_context import (CohortConfig, FEATURE_NAMES, GENOTYPE_PAIRS,
                            build_cohort_context, empty_cohort_context)


def genotype_index(a, b):
    a, b = sorted((a, b))
    return b * (b + 1) // 2 + a


def fixture(people=3, records=9, alleles=2):
    gp = np.zeros((people, records, 78))
    gp[:, :, 0] = 1.0
    return dict(initial_gp=gp, allele_count=np.full(records, alleles, dtype=int),
                positions=np.arange(records, dtype=int) * 1000,
                sample_ids=[f'person{i}' for i in range(people)],
                read_depth=np.full((people, records), 20.0))


def set_gt(gp, person, site, gt):
    gp[person, site] = 0
    gp[person, site, genotype_index(*gt)] = 1


class CohortContextTests(unittest.TestCase):
    def test_single_person_is_exact_empty_branch(self):
        data = fixture(1)
        result = build_cohort_context(**data)
        empty = empty_cohort_context(1, 9)
        self.assertTrue(np.array_equal(result.allele_context, empty.allele_context))
        self.assertTrue(np.array_equal(result.context_strength, empty.context_strength))
        self.assertEqual(result.donor_indices.shape, (1, 9, 0))
        self.assertEqual(result.allele_context.dtype, np.float32)
        self.assertFalse(result.provenance['enabled'])
        self.assertEqual(result.provenance['reason'], 'single_sample')

    def test_no_read_supported_donors_is_exact_zero(self):
        data = fixture(3)
        data['read_depth'][:] = 0
        result = build_cohort_context(**data)
        self.assertFalse(np.any(result.allele_context))
        self.assertFalse(np.any(result.context_strength))
        self.assertTrue(np.all(result.donor_indices == -1))
        self.assertEqual(result.provenance['active_target_records'], 0)

    def test_sample_permutation_is_bitwise_invariant(self):
        data = fixture(4)
        data['sample_ids'] = ['zeta', 'beta', 'alpha', 'gamma']
        rng = np.random.default_rng(724)
        data['initial_gp'][:, :, :3] = rng.dirichlet([.2, .7, .5], size=(4, 9))
        data['read_depth'] = rng.integers(1, 60, size=(4, 9)).astype(float)
        first = build_cohort_context(**data, config=CohortConfig(max_donors=2))
        permutation = np.array([2, 0, 3, 1])
        changed = dict(data, initial_gp=data['initial_gp'][permutation],
                       read_depth=data['read_depth'][permutation],
                       sample_ids=[data['sample_ids'][i] for i in permutation])
        second = build_cohort_context(**changed, config=CohortConfig(max_donors=2))
        inverse = np.argsort(permutation)
        self.assertTrue(np.array_equal(first.allele_context, second.allele_context[inverse]))
        self.assertTrue(np.array_equal(first.context_strength, second.context_strength[inverse]))
        self.assertTrue(np.array_equal(first.donor_weights, second.donor_weights[inverse]))
        self.assertEqual(first.provenance['initial_gp_sha256_canonical'],
                         second.provenance['initial_gp_sha256_canonical'])
        for target in range(4):
            for site in range(9):
                ids1 = [data['sample_ids'][i] for i in first.donor_indices[target, site] if i >= 0]
                ids2 = [changed['sample_ids'][i] for i in second.donor_indices[inverse[target], site] if i >= 0]
                self.assertEqual(ids1, ids2)

    def test_donor_ties_use_ids_and_never_self(self):
        data = fixture(4)
        data['sample_ids'] = ['zeta', 'gamma', 'alpha', 'beta']
        result = build_cohort_context(**data, config=CohortConfig(max_donors=2))
        self.assertEqual([data['sample_ids'][i] for i in result.donor_indices[0, 4]], ['alpha', 'beta'])
        for target in range(4):
            self.assertFalse(np.any(result.donor_indices[target] == target))

    def test_target_focal_gp_depth_quality_do_not_change_own_message(self):
        data = fixture(3)
        data['site_quality'] = np.ones((3, 9))
        original = build_cohort_context(**data)
        set_gt(data['initial_gp'], 0, 4, (1, 1))
        data['read_depth'][0, 4] = 999
        data['site_quality'][0, 4] = 0
        changed = build_cohort_context(**data)
        np.testing.assert_array_equal(original.allele_context[0, 4], changed.allele_context[0, 4])
        self.assertEqual(original.context_strength[0, 4], changed.context_strength[0, 4])
        np.testing.assert_array_equal(original.donor_weights[0, 4], changed.donor_weights[0, 4])

    def test_flanking_context_prefers_compatible_donor(self):
        data = fixture(3)
        for site in range(9):
            set_gt(data['initial_gp'], 0, site, (0, 1))
            set_gt(data['initial_gp'], 1, site, (0, 1))
            set_gt(data['initial_gp'], 2, site, (1, 1))
        set_gt(data['initial_gp'], 1, 4, (0, 0))
        result = build_cohort_context(**data)
        ids = result.donor_indices[0, 4]
        weights = dict(zip(ids, result.donor_weights[0, 4]))
        self.assertGreater(weights[1], weights[2])
        self.assertLess(result.allele_context[0, 4, 1, 0], .5)
        # Changing only flanking evidence can legitimately change compatibility.
        for site in [0, 1, 2, 3, 5, 6, 7, 8]:
            set_gt(data['initial_gp'], 0, site, (1, 1))
        changed = build_cohort_context(**data)
        self.assertGreater(changed.allele_context[0, 4, 1, 0], .5)

    def test_incompatible_single_donor_reduces_absolute_strength(self):
        data = fixture(2)
        for site in range(9):
            set_gt(data['initial_gp'], 0, site, (0, 1))
            set_gt(data['initial_gp'], 1, site, (0, 1))
        first = build_cohort_context(**data)
        for site in [0, 1, 2, 3, 5, 6, 7, 8]:
            set_gt(data['initial_gp'], 1, site, (1, 1))
        second = build_cohort_context(**data)
        self.assertLess(second.context_strength[0, 4], first.context_strength[0, 4])
        # Focal genotype is unchanged; only local compatibility should weaken it.
        np.testing.assert_array_equal(first.allele_context[0, 4, :, 0],
                                      second.allele_context[0, 4, :, 0])

    def test_read_only_flanks_override_external_gp_similarity(self):
        data = fixture(3)
        # GP makes the target look like donor1; read-only flank PL favors donor2.
        for site in range(9):
            set_gt(data['initial_gp'], 2, site, (1, 1))
        set_gt(data['initial_gp'], 1, 4, (0, 0))
        pl = np.full((3, 9, 78), 999.0)
        pl[:, :, :3] = 80
        pl[0, :, 2] = 0
        pl[1, :, 0] = 0
        pl[2, :, 2] = 0
        without = build_cohort_context(**data, config=CohortConfig(max_donors=1))
        with_pl = build_cohort_context(**data, read_pl=pl, config=CohortConfig(max_donors=1))
        self.assertEqual(without.donor_indices[0, 4, 0], 1)
        self.assertEqual(with_pl.donor_indices[0, 4, 0], 2)
        self.assertEqual(with_pl.provenance['flank_source'], 'read_only_PL')
        with_gl = build_cohort_context(**data, read_gl=-pl * np.log(10) / 10,
                                      config=CohortConfig(max_donors=1))
        np.testing.assert_array_equal(with_pl.allele_context, with_gl.allele_context)

    def test_no_matching_contig_prevents_cross_chromosome_context(self):
        data = fixture(3)
        data['contigs'] = ['chr1'] * 9
        data['contigs'][4] = 'chr2'
        result = build_cohort_context(**data)
        for site in [0, 1, 2, 3, 5, 6, 7, 8]:
            set_gt(data['initial_gp'], 0, site, (1, 1))
        changed = build_cohort_context(**data)
        np.testing.assert_array_equal(result.allele_context[0, 4], changed.allele_context[0, 4])
        self.assertEqual(result.allele_context[0, 4, 0, 3], .5)

    def test_exclusion_radius_prevents_near_focal_reuse(self):
        data = fixture(3, records=3)
        data['positions'] = np.array([1000, 1001, 1100])
        original = build_cohort_context(**data)
        set_gt(data['initial_gp'], 0, 1, (1, 1))
        set_gt(data['initial_gp'], 2, 2, (1, 1))
        # donor2's nearby focal GP may change another site's own message;
        # donor genotypes at record0 remain unchanged, so record0 is invariant.
        changed = build_cohort_context(**data)
        np.testing.assert_array_equal(original.allele_context[0, 0], changed.allele_context[0, 0])

    def test_uncertain_donor_does_not_equal_certain_donor(self):
        data = fixture(3, records=1)
        set_gt(data['initial_gp'], 1, 0, (1, 1))
        data['initial_gp'][2, 0, :3] = 1 / 3
        result = build_cohort_context(**data)
        np.testing.assert_array_equal(result.donor_indices[0, 0], [1, -1])
        self.assertEqual(result.allele_context[0, 0, 1, 0], 1)
        self.assertEqual(result.allele_context[0, 0, 1, 5], .5)

    def test_dirty_high_depth_donor_has_low_weight(self):
        data = fixture(3, records=1)
        set_gt(data['initial_gp'], 2, 0, (1, 1))
        data['read_depth'][1, 0] = 4
        data['read_depth'][2, 0] = 1000
        data['site_quality'] = np.ones((3, 1))
        data['site_quality'][2, 0] = .001
        result = build_cohort_context(**data)
        weights = dict(zip(result.donor_indices[0, 0], result.donor_weights[0, 0]))
        self.assertLess(weights[2], weights[1] / 100)
        self.assertLess(result.allele_context[0, 0, 1, 0], .01)

    def test_mixed_depth_is_sitewise_and_saturates(self):
        data = fixture(3, records=2)
        data['positions'] = np.array([1000, 100000])
        data['read_depth'][1] = [2, 2000]
        data['read_depth'][2] = [40, 2]
        set_gt(data['initial_gp'], 1, 0, (1, 1))
        first = build_cohort_context(**data)
        weights = dict(zip(first.donor_indices[0, 0], first.donor_weights[0, 0]))
        self.assertGreater(weights[2], weights[1])
        data['read_depth'][1, 1] = 1
        second = build_cohort_context(**data)
        np.testing.assert_array_equal(first.allele_context[0, 0], second.allele_context[0, 0])
        data['read_depth'][1, 0] = 100
        third = build_cohort_context(**data)
        w100 = dict(zip(third.donor_indices[0, 0], third.donor_weights[0, 0]))[1]
        data['read_depth'][1, 0] = 10000
        fourth = build_cohort_context(**data)
        w10000 = dict(zip(fourth.donor_indices[0, 0], fourth.donor_weights[0, 0]))[1]
        self.assertLess(w10000 / w100, 1.04)

    def test_singleton_is_not_removed_or_overwritten(self):
        data = fixture(3, alleles=3)
        set_gt(data['initial_gp'], 0, 4, (0, 2))
        before = data['initial_gp'].copy()
        result = build_cohort_context(**data)
        np.testing.assert_array_equal(before, data['initial_gp'])
        self.assertEqual(result.allele_context[0, 4, 2, 0], 0)
        self.assertEqual(result.allele_context.shape, (3, 9, 12, 6))
        self.assertTrue(np.any(result.allele_context[0, 4, 2, 2:]))
        self.assertFalse(np.any(result.allele_context[:, :, 3:]))

    def test_vcf_triangular_dosage_and_heterozygous_carriers(self):
        data = fixture(2, records=1, alleles=12)
        set_gt(data['initial_gp'], 1, 0, (2, 11))
        result = build_cohort_context(**data)
        self.assertEqual(tuple(GENOTYPE_PAIRS[genotype_index(2, 11)]), (2, 11))
        self.assertEqual(result.allele_context[0, 0, 2, 0], .5)
        self.assertEqual(result.allele_context[0, 0, 11, 0], .5)
        self.assertEqual(result.allele_context[0, 0, 2, 1], .5)  # one effective carrier
        self.assertEqual(result.allele_context[0, 0, 0, 0], 0)

    def test_discordant_donors_are_exposed(self):
        data = fixture(3, records=1)
        set_gt(data['initial_gp'], 2, 0, (1, 1))
        result = build_cohort_context(**data)
        self.assertEqual(result.allele_context[0, 0, 0, 0], .5)
        self.assertEqual(result.allele_context[0, 0, 0, 4], 1)
        self.assertEqual(result.allele_context[0, 0, 1, 4], 1)

    def test_missing_read_likelihoods_give_neutral_flanks(self):
        data = fixture(3)
        gl = np.full_like(data['initial_gp'], -np.inf)
        result = build_cohort_context(**data, read_gl=gl)
        self.assertTrue(np.isfinite(result.allele_context).all())
        self.assertTrue(np.all(result.allele_context[:, :, :2, 3] == .5))
        self.assertEqual(result.provenance['messages_without_informative_flanks'], 27)

    def test_all_features_bounded_and_provenance_serializable(self):
        data = fixture(4, alleles=5)
        rng = np.random.default_rng(87)
        data['initial_gp'][:, :, :15] = rng.dirichlet(np.ones(15), size=(4, 9))
        result = build_cohort_context(**data)
        self.assertTrue(np.isfinite(result.allele_context).all())
        self.assertTrue(np.all((result.allele_context >= 0) & (result.allele_context <= 1)))
        self.assertTrue(np.all((result.context_strength >= 0) & (result.context_strength <= 1)))
        self.assertEqual(result.provenance['feature_names'], list(FEATURE_NAMES))
        json.dumps(result.provenance)

    def test_invalid_probabilities_and_recycling_rejected(self):
        mutations = ['nan', 'negative', 'not_normalized', 'padding']
        for mutation in mutations:
            data = fixture()
            if mutation == 'nan': data['initial_gp'][0, 0, 0] = np.nan
            elif mutation == 'negative': data['initial_gp'][0, 0, 0] = -1
            elif mutation == 'not_normalized': data['initial_gp'][0, 0, 0] = .5
            else:
                data['initial_gp'][0, 0, 0] = .5
                data['initial_gp'][0, 0, 3] = .5
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_cohort_context(**data)
        with self.assertRaises(ValueError):
            build_cohort_context(**fixture(), gp_source='post_cohort')

    def test_invalid_metadata_and_likelihoods_rejected(self):
        for update in [dict(sample_ids=['same'] * 3), dict(allele_count=np.full(9, 13)),
                       dict(read_depth=np.full((3, 9), -1)), dict(site_quality=np.full((3, 9), 2)),
                       dict(positions=np.arange(9) + .5), dict(contigs=['chr1'])]:
            with self.subTest(update=list(update)), self.assertRaises(ValueError):
                build_cohort_context(**dict(fixture(), **update))
        data = fixture()
        with self.assertRaises(ValueError):
            build_cohort_context(**data, read_pl=np.zeros((3, 9, 78)), read_gl=np.zeros((3, 9, 78)))
        with self.assertRaises(ValueError):
            build_cohort_context(**data, read_pl=np.full((3, 9, 78), -1))

    def test_empty_catalog_is_supported(self):
        result = build_cohort_context(**fixture(2, records=0))
        self.assertEqual(result.allele_context.shape, (2, 0, 12, 6))
        self.assertEqual(result.context_strength.shape, (2, 0))


if __name__ == '__main__':
    unittest.main()
