import sys
from pathlib import Path
import unittest
import tempfile
import json
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trust import MAX_GP, model_class, bundle, likelihood_posterior, catalog_signature, joint_ld_data
import pysam


class TrustContracts(unittest.TestCase):
    def test_ragged_inference_preserves_full_feature_model(self):
        import os
        from unittest.mock import patch
        from trust import Predictor, residual_model_class
        torch.manual_seed(629)
        predictor=Predictor.__new__(Predictor)
        predictor.residual=True;predictor.model=residual_model_class()().eval()
        predictor.temperature=1.;predictor.disabled=[]
        torch.nn.init.normal_(predictor.model.residual[-1].weight,std=.1)
        rows=[]
        for ng in (3,6,91):
            row=dict(features=np.random.default_rng(ng).normal(size=(10,42)).astype('f'),
                     probabilities=np.zeros((10,MAX_GP),np.float32),available=np.ones(10,bool),
                     n_genotypes=ng,genotype_features=np.zeros((MAX_GP,20),np.float32))
            row['features'][:,2]=ng>3
            p=np.random.default_rng(ng+1).random((10,ng)).astype('f');p/=p.sum(1,keepdims=True)
            row['probabilities'][:,:ng]=p;rows.append(row)
        with patch.dict(os.environ,VARDEPTH_COMPACT_NEURAL='0'):expected,ew=predictor(rows)
        ragged=[dict(r,probabilities=r['probabilities'][:,:r['n_genotypes']].copy(),genotype_features=r['genotype_features'][:r['n_genotypes']].copy()) for r in rows]
        with patch.dict(os.environ,VARDEPTH_COMPACT_NEURAL='1'):actual,aw=predictor(ragged)
        np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=2e-6)
        np.testing.assert_array_equal(actual.argmax(1),expected.argmax(1))
        np.testing.assert_array_equal(aw,ew)

    def record(self, gt):
        h = pysam.VariantHeader(); h.contigs.add('chrSynthetic', length=100)
        for key, number, typ in [('GT', 1, 'String'), ('GP', 'G', 'Float'), ('DP', 1, 'Integer')]:
            h.formats.add(key, number, typ, key)
        h.add_sample('singleton')
        r = h.new_record(contig='chrSynthetic', start=10, stop=11, alleles=('A', 'C', 'AT'))
        r.samples['singleton']['GT'] = gt
        p = [1e-5] * 6; a, b = sorted(gt); p[b * (b + 1) // 2 + a] = .99995
        r.samples['singleton']['GP'] = p; r.samples['singleton']['DP'] = 100
        return r

    def refinement(self, gt):
        p = [1e-5] * 6; a, b = sorted(gt); p[b * (b + 1) // 2 + a] = .99995
        return dict(alleles=['A', 'C', 'AT'], start0=10, added_alleles=0,
                    samples={'singleton': dict(gp=p, dp=100, exact_ad=[60, 40, 0],
                         trusted=True, support_fraction=1., pl=[0] * 6)})

    def test_spanning_evidence_can_score_whole_complex_genotype(self):
        b = bundle(self.record((0, 0)), 'singleton', self.refinement((0, 1)))
        self.assertTrue(b['available'][3])
        self.assertEqual(b['features'].shape, (10, 42))

    def test_untrusted_read_expert_is_available_but_zero_depth_is_not(self):
        refine = self.refinement((0, 2)); refine['samples']['singleton']['trusted'] = False
        b = bundle(self.record((0, 0)), 'singleton', refine)
        self.assertTrue(b['available'][3]); self.assertTrue(b['available'][5])
        self.assertEqual(b['features'][3, 9], 0)
        refine['samples']['singleton']['dp'] = 0
        b = bundle(self.record((0, 0)), 'singleton', refine)
        self.assertFalse(b['available'][3]); self.assertFalse(b['available'][5])

    def test_weak_prior_releases_strong_novel_allele_evidence(self):
        strong = likelihood_posterior([60, 20, 0], [.999, .001], 1.)
        weak = likelihood_posterior([60, 20, 0], [.999, .001], .25)
        self.assertEqual(weak.argmax(), 2)
        self.assertGreater(weak[2], strong[2])
        np.testing.assert_allclose(weak.sum(), 1.)

    def test_diploid_compatibility_handles_ambiguous_exact_matches(self):
        refined=self.refinement((0,1));ex=refined['samples']['singleton']
        ex['exact_ad']=[0,0,0]
        ex['exact_genotype_counts']=[[50,0,0],[15,17,30],[50,0,0],[0,0,0],[0,0,0],[0,0,0]]
        b=bundle(self.record((0,0)),'singleton',refined)
        self.assertAlmostEqual(float(b['features'][3,23]),.62,places=6)
        self.assertAlmostEqual(float(b['features'][3,24]),.15,places=6)
        self.assertAlmostEqual(float(b['features'][3,25]),15/32,places=6)
        ex['exact_genotype_counts'][1]=[0,0,62]
        b=bundle(self.record((0,0)),'singleton',refined)
        self.assertAlmostEqual(float(b['features'][3,23]),.62,places=6)
        self.assertEqual(float(b['features'][3,24]),0.)
        self.assertEqual(float(b['features'][3,25]),0.)

    def test_shared_flanks_preserve_catalog_but_allele_order_does_not(self):
        short = self.refinement((0, 1))
        expanded = dict(short, start0=8, alleles=['GG' + a + 'CC' for a in short['alleles']])
        self.assertEqual(catalog_signature(short), catalog_signature(expanded))
        b = bundle(self.record((0, 0)), 'singleton', expanded, spanning=short)
        self.assertTrue(b['available'][8])
        changed = dict(expanded, alleles=[expanded['alleles'][0], expanded['alleles'][2], expanded['alleles'][1]])
        with self.assertRaisesRegex(ValueError, 'catalog'):
            bundle(self.record((0, 0)), 'singleton', changed, spanning=short)

    def test_expanded_catalog_does_not_invent_old_source_likelihoods(self):
        old = self.refinement((0, 2)); expanded = self.refinement((0, 2))
        expanded['alleles'] = ['A', 'C', 'AT', 'AGG']; expanded['added_alleles'] = 1
        ex = expanded['samples']['singleton']; ex['gp'] = [1e-6] * 10; ex['gp'][6] = 1.
        ex['pl'] = [60.] * 10; ex['pl'][6] = 0.; ex['exact_ad'] = [30, 0, 0, 70]
        b = bundle(self.record((0, 0)), 'singleton', expanded, joint_ld=[1, 0, 0, 0, 0, 0], spanning=old)
        self.assertEqual(b['n_genotypes'], 10)
        self.assertEqual(int(b['probabilities'][3].argmax()), 6)
        for source in [0, 1, 7, 8, 9]:
            self.assertEqual(float(b['probabilities'][source, 6:10].sum()), 0.)
        self.assertAlmostEqual(float(b['probabilities'][9].sum()), 1., places=6)

    def test_joint_ld_requires_sample_and_exact_genotype_dimensions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run = root / 'trial'; run.mkdir()
            (root / 'PREPARED.json').write_text(json.dumps(dict(sample='singleton',
                records_metadata=[dict(record_id='test', alleles=['A', 'C'])])))
            np.savez(run / 'posterior.npz', gp=np.array([.1, .6, .3]), offsets=np.array([0, 3]))
            np.testing.assert_allclose(joint_ld_data(run, 'singleton')['test'], [.1, .6, .3])
            with self.assertRaisesRegex(ValueError, 'sample identity'):
                joint_ld_data(run, 'different_person')
            np.savez(run / 'posterior.npz', gp=np.array([.1, .6, .3]), offsets=np.array([0, 2]))
            with self.assertRaisesRegex(ValueError, 'genotype vector'):
                joint_ld_data(run, 'singleton')

    def test_indel_removal_and_discovery_remain_possible(self):
        for original, refined in [((0, 2), (0, 0)), ((0, 0), (0, 2))]:
            b = bundle(self.record(original), 'singleton', self.refinement(refined))
            self.assertTrue(b['available'][3])

    def test_missing_sources_and_padded_genotypes(self):
        torch.manual_seed(9)
        model = model_class()()
        features = torch.randn(2, 10, 42)
        probs = torch.zeros(2, 10, MAX_GP)
        probs[0, 0, :3] = torch.tensor([.1, .7, .2])
        probs[1, 0, :6] = torch.tensor([.1, .1, .1, .1, .5, .1])
        available = torch.zeros(2, 10, dtype=torch.bool); available[:, 0] = True
        # Bad predictions in missing sources cannot affect the result.
        probs[:, 1:, :] = 100
        out, weights = model(features, probs, available, torch.tensor([3, 6]))
        torch.testing.assert_close(out[0, :3], probs[0, 0, :3])
        torch.testing.assert_close(out[1, :6], probs[1, 0, :6])
        self.assertEqual(float(out[0, 3:].sum()), 0)
        self.assertEqual(float(out[1, 6:].sum()), 0)
        self.assertEqual(float(weights[:, 1:].sum()), 0)

    def test_no_independence_double_counting(self):
        model = model_class()()
        p = torch.zeros(1, 10, MAX_GP); p[:, :, :3] = torch.tensor([.1, .6, .3])
        out, _ = model(torch.zeros(1, 10, 42), p, torch.ones(1, 10, dtype=torch.bool), torch.tensor([3]))
        # Repeating the same evidence in all sources must not sharpen it.
        torch.testing.assert_close(out[0, :3], torch.tensor([.1, .6, .3]))

    def test_row_order_invariance_and_convex_hull(self):
        torch.manual_seed(11)
        model = model_class()()
        x = torch.randn(13, 10, 42)
        p = torch.softmax(torch.randn(13, 10, 3), -1)
        full = torch.zeros(13, 10, MAX_GP); full[:, :, :3] = p
        a = torch.ones(13, 10, dtype=torch.bool); n = torch.full((13,), 3)
        out, _ = model(x, full, a, n)
        perm = torch.randperm(13)
        reverse, _ = model(x[perm], full[perm], a[perm], n[perm])
        torch.testing.assert_close(out[perm], reverse)
        self.assertTrue(torch.all(out[:, :3] >= p.amin(1) - 1e-7))
        self.assertTrue(torch.all(out[:, :3] <= p.amax(1) + 1e-7))
        self.assertTrue(np.isfinite(out.detach().numpy()).all())


if __name__ == '__main__':
    unittest.main()
