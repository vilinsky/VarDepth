import os
import unittest
from unittest.mock import patch
import numpy as np
from base_adapter import base_modules

class BaseAccelerationParity(unittest.TestCase):
    def test_cpu_batch_matches_original(self):
        caller=base_modules();lib=caller.core()
        seq=['ACGT'*20,'A'*31,'GATTACA'*9];qs=[[20]*len(s) for s in seq];haps=['ACGT'*25,'A'*36,'GATTACA'*11]
        expected=np.stack([caller.pair_scores(lib,s,q,haps) for s,q in zip(seq,qs)])
        with patch.dict(os.environ,VARDEPTH_PAIRHMM_BACKEND='cpu'):
            # Prefix sharing changes normalization order in FP64 (observed ~1e-13).
            # Require tight likelihood agreement, and gate emitted VCF separately.
            np.testing.assert_allclose(caller.pair_scores_many(lib,seq,qs,haps),expected,atol=2e-12,rtol=2e-14)
            self.assertEqual(caller.pair_scores_many(lib,[],[],haps).shape,(0,3))
    @unittest.skipUnless(os.environ.get('VARDEPTH_TEST_CUDA')=='1','CUDA unavailable in this test run')
    def test_cuda_batch_matches_base_model(self):
        caller=base_modules();lib=caller.core();rng=np.random.default_rng(712)
        seq=[''.join(rng.choice(list('ACGT'),n)) for n in [12,151,700]]
        qs=[rng.integers(2,61,len(s)).tolist() for s in seq];haps=[''.join(rng.choice(list('ACGT'),n)) for n in [17,197,850]]
        expected=np.stack([caller.pair_scores(lib,s,q,haps) for s,q in zip(seq,qs)])
        with patch.dict(os.environ,VARDEPTH_PAIRHMM_BACKEND='cuda'):
                np.testing.assert_allclose(caller.pair_scores_many(lib,seq,qs,haps),expected,atol=2e-9,rtol=0)

if __name__=='__main__':unittest.main()
