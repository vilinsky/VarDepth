import os
import unittest
from unittest.mock import patch
import numpy as np
from accelerator import cpu_pairs,cpu_simd_pairs,validate_pairs,score_many,score_pairs,CudaPairHMM,pack_inputs
from whole_pairhmm import Prepared

class PairBatchTests(unittest.TestCase):
    def test_cuda_input_layout_contract_and_padding(self):
        import math
        pairs=validate_pairs([('ACNT',[0,20,60,255],'GATTACA'),('G',[3],'G'),('TGA',[10,20,30],'ACTG')])
        ids=[2,0,1];old=pack_inputs(pairs,ids,8,9);new=pack_inputs(pairs,ids,8,9,pair_major=True)
        for x,y in zip(old,new):np.testing.assert_array_equal(x.T if x.ndim==2 else x,y)
        self.assertTrue(all(x.flags.c_contiguous for x in old+new))
        for j,i in enumerate(ids):
            s,q,h=pairs[i]
            self.assertEqual(bytes(new[0][j,:len(s)]).decode(),s)
            self.assertEqual(bytes(new[1][j,:len(h)]).decode(),h)
            np.testing.assert_array_equal(new[2][j,:len(s)],[math.pow(10.,-min(60,int(v))/10.) for v in q])
            self.assertFalse(new[0][j,len(s):].any());self.assertFalse(new[1][j,len(h):].any());self.assertFalse(new[2][j,len(s):].any())
    def test_cpu_matches_existing_simd_and_empty(self):
        reads=['ACGT'*8,'A'*30,'GATTACA'];quals=[[30]*len(x) for x in reads]
        haps=['TT'+r+'GG' for r in reads]+['ACGT'*10]
        expected=np.array([Prepared(haps,mode=0)(r,q) for r,q in zip(reads,quals)])
        actual=score_many(reads,quals,haps,selection='cpu')
        np.testing.assert_allclose(actual,expected,atol=1e-11,rtol=0)
        self.assertEqual(score_many([],[],haps,selection='cpu').shape,(0,4))
        self.assertEqual(score_many(reads,quals,[],selection='cpu').shape,(3,0))
    def test_input_guards(self):
        for pairs in [[('A',[256],'A')],[('A',[-1],'A')],[('A',[3.5],'A')],[('A',[],'A')],[('X',[30],'A')]]:
            with self.assertRaises(ValueError):validate_pairs(pairs)
        with self.assertRaises(ValueError):score_pairs([('A',[30],'A')],selection='fake')
    def test_empty_sequences(self):
        self.assertTrue(np.isneginf(cpu_pairs([('',[],'A'),('A',[30],'')])).all())
    def test_simd_fallback_preserves_order_and_qualities(self):
        pairs=[('ACGT'*8,[30]*32,'ACGT'*9),('ACGT'*8,[3]*32,'ACGT'*9),
               ('GATTACA',[20]*7,'GGGATTACAT'),('ACGT'*8,[30]*32,'TCGT'*9),
               ('',[],'A'),('A',[30],'')]
        np.testing.assert_allclose(cpu_simd_pairs(pairs),cpu_pairs(pairs),atol=1e-11,rtol=0)
        self.assertEqual(cpu_simd_pairs([]).shape,(0,))
    @unittest.skipUnless(os.environ.get('VARDEPTH_TEST_CUDA')=='1','Explicit real CUDA test required')
    def test_cuda_model_parity_and_chunking(self):
        rng=np.random.default_rng(31);pairs=[]
        for n,m in [(1,1),(7,9),(30,35),(151,201),(300,450),(700,750)]:
            for k in range(6):
                s=''.join(rng.choice(list('ACGT'),n));h=''.join(rng.choice(list('ACGT'),m));q=rng.integers(0,61,n)
                pairs.append((s,q,h))
        engine=CudaPairHMM(memory_mb=8)
        for anchored in [False,True]:
            np.testing.assert_allclose(engine.score(pairs,anchored),cpu_pairs(pairs,anchored),atol=2e-9,rtol=0)
        pairs=[('ACGT'*10,[30]*40,'ACGT'*12)]*12000
        np.testing.assert_allclose(engine.score(pairs),cpu_pairs(pairs),atol=2e-9,rtol=0)
        self.assertGreater(engine.last['batches'],1)

    @unittest.skipUnless(os.environ.get('VARDEPTH_TEST_CUDA')=='1','Explicit real CUDA test required')
    def test_mixed_precision_guarded_errors(self):
        rng=np.random.default_rng(17391);pairs=[]
        for n,m in [(1,1),(7,9),(30,35),(151,201),(300,450),(512,1000),(700,750)]:
            for k in range(12):
                seq=''.join(rng.choice(list('ACGT'),n));hap=''.join(rng.choice(list('ACGT'),m))
                pairs.append((seq,rng.integers(1,61,n),hap))
        pairs.extend([('A'*100,[60]*100,'A'*100),('A'*200,[35]*200,'A'*450),('ACGT'*100,[60]*400,'TGCA'*160)])
        engine=CudaPairHMM(memory_mb=8,precision='mixed')
        for anchored in (False,True):
            actual=engine.score(pairs,anchored);expected=cpu_pairs(pairs,anchored)
            np.testing.assert_allclose(actual,expected,atol=2e-3,rtol=0)
            self.assertGreaterEqual(engine.last['float64_fallback_pairs'],12)
        self.assertEqual(engine.last['precision'],'mixed')

if __name__=='__main__':unittest.main()
