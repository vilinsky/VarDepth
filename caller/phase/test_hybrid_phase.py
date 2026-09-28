import itertools
import sys
import unittest
from pathlib import Path
import numpy as np
import pysam
sys.path.insert(0,str(Path(__file__).parent/'caller/caller'))
from hybrid_phase import phase_edges,transition,Edge,phase_forest,allele_from_read,molecular_edges

class PhaseTests(unittest.TestCase):
    def test_transition_dense_equivalence(self):
        rng=np.random.default_rng(8);a=rng.random((5,5));r=.13;t=(1-r)*np.eye(5)+r/5
        np.testing.assert_allclose(transition(a,r),t@a@t.T,rtol=1e-14,atol=1e-14)

    def test_exact_bruteforce_hidden_paths(self):
        h=np.array([[0,1],[1,0],[0,1]],float);g=np.array([1,0,1]);r=[0,.1,.15];mu=.07
        same=diff=0.
        for donor in itertools.product(range(2),repeat=6):
            w=1/4
            for t in [1,2]:
                for c in range(2):w*=((1-r[t])*(donor[2*t+c]==donor[2*(t-1)+c])+r[t]/2)
            for alleles in itertools.product(range(2),repeat=6):
                if any(sum(alleles[2*t:2*t+2])!=g[t] for t in range(3)):continue
                p=w
                for t in range(3):
                    for c in range(2):
                        x=mu+(1-2*mu)*h[t,donor[2*t+c]];p*=x if alleles[2*t+c] else 1-x
                if alleles[0]==alleles[4]:same+=p
                else:diff+=p
        self.assertAlmostEqual(phase_edges(h,g,r,mu)[0][2],diff/(same+diff),12)

    def test_independent_donors_uninformative(self):
        self.assertAlmostEqual(phase_edges([[0,1],[0,1]],[1,1],[0,1])[0][2],.5,12)

    def test_donor_label_invariance(self):
        h=np.array([[0,0,1],[1,0,0],[0,1,1]])
        a=phase_edges(h,[1,1,1],[0,.02,.04]);b=phase_edges(h[:,::-1],[1,1,1],[0,.02,.04])
        np.testing.assert_allclose(a,b,atol=1e-12)

    def test_gt_swap_complements_parity(self):
        h=np.array([[0,1],[0,1]]);a=phase_edges(h,[1,1],[0,.01])[0][2]
        h[1]=1-h[1];b=phase_edges(h,[1,1],[0,.01])[0][2]
        self.assertAlmostEqual(a+b,1,12)

    def test_forest_budget(self):
        result,meta=phase_forest(3,[Edge(0,1,0,.006,'panel'),Edge(1,2,1,.006,'reads')])
        self.assertEqual(len(result),2);self.assertEqual(meta['budget_rejected_edges'],1)

    def test_strong_conflict_abstains(self):
        result,meta=phase_forest(3,[Edge(0,1,0,.001,'panel'),Edge(1,2,0,.001,'reads'),Edge(0,2,1,.002,'reads')])
        self.assertFalse(result);self.assertEqual(meta['withheld_components'],1)

    def test_trans_orientation(self):
        result,_=phase_forest(2,[Edge(0,1,1,.001,'reads')]);self.assertEqual(result[0][1]^result[1][1],1)

    def test_branches_do_not_consume_unrelated_pair_budget(self):
        result,_=phase_forest(6,[Edge(0,j,j%2,.003,'reads') for j in range(1,6)])
        self.assertEqual(len(result),6)
        self.assertAlmostEqual(result[0][2],.006)

    def test_local_conflict_preserves_unaffected_branch(self):
        edges=[Edge(0,1,0,.001,'panel'),Edge(1,2,0,.001,'panel'),Edge(0,2,1,.002,'reads'),
               Edge(2,3,0,.001,'panel'),Edge(3,4,1,.001,'panel')]
        result,_=phase_forest(5,edges)
        self.assertEqual(set(result),{3,4});self.assertEqual(result[3][1]^result[4][1],1)

    def read(self,seq,cigar,start=98):
        r=pysam.AlignedSegment();r.query_name='fragment';r.query_sequence=seq;r.reference_id=0;r.reference_start=start
        r.mapping_quality=60;r.cigarstring=cigar;r.query_qualities=pysam.qualitystring_to_array('I'*len(seq));return r

    def test_snv(self):self.assertEqual(allele_from_read(self.read('AAGAA','5M'),100,('A','G'))[0],1)
    def test_insertion(self):self.assertEqual(allele_from_read(self.read('AACGTAA','3M2I2M'),100,('C','CGT'))[0],1)
    def test_deletion(self):self.assertEqual(allele_from_read(self.read('AACAA','3M2D2M'),100,('CGT','C'))[0],1)
    def test_reference_indel_allele(self):self.assertEqual(allele_from_read(self.read('AACGTAA','7M'),100,('CGT','C'))[0],0)
    def test_uncovered_deletion_is_missing(self):self.assertIsNone(allele_from_read(self.read('AAA','2M3D1M'),100,('CG','C')))
    def test_reference_skip_is_missing(self):self.assertIsNone(allele_from_read(self.read('AACAA','3M2N2M'),100,('CGT','C')))
    def test_softclip_not_insertion(self):self.assertIsNone(allele_from_read(self.read('AAACGT','3M3S'),100,('A','ACGT')))
    def test_shifted_repeat_alt_is_not_reference(self):
        self.assertIsNone(allele_from_read(self.read('AAAAAAAAAAAA','6M2I4M'),100,('A','AAA')))
    def test_molecule_count_and_parity(self):
        out=molecular_edges([{0:(0,.01),1:(1,.01)}]*3);self.assertEqual(out[0].support,3);self.assertEqual(out[0].parity,1)
        self.assertFalse(molecular_edges([{0:None,1:(1,.01)}]*10))

if __name__=='__main__':unittest.main()
