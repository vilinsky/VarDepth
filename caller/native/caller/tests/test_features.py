import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from features import genotypes,genotype_index,normalize_record,pack_fragments,build_row,empty_arrays
from call import decode_probabilities,check_probabilities,add_cohort_context


class FeatureTests(unittest.TestCase):
    def test_all_triangular_genotypes(self):
        for count in range(2,13):
            for index,pair in enumerate(genotypes(count)):
                self.assertEqual(genotype_index(*pair),index)
                gp=np.zeros(78);gp[index]=1
                gt,gq,ds=decode_probabilities(gp,count)
                self.assertEqual(gt,pair);self.assertEqual(gq,99)
                self.assertEqual(sum(ds),sum(x!=0 for x in pair))

    def test_multiallelic_normalization(self):
        ref='G'*20+'TC'+'A'*20
        start,alleles=normalize_record(20,['TC','TA','TG'],lambda a,b:ref[a:b])
        self.assertEqual((start,alleles),(21,('C','A','G')))
        ref='CAAAGT'
        start,alleles=normalize_record(2,['A','AA'],lambda a,b:ref[a:b])
        self.assertEqual((start,alleles),(0,('C','CA')))

    def test_sampling_ignores_evidence_and_preserves_rows(self):
        lh=np.ones((100,3));lh[:,1]=np.linspace(.01,.99,100)
        keys=[('lib',str(i)) for i in range(100)];meta=np.arange(800,dtype=float).reshape(100,8)
        f,m,selected=pack_fragments(lh,keys,meta,'chr1:1:A:C,G','sample')
        g,n,other=pack_fragments(lh[:,::-1],keys,meta,'chr1:1:A:C,G','sample')
        self.assertEqual(selected,other);self.assertEqual(int(m.sum()),64)
        np.testing.assert_array_equal(f[:64,12:],meta[selected])
        self.assertTrue(np.all(f[:64,:3]<=0))
        self.assertTrue(np.all(f[:64,:3]>=-20))

    def test_allele_permutation_corresponds_to_likelihood(self):
        lh=np.array([[1.,.1,.2],[.01,1.,.4]])
        f,_,_=pack_fragments(lh,[('l','a'),('l','b')],np.zeros((2,8)),'r','s')
        g,_,_=pack_fragments(lh[:,[0,2,1]],[('l','a'),('l','b')],np.zeros((2,8)),'r','s')
        np.testing.assert_array_equal(f[:2,[0,2,1]],g[:2,:3])

    def test_posterior_and_likelihood_are_separate(self):
        rec={'alleles':['A','C'],'start0':100,'record_id':'r','panel_af':[.9,.1],'read_support':3}
        payload={'gp':np.array([.1,.8,.1]),'pl':np.array([0.,-1.,-2.]),'dp':0,'ad':[0,0]}
        row,_=build_row(rec,payload,np.zeros((0,2)),[],np.zeros((0,8)),lambda a,b:'A'*(b-a),'s')
        self.assertEqual(np.argmax(row['base_gp']),1);self.assertEqual(np.argmin(row['read_pl'][:3]),0)
        self.assertEqual(row['genotype_mask'].sum(),3);self.assertEqual(row['fragment_mask'].sum(),0)

    def test_reject_invalid_posterior_or_padding(self):
        gp=np.zeros((1,78));gp[0,3]=1
        with self.assertRaises(ValueError):check_probabilities(gp,[2])
        gp[0,0]=np.nan
        with self.assertRaises(ValueError):check_probabilities(gp,[2])

    def test_empty_catalog(self):
        arrays=empty_arrays();meta={'samples':['s'],'records':[],'rows':[]}
        out,p=add_cohort_context(arrays,meta)
        self.assertEqual(out['cohort_context'].shape,(0,12,6))
        check_probabilities(np.zeros((0,78)),[])

    def test_single_sample_cohort_is_neutral(self):
        arrays=empty_arrays()
        arrays.update(base_gp=np.pad(np.array([[.8,.1,.1]]),((0,0),(0,75))),read_pl=np.zeros((1,78)),
                      depth=np.array([5]),sample_index=np.array([0]),record_index=np.array([0]))
        meta={'samples':['s'],'records':[{'alleles':['A','C'],'start0':100,'chrom':'chr1'}],'rows':[{}]}
        out,p=add_cohort_context(arrays,meta)
        self.assertEqual(out['cohort_context'].shape,(1,12,6))
        self.assertTrue(np.all(out['cohort_context']==0));self.assertTrue(np.all(out['context_strength']==0))


if __name__=='__main__':unittest.main()
