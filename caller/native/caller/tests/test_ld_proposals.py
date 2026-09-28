import itertools
from pathlib import Path
import random
import sys
import unittest
import numpy as np
import pysam

HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from base_adapter import base_modules
base=base_modules()
from local_alleles import make_blocks,sequence_likelihoods
from ld_proposals import select_proposals,changed_blocks,preserve_read_supported_alleles


class LDProposalTests(unittest.TestCase):
    def setUp(self):
        rng=random.Random(33);self.reference=''.join(rng.choice('ACGT') for _ in range(300))
        self.fetch=lambda a,b:self.reference[max(0,a):b]
        self.variants=[(100+j,self.reference[100+j],next(x for x in 'ACGT' if x!=self.reference[100+j])) for j in range(4)]
        vectors=list(itertools.product([0,1],repeat=4));matrix=[]
        for vector in vectors:matrix += [vector]*(1 if vector==(1,1,1,1) else 20)
        h=np.array(matrix,dtype=float).T
        self.panel={v:(h[j],float(h[j].mean())) for j,v in enumerate(self.variants)}
        self.before=make_blocks(self.variants,self.fetch,panel=self.panel,max_alleles=12,groups_override=[[0,1,2,3]])
        self.proposal={(0,1,2,3):[(1,1,1,1)]}

    def test_ld_path_changes_sequence_and_real_read_likelihood(self):
        # One molecular read is insufficient for min-support=2 discovery, but
        # an independently selected LD donor can recover its rare panel path.
        after=make_blocks(self.variants,self.fetch,panel=self.panel,max_alleles=12,
                          groups_override=[[0,1,2,3]],proposal_vectors=self.proposal)
        rare=''.join(v[2] for v in self.variants)
        self.assertNotIn(rare,self.before[0]['alleles']);self.assertIn(rare,after[0]['alleles'])
        self.assertEqual(changed_blocks(self.before,after),[0])
        r=pysam.AlignedSegment();r.query_name='single-molecule';r.reference_id=0;r.reference_start=85
        seq=list(self.reference[85:135]);seq[15:19]=rare
        r.query_sequence=''.join(seq);r.query_qualities=pysam.qualitystring_to_array('I'*50);r.mapping_quality=60;r.cigarstring='50M'
        lib=base.core();old=sequence_likelihoods(self.before[0],[('lib',r)],self.fetch,20,lib)
        new=sequence_likelihoods(after[0],[('lib',r)],self.fetch,20,lib)
        self.assertEqual(new['dp'],1);self.assertEqual(new['fragment_keys'],old['fragment_keys'])
        self.assertEqual(after[0]['alleles'][int(np.argmax(new['lh'][0]))],rare)
        self.assertLess(new['lh'][0,0],old['lh'][0,0])
        # Global population AF must not become the chosen donor's frequency.
        self.assertAlmostEqual(after[0]['allele_af'][after[0]['alleles'].index(rare)],1/301)

    def test_posterior_selects_at_most_two_unique_vectors(self):
        initial={'s':(None,None,{'ld_sequence_proposals':{0:[{'vector':[1,1,1,1],'weight':1.9},{'vector':[1,1,0,0],'weight':.1}]}}),
                 't':(None,None,{'ld_sequence_proposals':{0:[{'vector':[1,1,1,1],'weight':1.8},{'vector':[0,1,1,1],'weight':.8}]}})}
        proposed=select_proposals(initial,self.before)
        self.assertEqual(proposed[(0,1,2,3)],[(1,1,1,1),(0,1,1,1)])

    def test_already_available_ld_path_is_noop(self):
        vector=self.before[0]['representations'][1][0]
        after=make_blocks(self.variants,self.fetch,panel=self.panel,max_alleles=12,
                          groups_override=[[0,1,2,3]],proposal_vectors={(0,1,2,3):[vector]})
        self.assertEqual(changed_blocks(self.before,after),[])

    def test_proposals_cannot_displace_read_supported_sequence(self):
        after=make_blocks(self.variants,self.fetch,panel=self.panel,max_alleles=12,
                          groups_override=[[0,1,2,3]],proposal_vectors=self.proposal)
        removed=set(self.before[0]['alleles'])-set(after[0]['alleles']);self.assertTrue(removed)
        allele=next(iter(removed));idx=self.before[0]['alleles'].index(allele)
        vector=self.before[0]['representations'][idx][0];j=next(j for j,v in enumerate(vector) if v)
        guarded=preserve_read_supported_alleles(self.before,after,self.variants,{self.variants[j]:2})
        self.assertEqual(changed_blocks(self.before,guarded),[])


if __name__=='__main__':unittest.main()
