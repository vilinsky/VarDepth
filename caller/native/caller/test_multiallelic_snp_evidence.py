import sys,unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pysam
sys.path.insert(0,str(Path(__file__).parent/'base'))
from read_index import ReadIndex
from local_alleles import sequence_likelihoods,single_base_likelihoods
def read(name,base,q=30):
 r=pysam.AlignedSegment();r.query_name=name;r.query_sequence='AA'+base+'AA'
 if q is not None:r.query_qualities=[30,30,q,30,30]
 r.reference_id=0;r.reference_start=100;r.mapping_quality=60;r.cigartuples=[(0,5)];return r
class MultiallelicSnpEvidence(unittest.TestCase):
 def test_short_window_does_not_erase_observed_multiallelic_base(self):
  block=dict(start=102,end=103,left=102,right=103,alleles=['C','G','T'])
  x=single_base_likelihoods(block,ReadIndex([('lib',read('x','G'))]),lambda a,b:'C',20)
  self.assertEqual(x['dp'],1);self.assertEqual(x['ad'].tolist(),[0,1,0])
  e=1e-3;mq=1e-6;expected=np.array([(1-mq)*e/3+mq/4,(1-mq)*(1-e)+mq/4,(1-mq)*e/3+mq/4])
  np.testing.assert_allclose(x['lh'][0],expected/expected.max(),rtol=1e-14)
 def test_overlapping_mates_count_once_and_bq_filter_applies(self):
  block=dict(start=102,end=103,alleles=['C','G','T'])
  reads=[('lib',read('same','C',21)),('lib',read('same','G',35)),('lib',read('bad','T',5))]
  x=single_base_likelihoods(block,ReadIndex(reads),lambda a,b:'C',20)
  self.assertEqual(x['dp'],1);self.assertEqual(x['ad'].tolist(),[0,1,0])
 def test_library_identity_and_missing_quality(self):
  block=dict(start=102,end=103,alleles=['C','G','T']);bad=read('bad','G',None)
  x=single_base_likelihoods(block,ReadIndex([('a',read('same','G')),('b',read('same','T')),('a',bad)]),lambda a,b:'C',20)
  self.assertEqual(x['dp'],2)
 def test_available_sequence_evidence_is_preserved(self):
  block=dict(start=102,end=103,alleles=['C','G','T'])
  pending=[('lib',None,[],0,15,'A'*15,b'?'*15)]
  context=dict(pending=pending,haps=['AACAA','AAGAA','AATAA'])
  scores=np.array([[-2.,-1.,-3.]])
  sentinel={'source':'sequence'}
  with patch('local_alleles._sequence_inputs',return_value=context), patch('caller.pair_scores_many',return_value=scores) as score, patch('local_alleles._finish_sequence',return_value=sentinel), patch('local_alleles.single_base_likelihoods') as base:
   result=sequence_likelihoods(block,[],None,20,None)
   self.assertIs(result,sentinel);score.assert_called_once();base.assert_not_called()
 def test_rescue_requires_twenty_distinct_molecules(self):
  block=dict(start=102,end=103,left=102,right=103,alleles=['C','G','T'])
  for count,expected in [(19,0),(20,20)]:
   reads=ReadIndex([('lib',read(str(i),'G')) for i in range(count)])
   with patch('caller.pair_scores_many',return_value=np.empty((0,3))):
    x=sequence_likelihoods(block,reads,lambda a,b:'C',20,None)
   self.assertEqual(x['dp'],expected)
if __name__=='__main__':unittest.main()
