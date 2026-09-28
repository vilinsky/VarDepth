import unittest
from proposals import merge_catalogs
class PreservationTests(unittest.TestCase):
 def test_shared_context_padding_preserves_order(self):
  ref='ACGTACGTACGT';fetch=lambda a,b:ref[a:b]
  before=dict(start0=3,end0=6,alleles=['TAC','TAAC'],added_alleles=1)
  new=dict(start0=2,end0=7,alleles=['GTACG','GTAACG','GTATCG'],added_alleles=2)
  merged,n=merge_catalogs(before,new,fetch)
  self.assertEqual(merged['alleles'],['GTACG','GTAACG','GTATCG']);self.assertEqual(n,1)
  self.assertEqual(merged['added_alleles'],2)
 def test_capacity_never_evicts_old_alleles(self):
  before=dict(start0=0,end0=1,alleles=['A','AT','AG'],added_alleles=1)
  new=dict(start0=0,end0=1,alleles=['A','AC','ACC'],added_alleles=2)
  merged,n=merge_catalogs(before,new,lambda a,b:'A'[a:b],maximum=4)
  self.assertEqual(merged['alleles'],['A','AT','AG','AC']);self.assertEqual(n,1)
if __name__=='__main__':unittest.main()
