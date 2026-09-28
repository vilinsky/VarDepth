import unittest
from ownership_rules import implied,redundant_records
E=(100,'AT','A');S=(50,'G','T');Q=(105,'C','A')
F=frozenset
class GenotypeOwnership(unittest.TestCase):
 def test_superset_with_outside_snp(self):self.assertTrue(implied((F([S]),F([S,E])),(F(),F([E])),90,120))
 def test_chromosome_swap(self):self.assertTrue(implied((F([E]),F()),(F(),F([E])),90,120))
 def test_dosage_mismatch(self):self.assertFalse(implied((F([E]),F([E])),(F(),F([E])),90,120))
 def test_cis_trans_not_equivalent(self):self.assertFalse(implied((F([E,Q]),F()),(F([E]),F([Q])),90,120))
 def test_extra_inside_edit_not_ignored(self):self.assertFalse(implied((F([Q]),F([E])),(F(),F([E])),90,120))
 def test_crossing_boundary_not_truncated(self):self.assertFalse(implied((F(),F([(85,'A'*20,'A')]),),(F(),F()),90,120))
 def test_all_samples_must_agree_and_owner_survives(self):
  def d(pair,rank,second=None):return dict(events=F([E]),pairs={'a':pair,'b':second or pair},left=90,right=120,rank=rank)
  child=(F(),F([E]));parent=(F([S]),F([S,E]))
  ds={0:d(parent,(100,2,.999)),1:d(child,(30,1,.999)),2:d(child,(20,1,.999)),3:d(child,(10,1,.999),second=(F([E]),F([E])))}
  self.assertEqual(redundant_records(ds),{1:0,2:0})
if __name__=='__main__':unittest.main()
