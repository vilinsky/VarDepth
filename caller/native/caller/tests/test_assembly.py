import unittest
from assembly import assemble
class AssemblyTests(unittest.TestCase):
 def test_composes_two_separately_seen_insertions(self):
  ref='ACGTCAGTGCTAGCATGCTAGTCAGTACGATCGTACG'
  left=ref[2:10]+'AA'+ref[10:22];right=ref[14:26]+'GG'+ref[26:34]
  obs=[(name,2,22,left) for name in ['a','b']]+[(name,14,34,right) for name in ['c','d']]
  true=ref[:10]+'AA'+ref[10:26]+'GG'+ref[26:]
  one,_=assemble(ref,[ref],obs,lambda x:True,rounds=1)
  two,stats=assemble(ref,[ref],obs,lambda x:True,rounds=2)
  self.assertNotIn(true,one);self.assertIn(true,two)
  self.assertEqual(stats['proposals'][true]['distinct_molecules'],4)
  self.assertEqual(two[0],ref)
 def test_mates_do_not_supply_two_independent_molecules(self):
  ref='ACGTACGTACGT';obs=[('same',1,11,ref[1:5]+'AA'+ref[5:11])]*2
  result,_=assemble(ref,[ref],obs,lambda x:True)
  self.assertEqual(result,[ref])
 def test_order_invariance_and_maximum(self):
  ref='ACGTCAGTGCTAGCATGCTAGTCAGTACGATCGTACG'
  obs=[(name,2,32,ref[2:10]+ins+ref[10:32]) for ins in ['AA','GG','TT'] for name in [ins+'a',ins+'b']]
  a,_=assemble(ref,[ref],obs,lambda x:True,maximum=2)
  b,_=assemble(ref,[ref],obs[::-1],lambda x:True,maximum=2)
  self.assertEqual(a,b);self.assertEqual(len(a),2)
if __name__=='__main__':unittest.main()
