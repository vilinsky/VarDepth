import unittest
from types import SimpleNamespace
from scaffold_attach38 import attach,overlapping_indices

class AttachmentTests(unittest.TestCase):
    def scaffold(self):return {0:(0,0,.002),1:(0,1,.002),2:(0,0,.002)}
    def test_pool_fragmented_anchors(self):
        m=[{0:(0,.005),3:(1,.02)},{1:(0,.005),3:(0,.02)},{2:(0,.005),3:(1,.02)}]
        out,stats,rows=attach(self.scaffold(),m)
        self.assertEqual(out[3][1],1);self.assertEqual(rows[0]['molecules'],3)
        for i in self.scaffold():self.assertEqual(out[i][:2],self.scaffold()[i][:2])
    def test_many_snps_not_many_molecules(self):
        m=[{0:(0,.005),1:(1,.005),2:(0,.005),3:(0,.02)}]
        self.assertNotIn(3,attach(self.scaffold(),m)[0])
    def test_conflicting_molecule_withheld(self):
        m=[{0:(0,.005),1:(0,.005),3:(0,.02)}]*10
        out,stats,_=attach(self.scaffold(),m)
        self.assertNotIn(3,out);self.assertEqual(stats['discordant_molecule_blocks'],10)
    def test_one_allele_only_withheld(self):
        m=[{0:(0,.005),3:(0,.02)}]*10
        self.assertNotIn(3,attach(self.scaffold(),m)[0])
    def test_no_recursive_attachment(self):
        m=[{0:(0,.005),3:(1,.02)},{1:(0,.005),3:(0,.02)},{2:(0,.005),3:(1,.02)}]
        m += [{3:(0,.005),4:(0,.005)}]*3
        out,_,_=attach(self.scaffold(),m)
        self.assertIn(3,out);self.assertNotIn(4,out)
    def test_cannot_override_prior_quarantine(self):
        m=[{0:(0,.005),3:(1,.02)},{1:(0,.005),3:(0,.02)},{2:(0,.005),3:(1,.02)}]
        self.assertNotIn(3,attach(self.scaffold(),m,forbidden={3})[0])
    def test_two_leaf_error_reserved(self):
        m=[{0:(0,.005),3:(1,.02)},{1:(0,.005),3:(0,.02)},{2:(0,.005),3:(1,.02)}]
        scaffold={i:(r,b,.0099) for i,(r,b,e) in self.scaffold().items()}
        self.assertNotIn(3,attach(scaffold,m)[0])
    def test_nested_overlap(self):
        rows=[SimpleNamespace(start=0,stop=100),SimpleNamespace(start=1,stop=2),SimpleNamespace(start=5,stop=6),SimpleNamespace(start=100,stop=101)]
        self.assertEqual(overlapping_indices(rows),{0,1,2})
    def test_snv_neighbours_not_overlapping(self):
        self.assertEqual(overlapping_indices([SimpleNamespace(start=i,stop=i+1) for i in range(5)]),set())

if __name__=='__main__':unittest.main()
