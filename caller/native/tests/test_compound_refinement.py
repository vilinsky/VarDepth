"""Eligibility guards; synthetic tests are not genomic accuracy evidence."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'caller'))
from indel_refine import refinement_eligible

class CompoundEligibility(unittest.TestCase):
    def test_compound_snp_uses_read_refinement(self):
        self.assertTrue(refinement_eligible({'alleles':['ACGT','ATAT']},20))
    def test_single_base_and_low_depth_remain_outside(self):
        for alleles,depth in [(['A','T'],100),(['A','C','G'],100),(['ACGT','ATAT'],19)]:
            self.assertFalse(refinement_eligible({'alleles':alleles},depth))
    def test_existing_indel_eligibility_preserved(self):
        for alleles in [['A','AT'],['AT','A'],['ACTG','ACT','ACTT']]:
            self.assertTrue(refinement_eligible({'alleles':alleles},20))
            self.assertFalse(refinement_eligible({'alleles':alleles},19))
if __name__=='__main__':unittest.main()
