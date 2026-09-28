import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'caller/base'))
from local_alleles import supported_candidates,overlap_groups

class StrongCandidates(unittest.TestCase):
    def test_rescue_only_independently_supported_snps(self):
        bridges=[(p,'A'*31,'A') for p in range(0,151,20)]
        strong=(160,'A','C');weak=(159,'A','G');indel=(158,'A','AC')
        variants=sorted(bridges+[strong,weak,indel])
        kept,dropped=supported_candidates(variants,{strong:20,weak:19,indel:80},{})
        self.assertEqual(kept,[strong]);self.assertEqual(sorted(kept+dropped),variants)
        for ids in overlap_groups(kept):self.assertLessEqual(len(ids),3)

    def test_low_evidence_oversized_component_keeps_original_behavior(self):
        variants=[(p,'A'*31,'A') for p in range(0,151,20)]+[(160,'A','C')]
        kept,dropped=supported_candidates(sorted(variants),{variants[-1]:2},{})
        self.assertFalse(kept);self.assertEqual(dropped,sorted(variants))

    def test_regular_short_candidates_unchanged(self):
        variants=[(10,'A','C'),(11,'AC','A'),(50,'T','A')]
        self.assertEqual(supported_candidates(variants,{},{}),(variants,[]))

if __name__=='__main__':unittest.main()
