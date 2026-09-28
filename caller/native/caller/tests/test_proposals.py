import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pysam
from baseline_proposals import splice_haplotype, propose

class SpliceTests(unittest.TestCase):
    def test_reference_background_insertion(self):
        self.assertEqual(splice_haplotype('ACGTAACG', np.arange(8), 2, 6, 'GTGAA'), 'ACGTGAACG')

    def test_template_outside_observation_preserved(self):
        # A template deletion to the left shifts coordinates in the haplotype.
        mapping = np.array([0, 1, -1, 2, 3, 4, 5, 6])
        self.assertEqual(splice_haplotype('ACTAACG', mapping, 3, 6, 'TGAA'), 'ACTGAACG')

    def test_deleted_anchor_rejected(self):
        with self.assertRaises(ValueError):
            splice_haplotype('ACG', np.array([0, -1, 1, 2]), 1, 3, 'TT')

    def proposal_fixture(self, names):
        reference = 'ACGTCAGTGCATGACTACGATCGTACCTGATCGATGCTAGTCAGGCTACATGCAGTACGATGCATAGCTCGATCGTAGCTAGTCAGCTAGCA'
        start, end = 5, 85
        original = dict(start0=start, end0=end, alleles=[reference[start:end], reference[start:40] + reference[42:end]])
        row = dict(original, chrom='chr1', added_alleles=0)
        reads = []
        for name in names:
            read = pysam.AlignedSegment(); read.query_name = name; read.reference_id = 0
            read.reference_start = 20; read.mapping_quality = 60; read.cigartuples = [(0, 20), (1, 2), (0, 25)]
            read.query_sequence = reference[20:40] + 'AA' + reference[40:65]
            read.query_qualities = pysam.qualitystring_to_array('I' * len(read.query_sequence)); reads.append(read)
        class Bam:
            def fetch(self, *args): return iter(reads)
        return propose(row, original, {'sample': [(Bam(), {}, 'library')]}, lambda a, b: reference[max(0, a):b])

    def test_two_partial_molecules_propose_new_allele(self):
        row, stats = self.proposal_fixture(['a', 'b'])
        self.assertGreater(stats['added'], 0)
        self.assertEqual(row['added_alleles'], stats['added'])
        self.assertTrue(all(r['distinct_molecules'] == 2 for r in stats['proposals'].values()))

    def test_duplicate_molecule_does_not_propose(self):
        row, stats = self.proposal_fixture(['same', 'same'])
        self.assertEqual(stats['added'], 0)

if __name__ == '__main__': unittest.main()
