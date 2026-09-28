import sys
import tempfile
from pathlib import Path
import unittest
import pysam

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from indel_refine import fragment_seeded_haplotypes


class FragmentProposals(unittest.TestCase):
    reference = 'ACGT' * 100
    record = dict(chrom='chrSynthetic', start0=100, end0=150)

    def bam(self, path, names):
        h = dict(HD=dict(VN='1.6', SO='coordinate'), SQ=[dict(SN='chrSynthetic', LN=400)])
        with pysam.AlignmentFile(path, 'wb', header=h) as out:
            for start, flag, cigar in [(100, 99, [(0, 20), (1, 1), (0, 10)]), (125, 147, [(0, 25)])]:
                for name in names:
                    r = pysam.AlignedSegment(out.header); r.query_name = name
                    r.reference_id = 0; r.reference_start = start; r.flag = flag
                    r.mapping_quality = 60; r.cigartuples = cigar
                    r.query_sequence = (self.reference[100:120] + 'A' + self.reference[120:130]
                                        if start == 100 else self.reference[125:135] + 'C' + self.reference[136:150])
                    r.query_qualities = [35] * len(r.query_sequence)
                    r.set_tag('RG', 'undeclared'); out.write(r)
        pysam.index(str(path)); return str(path)

    def proposals(self, paths):
        return fragment_seeded_haplotypes(self.record, {'sample': paths}, lambda s, e: self.reference[s:e])

    def test_mates_link_indel_and_snp(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.bam(Path(tmp) / 'a.bam', ['fragment1', 'fragment2'])
            result = self.proposals([p])
            expected = self.reference[100:120] + 'A' + self.reference[120:135] + 'C' + self.reference[136:150]
            self.assertEqual(result, {expected: 2})

    def test_duplicate_alignments_are_not_independent_molecules(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.bam(Path(tmp) / 'a.bam', ['same', 'same'])
            self.assertEqual(self.proposals([p]), {})

    def test_unknown_read_groups_do_not_merge_distinct_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [self.bam(Path(tmp) / name, ['same']) for name in ['a.bam', 'b.bam']]
            self.assertEqual(list(self.proposals(paths).values()), [2])


if __name__ == '__main__':
    unittest.main()
