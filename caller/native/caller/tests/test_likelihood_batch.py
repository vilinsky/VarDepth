"""Across-locus batching preserves molecular evidence, order and empty loci."""
import os
import random
import unittest
from unittest.mock import patch

import numpy as np
import pysam
from base_adapter import base_modules


class AcrossLocusBatchTests(unittest.TestCase):
    def setUp(self):
        self.caller = base_modules()
        self.lib = self.caller.core()
        rng = random.Random(173)
        self.reference = ''.join(rng.choice('ACGT') for _ in range(1200))
        self.fetch = lambda a, b: self.reference[a:b]
        self.variants = []
        for p in (100, 350, 800):
            self.variants.extend([(p, self.reference[p], self.reference[p] + 'GG'),
                                  (p, self.reference[p:p+3], self.reference[p])])
        self.reads = []
        for p in (100, 350):
            for j, cigar in enumerate(('90M', '45M2I45M', '43M2D47M', '90M')):
                r = pysam.AlignedSegment()
                r.query_name = str(p) + '-' + str(j if j != 3 else 0)
                r.reference_id = 0
                r.reference_start = p - 44
                r.cigarstring = cigar
                s = self.reference[p-44:p+46]
                if j == 1: s = s[:45] + 'GG' + s[45:]
                if j == 2: s = self.reference[p-44:p-1] + self.reference[p+1:p+48]
                r.query_sequence = s
                r.query_qualities = [35 if j != 3 else 21] * len(s)
                r.mapping_quality = 60 if j != 3 else 25
                self.reads.append(('library', r))

    def compare(self, gpu):
        import accelerator
        from likelihood_batch import indel_inputs, finish_indel, indel_batches
        from local_alleles import make_blocks, sequence_likelihoods, sequence_likelihoods_many
        blocks = make_blocks(self.variants, self.fetch, groups_override=[[0, 1], [2, 3], [4, 5]])
        # Include an empty block in the middle, so offset mistakes cannot hide.
        blocks = [blocks[1], blocks[2], blocks[0]]
        expected = {}
        with patch.dict(os.environ, VARDEPTH_PAIRHMM_BACKEND='cpu'):
            for i, v in enumerate(self.variants):
                pending, haps = indel_inputs(v, self.reads, self.fetch, 20)
                ll = self.caller.pair_scores_many(self.lib, [x[2] for x in pending], [x[3] for x in pending], haps)
                expected[i] = finish_indel(pending, ll)
            expected_blocks = [sequence_likelihoods(b, self.reads, self.fetch, 20, self.lib) for b in blocks]
        self.assertEqual([b['dp'] for b in expected_blocks], [3, 0, 3])
        # A CPU scorer exercises batching on hosts without CUDA. The CUDA test
        # below runs the identical regrouping on the actual device.
        scorer = accelerator.score_pairs if gpu else accelerator.cpu_pairs
        for limit in (1, 16, 32768):
            with patch.dict(os.environ, VARDEPTH_PAIRHMM_BACKEND='cuda'), patch.object(accelerator, 'score_pairs', side_effect=scorer):
                actual = indel_batches(self.variants, self.reads, self.fetch, 20, max_pairs=limit)
                actual_blocks = sequence_likelihoods_many(blocks, self.reads, self.fetch, 20, self.lib, max_pairs=limit)
            self.assertEqual(list(actual), list(expected))
            for i in expected:
                self.assertEqual(list(actual[i]), list(expected[i]))
                for key in expected[i]:
                    np.testing.assert_allclose(actual[i][key][0], expected[i][key][0], atol=2e-9, rtol=0)
                    np.testing.assert_allclose(actual[i][key][1:], expected[i][key][1:], atol=2e-9, rtol=0)
            for a, b in zip(actual_blocks, expected_blocks):
                for name in ('dp', 'fragment_keys', 'partial_alignments'):
                    self.assertEqual(a[name], b[name])
                np.testing.assert_array_equal(a['ad'], b['ad'])
                np.testing.assert_allclose(a['lh'], b['lh'], atol=2e-9, rtol=0)

    def test_across_locus_order_empty_and_molecule_parity(self):
        self.compare(gpu=False)

    @unittest.skipUnless(os.environ.get('VARDEPTH_TEST_CUDA') == '1', 'CUDA unavailable in this test run')
    def test_cuda_across_locus_parity(self):
        self.compare(gpu=True)

    def compare_whole(self, gpu):
        import accelerator
        from whole_read import score, score_rows
        class MemoryBam:
            def __init__(self, reads): self.reads = reads
            def fetch(self, chrom, start, end):
                return [r for lib, r in self.reads if r.reference_start < end and r.reference_end > start]
        rows = [dict(chrom='chr1', start0=p, end0=p+3,
                     alleles=[self.reference[p:p+3], self.reference[p]+'GG'+self.reference[p+1:p+3], self.reference[p]])
                for p in (350, 800, 100)]
        bams = {'S': [(MemoryBam(self.reads), {}, 'library')], 'EMPTY': [(MemoryBam([]), {}, 'empty')]}
        with patch.dict(os.environ, VARDEPTH_PAIRHMM_BACKEND='cpu'):
            expected = [score(row, bams, self.fetch) for row in rows]
        scorer = accelerator.score_pairs if gpu else accelerator.cpu_pairs
        for limit in (1, 32768):
            with patch.dict(os.environ, VARDEPTH_PAIRHMM_BACKEND='cuda'), patch.object(accelerator, 'score_pairs', side_effect=scorer):
                actual = score_rows(rows, bams, lambda chrom:self.fetch, max_pairs=limit)
            for a, b in zip(actual, expected):
                self.assertEqual(list(a), list(b))
                for sample in a:
                    for key in b[sample]:
                        if key in ('gp','pl'):
                            np.testing.assert_allclose(a[sample][key],b[sample][key],atol=2e-8,rtol=0)
                        else:self.assertEqual(a[sample][key],b[sample][key])

    def test_whole_read_batch_row_sample_and_empty_parity(self):
        self.compare_whole(gpu=False)

    @unittest.skipUnless(os.environ.get('VARDEPTH_TEST_CUDA') == '1', 'CUDA unavailable in this test run')
    def test_cuda_whole_read_batch_parity(self):
        self.compare_whole(gpu=True)

    def compare_partial(self,gpu):
        import accelerator
        from partial_anchor import refine,refine_rows
        from indel_refine import kernel
        class MemoryBam:
            def __init__(self,reads):self.reads=reads
            def fetch(self,chrom,start,end):
                return [r for lib,r in self.reads if r.reference_start<end and r.reference_end>start]
        rows=[];original={}
        for p in (350,800,100):
            alleles=[self.reference[p:p+3],self.reference[p]+'GG'+self.reference[p+1:p+3],self.reference[p]]
            row=dict(record_id=str(p),chrom='chr1',start0=p,end0=p+3,alleles=alleles,added_alleles=0,
                     samples={s:dict(dp=10,proposed_sequences={}) for s in ('S','EMPTY','S2')})
            rows.append(row);original[str(p)]=dict(panel_af=[.9,.08,.02])
        bams={s:[(MemoryBam(self.reads if s!='EMPTY' else []),{},'library')] for s in ('S','EMPTY','S2')}
        lib=kernel()
        with patch.dict(os.environ,VARDEPTH_PAIRHMM_BACKEND='cpu'):
            expected=[refine(row,original[row['record_id']],bams,self.fetch,lib) for row in rows]
        scorer=accelerator.score_pairs if gpu else accelerator.cpu_pairs
        for limit in (1,16,32768):
            with patch.dict(os.environ,VARDEPTH_PAIRHMM_BACKEND='cuda'),patch.object(accelerator,'score_pairs',side_effect=scorer):
                actual=refine_rows(rows,original,bams,lambda chrom:self.fetch,lib,max_pairs=limit)
            for (a,ad),(b,bd) in zip(actual,expected):
                self.assertEqual(ad,bd)
                self.assertEqual(a.keys(),b.keys())
                for key in a:
                    if key!='samples':self.assertEqual(a[key],b[key])
                for sample in a['samples']:
                    for key in b['samples'][sample]:
                        if key in ('gp','pl'):
                            np.testing.assert_allclose(a['samples'][sample][key],b['samples'][sample][key],atol=2e-8,rtol=0)
                        else:self.assertEqual(a['samples'][sample][key],b['samples'][sample][key])

    def test_partial_across_rows_samples_and_empty_parity(self):self.compare_partial(False)

    @unittest.skipUnless(os.environ.get('VARDEPTH_TEST_CUDA')=='1','CUDA unavailable in this test run')
    def test_cuda_partial_across_rows_parity(self):self.compare_partial(True)


if __name__ == '__main__': unittest.main()
