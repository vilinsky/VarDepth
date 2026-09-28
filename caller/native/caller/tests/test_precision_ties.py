"""A public repeat exposed a discontinuous PL argmin in the fusion model.

The fixture contains reads/reference, never a truth genotype. The production
guard depends only on numerical margins, not this coordinate or sample ID.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pysam
from accelerator import cpu_pairs, score_pairs
from indel_refine import kernel
from partial_anchor import _refine_steps


class PrecisionTieTests(unittest.TestCase):
    def setUp(self):
        self.data=json.loads((Path(__file__).parent/'fixtures/partial-tie-public.json').read_text())
        self.temp=tempfile.TemporaryDirectory()
        path=Path(self.temp.name)/'public.bam'
        header=pysam.AlignmentHeader.from_dict(self.data['header'])
        with pysam.AlignmentFile(path,'wb',header=header) as out:
            for line in self.data['reads_sam']:
                out.write(pysam.AlignedSegment.fromstring(line,header))
        pysam.index(str(path))
        self.bam=pysam.AlignmentFile(path)
        rg={r['ID']:r.get('LB',r['ID']) for r in self.data['header'].get('RG',[])}
        self.inputs={'HG001':[(self.bam,rg,str(path))]}

    def tearDown(self):
        self.bam.close()
        self.temp.cleanup()

    def fetch(self,a,b):
        offset=self.data['reference_start']
        self.assertTrue(offset<=a<=b<=offset+len(self.data['reference']))
        return self.data['reference'][a-offset:b-offset]

    def run_refinement(self,backend,scorer=None,canonical=False):
        with patch.dict(os.environ,VARDEPTH_PAIRHMM_BACKEND=backend):
            g=_refine_steps(self.data['row'],self.data['original'],self.inputs,self.fetch,kernel(),force_cpu=canonical)
            try:
                jobs=next(g)
                self.assertIsNotNone(scorer)
                while True:jobs=g.send(scorer(jobs))
            except StopIteration as done:return done.value

    def check_guard(self,scorer):
        expected,details=self.run_refinement('cpu',canonical=True)
        actual,audit=self.run_refinement('cuda',scorer)
        self.assertEqual(expected,actual)
        guard=audit['HG001'].pop('numerical_guard')
        self.assertEqual(details,audit)
        self.assertEqual(guard['backend'],'cpu_float64')
        self.assertLessEqual(min(guard['observed_gaps']),guard['log_likelihood_margin'])
        # Sample and allele expansion semantics are preserved by the fallback.
        self.assertEqual(set(actual['samples']),{'HG001'})
        self.assertEqual(actual['samples']['HG001']['dp'],expected['samples']['HG001']['dp'])

    def test_cpu_scored_tie_uses_canonical_cpu(self):
        self.check_guard(lambda jobs:cpu_pairs(jobs,anchored=True))

    def test_batched_cpu_tie_uses_canonical_cpu(self):
        expected,details=self.run_refinement('cpu',canonical=True)
        actual,audit=self.run_refinement('cpu')
        self.assertEqual(expected,actual)
        guard=audit['HG001'].pop('numerical_guard')
        self.assertEqual(details,audit)
        self.assertEqual(guard['backend'],'cpu_float64')

    def test_float32_perturbed_tie_uses_canonical_cpu(self):
        self.check_guard(lambda jobs:cpu_pairs(jobs,anchored=True).astype(np.float32).astype(np.float64))

    @unittest.skipUnless(os.environ.get('VARDEPTH_TEST_CUDA')=='1','CUDA unavailable')
    def test_real_cuda_tie_uses_canonical_cpu(self):
        self.check_guard(lambda jobs:score_pairs(jobs,anchored=True))


if __name__=='__main__':unittest.main()
