"""Spanning-read batching retains per-sample and per-molecule semantics."""
import os,unittest
from unittest.mock import patch
import numpy as np
import indel_refine
from accelerator import cpu_pairs,score_pairs
from batch_coroutines import score_generators

class SpanningBatchTests(unittest.TestCase):
    def compare(self,gpu):
        reference='GCTCGTACGAGTCTACGTGCGATCGACGTAGCTACGATCGT';fetch=lambda a,b:reference[a:b]
        records=[]
        for p in (8,18,28):
            records.append(dict(record_id=str(p),chrom='chr1',start0=p,end0=p+3,evidence_start0=p,evidence_end0=p+3,
                                alleles=[reference[p:p+3],reference[p]+'GG'+reference[p+1:p+3]],panel_af=[.8,.2]))
        def observe(paths,chrom,start,end,fetch,**kw):
            if paths==['empty']:return []
            ref=fetch(start,end);alts=[ref,ref[:1]+'GG'+ref[1:]]
            return [dict(sequence=fetch(start-3,start)+inner+fetch(end,end+3),inner=inner,
                         qualities=[35]*(len(inner)+6),mq=60,reverse=False,proposal_quality=True)
                    for inner in alts for repeat in range(4)]
        bams={'S':['reads'],'E':['empty']};lib=indel_refine.kernel()
        with patch.object(indel_refine,'observations',side_effect=observe),patch.dict(os.environ,VARDEPTH_PAIRHMM_BACKEND='cpu'):
            expected=[indel_refine.refine_record(r,bams,fetch,lib) for r in records]
        scorer=score_pairs if gpu else cpu_pairs
        for limit in (1,32768):
            with patch.object(indel_refine,'observations',side_effect=observe),patch.dict(os.environ,VARDEPTH_PAIRHMM_BACKEND='cuda'):
                actual=score_generators((indel_refine._refine_record_steps(r,bams,fetch,lib) for r in records),lambda jobs:scorer(jobs,anchored=True),limit)
            for a,b in zip(actual,expected):
                for key in b:
                    if key!='samples':self.assertEqual(a[key],b[key])
                for sample in b['samples']:
                    for key in b['samples'][sample]:
                        if key in ('gp','pl'):np.testing.assert_allclose(a['samples'][sample][key],b['samples'][sample][key],atol=2e-8,rtol=0)
                        else:self.assertEqual(a['samples'][sample][key],b['samples'][sample][key])
    def test_grouped_full_objective_parity(self):self.compare(False)
    @unittest.skipUnless(os.environ.get('VARDEPTH_TEST_CUDA')=='1','CUDA unavailable')
    def test_grouped_cuda_parity(self):self.compare(True)

if __name__=='__main__':unittest.main()
