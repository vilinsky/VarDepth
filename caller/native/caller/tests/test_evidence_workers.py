import json,multiprocessing,os,random,tempfile,unittest
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
import pysam
import numpy as np
from gpu_evidence_pipeline import run_tasks
from accelerator import cpu_pairs
from evidence_workers import initialize,full_cpu,proposal,graph

class EvidenceWorkerTests(unittest.TestCase):
    def test_spawn_private_handles_preserve_order_and_results(self):
        rng=random.Random(19381);reference=''.join(rng.choice('ACGT') for _ in range(1200))
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,VARDEPTH_PAIRHMM_BACKEND='cpu'):
            d=Path(directory);ref=d/'ref.fa';ref.write_text('>chr1\n'+reference+'\n');pysam.faidx(str(ref))
            bam=d/'reads.bam'
            header=dict(HD=dict(VN='1.6',SO='coordinate'),SQ=[dict(SN='chr1',LN=len(reference))],RG=[dict(ID='r',SM='S',PL='ILLUMINA',LB='lib')])
            with pysam.AlignmentFile(str(bam),'wb',header=header) as stream:
                for p in (200,500):
                    for j in range(12):
                        r=pysam.AlignedSegment();r.query_name=f'{p}-{j}';r.reference_id=0;r.reference_start=p-40
                        sequence=reference[p-40:p+60]
                        if j%2:sequence=sequence[:51]+'GG'+sequence[51:];r.cigarstring='51M2I49M'
                        else:r.cigarstring='100M'
                        r.query_sequence=sequence;r.query_qualities=[35]*len(sequence);r.mapping_quality=60;r.set_tag('RG','r');stream.write(r)
            pysam.index(str(bam));arguments=['S='+str(bam)];tasks=[]
            for p in (500,800,200):
                alleles=[reference[p:p+31],reference[p:p+11]+'GG'+reference[p+11:p+31]]
                row=dict(record_id=str(p),chrom='chr1',start0=p,end0=p+31,alleles=alleles,added_alleles=0,samples={'S':dict(dp=12,proposed_sequences={})})
                original=dict(row,panel_af=[.8,.2]);tasks.append((row,original,{'S':30 if p!=800 else 0}))
            # Single worker and multiple workers use independently opened inputs.
            with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(arguments,str(ref))) as pool:
                expected=list(pool.map(full_cpu,tasks))
            with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(arguments,str(ref))) as pool:
                actual=list(pool.map(full_cpu,tasks,chunksize=1))
                proposals=list(pool.map(proposal,[(r,o) for r,o,c in tasks]))
                graphs=list(pool.map(graph,[(r,c) for r,o,c in tasks]))
            self.assertEqual(json.dumps(actual,sort_keys=True),json.dumps(expected,sort_keys=True))
            self.assertEqual([x[0]['record_id'] for x in actual],['500','800','200'])
            self.assertEqual(len(proposals),3);self.assertEqual(len(graphs),3)
            streamed=run_tasks(tasks,arguments,str(ref),workers=2,slots_per_worker=2,max_pairs=64,scorer=cpu_pairs)
            def close(a,b):
                if isinstance(a,dict):
                    self.assertEqual(set(a),set(b))
                    for key in a:close(a[key],b[key])
                elif isinstance(a,(list,tuple)):
                    self.assertEqual(len(a),len(b))
                    for x,y in zip(a,b):close(x,y)
                elif isinstance(a,float):np.testing.assert_allclose(a,b,atol=1e-9,rtol=0)
                else:self.assertEqual(a,b)
            close(streamed,expected)
            self.assertIsNone(actual[1][2]);self.assertIsNotNone(actual[0][2])

if __name__=='__main__':unittest.main()
