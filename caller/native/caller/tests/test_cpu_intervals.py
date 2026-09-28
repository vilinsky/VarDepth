import tempfile,unittest
from pathlib import Path
import numpy as np,pysam
from bam_cache import RegionReader
from evidence_workers import locality_batches
from mixture_cpu import general
from local_alleles import _likelihood_matrix

class CPUIntervals(unittest.TestCase):
    def test_interval_cache_preserves_order_and_boundary_overlaps(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reads.bam'
            header={'HD':{'VN':'1.6','SO':'coordinate'},'SQ':[{'SN':'chr1','LN':1000}]}
            with pysam.AlignmentFile(path,'wb',header=header) as out:
                for i,start in enumerate([0,8,8,16,31,32,33,60,61,100,101,600]):
                    read=pysam.AlignedSegment(out.header);read.query_name=f'r{i}';read.reference_id=0;read.reference_start=start
                    read.cigartuples=[(0,10),(3,70),(0,10)] if i%3==0 else [(4,3),(0,14),(1,3)]
                    read.query_sequence='A'*20;read.query_qualities=[30]*20;read.mapping_quality=60;out.write(read)
            pysam.index(str(path))
            with pysam.AlignmentFile(path) as raw:
                cached=RegionReader(pysam.AlignmentFile(path),tile_bp=32,capacity=2)
                for start in [0,31,32,61,100,599,9,33,101,32]:
                    for size in [1,10,32,99]:
                        expected=[r.to_string() for r in raw.fetch('chr1',start,start+size)]
                        actual=[r.to_string() for r in cached.fetch('chr1',start,start+size)]
                        self.assertEqual(expected,actual,(start,size))
                cached.close()

    def test_locality_batches_preserve_every_locus(self):
        rows=[dict(record_id=str(i),chrom='chr1',start0=i*997,alleles=['A','AT']) for i in range(113)]
        original={r['record_id']:r for r in rows};coverage={str(i):{'sample':i%31+1} for i in range(113)}
        for workers in [1,4,32,64]:
            jobs=locality_batches(rows,original,coverage,workers)
            flat=[(i,task) for batch in jobs for i,task in batch]
            self.assertEqual(sorted(i for i,task in flat),list(range(len(rows))))
            for i,task in flat:self.assertIs(task[0],rows[i])

    def test_general_mixture_retains_all_reads_and_extreme_likelihoods(self):
        rng=np.random.default_rng(3306)
        for n in [2,3,6,12,24]:
            patterns=np.exp(rng.uniform(-690,0,(9,n)));x=patterns[rng.integers(0,9,131)]
            for prior in [None,0.,.001,.3]:
                expected=_likelihood_matrix(x,prior)
                np.testing.assert_allclose(general(x,prior),expected,rtol=1e-12,atol=1e-9)
                ordered=np.array([expected[a,b] for b in range(n) for a in range(b+1)])
                np.testing.assert_allclose(general(x,prior,True),ordered,rtol=1e-12,atol=1e-9)
        with self.assertRaises(ValueError):general(np.array([[np.nan,1.]]),.01)

if __name__=='__main__':unittest.main()
