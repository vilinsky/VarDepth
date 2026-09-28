import importlib.util,sys,unittest
from pathlib import Path
import numpy as np
import pysam
HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE/'base')]
from alignment_queries import target_queries,point_query,window_pairs,outer_safe_anchors

class CigarQueries(unittest.TestCase):
    def test_random_coordinate_equivalence(self):
        rng=np.random.default_rng(8870)
        for i in range(500):
            read=pysam.AlignedSegment();read.reference_start=1000
            ops=[(4,3)]+[(int(rng.choice([0,0,0,1,2,3,7,8])),int(rng.integers(1,15))) for _ in range(12)]+[(4,2)]
            read.cigartuples=ops;n=sum(n for op,n in ops if op in (0,1,4,7,8))
            read.query_sequence='A'*n;read.query_qualities=[30]*n
            original=read.get_aligned_pairs(matches_only=True)
            selected=sorted(set(rng.integers(980,1160,20).tolist()))
            self.assertEqual(list(target_queries(read,selected)),[(q,p) for q,p in original if p in selected])
            for p in selected:self.assertEqual(point_query(read,p),next((q for q,pos in original if pos==p),None))
            self.assertEqual(window_pairs(read,1010,1060),[(q,p) for q,p in original if 1010<=p<1060])
            safe=sorted(set(rng.integers(0,100,30).tolist()));aligned={p-1000:q for q,p in original}
            possible=[p for p in safe if p in aligned and all(aligned.get(p+j)==aligned[p]+j for j in range(3))]
            result=outer_safe_anchors(read,1000,safe,3)
            expected=(possible[0],possible[-1]+3,aligned[possible[0]],aligned[possible[-1]+2]+1) if possible else (None,)*4
            self.assertEqual(result,expected)
if __name__=='__main__':unittest.main()
