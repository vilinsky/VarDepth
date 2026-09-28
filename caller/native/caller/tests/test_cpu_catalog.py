import unittest,json,ctypes,sys,platform
import numpy as np,pysam
from row_patterns import unique_rows
from local_alleles import make_blocks
from alignment_queries import window_bounds

class CatalogReuse(unittest.TestCase):
 def test_retained_alignment_keeps_raw_order_and_observed_mapping_quality(self):
  from caller import likelihoods
  from read_index import ReadIndex
  reads=[]
  for start in (115,110):
   read=pysam.AlignedSegment();read.query_name='same-molecule';read.reference_start=start;read.cigartuples=[(0,30)]
   read.query_sequence='A'*30;read.query_qualities=[30]*30;read.mapping_quality=60;reads.append(('lib',read))
  ranks={id(r):i for i,(_,r) in enumerate(reads)}
  for mq in (60,50):
   reads[0][1].mapping_quality=mq;retained={}
   result=likelihoods([(120,'A','G')],ReadIndex(list(reversed(reads))),lambda a,b:'A'*(b-a),20,None,alignments=retained,alignment_ranks=ranks)
   observation=result[4][0][('lib','same-molecule')]
   candidates=[r for _,r in reads if (r.query_sequence[120-r.reference_start],30,r.mapping_quality)==observation]
   self.assertIs(retained[0][('lib','same-molecule')][0],candidates[0])
 @unittest.skipUnless(sys.platform=='linux' and platform.machine() in ('x86_64','AMD64'),'Measured canonical SIMD platform')
 def test_canonical_batch_preserves_scalar_anchored_values(self):
  from cpu_batch import pairs
  from indel_refine import kernel
  rng=np.random.default_rng(7713);jobs=[];expected=[];lib=kernel()
  for i in range(31):
   n=int(rng.integers(12,140));s=''.join(rng.choice(list('ACGT'),n));h=s[:n//2]+'TA'+s[n//2:]
   q=rng.integers(0,61,n).tolist();jobs.append((s,q,h))
   expected.append(lib.anchored_pairhmm(s.encode(),(ctypes.c_ubyte*n)(*q),n,h.encode(),len(h),.001,.1))
  for repeat in range(2):np.testing.assert_array_equal(pairs(jobs,mode=1,canonical=True),expected)
 def test_binary_pattern_order_and_inverse(self):
  rng=np.random.default_rng(70819)
  for width in (1,2,7,8,9,16,17,63,64,65):
   for dtype in (np.uint8,np.float64):
    x=rng.integers(0,2,(93,width)).astype(dtype);x[4:10]=x[0]
    for got,expected in zip(unique_rows(x,return_counts=True,return_inverse=True),np.unique(x,axis=0,return_counts=True,return_inverse=True)):
     np.testing.assert_array_equal(got,expected)
 def test_catalog_cache_keeps_ld_promotion_semantics(self):
  reference='ACGT'*100;fetch=lambda a,b:reference[a:b]
  variants=[(10,'G','GT'),(12,'A','C'),(14,'G','T')]
  panel={v:(np.array([0,0,0,0,0,1,1,1],np.uint8),.375) for v in variants}
  args=dict(panel=panel,support={v:2 for v in variants},max_alleles=3,groups_override=[[0,1,2]])
  cache={};make_blocks(variants,fetch,catalog_cache=cache,**args)
  self.assertTrue(cache)
  normalize=lambda x:json.dumps(x,sort_keys=True,default=lambda v:v.tolist())
  for proposed in ([],[(1,1,1)],[(0,1,1),(1,0,1)],[(1,0,0)]):
   options=dict(args,proposal_vectors={(0,1,2):proposed})
   expected=make_blocks(variants,fetch,**options)
   actual=make_blocks(variants,fetch,catalog_cache=cache,**options)
   self.assertEqual(normalize(expected),normalize(actual))
 def test_window_bounds_match_htslib_with_gaps_and_clips(self):
  rng=np.random.default_rng(4189)
  for trial in range(100):
   cigar=[(int(rng.choice([0,1,2,3,4,5,7,8])),int(rng.integers(1,15))) for _ in range(9)]
   n=sum(n for op,n in cigar if op in (0,1,4,7,8))
   if not n:continue
   read=pysam.AlignedSegment();read.reference_start=100;read.cigartuples=cigar;read.query_sequence='A'*n
   for start in (80,100,113,149,200):
    end=start+17;expected=[(q,p) for q,p in read.get_aligned_pairs(matches_only=True) if start<=p<end]
    self.assertEqual(window_bounds(read,start,end),(expected[0][0],expected[-1][0]+1) if expected else None)
