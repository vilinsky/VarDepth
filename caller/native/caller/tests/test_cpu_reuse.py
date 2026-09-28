"""Numerical regression tests for sharing work, without pruning evidence."""
import unittest
import numpy as np
import pysam
from cpu_batch import _uncached_pairs
from trie_vector import score
from mixture_cpu import two
from local_alleles import _likelihood_matrix
from graph_assemble import graph,_graph_python
from graph_extend import quality_segments
from features import alignment_metadata

class CpuReuse(unittest.TestCase):
 def test_prefix_kernel_preserves_all_paths(self):
  rng=np.random.default_rng(441901);jobs=[]
  for n in [1,3,30,151]:
   s=''.join(rng.choice(list('ACGT'),n));q=rng.integers(0,61,n).tolist()
   for h in [s,s[:n//2]+'T'+s[n//2:],s[:max(1,n-2)],'C'*n]:jobs.append((s,q,h))
  jobs.extend([('',[],'A'),('A',[30],''),('G'*600,[0]*600,'G'*601)])
  for mode in range(3):
   np.testing.assert_allclose(score(jobs,mode,_uncached_pairs),_uncached_pairs(jobs,mode,0),atol=1e-9,rtol=1e-12)
 def test_mixture_extreme_and_duplicate_rows(self):
  rng=np.random.default_rng(9923)
  for n in [0,1,8,100]:
   for prior in [0.,.01,.3]:
    x=rng.choice([1.,.01,1e-200],(n,2))
    np.testing.assert_allclose(two(x,prior),_likelihood_matrix(x,prior),atol=1e-9,rtol=1e-12)
 def test_kmer_molecule_count_and_order(self):
  rng=np.random.default_rng(18001)
  for k in [3,31,41,51,63]:
   ref=''.join(rng.choice(list('ACGT'),160));obs=[(j//2,ref[:70]+('A' if j%3 else 'N')+ref[71:]) for j in range(20)]
   self.assertEqual(graph(ref,obs,k),_graph_python(ref,obs,k))
 def test_quality_runs_match_basewise_filter(self):
  rng=np.random.default_rng(7124)
  for _ in range(100):
   s=''.join(rng.choice(list('ACGTN'),150));q=rng.integers(0,61,150);expected=[];begin=0;k=7
   for i in range(len(s)+1):
    if i==len(s) or q[i]<20 or s[i] not in 'ACGT':
     if i-begin>=k:expected.append(s[begin:i])
     begin=i+1
   self.assertEqual(quality_segments(s,q.astype(np.uint8).tobytes(),k),expected)
 def test_scoped_alignment_metadata(self):
  r=pysam.AlignedSegment();r.query_sequence='ACGT'*40;r.cigarstring='4S150M6S';r.query_qualities=[31,40,22,17]*40;r.mapping_quality=51;r.set_tag('NM',4)
  cache={}
  for a,b,f in [(0,None,1.),(11,12,1.),(13,55,.4),(4,4,0.)]:
   np.testing.assert_array_equal(alignment_metadata(r,a,b,f),alignment_metadata(r,a,b,f,cache=cache))
if __name__=='__main__':unittest.main()
