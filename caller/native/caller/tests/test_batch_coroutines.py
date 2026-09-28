import unittest
import numpy as np
from batch_coroutines import score_generators

class CoroutineTests(unittest.TestCase):
    def test_larger_followup_samples_are_rebatched(self):
        calls=[]
        def worker(i):
            a=yield [i]
            b=yield [i]*7
            return (float(a[0]),float(sum(b)))
        def score(jobs):calls.append(len(jobs));return np.asarray(jobs,dtype=float)
        result=score_generators((worker(i) for i in range(20)),score,max_pairs=10)
        self.assertEqual(result,[(float(i),float(i*7)) for i in range(20)])
        self.assertLessEqual(max(calls),10)
    def test_bad_scorer_shape_fails(self):
        def worker():yield [1,2]
        with self.assertRaises(ValueError):score_generators([worker()],lambda jobs:[])

if __name__=='__main__':unittest.main()
