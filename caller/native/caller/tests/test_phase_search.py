import unittest
import numpy as np
from quantum_phase import PhaseProblem,Observation,exact
from phase_search import eliminate,eliminate_unique,anneal,factors

class PhaseSearchTests(unittest.TestCase):
    def test_exact_factor_elimination_full_objective(self):
        rng=np.random.default_rng(1707)
        for n in (2,5,9):
            fragments=[]
            for j in range(n*3):
                sites=rng.choice(n,min(n,4),replace=False)
                fragments.append([Observation(int(i),int(rng.integers(0,2)),float(rng.uniform(.01,.4))) for i in sites])
            p=PhaseProblem(list(range(n)),fragments);a=eliminate(p);b=exact(p)
            self.assertTrue(a['solved']);self.assertAlmostEqual(a['nll'],b['nll'],places=10)
            self.assertEqual(a['bits'][0],0)
            unique=eliminate_unique(p)
            self.assertTrue(unique['uniqueness_certified'])
            self.assertAlmostEqual(unique['nll'],b['nll'],places=10)
            self.assertAlmostEqual(unique['next_solution_gap'],b['next_solution_gap'],places=10)
            self.assertEqual(unique['exact_solves'],n)
            sa=anneal(p,restarts=4,sweeps=50)
            self.assertGreaterEqual(sa['nll']+1e-9,b['nll'])
            self.assertAlmostEqual(sa['nll'],p.full_nll(sa['bits']),places=10)

    def test_local_chain_scales_and_width_guard(self):
        p=PhaseProblem(list(range(100)),[[Observation(i,0,.1),Observation(i+1,i%2,.1)] for i in range(99)])
        a=eliminate(p,max_width=2)
        self.assertTrue(a['solved']);self.assertLessEqual(a['max_width'],2)
        dense=PhaseProblem(list(range(6)),[[Observation(i,i%2,.1) for i in range(6)]])
        self.assertFalse(eliminate(dense,max_width=2)['solved'])

    def test_ties_constraints_and_bounded_runner_up(self):
        p=PhaseProblem(['a','b','c'],[[Observation(0,0,.1),Observation(1,0,.1)]])
        result=eliminate_unique(p)
        self.assertTrue(result['solved']);self.assertAlmostEqual(result['next_solution_gap'],0.)
        fixed=eliminate(p,fixed={1:1,2:1})
        self.assertEqual(fixed['bits'],[0,1,1]);self.assertAlmostEqual(fixed['nll'],p.full_nll([0,1,1]))
        for constraint in ({0:1},{3:0},{1:2}):
            with self.assertRaises(ValueError):eliminate(p,fixed=constraint)
        dense=PhaseProblem(list(range(6)),[[Observation(i,i%2,.1) for i in range(6)]])
        self.assertFalse(eliminate_unique(dense,max_width=2)['solved'])

if __name__=='__main__':unittest.main()
