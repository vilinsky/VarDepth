import multiprocessing as mp
import unittest
import numpy as np
from gpu_evidence_pipeline import run_tasks


def requests(task):
    values=yield bool(task%2),[task]*2
    initial=sum(values)
    values=yield not bool(task%2),[task+1]*(task%5+1)
    return task,initial+sum(values)


def failing(task):
    if task==3:raise ValueError('deliberate worker failure')
    yield False,[task]
    return task


def scoring(jobs,anchored):return np.asarray(jobs,dtype=float)*(2 if anchored else 3)


class ActorProtocolTests(unittest.TestCase):
    def test_order_backpressure_modes_and_followup_requests(self):
        tasks=list(range(20,-1,-1));seen=[]
        def score(jobs,anchored):seen.append((len(jobs),anchored));return scoring(jobs,anchored)
        actual=run_tasks(tasks,None,None,workers=3,slots_per_worker=3,max_pairs=4,scorer=score,factory=requests)
        expected=[(x,2*x*(2 if x%2 else 3)+(x%5+1)*(x+1)*(3 if x%2 else 2)) for x in tasks]
        self.assertEqual(actual,expected);self.assertEqual({a for n,a in seen},{True,False})
        self.assertTrue(all(n<=5 for n,a in seen))
    def test_worker_error_cleans_children(self):
        before={p.pid for p in mp.active_children()}
        with self.assertRaisesRegex(RuntimeError,'deliberate worker failure'):
            run_tasks(range(12),None,None,workers=2,slots_per_worker=2,scorer=scoring,factory=failing)
        self.assertEqual({p.pid for p in mp.active_children()},before)
    def test_bad_scorer_cleans_children(self):
        before={p.pid for p in mp.active_children()}
        with self.assertRaisesRegex(ValueError,'wrong shape'):
            run_tasks(range(12),None,None,workers=2,slots_per_worker=2,scorer=lambda jobs,a:np.zeros(len(jobs)+1),factory=requests)
        self.assertEqual({p.pid for p in mp.active_children()},before)
    def test_bounds_and_empty(self):
        self.assertEqual(run_tasks([],None,None,scorer=scoring,factory=requests),[])
        for kw in [dict(workers=0),dict(slots_per_worker=0),dict(max_pairs=0)]:
            with self.assertRaises(ValueError):run_tasks([],None,None,**kw)

if __name__=='__main__':unittest.main()
