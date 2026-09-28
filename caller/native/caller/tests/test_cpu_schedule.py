import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cpu_schedule import allocate

class CpuSchedule(unittest.TestCase):
    def test_budget_invariants(self):
        for budget in [1,2,4,8,16,32,64]:
            for chunks in [1,2,7,30]:
                for components in [1,4]:
                    p=allocate(budget,chunks,components)
                    self.assertLessEqual(p['peak_evidence_workers'],budget)
                    self.assertLessEqual(p['jobs'],chunks)
                    self.assertLessEqual(p['component_jobs'],components)
                    self.assertGreaterEqual(p['evidence_jobs'],1)
    def test_explicit_overcommit_rejected(self):
        with self.assertRaises(ValueError):allocate(4,3,4,jobs=2,component_jobs=2,evidence_jobs=2)
    def test_explicit_control_preserved(self):
        p=allocate(8,1,1,1,1,4)
        self.assertEqual((p['jobs'],p['component_jobs'],p['evidence_jobs']),(1,1,4))
    def test_independent_work_uses_each_available_core(self):
        p=allocate(8,30,4)
        self.assertEqual((p['jobs'],p['component_jobs'],p['evidence_jobs']),(8,1,1))
    def test_neural_threads_are_in_the_total_budget(self):
        p=allocate(8,30,4,neural_threads=4)
        self.assertLessEqual(p['peak_neural_threads'],8)
        with self.assertRaises(ValueError):allocate(8,30,4,jobs=8,neural_threads=4)
if __name__=='__main__':unittest.main()
