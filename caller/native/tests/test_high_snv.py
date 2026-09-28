import math, sys, unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'caller'))
from high_snv_model import model_class, eligible, inputs

class HighSnpTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(731); torch.set_num_threads(1)
        self.model = model_class()().eval()
        self.x = torch.rand(5, 10, 42); self.x[:, :, 2] = 0
        self.x[:, :, 3] = math.log(2)/math.log(101)
        self.x[:, :, 0] = torch.tensor([0, math.log(2)/math.log(101), 1, 0, 0])[:, None]
        self.x[3, :, 2] = 1; self.x[4, :, 3] = math.log(3)/math.log(101)
        self.p = torch.softmax(torch.randn(5, 10, 3), -1)
        self.a = torch.ones(5, 10, dtype=torch.bool); self.n = torch.full((5,), 3)
        self.gf = torch.rand(5, 3, 20)

    def test_scope_and_posterior(self):
        self.assertEqual(eligible(self.x, self.n).tolist(), [False, False, True, False, False])
        with torch.no_grad():
            old, _ = self.model.base(self.x, self.p, self.a, self.n, self.gf)
            p, _ = self.model(self.x, self.p, self.a, self.n, self.gf)
        torch.testing.assert_close(p, old, rtol=2e-6, atol=1e-7)
        self.assertTrue(torch.equal(old[[0,1,3,4]], p[[0,1,3,4]]))

    def test_trained_correction_keeps_other_regimes_bitwise(self):
        with torch.no_grad(): self.model.correction[-1].weight.normal_()
        with torch.no_grad():
            old, _ = self.model.base(self.x, self.p, self.a, self.n, self.gf)
            p, _ = self.model(self.x, self.p, self.a, self.n, self.gf)
        self.assertTrue(torch.equal(old[[0,1,3,4]], p[[0,1,3,4]]))
        torch.testing.assert_close(p.sum(1), torch.ones(5))
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue((p >= 0).all())

if __name__ == '__main__': unittest.main()
