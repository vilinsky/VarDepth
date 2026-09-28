import sys, unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'caller'))
from high_snv_model import read_likelihood_guard

class ReadGuardTests(unittest.TestCase):
    def test_read_conflict_missing_and_clipped_tie_restore_full_parent(self):
        parent = torch.tensor([[.1, .6, .3]]).repeat(4, 1)
        proposal = torch.tensor([[.1, .3, .6]]).repeat(4, 1)
        features = torch.zeros(4, 3, 20)
        features[:, :, 18] = 1
        features[:, :, 19] = torch.tensor([[1., 0., 1.], [1., 2., 0.],
                                           [1., 2., 0.], [1., 10., 10.]])
        features[2, 2, 18] = 0
        actual = read_likelihood_guard(parent, proposal, features)
        self.assertTrue(torch.equal(actual[[0, 2, 3]], parent[[0, 2, 3]]))
        self.assertTrue(torch.equal(actual[1], proposal[1]))

    def test_unchanged_map_preserves_proposed_probabilities(self):
        parent = torch.tensor([[.1, .6, .3]])
        proposal = torch.tensor([[.2, .7, .1]])
        actual = read_likelihood_guard(parent, proposal, torch.zeros(1, 3, 20))
        self.assertTrue(torch.equal(actual, proposal))

if __name__ == '__main__': unittest.main()
