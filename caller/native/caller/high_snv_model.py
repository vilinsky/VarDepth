"""Bounded read-dominant SNP correction over a frozen dependent-expert model."""
import math
import torch
from torch import nn
from trust import residual_model_class

def eligible(x, n):
    return ((x[:, 0, 2] < .5) & (n == 3)
            & ((x[:, 0, 3] - math.log(2)/math.log(101)).abs() < 1e-6)
            & (x[:, 0, 0] >= math.log1p(19.5)/math.log(101)))

def inputs(x, probabilities, available, genotype_features, parent):
    """No positions, nominal coverage, individual IDs or truth-derived features."""
    batch = len(x)
    source = probabilities[:, :, :3].transpose(1, 2)
    return torch.cat([x[:, 0, :14, None].transpose(1, 2).expand(-1, 3, -1),
        source, source.clamp_min(1e-12).log()/30,
        available[:, None, :].expand(-1, 3, -1).float(), genotype_features[:, :3],
        parent[:, :3, None].clamp_min(1e-12).log()/30,
        torch.tensor([1., 0., 0.], device=x.device)[None, :, None].expand(batch, -1, -1)], dim=-1)

def model_class():
    class HighSnvResidual(nn.Module):
        def __init__(self, inputs=42, hidden=48, architecture='high_snv_residual'):
            super().__init__()
            if architecture not in ('high_snv_residual', 'high_snv_read_guard'): raise ValueError('Unexpected architecture')
            self.read_guard = architecture == 'high_snv_read_guard'
            self.base = residual_model_class()(inputs=inputs, hidden=hidden)
            for p in self.base.parameters(): p.requires_grad_(False)
            self.correction = nn.Sequential(nn.Linear(66, 32), nn.Tanh(), nn.Dropout(.05),
                                            nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
            nn.init.zeros_(self.correction[-1].weight); nn.init.zeros_(self.correction[-1].bias)

        def forward(self, x, probabilities, available, n_genotypes, genotype_features):
            self.base.eval()
            with torch.no_grad(): parent, weights = self.base(x, probabilities, available, n_genotypes, genotype_features)
            active = eligible(x, n_genotypes)
            if not active.any(): return parent, weights
            features = inputs(x[active], probabilities[active], available[active], genotype_features[active], parent[active])
            logp = parent[active, :3].clamp_min(1e-12).log() + 2*torch.tanh(self.correction(features).squeeze(-1))
            proposed = logp.softmax(-1)
            if self.read_guard:
                proposed = read_likelihood_guard(parent[active, :3], proposed, genotype_features[active, :3])
            posterior = parent.clone(); posterior[active, :3] = proposed
            return posterior, weights
    return HighSnvResidual


def read_likelihood_guard(parent, proposed, genotype_features):
    """Restore the complete parent posterior for unsupported genotype changes.

    Feature 18 marks available read PL; feature 19 is min(10, PL/60).
    Strict improvement avoids accepting ambiguous ties introduced by clipping.
    Unchanged MAP genotypes retain the proposed posterior. This is a conservative
    deployment gate, not a claim of calibrated probabilities or independent data.
    """
    old = parent.argmax(-1, keepdim=True)
    new = proposed.argmax(-1, keepdim=True)
    available = genotype_features[:, :, 18]
    pl = genotype_features[:, :, 19]
    supported = ((available.gather(1, old) > .5)
                 & (available.gather(1, new) > .5)
                 & (pl.gather(1, new) < pl.gather(1, old)))
    keep = (old == new) | supported
    return torch.where(keep, proposed, parent)
