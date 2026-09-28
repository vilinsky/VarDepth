"""Read-only summaries of actual diploid block-HMM posterior states."""
import numpy as np

NAMES = ['joint_entropy', 'donor_entropy', 'joint_ess_fraction',
         'donor_ess_fraction', 'joint_max', 'donor_max', 'diagonal_mass',
         'adjacent_donor_total_variation', 'log_state_count', 'posterior_known']


def summarize(state_posterior, states):
    original = np.asarray(state_posterior, dtype=np.float64)
    if states < 1 or original.ndim != 2 or original.shape[1] != states**2:
        raise ValueError('Invalid diploid posterior dimensions')
    if not np.isfinite(original).all() or (original < 0).any():
        raise ValueError('Invalid posterior mass')
    totals = original.sum(1, keepdims=True)
    if not np.allclose(totals, 1., atol=1e-8, rtol=1e-6):
        raise ValueError('Posterior mass is not normalized')
    # Never change an inference buffer through a view.
    p = original / totals
    matrix = p.reshape(-1, states, states)
    donor = (matrix.sum(1) + matrix.sum(2)) * .5
    joint_entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(1) / np.log(max(2, states**2))
    donor_entropy = -(donor * np.log(np.maximum(donor, 1e-300))).sum(1) / np.log(max(2, states))
    change = np.zeros(len(p))
    change[1:] = .5 * np.abs(np.diff(donor, axis=0)).sum(1)
    return np.column_stack([joint_entropy, donor_entropy,
        1 / ((p*p).sum(1) * states**2), 1 / ((donor*donor).sum(1) * states),
        p.max(1), donor.max(1), np.trace(matrix, axis1=1, axis2=2), change,
        np.full(len(p), np.log1p(states)/np.log(65)), np.ones(len(p))]).astype(np.float32)
