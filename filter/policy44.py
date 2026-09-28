"""Coverage policies; inputs contain no assay, location or truth identity."""
import numpy as np
from features43 import infer


def depth_groups(x, edges):
    # Recover integer VCF depth from log1p float32 features before binning.
    # A depth of 10 must not accidentally become 9.999999 after expm1.
    x = np.asarray(x)
    depth = np.maximum(np.nan_to_num(np.expm1(x[:, 0]), nan=0.),
                       np.nan_to_num(np.expm1(x[:, 1]), nan=0.))
    return np.searchsorted(np.asarray(edges[1:]), np.rint(depth), side='right')


def calibrated(model, x, groups):
    raw = infer(model, x)
    result = np.empty(len(x), dtype=float)
    for group in np.unique(groups):
        mask = groups == group
        c = model['depth_calibrations'][str(int(group))]
        result[mask] = np.interp(raw[mask], c['x'], c['y'])
    return result


def decisions(model, x, edges, skip_unconditional=True):
    """Return accepted and probabilities; NaN means score intentionally not run."""
    group = depth_groups(x, edges)
    cuts = np.asarray(model['depth_thresholds'])[group]
    score_mask = cuts > 0 if skip_unconditional else np.ones(len(x), bool)
    probability = np.full(len(x), np.nan)
    probability[score_mask] = calibrated(model, np.asarray(x)[score_mask], group[score_mask])
    accepted = cuts == 0
    accepted[score_mask] = probability[score_mask] >= cuts[score_mask]
    return accepted, probability, group


def genomic_f1(qtp, ttp, fp, fn):
    p = np.divide(qtp, qtp + fp, out=np.zeros_like(np.asarray(qtp, float)), where=qtp + fp > 0)
    r = np.divide(ttp, ttp + fn, out=np.zeros_like(np.asarray(ttp, float)), where=ttp + fn > 0)
    return np.divide(2*p*r, p+r, out=np.zeros_like(p), where=p+r > 0)
