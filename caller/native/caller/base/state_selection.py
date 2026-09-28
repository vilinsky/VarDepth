"""Independent, bounded diploid-aware donor selection for a copying HMM.

The selection score uses independent read GL as a proposal only. Final inference
must recompute the joint model once, without multiplying these scores into GP.
This is a deterministic approximation, not PBWT or a full posterior sampler.
"""
import numpy as np


def marginal_scores(haplotypes, log_gl):
    """Rank a donor after integrating the other chromosome with a flat prior."""
    gl = np.exp(log_gl - np.max(log_gl, axis=1, keepdims=True))
    informative = np.ptp(gl, axis=1) > 1e-12
    h = haplotypes[informative]
    gl = gl[informative]
    if not len(gl):
        return np.zeros(haplotypes.shape[1])
    ref = np.maximum(1e-300, (gl[:, 0] + gl[:, 1]) / 2)
    alt = np.maximum(1e-300, (gl[:, 1] + gl[:, 2]) / 2)
    return (np.log(alt) - np.log(ref)) @ h


def partner_scores(haplotypes, log_gl, anchor, mutation=1e-3, chunk=512):
    """Log likelihood of every constant donor pair containing anchor.

    Missing/unphased alleles represented as probabilities are integrated. The
    mutation floor only regularizes proposal selection and is not final GL.
    """
    gl = np.exp(log_gl - np.max(log_gl, axis=1, keepdims=True))
    informative = np.ptp(gl, axis=1) > 1e-12
    h = haplotypes[informative]
    gl = gl[informative]
    if not len(gl):
        return np.zeros(haplotypes.shape[1])
    a = mutation + (1 - 2 * mutation) * h[:, anchor]
    # P(R | anchor, partner=0/1), integrating uncertain anchor alleles.
    ref = gl[:, 0] * (1 - a) + gl[:, 1] * a
    alt = gl[:, 1] * (1 - a) + gl[:, 2] * a
    from selection_cpu import scores as binary_scores
    exact=binary_scores(h,ref,alt,mutation)
    if exact is not None:return exact
    scores = np.empty(h.shape[1])
    for begin in range(0, h.shape[1], chunk):
        b = mutation + (1 - 2 * mutation) * h[:, begin:begin + chunk]
        likelihood = ref[:, None] * (1 - b) + alt[:, None] * b
        scores[begin:begin + chunk] = np.log(np.maximum(likelihood, 1e-300)).sum(axis=0)
    return scores


def select_haplotypes(haplotypes, log_gl, count, positions=None, mode='paired', windows=4):
    """Choose complementary pairs globally and in fixed physical windows.

    Half of the budget remains globally marginal-ranked. The other half recruits
    best partners for deterministic marginal seeds across equal-size windows.
    At least one complete paired proposal is retained when count >= 2.
    No truth data or cohort target genotypes enter this function.
    """
    h = np.asarray(haplotypes, dtype=float)
    gl = np.asarray(log_gl, dtype=float)
    if h.ndim != 2 or gl.shape != (h.shape[0], 3) or not np.isfinite(h).all() or not np.isfinite(gl).all():
        raise ValueError('Expected finite marker-by-donor probabilities and marker-by-3 log GL')
    if h.size and (h.min() < 0 or h.max() > 1):
        raise ValueError('Donor probabilities must be in [0,1]')
    if count < 1 or windows < 1 or mode not in ('marginal', 'paired'):
        raise ValueError('Invalid state selection configuration')
    count = min(int(count), h.shape[1])
    if not count:
        return np.empty(0, dtype=int)
    # Every physical window uses a contiguous subset of these same informative
    # markers. Select/copy the large panel once, retaining the original float64
    # BLAS reduction for marginal ranking and a byte view for pair proposals.
    normalized=np.exp(gl-np.max(gl,axis=1,keepdims=True))
    informative=np.ptp(normalized,axis=1)>1e-12
    informative_h=h[informative];informative_gl=normalized[informative]
    marginal_terms=np.log(np.maximum(1e-300,(informative_gl[:,1]+informative_gl[:,2])/2))-np.log(np.maximum(1e-300,(informative_gl[:,0]+informative_gl[:,1])/2))
    binary=np.all((informative_h==0)|(informative_h==1))
    packed_all=np.ascontiguousarray(informative_h,dtype=np.uint8) if binary else None
    ranking = np.argsort(-(marginal_terms @ informative_h), kind='stable')
    if mode == 'marginal' or count == 1 or count == h.shape[1] or not h.shape[0]:
        return ranking[:count]
    masks = [np.ones(h.shape[0], dtype=bool)]
    if positions is not None and windows > 1:
        positions = np.asarray(positions)
        if positions.shape != (h.shape[0],) or np.any(np.diff(positions) < 0):
            raise ValueError('Positions must be sorted and match marker count')
        boundaries = np.linspace(positions[0], positions[-1] + 1, windows + 1)
        masks.extend((positions >= boundaries[i]) & (positions < boundaries[i+1]) for i in range(windows))
    proposals = []
    for mask in masks:
        if not mask.any() or not np.any(np.ptp(gl[mask], axis=1) > 1e-12):
            continue
        selected=np.flatnonzero(mask[informative])
        span=slice(int(selected[0]),int(selected[-1])+1) if len(selected) else slice(0,0)
        selected_h=informative_h[span];selected_gl=informative_gl[span]
        local_ranking=ranking if mask.all() else np.argsort(-(marginal_terms[span] @ selected_h),kind='stable')
        packed=packed_all[span] if binary else None
        # Two seeds reduce dependence on a single arbitrary tied best donor.
        local = []
        for anchor in local_ranking[:2]:
            if binary:
                from selection_cpu import scores_u8
                a=.001+(1-2*.001)*selected_h[:,int(anchor)]
                ref=selected_gl[:,0]*(1-a)+selected_gl[:,1]*a
                alt=selected_gl[:,1]*(1-a)+selected_gl[:,2]*a
                score=scores_u8(packed,ref,alt,.001)
            else:score=partner_scores(h[mask],gl[mask],int(anchor))
            partners = np.argsort(-score, kind='stable')
            # A homozygous best pair is valid; use further partners for diversity.
            local.append((int(anchor), [int(x) for x in partners[:count]]))
        proposals.append(local)
    chosen = []
    chosen_set = set()
    def add(index):
        if len(chosen) < count and index not in chosen_set:
            chosen.append(index)
            chosen_set.add(index)
    if proposals:
        anchor, partners = proposals[0][0]
        add(anchor)
        add(partners[0])
    for index in ranking:
        if len(chosen) >= max(2, count // 2):
            break
        add(int(index))
    for rank in range(count):
        for local in proposals:
            for anchor, partners in local:
                add(anchor)
                if rank < len(partners):
                    add(partners[rank])
        if len(chosen) == count:
            break
    for index in ranking:
        if len(chosen) == count:
            break
        add(int(index))
    return np.asarray(chosen, dtype=int)
