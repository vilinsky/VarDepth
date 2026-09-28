"""One-pass donor features for a learned neural-caller context gate.

Inputs must be the *initial, external-only* genotype probabilities on one shared
allele catalog. This module never changes those probabilities, assigns donor
truth, multiplies posterior distributions, or consumes previous cohort output.
One row is one biological person: upstream adapters must combine that person's
libraries/read groups before calling this function.

Nearby unphased genotype compatibility is a proxy for local haplotype sharing,
not a phased LD model. Read-only PL/GL are preferred for this compatibility;
external-only GP are the explicitly reported fallback. The coordinate exclusion
zone removes the focal site and close neighbors but cannot prove read-molecule
independence from site summaries. Accordingly, outputs are neural features,
never an independent likelihood or a calibrated posterior.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Sequence

import numpy as np

MAX_ALLELES = 12
MAX_GENOTYPES = MAX_ALLELES * (MAX_ALLELES + 1) // 2
FEATURE_NAMES = (
    "donor_allele_frequency",
    "effective_carrier_support",
    "donor_genotype_certainty",
    "flanking_genotype_compatibility",
    "allele_disagreement",
    "donor_availability",
)
# VCF Number=G: (0,0), (0,1), (1,1), (0,2), ...
GENOTYPE_PAIRS = np.asarray([(a, b) for b in range(MAX_ALLELES)
                             for a in range(b + 1)], dtype=np.int16)
_DOSAGE = np.zeros((MAX_GENOTYPES, MAX_ALLELES), dtype=np.float64)
for _index, (_a, _b) in enumerate(GENOTYPE_PAIRS):
    _DOSAGE[_index, _a] += 0.5
    _DOSAGE[_index, _b] += 0.5
_CARRIER = (_DOSAGE > 0).astype(np.float64)


@dataclass(frozen=True)
class CohortConfig:
    window_bp: int = 20_000
    focal_exclusion_bp: int = 500
    max_flank_records: int = 128
    max_donors: int = 8
    depth_half_saturation: float = 4.0
    min_donor_reliability: float = 1e-4
    incompatibility_penalty: float = 4.0
    flank_weight_half_saturation: float = 2.0

    def validate(self) -> None:
        for name in ("window_bp", "focal_exclusion_bp", "max_flank_records", "max_donors"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.window_bp <= self.focal_exclusion_bp or self.focal_exclusion_bp < 0:
            raise ValueError("window_bp must exceed nonnegative focal_exclusion_bp")
        if self.max_flank_records < 1 or self.max_donors < 1:
            raise ValueError("max_flank_records and max_donors must be positive")
        for name in ("depth_half_saturation", "incompatibility_penalty", "flank_weight_half_saturation"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.min_donor_reliability) or not 0 <= self.min_donor_reliability < 1:
            raise ValueError("min_donor_reliability must be in [0,1)")


@dataclass
class CohortContext:
    allele_context: np.ndarray       # [people, records, 12, 6], float32
    context_strength: np.ndarray     # [people, records], float32
    provenance: dict[str, Any]       # JSON-serializable summary
    donor_indices: np.ndarray        # [people, records, K], original input row indices; -1 unused
    donor_weights: np.ndarray        # same shape, unnormalized reliability/compatibility weights


def empty_cohort_context(n_people: int, n_records: int, *, donor_slots: int = 0,
                         reason: str = "no_eligible_donors") -> CohortContext:
    """The exact zero feature branch also used for automatic single-person runs."""
    if min(n_people, n_records, donor_slots) < 0:
        raise ValueError("Output dimensions must be nonnegative")
    return CohortContext(
        np.zeros((n_people, n_records, MAX_ALLELES, len(FEATURE_NAMES)), dtype=np.float32),
        np.zeros((n_people, n_records), dtype=np.float32),
        {"enabled": False, "reason": reason, "feature_names": list(FEATURE_NAMES)},
        np.full((n_people, n_records, donor_slots), -1, dtype=np.int32),
        np.zeros((n_people, n_records, donor_slots), dtype=np.float32),
    )


def _certainty(probabilities: np.ndarray, genotype_count: np.ndarray) -> np.ndarray:
    log_prob = np.zeros_like(probabilities)
    np.log(probabilities, out=log_prob, where=probabilities > 0)
    entropy = -np.sum(probabilities * log_prob, axis=-1)
    maximum = np.log(np.maximum(genotype_count, 2))
    result = 1.0 - entropy / maximum[None, :]
    result[:, genotype_count == 1] = 1.0
    result = np.clip(result, 0.0, 1.0)
    result[result < 1e-12] = 0.0  # uniform rows carry exactly no information
    return result


def _read_probabilities(values: np.ndarray, valid: np.ndarray, shape: tuple[int, ...],
                        is_phred: bool) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    if value.shape != shape or np.isnan(value).any():
        raise ValueError("read PL/GL must match initial_gp shape and contain no NaNs")
    if is_phred:
        if np.isneginf(value).any() or np.any(value[np.isfinite(value)] < 0):
            raise ValueError("read_pl must be nonnegative Phred values (+inf is allowed)")
        value = value * (-np.log(10.0) / 10.0)
    elif np.isposinf(value).any():
        raise ValueError("read_gl must be natural-log likelihoods without +inf")
    value = np.where(valid[None, :, :], value, -np.inf)
    best = np.max(value, axis=-1, keepdims=True)
    # Missing likelihood rows have no genotype information: uniform, certainty 0.
    value = np.where(np.isfinite(best), value, np.where(valid[None, :, :], 0.0, -np.inf))
    best = np.max(value, axis=-1, keepdims=True)
    probability = np.exp(value - best)
    return probability / probability.sum(axis=-1, keepdims=True)


def build_cohort_context(
    initial_gp: np.ndarray,
    allele_count: np.ndarray,
    positions: np.ndarray,
    sample_ids: Sequence[str],
    *,
    read_depth: np.ndarray,
    site_quality: np.ndarray | None = None,
    contigs: Sequence[str] | None = None,
    read_pl: np.ndarray | None = None,
    read_gl: np.ndarray | None = None,
    config: CohortConfig | None = None,
    gp_source: str = "external_only",
) -> CohortContext:
    """Build target-excluded donor features automatically when there are >=2 people.

    GP uses 78 padded VCF-triangular slots; at an A-allele record only the first
    A*(A+1)/2 slots may carry probability. Rows must sum to one (1e-5 tolerance).
    DP is nonnegative per-person/per-site distinct-fragment depth. Optional
    site_quality is a read-only [0,1] quality factor (e.g. mapping/artifact QC),
    not a truth-derived label or post-cohort confidence. Multiple sequencing
    depths are handled per site through DP/(DP + half_saturation), so an exome
    island cannot set the reliability scale for shallow WGS elsewhere.

    PL is Phred-scaled; GL is natural log, not log10. Pass at most one. No
    ordering of people is privileged: all reductions and score ties use sorted
    sample IDs, then results are restored to input order.

    Six channels, bounded in [0,1], at each valid allele:
      0 weighted donor dosage/2;
      1 expected effective carrier count / (1 + that count), where effective
        donor N=(sum w)^2/sum(w^2), never sequencing depth;
      2 weighted normalized genotype certainty (1 - entropy/log genotype_count);
      3 weighted flanking genotype affinity, 0.5 when no informative flanks;
      4 four times weighted variance of donor allele dosage/2;
      5 eligible donor count / (people - 1), before deterministic top-K.
    Channels 2,3,5 describe the selected donor message and repeat over valid
    alleles. Padding is exactly zero. Focal target GP/DP/quality never contribute
    to its own message; changes to its flanks may change donor compatibility.
    A target singleton allele is retained in the shared catalog with donor AF=0;
    it is never removed or assigned zero genotype probability by this function.
    """
    cfg = config or CohortConfig()
    cfg.validate()
    if gp_source != "external_only":
        raise ValueError("Only initial external_only GP are allowed; no post-cohort recycling")
    if read_pl is not None and read_gl is not None:
        raise ValueError("Provide read_pl or read_gl, not both")
    gp = np.asarray(initial_gp, dtype=np.float64)
    if gp.ndim != 3 or gp.shape[2] != MAX_GENOTYPES or gp.shape[0] < 1:
        raise ValueError("initial_gp must have shape [people>=1, records, 78]")
    n_people, n_records, _ = gp.shape
    ids = list(sample_ids)
    if len(ids) != n_people or any(not isinstance(s, str) or not s for s in ids) or len(set(ids)) != n_people:
        raise ValueError("sample_ids must be unique nonempty biological-person IDs")
    counts = np.asarray(allele_count)
    pos = np.asarray(positions)
    if counts.shape != (n_records,) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("allele_count must be an integer vector of length records")
    if np.any((counts < 1) | (counts > MAX_ALLELES)):
        raise ValueError("allele_count must be in [1,12]")
    if pos.shape != (n_records,) or not np.issubdtype(pos.dtype, np.integer) or np.any(pos < 0):
        raise ValueError("positions must be a nonnegative integer vector of length records")
    counts = counts.astype(np.int64, copy=False)
    pos = pos.astype(np.int64, copy=False)
    chromosomes = np.asarray([""] * n_records if contigs is None else list(contigs), dtype=str)
    if chromosomes.shape != (n_records,):
        raise ValueError("contigs must have length records")
    genotype_count = counts * (counts + 1) // 2
    valid = np.arange(MAX_GENOTYPES)[None, :] < genotype_count[:, None]
    if not np.isfinite(gp).all() or np.any(gp < 0):
        raise ValueError("initial_gp must contain finite nonnegative probabilities")
    if np.any(np.where(valid[None, :, :], 0.0, gp) > 1e-8):
        raise ValueError("initial_gp has probability in padded/nonexistent genotype states")
    if not np.allclose(gp.sum(axis=-1), 1.0, rtol=0, atol=1e-5):
        raise ValueError("initial_gp rows must sum to one")
    depth = np.asarray(read_depth, dtype=np.float64)
    if depth.shape != (n_people, n_records) or not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("read_depth must be finite nonnegative [people, records]")
    quality = np.ones_like(depth) if site_quality is None else np.asarray(site_quality, dtype=np.float64)
    if quality.shape != depth.shape or not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 1)):
        raise ValueError("site_quality must be finite [0,1] values shaped [people, records]")
    # Canonical ordering guarantees deterministic top-K and floating reductions.
    order = np.asarray(sorted(range(n_people), key=lambda i: ids[i]), dtype=int)
    gp = np.where(valid[None, :, :], gp[order], 0.0)
    gp /= gp.sum(axis=-1, keepdims=True)
    depth, quality = depth[order], quality[order]
    if read_pl is not None or read_gl is not None:
        values = read_pl if read_pl is not None else read_gl
        flank_gp = _read_probabilities(values, valid, gp.shape, read_pl is not None)[order]
        flank_source = "read_only_PL" if read_pl is not None else "read_only_natural_log_GL"
    else:
        flank_gp = gp
        flank_source = "initial_external_GP_fallback"
    slots = min(cfg.max_donors, n_people - 1)
    result = empty_cohort_context(n_people, n_records, donor_slots=slots,
                                  reason="single_sample" if n_people == 1 else "no_eligible_donors")
    result.provenance.update({
        "version": "cohort_features_v1", "n_people": n_people, "n_records": n_records,
        "gp_source": gp_source, "flank_source": flank_source,
        "sample_ids_canonical": [ids[i] for i in order],
        "genotype_order": "VCF triangular: b*(b+1)/2+a, a<=b",
        "config": asdict(cfg), "passes": 1, "self_donors": False,
        "posterior_multiplication": False, "phase_inferred": False,
        "initial_gp_sha256_canonical": hashlib.sha256(gp.tobytes()).hexdigest(),
        "library_policy": "one upstream-combined row per biological person; sitewise depth saturation",
        "limitations": ["unphased flanking compatibility is not a phased copying HMM",
                        "coordinate separation does not establish molecular independence",
                        "features and context_strength are not calibrated posterior probabilities"],
    })
    if n_people == 1 or n_records == 0:
        return result
    dosage = gp @ _DOSAGE
    carrier = gp @ _CARRIER
    certainty = _certainty(gp, genotype_count)
    depth_factor = depth / (depth + cfg.depth_half_saturation)
    reliability = certainty * depth_factor * quality
    flank_reliability = _certainty(flank_gp, genotype_count) * depth_factor * quality
    flank_dosage = flank_gp @ _DOSAGE
    # Cohort variation only chooses useful flanking records; no focal data enter.
    mean_flank = flank_dosage.mean(axis=0)
    informativeness = np.maximum(0.05, 1.0 - np.sum(mean_flank**2, axis=-1))
    active_messages = 0
    messages_without_flanks = 0
    from evidence_trace import enabled as trace_enabled
    trace_flanks = {} if trace_enabled() else None
    for site in range(n_records):
        distance = np.abs(pos - pos[site])
        flank_indices = np.flatnonzero((chromosomes == chromosomes[site]) &
                                      (distance > cfg.focal_exclusion_bp) &
                                      (distance <= cfg.window_bp))
        if flank_indices.size:
            near_order = np.lexsort((flank_indices, pos[flank_indices], distance[flank_indices]))
            flank_indices = flank_indices[near_order[:cfg.max_flank_records]]
        if trace_flanks is not None:
            trace_flanks[str(site)] = flank_indices.tolist()
        for target in range(n_people):
            eligible = np.flatnonzero((np.arange(n_people) != target) &
                                     (reliability[:, site] > cfg.min_donor_reliability))
            if not eligible.size:
                continue
            similarity = np.full(eligible.size, 0.5, dtype=float)
            flank_evidence = np.zeros(eligible.size, dtype=float)
            if flank_indices.size:
                target_prob = flank_gp[target, flank_indices]
                donor_prob = flank_gp[eligible[:, None], flank_indices[None, :]]
                # Bhattacharyya affinity of uncertain diploid genotype states.
                affinity = np.sum(np.sqrt(donor_prob * target_prob[None, :, :]), axis=-1)
                marker_weight = np.sqrt(flank_reliability[eligible[:, None], flank_indices[None, :]] *
                                        flank_reliability[target, flank_indices][None, :])
                marker_weight *= informativeness[flank_indices][None, :]
                marker_weight *= np.exp(-distance[flank_indices] / cfg.window_bp)[None, :]
                total_marker_weight = marker_weight.sum(axis=1)
                supported = total_marker_weight > 1e-12
                similarity[supported] = ((affinity * marker_weight).sum(axis=1)[supported] /
                                         total_marker_weight[supported])
                flank_evidence = total_marker_weight / (total_marker_weight + cfg.flank_weight_half_saturation)
            # No flank support -> neutral compatibility, weak context, not a fake haplotype match.
            effective_similarity = 0.5 + flank_evidence * (similarity - 0.5)
            weights = reliability[eligible, site] * np.exp(-cfg.incompatibility_penalty *
                                                          (1.0 - effective_similarity))
            # eligible is already in canonical ID order; stable sort resolves exact ties.
            selected = np.argsort(-weights, kind="stable")[:slots]
            donors = eligible[selected]
            w = weights[selected]
            norm = w / w.sum()
            af = norm @ dosage[donors, site]
            neff = float(1.0 / np.sum(norm**2))
            effective_carriers = neff * (norm @ carrier[donors, site])
            support = effective_carriers / (1.0 + effective_carriers)
            donor_certainty = float(norm @ certainty[donors, site])
            compatibility = float(norm @ similarity[selected])
            disagreement = np.clip(4.0 * (norm @ ((dosage[donors, site] - af)**2)), 0.0, 1.0)
            availability = eligible.size / (n_people - 1)
            n_alleles = counts[site]
            original_target = order[target]
            feature = result.allele_context[original_target, site, :n_alleles]
            feature[:, 0] = af[:n_alleles]
            feature[:, 1] = support[:n_alleles]
            feature[:, 2] = donor_certainty
            feature[:, 3] = compatibility
            feature[:, 4] = disagreement[:n_alleles]
            feature[:, 5] = availability
            flank_support = float(norm @ flank_evidence[selected])
            mean_reliability = float(norm @ w)  # retain absolute compatibility, even with one donor
            # Absolute quality matters, unlike a normalized AF where even one weak donor sums to 1.
            strength = mean_reliability * neff / (1.0 + neff) * (0.25 + 0.75 * flank_support)
            result.context_strength[original_target, site] = np.clip(strength, 0.0, 1.0)
            result.donor_indices[original_target, site, :donors.size] = order[donors]
            result.donor_weights[original_target, site, :donors.size] = w
            active_messages += 1
            messages_without_flanks += int(flank_support == 0.0)
    result.provenance.update({"enabled": True, "reason": "automatic_multi_person",
                              "active_target_records": active_messages,
                              "messages_without_informative_flanks": messages_without_flanks})
    if trace_flanks is not None:
        result.provenance['lineage_flank_indices'] = trace_flanks
    return result
