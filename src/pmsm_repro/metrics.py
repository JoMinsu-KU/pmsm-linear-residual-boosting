"""Profile-level metrics and descriptive comparisons used in the manuscript."""
from __future__ import annotations

import numpy as np

MACRO_FIELDS = ("rmse", "mae", "bias", "under_rmse", "over_mae",
                "max_under", "max_over", "max_absolute_error")


def profile_metrics(y, prediction):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or not len(y) or p.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Nonempty finite paired one-dimensional predictions/targets required")
    e = p - y
    under, over = np.maximum(-e, 0), np.maximum(e, 0)
    return dict(n=len(y), mse=float(np.mean(e * e)), rmse=float(np.sqrt(np.mean(e * e))),
                mae=float(np.mean(np.abs(e))), bias=float(e.mean()),
                under_rmse=float(np.sqrt(np.mean(under * under))), over_mae=float(over.mean()),
                max_under=float(under.max()), max_over=float(over.max()),
                max_absolute_error=float(np.max(np.abs(e))))


metrics = profile_metrics


def aggregate_profiles(rows):
    """Unweighted means across profiles; pooled RMSE separately weights endpoints."""
    rows = list(rows)
    if not rows or any(r["n"] <= 0 for r in rows):
        raise ValueError("Nonempty profile metric records required")
    n = sum(r["n"] for r in rows)
    return dict(profiles=len(rows), n=n,
                **{"macro_" + key: float(np.mean([r[key] for r in rows])) for key in MACRO_FIELDS},
                pooled_rmse=float(np.sqrt(sum(r["n"] * r["mse"] for r in rows) / n)))


def paired_comparison(left, right):
    """Descriptive paired RMSE differences; no significance or confirmation claim."""
    left, right = list(left), list(right)
    if not left or len(left) != len(right) or any(a["n"] != b["n"] for a, b in zip(left, right)):
        raise ValueError("Paired profiles/endpoints required")
    delta = np.asarray([a["rmse"] - b["rmse"] for a, b in zip(left, right)])
    deleted = [float(np.mean(np.delete(delta, j))) for j in range(len(delta))] if len(delta) > 1 else []
    reference = float(np.mean([r["rmse"] for r in right]))
    return dict(macro_delta=float(delta.mean()),
                relative_improvement_percent=float(-100 * delta.mean() / reference) if reference else None,
                wins=int(np.sum(delta < 0)), ties=int(np.sum(delta == 0)), losses=int(np.sum(delta > 0)),
                profile_deltas=delta.tolist(), leave_one_profile_out_macro_deltas=deleted,
                formal_test_performed=False)


def seed_metric_summary(seed_summaries):
    """Average seed metrics, never average seed predictions implicitly."""
    rows = list(seed_summaries)
    if not rows:
        raise ValueError("At least one seed summary required")
    if any(set(r) != set(rows[0]) for r in rows):
        raise ValueError("Aligned seed metric fields required")
    return {k: dict(mean=float(np.mean([r[k] for r in rows])),
                    minimum=float(min(r[k] for r in rows)), maximum=float(max(r[k] for r in rows)))
            for k in rows[0] if k not in ("n", "profiles")}


def ensemble_decomposition(y, direct, expert):
    """Exact same-endpoint MSE algebra; not a new theorem or macro-RMSE guarantee."""
    profile_metrics(y, direct)
    profile_metrics(y, expert)
    y, direct, expert = map(lambda a: np.asarray(a, dtype=np.float64), (y, direct, expert))
    ed, er = direct - y, expert - y
    half = .5 * (direct + expert)
    md, mr, mh = (float(np.mean(a * a)) for a in (ed, er, half - y))
    disagreement = float(np.mean((direct - expert) ** 2))
    quality, diversity = .5 * (mr - md), -.25 * disagreement
    denominator = np.sqrt(mh) + np.sqrt(md)
    return dict(mse_direct=md, mse_expert=mr, mse_half=mh, delta_mse=mh - md,
                expert_quality_term=quality, diversity_term=diversity,
                disagreement=disagreement, raw_error_cross_product=float(np.mean(ed * er)),
                centered_error_covariance=float(np.mean((ed - ed.mean()) * (er - er.mean()))),
                delta_rmse=float(np.sqrt(mh) - np.sqrt(md)),
                rmse_expert_quality_term=float(quality / denominator) if denominator else 0.,
                rmse_diversity_term=float(diversity / denominator) if denominator else 0.)


def residual_decomposition(y, prior, residual_prediction):
    profile_metrics(y, prior)
    profile_metrics(y, residual_prediction)
    y, prior, final = map(lambda a: np.asarray(a, dtype=np.float64), (y, prior, residual_prediction))
    error, correction = prior - y, final - prior
    energy = float(np.mean(correction ** 2))
    return dict(prior_mse=float(np.mean(error ** 2)), hybrid_mse=float(np.mean((final - y) ** 2)),
                delta_mse=float(np.mean((final - y) ** 2) - np.mean(error ** 2)),
                error_correction_cross_term=float(2 * np.mean(error * correction)), correction_energy=energy,
                correction_mean=float(correction.mean()), correction_rms=float(np.sqrt(energy)))
