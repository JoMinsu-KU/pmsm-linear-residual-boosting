"""Descriptive paper diagnostics from the current run's supplied rows only.

The original condition analysis used Spearman(B, own-minus-original RMSE),
with fewer than three profiles reported as null. Pearson is reported separately.
No archived result, target, file or fitted model is read here.
"""
from collections import defaultdict
import math

import numpy as np


def _index(rows, fields):
    indexed = {}
    for row in rows:
        key = tuple(row[k] for k in fields)
        if key in indexed:
            raise ValueError("Duplicate diagnostic input identity: " + repr(key))
        indexed[key] = row
    return indexed


def _metric(row):
    n, value = int(row["n"]), float(row["rmse"])
    if n <= 0 or not math.isfinite(value) or value < 0:
        raise ValueError("Positive endpoint count and finite nonnegative RMSE required")
    return n, value


def _association(x, y, kind, missing_reason=None):
    n = len(x)
    result = dict(value=None, n=n, null_reason=missing_reason,
                  p_value=None, causal_or_confirmatory=False)
    if missing_reason:
        return result
    minimum = 3 if kind == "spearman" else 2
    if n < minimum:
        result["null_reason"] = "fewer_than_three_profiles" if minimum == 3 else "fewer_than_two_profiles"
        return result
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if kind == "spearman":
        def average_rank(v):
            _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
            starts = np.cumsum(counts) - counts
            return (starts + (counts - 1) / 2)[inverse].astype(float)
        x, y = average_rank(x), average_rank(y)
    x, y = x - np.mean(x), y - np.mean(y)
    den = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if den == 0:
        result["null_reason"] = "constant_profile_rank" if kind == "spearman" else "zero_variance"
    else:
        result["value"] = float(np.dot(x, y) / den)
    return result


def summarize_diagnostics(profiles, segments, diagnostics):
    """Return complete-pair cohort summaries; missing rows never form a hidden subset.

    Temporal contrasts are means of within-profile segment RMSE differences,
    not pooled-row RMSE differences or equal-duration segments. Initialization
    change is own_boundary minus original; negative values denote improvement.
    """
    pidx = _index(profiles, ("dataset", "arm", "role", "profile", "method"))
    sidx = _index(segments, ("dataset", "arm", "role", "profile", "method", "fifth"))
    didx = _index(diagnostics, ("dataset", "arm", "role", "profile"))
    groups = defaultdict(set)
    for ds, arm, role, pid, method in pidx:
        groups[(ds, arm, role)].add(pid)
    for ds, arm, role, pid, method, fifth in sidx:
        if fifth not in range(1, 6):
            raise ValueError("Temporal fifth must be one through five")
        if (ds, arm, role, pid, method) not in pidx:
            raise ValueError("Segment has no corresponding full-profile result")

    temporal = []
    contrasts = (("linear_half", "d9"), ("linear_residual", "direct200"))
    for (ds, arm, role), ids in sorted(groups.items()):
        for fifth in range(1, 6):
            for left, right in contrasts:
                pairs, missing = [], []
                for pid in sorted(ids):
                    a = sidx.get((ds, arm, role, pid, left, fifth))
                    b = sidx.get((ds, arm, role, pid, right, fifth))
                    if a is None or b is None:
                        missing.append(dict(profile=int(pid), missing_methods=[m for m, v in ((left, a), (right, b)) if v is None]))
                        continue
                    na, va = _metric(a)
                    nb, vb = _metric(b)
                    if na != nb or any(a.get(k) != b.get(k) for k in ("first_source_index", "last_source_index")):
                        raise ValueError("Temporal contrast endpoints differ")
                    pairs.append(dict(profile=int(pid), n=na, left_rmse_c=va, right_rmse_c=vb,
                                      difference_c=va-vb, first_source_index=a.get("first_source_index"),
                                      last_source_index=a.get("last_source_index")))
                complete = len(pairs) == len(ids) and bool(ids)
                temporal.append(dict(dataset=ds, arm=arm, role=role, fifth=fifth,
                    left=left, right=right, contrast=left+"__minus__"+right,
                    profiles=len(pairs), expected_profiles=len(ids), n=sum(v["n"] for v in pairs),
                    macro_difference_c=math.fsum(v["difference_c"] for v in pairs)/len(pairs) if complete else None,
                    macro_left_rmse_c=math.fsum(v["left_rmse_c"] for v in pairs)/len(pairs) if complete else None,
                    macro_right_rmse_c=math.fsum(v["right_rmse_c"] for v in pairs)/len(pairs) if complete else None,
                    complete_pairs=complete, null_reason=None if complete else "missing_method_pairs",
                    missing_pairs=missing, per_profile=pairs))

    cohorts = sorted({(ds, role) for ds, arm, role in groups if ds in ("ipmsm", "paderborn") and arm in ("original", "own_boundary")})
    associations = []
    for ds, role in cohorts:
        ids = groups.get((ds, "original", role), set()) | groups.get((ds, "own_boundary", role), set())
        pairs, missing = [], []
        for pid in sorted(ids):
            old = pidx.get((ds, "original", role, pid, "linear_residual"))
            own = pidx.get((ds, "own_boundary", role, pid, "linear_residual"))
            descriptor = didx.get((ds, "original", role, pid), {}).get("boundary_history_rms_difference")
            missing_fields = [key for key, value in (("original_linear_residual", old),
                ("own_boundary_linear_residual", own), ("original_B", descriptor)) if value is None]
            if missing_fields:
                missing.append(dict(profile=int(pid), missing_fields=missing_fields))
                continue
            no, vo = _metric(old)
            nn, vn = _metric(own)
            if no != nn:
                raise ValueError("Initialization contrast requires identical endpoint counts")
            b = float(descriptor)
            if not math.isfinite(b) or b < 0:
                raise ValueError("Finite nonnegative B descriptor required")
            delta = vn - vo
            pairs.append(dict(profile=int(pid), n=no, B_rms_c=b, original_rmse_c=vo,
                own_boundary_rmse_c=vn, signed_change_c=delta,
                percent_change=100*delta/vo if vo > 0 else None,
                percent_null_reason=None if vo > 0 else "zero_original_rmse"))
        complete = len(pairs) == len(ids) and bool(ids)
        reason = None if complete else "missing_initialization_pairs_or_B"
        x, signed = [p["B_rms_c"] for p in pairs], [p["signed_change_c"] for p in pairs]
        percent_pairs = [p for p in pairs if p["percent_change"] is not None]
        percent_reason = reason or ("zero_original_rmse" if len(percent_pairs) != len(pairs) else None)
        px, py = [p["B_rms_c"] for p in percent_pairs], [p["percent_change"] for p in percent_pairs]
        associations.append(dict(dataset=ds, role=role, method="linear_residual",
            comparison="own_boundary minus original", expected_profiles=len(ids), profiles=len(pairs),
            n=sum(p["n"] for p in pairs), complete_pairs=complete, missing_pairs=missing,
            pearson_signed_change=_association(x, signed, "pearson", reason),
            spearman_signed_change=_association(x, signed, "spearman", reason),
            pearson_percent_change=_association(px, py, "pearson", percent_reason),
            spearman_percent_change=_association(px, py, "spearman", percent_reason),
            per_profile=pairs))
    return dict(schema="pmsm-descriptive-diagnostics-v2", temporal_fifth_contrasts=temporal,
                initialization_associations=associations,
                source="Only supplied generated per-profile, temporal-segment and descriptor rows; no archived reads.",
                temporal_interpretation="Equal-profile means of within-arm segment RMSE differences; fifths partition retained rows, not equal physical durations.",
                initialization_interpretation="B from the original arm versus own_boundary-minus-original linear-residual RMSE; negative change means improvement after refitting.",
                correlation_interpretation="Pearson requires at least2 profiles; historical-compatible Spearman uses average tied ranks and requires at least3. No p-values, causal claim, hypothesis test or independent confirmation.",
                incomplete_pairs="A missing pair makes the cohort summary null; no silent subset aggregation.",
                rwth_initialization_association="Not applicable: no ambient channel or channel-wise intervention.",
                causal_or_confirmatory=False)
