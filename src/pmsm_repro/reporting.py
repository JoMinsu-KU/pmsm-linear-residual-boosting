"""Publication tables and figures from explicitly supplied evaluation outputs.

No historical result is opened or substituted by this module. ``mode`` describes
provenance, not a choice of hidden data: reference snapshots must be passed by
the caller just like newly generated rows. Missing smoke-run evidence is marked
unavailable; ``strict=True`` rejects an incomplete paper report.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

METHOD_LABELS = {
    "linear_prior": "Ridge regression",
    "d9": "Selected direct boosting",
    "linear_residual": "Linear residual boosting",
    "linear_half": "Equal-weight ensemble",
    "direct200": "Matched direct boosting",
    "capacity_direct": "Expanded direct boosting",
    "free_prior": "History-only linear prior",
    "free_residual": "History-only residual boosting",
    "free_half": "History-only ensemble",
    "prior_half": "Selected-direct and ridge average",
    "direct200_half": "Selected-direct and matched-direct average",
    "mlp_mean": "Neural network metric mean",
}
METHOD_ALIASES = {
    "ridge": "linear_prior", "selected_direct": "d9", "D": "d9",
    "P": "linear_prior", "L": "linear_residual", "H": "linear_half",
    "equal_weight_ensemble": "linear_half", "matched_direct": "direct200",
    "expanded_direct": "capacity_direct", "capacity": "capacity_direct",
    "history_prior": "free_prior", "history_residual": "free_residual",
    "history_ensemble": "free_half", "free25_residual": "free_residual",
    "free25_half": "free_half", "neural_mean": "mlp_mean",
}
ARM_ALIASES = {"ambient": "original", "channelwise": "own_boundary",
               "channel_wise": "own_boundary", "own": "own_boundary",
               "first_heating": "first_load"}
ARM_LABELS = {"original": "Original initialization", "own_boundary": "Channel-wise initialization",
              "first_load": "First-observation heating initialization"}
ROLE_ALIASES = {"historical_evaluation": "confirmation", "pilot_evaluation": "pilot"}
ROLE_LABELS = {"calibration": "Calibration", "pilot": "Pilot evaluation",
               "development": "Development", "confirmation": "Historical evaluation"}
DATASET_LABELS = {"ipmsm": "IPMSM", "paderborn": "Paderborn", "rwth": "RWTH"}
MAIN_METHODS = ("linear_prior", "d9", "linear_residual", "linear_half", "direct200",
                "capacity_direct", "free_residual", "free_half", "mlp_mean")
MAIN_COHORTS = (("ipmsm", "calibration"), ("ipmsm", "pilot"), ("ipmsm", "confirmation"),
                ("paderborn", "development"), ("paderborn", "confirmation"))
REQUIRED_SETTING_LABELS = (
    "Full linear prior", "Ridge penalty", "Residual booster", "Matched direct booster",
    "Common tree settings", "Residual and matched-tree L2", "Final average",
)

RELATED_WORK_ROWS = [
    ["Kirchgässner et al. (2021)", "Motor-temperature estimation with recursive histories and multiple supervised model families.", "Feature and baseline precedent for separating direct prediction, residual correction, and averaging."],
    ["Jing et al. (2023)", "Boosted rotor-temperature estimation with stator tooth, winding, and yoke temperatures.", "Earlier tree estimator with a different declared sensor budget."],
    ["Jin et al. (2024)", "Simplified thermal trend with neural compensation for unmodeled effects.", "Hybrid correction precedent; exact algebraic equivalence is unverified."],
    ["Hossain et al. (2026)", "Thermal-network prediction with averaged boosted residuals and a 30-second forecast.", "Close ensemble precedent with a different target horizon and averaging position."],
    ["Lee and Ha (2020)", "Temperature-increment learning, measured initialization, and estimated-state feedback.", "Temporal differences and initial-state information differ from contemporaneous residual fitting."],
    ["Cheng et al. (2025)", "Stator-temperature network with skip connections and centered condition clustering.", "Different target, residual meaning, and observation timing."],
    ["Aguilar-Zamorate et al. (2025)", "Two-state thermal model, measured temperature boundaries, and processor-in-the-loop evaluation.", "Thermal initialization and embedded validation precedent with different sensors and hardware."],
]
INPUT_LABELS = {
    "ipmsm": "Reference dq currents, dq voltages, speed, water-outlet and ambient temperatures",
    "paderborn": "Measured dq currents, dq voltages, speed, coolant and ambient temperatures",
    "rwth": "Current, speed, torque, coolant temperature",
}
CONTROL_ROWS = [
    ["Ridge regression", "33 features plus intercept", "Linear baseline"],
    ["Linear residual boosting", "Ridge plus residual trees", "Residual correction"],
    ["Selected direct boosting", "Training-selected trees", "Established direct control"],
    ["Equal-weight ensemble", "Average of selected direct and completed linear residual predictions", "Added averaging value"],
    ["Matched direct boosting", "Same settings as residual booster", "Same tree settings"],
    ["Expanded direct boosting", "Larger direct split budget", "Larger split budget"],
    ["History-only residual boosting / ensemble", "Filter-only prior, residual trees, optional average", "Prior representation"],
    ["Neural network", "33 → 64 → 32 → 1; supplied training seeds", "Neural alternative"],
]
FEATURE_ROWS = [
    ["Instantaneous terms", "Eight terms after both history blocks", "Eight terms before histories"],
    ["Boundary histories", "Ambient then coolant within each span; 10 terms", "Coolant at all five spans; five terms"],
    ["Heating histories", "Three drives within each span; 15 terms", "Four drives, each at all five spans; 20 terms"],
    ["History ordering", "Span-major", "Channel-major"],
    ["Sampling treatment", "One update per source row", "Timestamp-adjusted update"],
]


class IncompleteReportError(ValueError):
    """The supplied run does not contain all evidence required for the paper displays."""


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Report inputs must contain finite numbers, not NaN/inf.")
    return result


def _method(value: str, metadata: Mapping[str, Any]) -> str:
    value = METHOD_ALIASES.get(value, value)
    if value == "mlp_mse_seed1":
        seed = metadata.get("timed_neural_seed")
        if seed is None:
            raise ValueError("The timed neural alias requires metadata['timed_neural_seed'].")
        return "mse_mlp_" + str(int(seed))
    if value.startswith("neural_seed_"):
        return "mse_mlp_" + value.removeprefix("neural_seed_")
    if value not in METHOD_LABELS and not re.fullmatch(r"mse_mlp_\d+", value):
        raise ValueError(f"Unknown method {value!r}; provide an explicit scientific-name mapping.")
    return value


def method_label(method: str) -> str:
    if method.startswith("mse_mlp_"):
        return "Neural network (seed " + method.removeprefix("mse_mlp_") + ")"
    return METHOD_LABELS[method]


def _normalize(rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], kind: str) -> list[dict]:
    normalized = []
    for original in rows:
        row = dict(original)
        for key in ("dataset", "method"):
            if key not in row:
                raise ValueError(f"{kind} row lacks {key}.")
        row["dataset"] = str(row["dataset"]).lower()
        row["dataset"] = {"rwth48v": "rwth", "rwth_48v": "rwth"}.get(row["dataset"], row["dataset"])
        if row["dataset"] not in DATASET_LABELS:
            raise ValueError(f"Unknown dataset {row['dataset']!r}.")
        row["method"] = _method(str(row["method"]), metadata)
        # A missing arm is never silently treated as channel-wise.
        if "arm" not in row and "initialization" not in row:
            raise ValueError(f"{kind} row needs an explicit arm/initialization field.")
        row["arm"] = ARM_ALIASES.get(str(row.get("arm", row.get("initialization"))), str(row.get("arm", row.get("initialization"))))
        if row["arm"] not in ARM_LABELS:
            raise ValueError(f"Unknown initialization {row['arm']!r}.")
        if kind != "timing":
            role = str(row.get("role", row.get("cohort", "")))
            row["role"] = ROLE_ALIASES.get(role, role)
            if row["role"] not in ROLE_LABELS:
                raise ValueError(f"Unknown or missing cohort {row['role']!r}.")
        if kind == "profiles":
            if "profile" not in row:
                row["profile"] = row.get("profile_id")
            if row["profile"] is None:
                raise ValueError("Per-profile row has no profile identifier.")
            row["profile"] = str(row["profile"])
        if kind == "timing" and "artifact_bytes" not in row:
            row["artifact_bytes"] = row.get("required_file_bytes")
        for key, value in list(row.items()):
            if key in {"n", "profiles", "mse", "rmse", "mae", "bias", "under_rmse", "over_mae", "max_under", "max_over", "max_absolute_error", "macro_rmse", "pooled_rmse", "macro_under_rmse", "macro_over_mae", "macro_max_under", "worst_max_under", "median_us", "p95_us", "artifact_bytes"}:
                row[key] = _number(value)
        normalized.append(row)
    keys = ("dataset", "arm", "method") + (() if kind == "timing" else ("role",)) + (("profile",) if kind == "profiles" else ())
    seen = set()
    for row in normalized:
        key = tuple(row.get(k) for k in keys)
        if key in seen:
            raise ValueError(f"Duplicate {kind} key: {key}. Supply one aggregated timing row per method/panel.")
        seen.add(key)
    return normalized


def _with_neural_means(rows: list[dict]) -> list[dict]:
    """Average supplied seed metrics, never predictions or pooled seed errors."""
    out = [dict(row) for row in rows]
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row["method"].startswith("mse_mlp_"):
            groups.setdefault((row["dataset"], row["arm"], row["role"]), []).append(row)
    for key, seeds in groups.items():
        existing = next((r for r in out if (r["dataset"], r["arm"], r["role"], r["method"]) == (*key, "mlp_mean")), None)
        mean = dict(dataset=key[0], arm=key[1], role=key[2], method="mlp_mean", seed_count=len(seeds))
        for field in ("macro_rmse", "pooled_rmse", "macro_under_rmse", "macro_over_mae", "macro_max_under"):
            values = [r.get(field) for r in seeds]
            if all(v is not None for v in values):
                mean[field] = sum(values) / len(values)
        for field in ("n", "profiles"):
            values = {r.get(field) for r in seeds}
            if len(values) != 1:
                raise ValueError("Neural seeds must evaluate the same profile/row counts before averaging metrics.")
            mean[field] = values.pop()
        if existing:
            for field in ("macro_rmse", "pooled_rmse"):
                if mean.get(field) is not None and existing.get(field) is not None and not math.isclose(mean[field], existing[field], rel_tol=1e-9, abs_tol=1e-10):
                    raise ValueError("Supplied neural mean disagrees with the average of seed metrics.")
            existing["seed_count"] = len(seeds)
        else:
            out.append(mean)
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _csv(path: Path, fields: list[str], rows: Sequence[Sequence | Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for row in rows:
            writer.writerow([row.get(k, "") for k in fields] if isinstance(row, Mapping) else row)


def _tex(value: Any) -> str:
    if value is None or value == "":
        return "Not available"
    text = str(value)
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "°": r"$^{\circ}$", "μ": r"$\mu$", "→": r"$\rightarrow$", "–": "--", "−": "-"}
    return "".join(mapping.get(ch, ch) for ch in text)


def _table(out: Path, number: int, columns: list[str], rows: list[list], caption: str, scope: str) -> None:
    _csv(out / "tables" / f"table{number}.csv", columns, rows)
    wide = number in (1, 2, 4)
    env, width = ("table*", r"\textwidth") if wide else ("table", r"\columnwidth")
    spec = "".join(r">{\raggedright\arraybackslash}X" for _ in columns)
    lines = ["% Requires booktabs, tabularx, and array; generated from explicit supplied records; " + scope,
             r"\begin{" + env + "}[t]", r"\caption{" + _tex(caption) + "}",
             r"\label{tab:generated-" + str(number) + "}", r"\centering\small",
             r"\begin{tabularx}{" + width + "}{" + spec + "}", r"\toprule",
             " & ".join(r"\textbf{" + _tex(c) + "}" for c in columns) + r" \\", r"\midrule"]
    lines += [" & ".join(_tex(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{" + env + "}"]
    (out / "tables" / f"table{number}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: float | None, places: int = 6) -> str:
    return "" if value is None else f"{value:.{places}f}"


def _public_row(row: dict) -> dict:
    result = dict(row)
    result["method"] = method_label(row["method"])
    result["arm"] = ARM_LABELS[row["arm"]]
    if "role" in row:
        result["role"] = ROLE_LABELS[row["role"]]
    return result


def generate_report(output_dir: str | Path, accuracy: Sequence[Mapping[str, Any]],
                    profiles: Sequence[Mapping[str, Any]], timing: Sequence[Mapping[str, Any]],
                    metadata: Mapping[str, Any], mode: str = "generated", strict: bool = True) -> dict:
    """Write seven CSV/LaTeX tables, four PNG/PDF figures and supplementary CSVs.

    Accuracy keys: dataset, arm, role, method, profiles, n, macro_rmse,
    pooled_rmse; Table V also needs macro_under_rmse, macro_over_mae and
    macro_max_under. Profiles: dataset, arm, role, method, profile, n, rmse,
    max_under. Timing: dataset, arm, method, profile_id, median_us, p95_us,
    artifact_bytes. Numeric values must be finite. Neural method IDs identify
    the actual seed: ``mse_mlp_<seed>``.

    metadata['datasets'][dataset] supplies training_profiles (list or count),
    total_profiles, and optionally warmup_exclusion/permitted_inputs.
    metadata['model_settings'] maps REQUIRED_SETTING_LABELS to the actual run's
    text settings. metadata['execution_mode']='smoke' marks non-paper runs.
    ``strict=False`` emits honest missing-value/empty-panel markers for smoke
    runs. It never loads a prior saved result to fill those gaps.
    """
    if mode not in ("generated", "reference_snapshot"):
        raise ValueError("mode must be 'generated' or explicitly 'reference_snapshot'.")
    out = Path(output_dir)
    metadata = _jsonable(metadata)
    acc = _with_neural_means(_normalize(accuracy, metadata, "accuracy"))
    prof = _normalize(profiles, metadata, "profiles")
    times = _normalize(timing, metadata, "timing")
    missing: list[str] = []
    def require(value: Any, label: str) -> Any:
        if value is None or value == "":
            missing.append(label)
        return value
    index = {(r["dataset"], r["arm"], r["role"], r["method"]): r for r in acc}
    cost = {(r["dataset"], r["arm"], r["method"]): r for r in times}
    timing_profiles: dict[tuple, set] = {}
    for row in times:
        profile_id = row.get("profile_id")
        require(profile_id, f"timing/{row['dataset']}/{row['arm']}/{row['method']}/profile_id")
        timing_profiles.setdefault((row["dataset"], row["arm"]), set()).add(str(profile_id))
    if any(len(ids) != 1 for ids in timing_profiles.values()):
        raise ValueError("All methods in a timing panel must use the same fixed profile.")
    def metric(dataset: str, role: str, method: str, field: str = "macro_rmse", arm: str = "own_boundary") -> float | None:
        return require(index.get((dataset, arm, role, method), {}).get(field), f"{dataset}/{arm}/{role}/{method}/{field}")
    scope = "reference snapshot; not a new experimental run" if mode == "reference_snapshot" else str(metadata.get("execution_mode", "generated experiment"))

    # Prepare all table values before writing, so strict mode cannot leave a
    # plausible-looking partial paper export behind.
    table2 = []
    dsmeta = metadata.get("datasets", {})
    for ds in DATASET_LABELS:
        info = dsmeta.get(ds, {})
        train = info.get("training_profiles")
        train = len(train) if isinstance(train, (list, tuple)) else train
        require(train, f"metadata/{ds}/training_profiles")
        total = require(info.get("total_profiles"), f"metadata/{ds}/total_profiles")
        counts = []
        roles = ["calibration", "pilot", "confirmation"] if ds == "ipmsm" else ["development", "confirmation"] if ds == "paderborn" else ["calibration", "confirmation"]
        for role in roles:
            candidates = [r for r in acc if r["dataset"] == ds and r["role"] == role and r.get("profiles") is not None and r.get("n") is not None]
            pairs = {(int(r["profiles"]), int(r["n"])) for r in candidates}
            if len(pairs) > 1:
                raise ValueError(f"Conflicting evaluation profile/row counts for {ds}/{role}.")
            value = next(iter(pairs), None)
            require(value, f"metadata/{ds}/{role}/evaluation_counts")
            counts.append((role, value))
        assign = "training " + (str(int(train)) if train is not None else "not supplied")
        assign += "; " + "; ".join(ROLE_LABELS[role].lower() + " " + (str(pair[0]) if pair else "not available") for role, pair in counts)
        rows = " / ".join(f"{pair[1]:,}" if pair else "not available" for _, pair in counts)
        mask = info.get("warmup_exclusion")
        require(mask, f"metadata/{ds}/warmup_exclusion")
        rows += "; " + ("no warmup exclusion" if mask == 0 else f"exclude first {mask} endpoints" if mask is not None else "mask not supplied")
        table2.append([DATASET_LABELS[ds] + (f", {int(total)} profiles" if total is not None else ""), info.get("permitted_inputs", INPUT_LABELS[ds]), assign, rows])
    table4 = []
    for method in MAIN_METHODS:
        table4.append([method_label(method), *[_fmt(metric(ds, role, method)) for ds, role in MAIN_COHORTS]])
    for ds, role in MAIN_COHORTS:
        mean = index.get((ds, "own_boundary", role, "mlp_mean"), {})
        if mean.get("seed_count", 0) != int(metadata.get("expected_neural_seed_count", 3)):
            missing.append(f"{ds}/{role}/complete neural seed metric set")
    table5 = []
    for title, field in [("Macro-RMSE", "macro_rmse"), ("Pooled RMSE", "pooled_rmse"),
                         ("Macro underestimation RMSE", "macro_under_rmse"), ("Mean maximum underestimation", "macro_max_under")]:
        table5.append([title, *[_fmt(metric("paderborn", "confirmation", m, field)) for m in ("linear_residual", "linear_half")]])
    worst = []
    for method in ("linear_residual", "linear_half"):
        rows = [r for r in prof if (r["dataset"], r["arm"], r["role"], r["method"]) == ("paderborn", "own_boundary", "confirmation", method)]
        value = max((r["max_under"] for r in rows if r.get("max_under") is not None), default=None)
        require(value, f"paderborn/confirmation/{method}/profile maximum underestimation")
        worst.append(_fmt(value))
    table5.append(["Worst maximum underestimation", *worst])
    table5.append(["Macro overestimation MAE", *[_fmt(metric("paderborn", "confirmation", m, "macro_over_mae")) for m in ("linear_residual", "linear_half")]])
    file_label = "Prediction component files, bytes" if mode == "generated" else "Required files, bytes"
    for title, field, precision in [("Median latency (μs)", "median_us", 3), ("95th-percentile latency (μs)", "p95_us", 3), (file_label, "artifact_bytes", 0)]:
        vals = [require(cost.get(("paderborn", "own_boundary", method), {}).get(field), f"timing/paderborn/{method}/{field}") for method in ("linear_residual", "linear_half")]
        table5.append([title, *[_fmt(v, precision) for v in vals]])
    settings = metadata.get("model_settings", {})
    table7 = [[label, require(settings.get(label), "metadata/model_settings/" + label)] for label in REQUIRED_SETTING_LABELS]

    # Plot data are exported too, to make the graphics independently inspectable.
    figure2 = []
    time_methods = ["d9", "linear_residual", "linear_half", "direct200", "capacity_direct", "free_residual", "free_half"]
    timed_neural = [r["method"] for r in times if r["dataset"] == "paderborn" and r["arm"] == "own_boundary" and r["method"].startswith("mse_mlp_")]
    if len(timed_neural) != 1:
        missing.append("one identified timed Paderborn neural seed")
    if len(timed_neural) > 1:
        raise ValueError("Figure 2 requires one explicitly identified timed neural seed.")
    time_methods += timed_neural
    for role in ("development", "confirmation"):
        for method in time_methods:
            t = cost.get(("paderborn", "own_boundary", method), {})
            x = require(t.get("median_us"), f"figure2/{method}/median_us")
            y = metric("paderborn", role, method)
            if x is not None and y is not None:
                if x <= 0:
                    raise ValueError("Logarithmic latency axis requires positive timings.")
                figure2.append({"cohort": ROLE_LABELS[role], "method": method_label(method), "median_us": x, "macro_rmse": y, "timing_profile": t.get("profile_id", "")})
    figure3, influence = [], []
    for role in ("development", "confirmation"):
        selected = [r for r in prof if r["dataset"] == "paderborn" and r["arm"] == "own_boundary" and r["role"] == role]
        methods = {method: {r["profile"]: r for r in selected if r["method"] == method} for method in ("linear_residual", "direct200")}
        if set(methods["linear_residual"]) != set(methods["direct200"]):
            raise ValueError("Figure 3 requires exactly paired residual and matched-direct profiles; no inner-join exclusion is allowed.")
        order = list(methods["linear_residual"])
        if not order:
            missing.append("figure3/" + role + "/paired profiles")
        deltas = []
        for pid in order:
            left, right = (methods[m][pid] for m in ("linear_residual", "direct200"))
            if left.get("n") != right.get("n"):
                raise ValueError("Paired methods have different endpoint counts.")
            delta = float(left["rmse"]) - float(right["rmse"])
            deltas.append(delta)
            figure3.append({"cohort": ROLE_LABELS[role], "profile": pid, "rmse_difference": delta})
        if deltas:
            mean = sum(deltas) / len(deltas)
            for pid, delta in zip(order, deltas):
                influence.append({"dataset": "paderborn", "cohort": ROLE_LABELS[role], "contrast": "Linear residual boosting minus matched direct boosting", "deleted_profile": pid, "full_mean_difference": mean, "deletion_mean_difference": (sum(deltas) - delta) / (len(deltas) - 1) if len(deltas) > 1 else "", "refitted": False})
    figure4 = []
    init_methods = ["d9", "linear_residual", "linear_half", "free_half", "mlp_mean"]
    for ds, role in [("ipmsm", "confirmation"), ("paderborn", "development"), ("paderborn", "confirmation")]:
        for method in init_methods:
            baseline = metric(ds, role, method, arm="original")
            channel = metric(ds, role, method)
            if baseline is not None and channel is not None:
                if baseline <= 0:
                    raise ValueError("Relative initialization change requires positive baseline RMSE.")
                figure4.append({"dataset": DATASET_LABELS[ds], "cohort": ROLE_LABELS[role], "method": method_label(method), "original_macro_rmse": baseline, "channelwise_macro_rmse": channel, "change_percent": 100 * (channel - baseline) / baseline})
    missing = sorted(set(missing))
    if strict and missing:
        raise IncompleteReportError("Incomplete paper report inputs:\n" + "\n".join(missing))

    out.mkdir(parents=True, exist_ok=True)
    columns4 = ["Method", *[DATASET_LABELS[ds] + " " + ROLE_LABELS[role].lower() for ds, role in MAIN_COHORTS]]
    tables = [
        (1, ["Study", "Established target or construction", "Relation to the present comparison"], RELATED_WORK_ROWS, "Selected prior work. Descriptive protocol table; no numerical cross-paper ranking."),
        (2, ["Resource", "Permitted operating inputs", "Profile assignments", "Retained evaluation rows and masking"], table2, "Data and evaluation roles from the supplied run. Historical evaluation records are exposed."),
        (3, ["Predictor", "Structure", "Purpose"], CONTROL_ROWS, "Predictor structures and control purposes. Actual fitted settings appear in Table VII."),
        (4, columns4, table4, "Channel-wise initialization profile-macro RMSE (°C). Neural rows average supplied seed metrics, not predictions."),
        (5, ["Quantity", "Linear residual boosting", "Equal-weight ensemble"], table5, "Paderborn historical evaluation errors and fixed-profile resident-row CPU cost. Error units are °C."),
        (6, ["Feature block", "IPMSM and Paderborn", "RWTH"], FEATURE_ROWS, "Feature-vector ordering and causal history representation."),
        (7, ["Component", "Setting"], table7, "Actual settings supplied by the execution metadata."),
    ]
    for number, columns, rows, caption in tables:
        _table(out, number, columns, rows, caption + " Provenance: " + scope + ".", scope)
    for name, rows in [("S1-channelwise-accuracy.csv", [r for r in acc if r["arm"] == "own_boundary"]), ("S2-profile-metrics.csv", prof), ("S3-streaming-cost.csv", times), ("S4-all-initializations.csv", acc)]:
        public = [_public_row(row) for row in rows]
        fields = list(dict.fromkeys(k for row in public for k in row)) or ["dataset", "arm", "method"]
        _csv(out / "tables" / name, fields, public)
    _csv(out / "tables" / "S5-profile-deletion.csv", ["dataset", "cohort", "contrast", "deleted_profile", "full_mean_difference", "deletion_mean_difference", "refitted"], influence)
    for number, fields, rows in [(2, ["cohort", "method", "median_us", "macro_rmse", "timing_profile"], figure2), (3, ["cohort", "profile", "rmse_difference"], figure3), (4, ["dataset", "cohort", "method", "original_macro_rmse", "channelwise_macro_rmse", "change_percent"], figure4)]:
        _csv(out / "figures" / f"figure{number}-data.csv", fields, rows)
    _draw_figures(out, figure2, figure3, figure4, scope)
    manifest = {"schema": "pmsm-report-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "mode": mode, "execution_mode": metadata.get("execution_mode", "unspecified"),
                "status": "COMPLETE" if not missing else "PARTIAL_EXPLICIT_MISSING_INPUTS",
                "strict": strict, "missing_inputs": missing,
                "input_counts": {"accuracy": len(accuracy), "profiles": len(profiles), "timing": len(timing)},
                "input_sha256": hashlib.sha256(json.dumps(_jsonable({"accuracy": list(accuracy), "profiles": list(profiles), "timing": list(timing), "metadata": metadata}), sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
                "neural_summary": "Arithmetic mean of supplied seed metrics; Figure 2 pairs accuracy with the actual timed seed.",
                "deletion_scope": "Descriptive removal from saved per-profile differences, no refit or confidence interval.",
                "historical_fallback_used": False,
                "files": [{"path": p.relative_to(out).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for folder in ("tables", "figures") for p in sorted((out / folder).iterdir()) if p.is_file()]}
    (out / "report-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _draw_figures(out: Path, points: list[dict], paired: list[dict], changes: list[dict], scope: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    from matplotlib.ticker import MaxNLocator, NullLocator, FuncFormatter
    import numpy as np
    from .plot_style import add_panel_labels, scatter_style, INITIALIZATION_BARS, write_style_receipt
    style = {"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    def save(fig, stem, panels=(), width_mm=177.53):
        # The fresh report keeps its provenance, including smoke labels. Panel
        # font substitution is recorded explicitly, without an archive claim.
        fig.text(.5, .012, scope, ha="center", va="bottom", fontsize=6.5, color="#555555")
        receipt = add_panel_labels(fig, panels, width_mm) if len(panels) else None
        for ext in ("pdf", "png"):
            fig.savefig(out / "figures" / f"{stem}.{ext}", dpi=220, facecolor="white")
        write_style_receipt(out / "figures", stem, receipt)
        plt.close(fig)
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(9, 3.4));ax.set(xlim=(0, 9), ylim=(0, 3.4));ax.axis("off")
        def box(x, y, w, h, text, color="#222222"):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.02", facecolor="#f6f6f6", edgecolor=color, linewidth=.9))
            ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=8.5, color=color)
        def arrow(x1, y1, x2, y2, color="#222222"):
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color=color, linewidth=.9))
        ax.text(.1, 3.15, "Training: current temperature target", weight="bold")
        for x, text in [(.15, r"Fit ridge $\hat T_{lin}$"), (2.5, r"Form $y-\hat T_{lin}(x)$"), (5.0, r"Fit residual booster $\hat r$")]:box(x, 2.55, 2, .42, text)
        arrow(2.2, 2.76, 2.45, 2.76);arrow(4.55, 2.76, 4.95, 2.76)
        ax.text(.15, 2.28, "Fit the direct predictor separately; freeze both complete predictors before evaluation.", fontsize=8.5)
        ax.text(.1, 1.98, "Inference: no target-temperature feedback", weight="bold")
        box(.1, .75, 1.35, .65, "Operating\nmeasurements");box(1.8, .75, 1.35, .65, "Causal history\n33 features")
        arrow(1.5, 1.08, 1.75, 1.08)
        for y, text in [(1.35, "Ridge prediction"), (.75, "Residual correction"), (.15, "Direct prediction")]:
            box(3.6, y, 1.65, .42, text);arrow(3.2, 1.08, 3.55, y+.21)
        ax.add_patch(Circle((5.65, 1.22), .14, facecolor="white", edgecolor="#222222"));ax.text(5.65, 1.22, "+", ha="center", va="center")
        arrow(5.3, 1.56, 5.55, 1.33);arrow(5.3, .96, 5.55, 1.11)
        box(6.05, .98, 1.0, .48, r"$\hat T_{LR}$");arrow(5.8, 1.22, 6., 1.22)
        box(7.55, .98, 1.1, .48, r"$\hat T_{avg}$");arrow(7.1, 1.22, 7.5, 1.22)
        ax.plot([5.3, 8.1], [.36, .36], color="#555555", lw=.9);arrow(8.1, .36, 8.1, .93)
        ax.text(6.9, .08, "Fixed equal-weight average", ha="center", fontsize=8)
        fig.subplots_adjust(left=.015, right=.99, top=.99, bottom=.1);save(fig, "figure1-method-overview")

        fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.6), sharey=True)
        for ax, role in zip(axes, ("Development", "Historical evaluation")):
            current = [p for p in points if p["cohort"] == role]
            for p in current:
                ax.scatter(p["median_us"], p["macro_rmse"], **scatter_style(p["method"]), s=38, label=p["method"])
            if not current:ax.text(.5, .5, "Not available in supplied run", transform=ax.transAxes, ha="center")
            ax.set_xscale("log");ax.set_title(role);ax.grid(alpha=.2);ax.set_xlabel("Median CPU latency (μs/row; log scale)")
            if current and max(p["median_us"] for p in current) / min(p["median_us"] for p in current) < 10:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                ax.xaxis.set_minor_locator(NullLocator())
        axes[0].set_ylabel("Profile-macro RMSE (°C)")
        legend = {label: handle for ax in axes for handle, label in zip(*ax.get_legend_handles_labels())}
        if legend:fig.legend(legend.values(), legend.keys(), loc="lower center", bbox_to_anchor=(.5,.045), ncol=2, fontsize=7.8, frameon=False)
        fig.text(.5, .94, "Paderborn: channel-wise initialization", ha="center", weight="bold")
        fig.subplots_adjust(left=.10, right=.98, bottom=.40, top=.86, wspace=.16);save(fig, "figure2-accuracy-latency", axes)

        fig, axes = plt.subplots(2, 1, figsize=(3.6, 5.4), sharey=True)
        for ax, role in zip(axes, ("Development", "Historical evaluation")):
            current = [r for r in paired if r["cohort"] == role];x=np.arange(len(current))
            bars = ax.bar(x, [r["rmse_difference"] for r in current], edgecolor="black", linewidth=.65)
            for bar in bars:
                negative = bar.get_height() < 0
                bar.set_facecolor("#bdbdbd" if negative else "white")
                bar.set_hatch("/" if negative else "xx")
            ax.axhline(0, color="black", lw=.7);ax.set_xticks(x, [r["profile"] for r in current]);ax.set_title(role)
            ax.set_ylabel("RMSE difference (°C)");ax.grid(axis="y", alpha=.15);ax.set_axisbelow(True)
            ax.set_xlabel("Profile identifier (supplied order)")
            ax.tick_params(axis="x", labelsize=7)
            if not current:ax.text(.5,.5,"Not available in supplied run",transform=ax.transAxes,ha="center")
        fig.text(.5,.98,"Linear residual boosting minus\nmatched direct boosting",ha="center",va="top",fontsize=8)
        fig.text(.5,.044,"Negative /: favors linear residual.\nPositive x: favors matched direct.",ha="center",fontsize=7)
        fig.subplots_adjust(left=.20,right=.98,top=.85,bottom=.22,hspace=.90);save(fig,"figure3-paired-profile-errors",axes,85.29)

        fig, axes = plt.subplots(3, 1, figsize=(3.6, 6.0), sharex=True)
        names = [method_label(m) for m in ("d9", "linear_residual", "linear_half", "free_half", "mlp_mean")]
        short_names = ["Selected direct", "Linear residual", "Equal-weight", "History-only avg.", "Neural mean"]
        values = [r["change_percent"] for r in changes]
        lower, upper = min([0., *values]), max([0., *values])
        for ax, (ds, role) in zip(axes, (("IPMSM", "Historical evaluation"), ("Paderborn", "Development"), ("Paderborn", "Historical evaluation"))):
            current = {r["method"]: r for r in changes if r["dataset"] == ds and r["cohort"] == role}
            for i, name in enumerate(names):
                if name in current:
                    v = current[name]["change_percent"]
                    fill, hatch = INITIALIZATION_BARS[i]
                    ax.barh(i, v, color=fill, hatch=hatch, edgecolor="black", linewidth=.65, height=.7)
                    inside = abs(v) >= .35 * (upper-lower) and v != 0
                    direction = (1 if v >= 0 else -1) * (-1 if inside else 1)
                    ax.annotate(f"{v:+.1f}%", (v,i), xytext=(4*direction,0), textcoords="offset points",
                                va="center", ha="left" if direction > 0 else "right", fontsize=8,
                                bbox=dict(facecolor="white", edgecolor="none", pad=.35) if inside else None)
                else:ax.text(.98,i,"not available",transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=8,color="#777777")
            ax.set_yticks(range(len(names)),short_names,fontsize=8);ax.invert_yaxis();ax.axvline(0,color="black",lw=.7);ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
            ax.set_title(ds+"\n"+role,fontsize=8.5)
        padding = max(1., .35*(upper-lower))
        for ax in axes:
            ax.set_xlim(lower-padding, upper+padding)
        axes[-1].set_xlabel("Relative macro-RMSE change (%)",fontsize=8)
        fig.text(.5,.045,"Channel-wise versus original; negative = better.",ha="center",fontsize=7)
        fig.subplots_adjust(left=.40,right=.97,top=.92,bottom=.19,hspace=.78);save(fig,"figure4-initialization-effect",axes,85.29)


def generate_from_directory(results_dir: str | Path, output_dir: str | Path, *,
                            mode: str = "generated", strict: bool = True) -> dict:
    """Read only the explicitly selected run directory; no parent-path fallback.

    Files: accuracy.csv, profiles.csv, timing.csv, metadata.json. The explicit
    ``reference_snapshot`` mode applies to those same names; it never changes
    the directory or substitutes included paper results.
    """
    root = Path(results_dir)
    def rows(name):
        with (root / name).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8-sig"))
    return generate_report(output_dir, rows("accuracy.csv"), rows("profiles.csv"), rows("timing.csv"), metadata, mode=mode, strict=strict)
