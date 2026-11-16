#!/usr/bin/env python3
"""Verify the saved v007 evidence using only the Python standard library.

This checks public bytes and recomputes aggregate arithmetic from the published
profile metrics. It does not reconstruct raw measurements, fit models, replay
inference, or turn historically exposed data into independent confirmation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def close(actual, expected, label):
    a, b = float(actual), float(expected)
    require(math.isfinite(a) and math.isfinite(b), f"Non-finite value: {label}")
    require(math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-10),
            f"Arithmetic mismatch: {label}: saved={a}, recomputed={b}")


def group(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    return result


def mean(rows, field):
    return math.fsum(float(r[field]) for r in rows) / len(rows)


def number_sum(rows, field):
    return sum(int(r[field]) for r in rows)


def compare_json_rows(saved, calculated, keys, label):
    lookup = {tuple(str(row[k]) for k in keys): row for row in calculated}
    require(len(saved) == len(calculated) == len(lookup), "JSON/CSV row count mismatch: " + label)
    for row in saved:
        other = lookup[tuple(str(row[k]) for k in keys)]
        for key, value in row.items():
            if isinstance(value, (int, float)):
                close(value, other[key], label + " " + key)
            else:
                require(str(value) == str(other[key]), "JSON/CSV label mismatch: " + label)


def safe_relative(value):
    return not Path(value).is_absolute() and not re.match(r"^[A-Za-z]:", value) and ".." not in Path(value).parts


def verify(root):
    manifest = read_json(root / "manifest.json")
    require(manifest["schema"] == "pmsm-v007-public-evidence-manifest-v1", "Unexpected manifest schema")
    require(manifest["independent_confirmation"] is False, "Exposure status changed")
    sources = {x["id"]: x for x in manifest["sources"]}
    require(len(sources) == len(manifest["sources"]), "Duplicate source ID")
    for source in sources.values():
        require(safe_relative(source["relative_path"]), "Non-relative original source path")
        require(re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is not None, "Invalid source digest")
    listed = set()
    private_pattern = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/(?:mnt/[a-z]|home|Users)/|\\Users\\|(?:^|[\s\"'])\\\\[A-Za-z0-9._-]+\\|(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
    for record in manifest["files"]:
        relative = record["path"]
        require(safe_relative(relative) and relative not in listed, "Unsafe or duplicate public file path")
        listed.add(relative)
        path = root / relative
        require(path.is_file(), f"Missing file: {relative}")
        require(path.stat().st_size == record["bytes"], f"Byte count mismatch: {relative}")
        require(sha256(path) == record["sha256"], f"SHA-256 mismatch: {relative}")
        require(all(sid in sources for sid in record["source_ids"]), f"Unknown source ID: {relative}")
        if path.suffix in {".json", ".csv", ".txt", ".md"}:
            require(private_pattern.search(path.read_text(encoding="utf-8-sig")) is None,
                    f"Machine-specific path or address: {relative}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == listed | {"manifest.json"}, "Unmanifested or missing public file")

    def rows(name, count):
        with (root / "analysis" / name).open(encoding="utf-8-sig", newline="") as handle:
            result = list(csv.DictReader(handle))
        require(len(result) == count, f"Unexpected row count: {name}")
        return result

    failure = rows("failure-profile-corrections.csv", 72)
    cohorts = rows("failure-cohort-corrections.csv", 10)
    failure_group = group(failure, ("dataset", "arm", "role"))
    require(len(group(failure, ("dataset", "arm", "role", "profile"))) == 72, "Duplicate failure profile")
    for row in failure:
        for mse, rmse in [("prior_mse_c2", "linear_prior_rmse_c"), ("residual_mse_c2", "linear_residual_rmse_c"), ("correction_energy_c2", "correction_rms_c")]:
            close(row[mse], float(row[rmse]) ** 2, mse)
        close(row["residual_minus_prior_mse_c2"], float(row["residual_mse_c2"]) - float(row["prior_mse_c2"]), "MSE change")
        close(row["residual_minus_prior_mse_c2"], float(row["correction_cross_term_c2"]) + float(row["correction_energy_c2"]), "correction decomposition")
        close(row["residual_minus_prior_rmse_c"], float(row["linear_residual_rmse_c"]) - float(row["linear_prior_rmse_c"]), "prior RMSE difference")
        close(row["residual_minus_matched_rmse_c"], float(row["linear_residual_rmse_c"]) - float(row["direct200_rmse_c"]), "matched RMSE difference")
    for row in cohorts:
        members = failure_group[(row["dataset"], row["arm"], row["role"])]
        close(row["profiles"], len(members), "failure cohort size")
        close(row["n"], number_sum(members, "n"), "failure cohort rows")
        for method in ("linear_prior", "linear_residual", "direct200", "d9"):
            close(row[f"{method}_macro_rmse_c"], mean(members, f"{method}_rmse_c"), "failure macro RMSE")
        for target, original in [("mean_cross_term_c2", "correction_cross_term_c2"), ("mean_correction_energy_c2", "correction_energy_c2"), ("mean_mse_change_c2", "residual_minus_prior_mse_c2"), ("mean_rowwise_improvement_percent", "rowwise_squared_error_reduced_percent")]:
            close(row[target], mean(members, original), target)
        close(row["residual_beats_prior_profiles"], sum(float(x["residual_minus_prior_rmse_c"]) < 0 for x in members), "prior win count")
        close(row["residual_beats_matched_profiles"], sum(float(x["residual_minus_matched_rmse_c"]) < 0 for x in members), "matched win count")
    for row in rows("failure-profile-support.csv", 144) + rows("failure-operating-ranges.csv", 36):
        ordered = [float(row[k]) for k in ("minimum", "q01", "q05", "median", "q95", "q99", "maximum")]
        require(ordered == sorted(ordered), "Unordered operating-range quantiles")
        for key in ("below_fit_min_percent", "above_fit_max_percent", "outside_fit_minmax_percent", "outside_fit_q05_q95_percent"):
            if row[key]:
                require(0 <= float(row[key]) <= 100, "Range percentage out of bounds")
        if row["outside_fit_minmax_percent"]:
            close(row["outside_fit_minmax_percent"], float(row["below_fit_min_percent"]) + float(row["above_fit_max_percent"]), "marginal exceedance sum")

    paired = rows("initialization-paired.csv", 540)
    arms = rows("initialization-profiles.csv", 1080)
    fifth_summary = rows("initialization-summary.csv", 75)
    whole = rows("initialization-whole-profile.csv", 108)
    whole_summary = rows("initialization-whole-summary.csv", 15)
    pkey = ("dataset", "role", "profile", "method", "fifth")
    arm_map = {tuple(r[k] for k in pkey) + (r["arm"],): r for r in arms}
    require(len(arm_map) == 1080, "Duplicate initialization arm/fifth")
    require(len(group(paired, pkey)) == 540, "Duplicate initialization pair")
    for row in paired:
        close(row["difference"], float(row["own_rmse"]) - float(row["original_rmse"]), "paired fifth difference")
        close(row["n"], int(row["last_source_index"]) - int(row["first_source_index"]) + 1, "fifth retained index count")
        for prefix, arm in (("original", "original"), ("own", "own_boundary")):
            other = arm_map[tuple(row[k] for k in pkey) + (arm,)]
            for field in ("n", "first_source_index", "last_source_index"):
                close(row[field], other[field], "arm pairing " + field)
            for metric in ("mse", "rmse"):
                close(row[prefix + "_" + metric], other[metric], "paired arm " + metric)
            close(row[prefix + "_mse"], float(row[prefix + "_rmse"]) ** 2, "fifth MSE")
    fifth_groups = group(paired, ("dataset", "role", "method", "fifth"))
    for row in fifth_summary:
        members = fifth_groups[tuple(row[k] for k in ("dataset", "role", "method", "fifth"))]
        n = number_sum(members, "n")
        close(row["profiles"], len(members), "fifth profile count")
        close(row["n"], n, "fifth sample count")
        for prefix in ("original", "own"):
            close(row[prefix + "_macro_rmse"], mean(members, prefix + "_rmse"), "fifth macro RMSE")
            pooled = math.sqrt(math.fsum(float(x[prefix + "_mse"]) * int(x["n"]) for x in members) / n)
            close(row[prefix + "_pooled_rmse"], pooled, "fifth pooled RMSE")
        close(row["macro_difference"], mean(members, "difference"), "fifth macro difference")
        close(row["improved_profiles"], sum(float(x["difference"]) < 0 for x in members), "initialization wins")
        close(row["worsened_profiles"], sum(float(x["difference"]) > 0 for x in members), "initialization losses")
    profile_groups = group(paired, ("dataset", "role", "profile", "method"))
    failure_map = {tuple(r[k] for k in ("dataset", "role", "profile", "arm")): r for r in failure}
    for row in whole:
        members = sorted(profile_groups[tuple(row[k] for k in ("dataset", "role", "profile", "method"))], key=lambda r: int(r["fifth"]))
        n = int(row["n"])
        require([int(x["fifth"]) for x in members] == [1, 2, 3, 4, 5], "Missing retained-row fifth")
        close(n, number_sum(members, "n"), "whole sample count")
        for index, fifth in enumerate(members):
            close(fifth["first_source_index"], 300 + n * index // 5, "fifth floor boundary start")
            close(fifth["last_source_index"], 300 + n * (index + 1) // 5 - 1, "fifth floor boundary end")
        for prefix, arm in (("original", "original"), ("own", "own_boundary")):
            mse = math.fsum(float(x[prefix + "_mse"]) * int(x["n"]) for x in members) / n
            close(row[prefix + "_mse"], mse, "whole MSE from fifths")
            close(row[prefix + "_rmse"], math.sqrt(mse), "whole RMSE from fifths")
            other = failure_map[(row["dataset"], row["role"], row["profile"], arm)]
            close(row[prefix + "_rmse"], other[row["method"] + "_rmse_c"], "failure/initialization agreement")
        close(row["difference"], float(row["own_rmse"]) - float(row["original_rmse"]), "whole paired difference")
    whole_groups = group(whole, ("dataset", "role", "method"))
    for row in whole_summary:
        members = whole_groups[tuple(row[k] for k in ("dataset", "role", "method"))]
        close(row["profiles"], len(members), "whole cohort count")
        close(row["n"], number_sum(members, "n"), "whole cohort rows")
        for prefix in ("original", "own"):
            close(row[prefix + "_macro_rmse"], mean(members, prefix + "_rmse"), "whole macro RMSE")

    failure_result = read_json(root / "analysis/failure-results.json")
    require(failure_result["independently_held_out"] is False, "Failure interpretation changed")
    compare_json_rows(failure_result["cohorts"], cohorts, ("dataset", "arm", "role"), "failure cohorts")
    pb = failure_group[("paderborn", "own_boundary", "confirmation")]
    sorted_diffs = sorted(float(x["residual_minus_matched_rmse_c"]) for x in pb)
    median = (sorted_diffs[6] + sorted_diffs[7]) / 2
    success = min((x for x in pb if float(x["residual_minus_matched_rmse_c"]) < 0), key=lambda x: (abs(float(x["residual_minus_matched_rmse_c"]) - median), int(x["profile"])))
    failure_example = min(failure_group[("ipmsm", "own_boundary", "confirmation")], key=lambda x: (-float(x["residual_minus_matched_rmse_c"]), int(x["profile"])))
    require([int(x["profile"]) for x in failure_result["selected_examples"]] == [int(success["profile"]), int(failure_example["profile"])], "Outcome-based example selection changed")
    compare_json_rows(failure_result["selected_examples"], [success, failure_example], ("dataset", "arm", "role", "profile"), "selected examples")
    close(failure_result["success_selection_full_cohort_median_difference_c"], median, "selection median")
    init_result = read_json(root / "analysis/initialization-results.json")
    require(init_result["all_data_previously_exposed"] is True, "Initialization exposure status changed")
    compare_json_rows(init_result["fifth_summary"], fifth_summary, ("dataset", "role", "method", "fifth"), "initialization fifth summaries")
    compare_json_rows(init_result["whole_summary"], whole_summary, ("dataset", "role", "method"), "initialization whole summaries")
    close(init_result["paired_profiles"], len(group(whole, ("dataset", "role", "profile"))), "paired profile total")
    close(init_result["retained_rows_per_arm"], sum(int(r["n"]) for r in whole if r["method"] == "linear_residual"), "retained row total")

    history = read_json(root / "history/confirmations.json")
    require(history["independent_confirmation_of_present_method"] is False, "Historical exposure relabeled")
    require(len(history["source_records"]) == 7 and len(history["confirmation_attempts"]) == 2, "Historical source/attempt count changed")
    for attempt in history["confirmation_attempts"]:
        rule, result = attempt["primary_rule"], attempt["result"]
        require(rule["mean_rmse_difference"] == "< 0" and rule["exact_one_sided_sign_p"] == "<= 0.025" and rule["minimum_percent_improvement"] is None, "Frozen confirmation rule changed")
        close(result["paired_mean_difference_c"], result["method_macro_rmse_c"] - result["direct_macro_rmse_c"], "historical mean difference")
        close(result["relative_error_increase_percent"], 100 * result["paired_mean_difference_c"] / result["direct_macro_rmse_c"], "historical relative change")
        n = result["profiles"] - result.get("ties", 0)
        p = sum(math.comb(n, k) for k in range(result["wins"], n + 1)) / (2 ** n)
        close(result["exact_one_sided_sign_p"], p, "historical exact sign test")
        require(result.get("passed", result.get("formal_passed")) is False, "Failed confirmation relabeled")
    second = history["confirmation_attempts"][1]
    subset = second["original_adjudication_primary_subset"]
    close(subset["mean_delta"], math.fsum(subset["per_profile_rmse_differences"]) / 10, "RWTH paired mean")
    require(subset["contribution_mean_condition"] == all(x["strictly_lower"] for x in subset["strict_point_mean_controls"]), "RWTH practical criterion changed")
    for control in subset["strict_point_mean_controls"]:
        require(control["strictly_lower"] == (control["primary_macro_rmse"] < control["comparator_macro_rmse"]), "RWTH control condition mismatch")
    secondary = second["subsequent_secondary_observation"]
    close(secondary["relative_improvement_percent"], 100 * (1 - secondary["macro_rmse_c"] / second["result"]["direct_macro_rmse_c"]), "secondary 8.4 percent observation")
    require(secondary["status"] == "PROMISING_SECONDARY_OBSERVATION_NOT_NEWLY_CONFIRMED_TOPIC", "Secondary observation relabeled")
    mapping = read_json(root / "figures/mapping.json")
    require({x["original_filename_number"]: x["v007_main_figure_number"] for x in mapping["figures"]} == {1: 1, 2: 2, 3: 3, 4: 5, 5: 4, 6: 6}, "Figure numbering changed")
    return {
        "schema": "pmsm-v007-public-aggregate-validation-v1", "status": "PASS",
        "scope": "Independent standard-library arithmetic checks of published profile metrics, public file digests and provenance links, exact historical criteria and figure numbering. Private original source digests are recorded but cannot be rechecked without those originals. No raw-array replay, training, inference or fresh confirmation.",
        "verifier_sha256": sha256(Path(__file__)),
        "checked": {"failure_profile_rows": 72, "failure_cohort_rows": 10, "support_profile_rows": 144, "operating_range_rows": 36, "initialization_paired_fifth_rows": 540, "initialization_arm_fifth_rows": 1080, "initialization_fifth_summary_rows": 75, "initialization_whole_profile_rows": 108, "initialization_whole_summary_rows": 15, "historical_confirmation_attempts": 2, "historical_source_digest_records": 7, "publication_figures": 6},
        "arithmetic_tolerance": {"relative": 1e-11, "absolute": 1e-10},
        "historically_exposed": True, "independent_confirmation": False,
        "unverified_from_aggregates": ["Raw source measurements and their marginal quantiles", "Archived model predictions", "Original model fitting", "Cross-motor independence or generalization"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=Path(__file__).resolve().parents[1] / "reference/v007")
    parser.add_argument("--output", type=Path, help="Optional JSON report path; no reference files are modified by default.")
    args = parser.parse_args()
    try:
        report = verify(args.reference_root)
    except (EvidenceError, KeyError, OSError, ValueError) as error:
        raise SystemExit(f"FAIL: {error}") from error
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
