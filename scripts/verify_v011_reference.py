"""Verify archived v011 figures and their saved inputs (standard library only)."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference/v011"
FIGURES = {
    1: "figure1-method-overview",
    2: "figure2-accuracy-latency",
    3: "figure3-paired-profile-errors",
    4: "figure5-temperature-trajectories",
    5: "figure4-initialization-effect",
    6: "figure6-initialization-segments",
}


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(reference=REFERENCE):
    reference = Path(reference).resolve()
    manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["files"]
    names = [entry["path"] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate reference manifest path")
    for entry in entries:
        path = reference / entry["path"]
        if not path.resolve().is_relative_to(reference) or path.is_symlink():
            raise ValueError("Unsafe reference path: " + entry["path"])
        if not path.is_file() or path.stat().st_size != entry["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Reference hash/size mismatch: " + entry["path"])
    actual = {p.relative_to(reference).as_posix() for p in reference.rglob("*")
              if p.is_file() and p.name != "manifest.json"}
    if actual != set(names):
        raise ValueError("Reference file inventory differs from manifest")
    required_figures = {f"figures/{stem}.{extension}" for stem in FIGURES.values()
                        for extension in ("pdf", "png", "svg")}
    if not required_figures.issubset(actual):
        raise ValueError("Incomplete six-figure archive")
    if manifest["paper_figure_mapping"] != {str(k): v for k, v in FIGURES.items()}:
        raise ValueError("Paper figure mapping mismatch")

    data = reference / "data"
    trajectories = rows(data / "figure5-plotted-data.csv")
    expected = {("paderborn", "57"): (14103, 300, 14402), ("ipmsm", "10"): (1236, 300, 1535)}
    counts = Counter((r["dataset"], r["profile"]) for r in trajectories)
    if counts != {key: value[0] for key, value in expected.items()}:
        raise ValueError("Unexpected trajectory profile selection or row count")
    examples = json.loads((data / "selected-examples.json").read_text(encoding="utf-8"))
    if [(r["dataset"], str(r["profile"])) for r in examples] != list(expected):
        raise ValueError("Selected example order changed")
    for example in examples:
        key = (example["dataset"], str(example["profile"]))
        selected = [r for r in trajectories if (r["dataset"], r["profile"]) == key]
        n, first, last = expected[key]
        if example["n"] != n or [int(r["within_profile_row"]) for r in selected] != list(range(first, last + 1)):
            raise ValueError("Trajectory rows must be complete and ordered")
        if any(r["arm"] != "own_boundary" or r["role"] != "confirmation" for r in selected):
            raise ValueError("Trajectory scope changed")
        for method in ("linear_prior", "linear_residual", "direct200"):
            errors = [float(r[method + "_c"]) - float(r["target_c"]) for r in selected]
            rmse = math.sqrt(math.fsum(e * e for e in errors) / n)
            bias = math.fsum(errors) / n
            if not math.isclose(rmse, example[method + "_rmse_c"], abs_tol=1e-10) or not math.isclose(bias, example[method + "_bias_c"], abs_tol=1e-10):
                raise ValueError("Trajectory values do not reproduce selected-example metrics")

    segments = rows(data / "initialization-summary.csv")
    cohorts = {("ipmsm", "calibration"), ("ipmsm", "pilot"), ("ipmsm", "confirmation"),
               ("paderborn", "development"), ("paderborn", "confirmation")}
    expected_keys = {(d, role, m, str(fifth)) for d, role in cohorts
                     for m in ("linear_residual", "direct200", "d9") for fifth in range(1, 6)}
    keys = {(r["dataset"], r["role"], r["method"], r["fifth"]) for r in segments}
    if len(segments) != 75 or keys != expected_keys:
        raise ValueError("Incomplete initialization segment grid")
    for row in segments:
        difference = float(row["own_macro_rmse"]) - float(row["original_macro_rmse"])
        if not math.isclose(difference, float(row["macro_difference"]), abs_tol=1e-10):
            raise ValueError("Initialization difference mismatch")
    return {"status": "PASS", "revision": "v011", "verified_files": len(entries),
            "archived_figures": 6, "trajectory_rows": len(trajectories),
            "initialization_cells": len(segments), "trajectory_metric_checks": 12,
            "scope": "File integrity, saved-example RMSE/bias, segment differences and manuscript figure mapping; no training."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    try:
        result = verify(args.reference)
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
