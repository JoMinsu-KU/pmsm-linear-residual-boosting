"""Independent saved-artifact audit of the exploratory C16 risk screen.

No scientific worker import, optimizer replay, new fit, or hypothesis test.
NumPy forward arithmetic uses a prospective FP32 roundoff envelope, not a
result-selected tolerance. Original formerly-confirmatory cohorts are exposed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for _thread_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_name] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP = ROOT / "followup-risk-20261110"
PROTOCOL_SHA = "10d816a74e763f64f4d47a0a485133d88f5189313e2d9138fd7f52f0cf80edac"
METRICS = ("rmse", "mae", "under_rmse", "under_mae", "over_mae", "under_q95",
           "under_q99", "max_under", "max_over", "under_gt5_longest_samples",
           "under_gt5_longest_seconds")
WEIGHT_SHAPES = {"0.weight": (64, 33), "0.bias": (64,), "2.weight": (32, 64),
                 "2.bias": (32,), "4.weight": (1, 32), "4.bias": (1,)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def path_in(root, value):
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Artifact escapes its declared root")
    return candidate


def array_digest(values):
    digest = hashlib.sha256()
    for key in sorted(values):
        value = np.ascontiguousarray(values[key])
        digest.update(json.dumps([key, value.dtype.str, list(value.shape)], separators=(",", ":")).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class Audit:
    def __init__(self, cap):
        self.start = time.monotonic()
        self.cap = cap
        self.checks = 0
        self.failures = []
        self.inputs = {}
        self.replayed = {}
        self.results = {}
        self._pins = {}

    def budget(self):
        if time.monotonic() - self.start > self.cap:
            raise TimeoutError("Independent audit time limit")

    def check(self, condition, label, detail=None):
        self.budget()
        self.checks += 1
        if not bool(condition):
            self.failures.append({"check": label, "detail": detail})
        return bool(condition)

    def require(self, condition, label, detail=None):
        if not self.check(condition, label, detail):
            raise ValueError(label)

    def pin(self, entry, base=ROOT):
        self.budget()
        p = path_in(base, entry["path"])
        expected = entry["sha256"]
        self.require(p.is_file(), "Pinned regular file exists: " + str(p))
        observed = sha(p)
        self.require(observed == expected, "Current file SHA: " + str(p))
        self.require(str(p) not in self.inputs or self.inputs[str(p)] == observed,
                     "One identity for every pinned input")
        self.inputs[str(p)] = observed
        self._pins[p] = observed
        return p

    def arrays(self, path):
        self.require(path in self._pins, "NPZ is pinned before decoding")
        with np.load(path, allow_pickle=False) as saved:
            result = {key: saved[key].copy() for key in saved.files}
        for key, value in result.items():
            self.require(value.dtype.kind in "biuf", "Numeric NPZ field " + key)
            self.require(np.isfinite(value).all(), "Finite NPZ field " + key)
        return result

    def same(self, actual, expected, label, atol=1e-9, rtol=1e-10):
        if isinstance(expected, dict):
            self.require(isinstance(actual, dict) and set(actual) == set(expected),
                         label + " exact keys", {"expected": sorted(expected),
                                                  "actual": sorted(actual) if isinstance(actual, dict) else None})
            for key in expected:
                self.same(actual[key], expected[key], label + "/" + str(key), atol, rtol)
        elif isinstance(expected, list):
            self.require(isinstance(actual, list) and len(actual) == len(expected), label + " length")
            for i, (x, y) in enumerate(zip(actual, expected)):
                self.same(x, y, label + "/" + str(i), atol, rtol)
        elif expected is None or isinstance(expected, (str, bool, int)):
            self.check(actual == expected, label, {"actual": actual, "expected": expected})
        else:
            good = isinstance(actual, (float, int)) and math.isfinite(actual)
            good = good and math.isclose(actual, float(expected), abs_tol=atol, rel_tol=rtol)
            self.check(good, label, {"actual": actual, "expected": float(expected)})

    def finish_pins(self):
        for path, expected in self._pins.items():
            self.budget()
            self.require(sha(path) == expected, "Pinned input unchanged at audit completion: " + str(path))


def load_config(a, args):
    cp = a.pin({"path": args.config, "sha256": args.config_sha256})
    cfg = read(cp)
    a.require(cfg.get("schema") == "c16-risk-dataset-v001", "Frozen dataset config schema")
    a.require(cfg.get("scientific_execution_admitted") is True, "Explicit numerical admission")
    a.require(cfg.get("all_evaluation_previously_exposed") is True, "All results exploratory")
    a.require(cfg["protocol"]["sha256"] == PROTOCOL_SHA, "Frozen risk protocol identity")
    a.pin(cfg["protocol"])
    original = read(a.pin(cfg["original_manifest"]))
    a.require(cfg["dataset"] in ("ipmsm", "paderborn") and original["dataset"] == cfg["dataset"], "Exact dataset identity")
    a.require(original["roles"]["fit"] == cfg["train_ids"], "Original FIT cohort retained")
    for source in cfg["dependency_sources"].values():
        a.pin(source)
    a.require(cfg["training"]["architecture"] == [33, 64, 32, 1], "Frozen MLP architecture")
    a.require(cfg["training"]["losses"] == ["mse", "asym4", "q90"], "Frozen losses")
    a.require(cfg["training"]["seeds"] == [2026111701, 2026111702, 2026111703], "Frozen paired seeds")
    a.require(cfg["training"]["epochs"] == 200 and cfg["training"]["warmup"] == 300
              and cfg["training"]["stride"] == 5, "Frozen endpoint/training rule")
    a.require(cfg["metrics"]["primary"] == "macro_under_rmse"
              and cfg["metrics"]["high_temperature_train_quantile"] == .95
              and cfg["metrics"]["under_exceedance_c"] == 5., "Frozen risk metrics")
    a.require(cfg["metrics"]["sample_interval_seconds"] == (.5 if cfg["dataset"] == "paderborn" else None),
              "Seconds known only for external data")
    a.require(len(cfg["train_ids"]) == len(set(cfg["train_ids"])), "Unique train IDs")
    used = set(cfg["train_ids"])
    calibrators = []
    for role in cfg["role_inputs"]:
        a.require(original["roles"][role["role"]] == role["ids"], "Original evaluation role retained")
        a.require(role["ids"] == [x["id"] for x in role["records"]], "Exact role record coverage")
        a.require(not used.intersection(role["ids"]) and len(role["ids"]) == len(set(role["ids"])),
                  "Disjoint train/calibration/evaluation profile roles")
        used.update(role["ids"])
        if role["usage"] == "calibration":
            calibrators.append(role["role"])
        else:
            a.require(role["usage"] == "exploratory_evaluation", "Held data are exploratory")
    a.require(calibrators == [cfg["calibration_role"]], "One exact calibration role")
    return cfg


def feature_matrix(record, qscale):
    return np.column_stack((record["boundary"], record["heat"] / np.asarray(qscale), record["instant"]))


def fit_reference(a, cfg):
    entries = {x["id"]: x for x in cfg["fit_cache"]}
    a.require(set(entries) == set(cfg["train_ids"]), "Exact FIT cache coverage")
    records = {}
    ys, instants, ewmas, ids, ii = [], [], [], [], []
    for pid in cfg["train_ids"]:
        entry = entries[pid]
        arrays = a.arrays(a.pin(entry))
        n = entry["rows"]
        record = {key: np.asarray(arrays[key], dtype=np.float64) for key in ("boundary", "heat", "instant", "ewma")}
        record["target"] = np.asarray(arrays[entry["target_key"]], dtype=np.float64)
        for key, d in (("boundary", 10), ("heat", 15), ("instant", 8), ("ewma", 23)):
            a.require(record[key].shape == (n, d), "FIT cached feature shape " + key)
        a.require(record["target"].shape == (n,), "FIT target shape")
        ix = np.arange(300, n, 5, dtype=np.int64)
        a.require(len(ix) > 0, "Nonempty FIT endpoints")
        ys.append(record["target"][ix]); instants.append(record["instant"][ix]); ewmas.append(record["ewma"][ix])
        ids.append(np.full(len(ix), pid, dtype=np.int64)); ii.append(ix)
        records[pid] = record
    y = np.concatenate(ys); instant = np.concatenate(instants); ewma = np.concatenate(ewmas)
    ri = max(float(np.sqrt(np.mean(instant[:, 0] ** 2))), 1e-8)
    rw = max(float(np.sqrt(np.mean(instant[:, 3] ** 2))), 1e-8)
    qscale = np.tile([ri * ri, rw * rw, ri * rw], 5)
    mid = float(np.median(y)); scale = max(float(np.quantile(y, .75) - np.quantile(y, .25)), 1.)
    rc_scale = dict(ids=cfg["train_ids"], qscale=qscale.tolist(), y_mid=mid, y_scale=scale,
                    ewma_mean=ewma.mean(0).tolist(), ewma_scale=np.maximum(ewma.std(0), 1e-8).tolist())
    x = np.concatenate([feature_matrix(records[p], qscale)[ix] for p, ix in zip(cfg["train_ids"], ii)])
    tr = dict(rc_scale=rc_scale, input_mean=x.mean(0).tolist(), input_std=np.maximum(x.std(0), 1e-8).tolist(),
              y_mid=mid, y_scale=scale, fit_ids=cfg["train_ids"], warmup=300, label_stride=5)
    w = np.concatenate([np.full(len(ix), len(y)/(len(ii)*len(ix)), dtype=np.float64) for ix in ii]).astype(np.float32)
    quantized = ((x - np.asarray(tr["input_mean"])) / np.asarray(tr["input_std"])).astype(np.float32)
    training = dict(x=quantized, y=((y-mid)/scale).astype(np.float32), w=w,
                    profile=np.concatenate(ids), index=np.concatenate(ii))
    return records, tr, training, float(np.quantile(y, .95, method="linear"))


def weight_geometry(a, weights, tag):
    a.require(set(weights) == set(WEIGHT_SHAPES), "Six exact MLP arrays " + tag)
    for key, shape in WEIGHT_SHAPES.items():
        a.require(weights[key].shape == shape and weights[key].dtype == np.float32
                  and np.isfinite(weights[key]).all(), "FP32 MLP array " + tag + "/" + key)
    a.require(sum(v.size for v in weights.values()) == 4289, "4289 model parameters " + tag)


def forward_oracle(a, weights, x):
    """Independent FP64 affine/ReLU calculation with propagated FP32 bound.

    Inputs have already undergone the producer's explicit FP32 quantization.
    gamma_(2n+2) includes products, additions and bias; ReLU is 1-Lipschitz.
    Assumes disabled TF32 and ordinary finite FP32 arithmetic, not GPU replay.
    """
    a.require(x.dtype == np.float32 and np.isfinite(x).all(), "Finite quantized forward inputs")
    value = x.astype(np.float64)
    error = np.zeros_like(value)
    unit = 2. ** -24
    for layer in ("0", "2", "4"):
        w = weights[layer + ".weight"].astype(np.float64)
        b = weights[layer + ".bias"].astype(np.float64)
        terms = 2 * value.shape[1] + 2
        gamma = terms * unit / (1 - terms * unit)
        next_value = value @ w.T + b
        error = error @ np.abs(w).T + gamma * ((np.abs(value) + error) @ np.abs(w).T + np.abs(b)) + 1e-12
        value = np.maximum(next_value, 0) if layer != "4" else next_value
    return value[:, 0], error[:, 0]


def verify_forward(a, record, tr, weights, index, prediction, tag):
    weight_geometry(a, weights, tag)
    x = ((feature_matrix(record, tr["rc_scale"]["qscale"])[index]
          - np.asarray(tr["input_mean"])) / np.asarray(tr["input_std"])).astype(np.float32)
    a.require(prediction.shape == (len(index),), "Complete prediction geometry " + tag)
    max_diff = max_bound = max_ratio = 0.
    for start in range(0, len(x), 4096):
        a.budget()
        value, error = forward_oracle(a, weights, x[start:start+4096])
        oracle = tr["y_mid"] + tr["y_scale"] * value
        bound = abs(tr["y_scale"]) * error + 1e-9
        delta = np.abs(prediction[start:start+4096] - oracle)
        a.check(np.isfinite(oracle).all() and np.isfinite(bound).all() and np.all(delta <= bound),
                "Independent forward roundoff bound " + tag + "/" + str(start))
        max_diff = max(max_diff, float(delta.max()))
        max_bound = max(max_bound, float(bound.max()))
        max_ratio = max(max_ratio, float(np.max(delta / bound)))
    a.replayed[tag] = dict(rows=len(index), all_values=True, max_absolute_error_c=max_diff,
                           max_forward_bound_c=max_bound, max_error_bound_ratio=max_ratio)


def longest_run(flags, index):
    best = current = 0
    previous = None
    for flag, local_index in zip(flags, index):
        if flag:
            current = current + 1 if previous is not None and local_index == previous + 1 else 1
            best = max(best, current)
            previous = int(local_index)
        else:
            current = 0
            previous = None
    return best


def metric_values(y, prediction, index, interval_seconds, mask=None):
    """Zeros enter every error distribution; masked gaps break episode runs."""
    mask = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return None
    error = np.asarray(y, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    under = np.maximum(error, 0.)
    over = np.maximum(-error, 0.)
    e, u, o = error[mask], under[mask], over[mask]
    run = longest_run(mask & (under > 5.), index)
    return dict(n=int(mask.sum()), rmse=float(np.sqrt(np.mean(e*e))), mae=float(np.mean(np.abs(e))),
                under_rmse=float(np.sqrt(np.mean(u*u))), under_mae=float(np.mean(u)), over_mae=float(np.mean(o)),
                under_q95=float(np.quantile(u, .95, method="linear")),
                under_q99=float(np.quantile(u, .99, method="linear")),
                max_under=float(u.max()), max_over=float(o.max()), under_gt5_longest_samples=int(run),
                under_gt5_longest_seconds=None if interval_seconds is None else float(run*interval_seconds))


def group_metrics(profiles, interval_seconds, high_threshold):
    rows = []
    for pid, y, prediction, index in profiles:
        normal = metric_values(y, prediction, index, interval_seconds)
        high = metric_values(y, prediction, index, interval_seconds, y >= high_threshold)
        rows.append(dict(id=pid, **normal, high_temperature=None if high is None else dict(id=pid, **high)))

    def aggregate(selected):
        result = dict(profile_count=len(selected), n=sum(x["n"] for x in selected))
        for key in METRICS:
            values = [x[key] for x in selected]
            result["macro_"+key] = None if not values or any(v is None for v in values) else float(np.mean(values))
        result["pooled_rmse"] = None if not selected else float(np.sqrt(sum(x["n"]*x["rmse"]**2 for x in selected)/result["n"]))
        return result

    summary = aggregate(rows)
    high_rows = [x["high_temperature"] for x in rows if x["high_temperature"] is not None]
    summary["high_temperature"] = dict(eligible_profile_ids=[x["id"] for x in high_rows],
                                        excluded_profile_ids=[x["id"] for x in rows if x["high_temperature"] is None],
                                        **aggregate(high_rows))
    return dict(per_profile=rows, summary=summary)


def balanced_residual_quantile(profiles, quantile):
    errors = np.concatenate([y-p for _, y, p, _ in profiles])
    weights = np.concatenate([np.full(len(y), 1./(len(profiles)*len(y))) for _, y, _, _ in profiles])
    order = np.argsort(errors, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = min(int(np.searchsorted(cumulative, quantile, side="left")), len(errors)-1)
    return float(errors[order[position]])


def existing_role_records(a, cfg):
    records = {}
    for role in cfg["role_inputs"]:
        prior = read(a.pin(role["prior_evaluation_audit"]))
        a.require(prior.get("status") == "PASS" and prior.get("stage") == "evaluation", "Original input evaluation audit PASS")
        pl_path = a.pin(role["prediction_lock"])
        pl = read(pl_path)
        a.require(pl["role"] == role["role"], "Original prediction role")
        by_id = {x["id"]: x for x in pl["records"]}
        a.require(list(by_id) == role["ids"], "Original prediction profile coverage")
        for entry in role["records"]:
            fp = a.pin(entry["features"]); tp = a.pin(entry["truth"])
            for pinned_path in (fp, tp, pl_path):
                suffix = pinned_path.relative_to(ROOT).as_posix()
                matches = [value for key, value in prior["inputs"].items() if key.replace("\\", "/").endswith("/"+suffix)]
                a.require(bool(matches) and all(value == a._pins[pinned_path] for value in matches),
                          "Original full audit binds reused input " + suffix)
            source = a.arrays(fp); truth = a.arrays(tp)
            n = entry["rows"]
            ix = np.arange(300, n, dtype=np.int64)
            a.require(np.array_equal(source["index"], ix) and np.array_equal(truth["index"], ix), "Exact 300: source/truth indices")
            a.require(truth["target"].shape == (len(ix),), "Aligned scalar truth")
            for key, d in (("boundary", 10), ("heat", 15), ("instant", 8)):
                a.require(source[key].shape == (n, d), "Original feature matrix geometry " + key)
            a.require(source["source_row"].shape == (n,), "Whole-profile source-row identity")
            a.require(by_id[entry["id"]]["sha256"] == entry["features"]["sha256"], "Prediction-lock feature bytes")
            records[(role["role"], entry["id"])] = dict(source=source, truth=truth["target"].astype(np.float64),
                                                        index=ix, source_row=source["source_row"][ix])
    return records


def summary_files(a, folder, expected_summary_sha):
    sp = a.pin({"path": "summary.json", "sha256": expected_summary_sha}, folder)
    summary = read(sp)
    files = summary["files"]
    a.require(isinstance(files, dict) and len(files) > 0, "Nonempty completed artifact map")
    for name, value in files.items():
        a.pin({"path": name, "sha256": value}, folder)
    return summary, files


def high_threshold_only(a, cfg):
    labels = []
    entries = {entry["id"]: entry for entry in cfg["fit_cache"]}
    a.require(set(entries) == set(cfg["train_ids"]), "Threshold exact FIT coverage")
    for pid in cfg["train_ids"]:
        entry = entries[pid]
        file = a.pin(entry)
        with np.load(file, allow_pickle=False) as saved:
            target = saved[entry["target_key"]].astype(np.float64)
        a.require(target.shape == (entry["rows"],) and np.isfinite(target).all(), "Threshold FIT target shape")
        labels.append(target[300::5])
    return float(np.quantile(np.concatenate(labels), .95, method="linear"))


def audit_analysis(a, args, cfg, folder):
    summary, files = summary_files(a, folder, args.input_sha256)
    a.require(summary["status"] == "EXPLORATORY_RISK_ANALYSIS_COMPLETE", "Completed risk analysis")
    a.require(summary["dataset"] == cfg["dataset"] and summary["independent_confirmation"] is False,
              "Analysis dataset and non-confirmatory scope")
    a.require({"component-inputs.json", "settings.json", "metrics.json", "policies.json", "configuration.json"} <= set(files),
              "Required analysis artifacts")
    inputs = read(folder/"component-inputs.json")
    a.require(inputs["dataset"] == cfg["dataset"] and inputs["config"]["sha256"] == args.config_sha256,
              "Component manifest config identity")
    a.require(a.pin(inputs["config"]) == path_in(ROOT, args.config), "Same explicit config path")
    configuration = read(folder/"configuration.json")
    original_components = a.pin({"path": configuration["components"], "sha256": configuration["components_sha256"]})
    a.same(read(original_components), inputs, "Exact copied source component manifest")
    a.require(summary.get("source_sha256", args.producer_sha256) == args.producer_sha256, "Analysis producer identity")
    threshold = high_threshold_only(a, cfg)
    a.same(summary["high_temperature_threshold_c"], threshold, "Training-only high-temperature threshold")
    source = existing_role_records(a, cfg)
    roles = [r["role"] for r in cfg["role_inputs"]]
    role_ids = {r["role"]: r["ids"] for r in cfg["role_inputs"]}
    components, metadata = {}, {}
    neural_audits = {}
    expected_coverage = [(r, p) for r in roles for p in role_ids[r]]
    for c in inputs["components"]:
        cid = c["id"]
        a.require(cid not in components, "Unique component identity")
        a.require(c["backbone"] in ("mlp", "hgb", "context"), "Known component family")
        if c["backbone"] == "hgb":
            a.require("source_receipt" in c, "Tree component has exact producer receipt")
            hp = a.pin(c["source_receipt"])
            hs = read(hp)
            a.require(hs["status"] == "HGB_FIT_PREDICTIONS_LOCKED" and hs["dataset"] == cfg["dataset"]
                      and hs["config_sha256"] == args.config_sha256 and hs["loss"] == c["loss"]
                      and hs["seed"] == c["seed"], "HGB exact dataset/config/loss/seed receipt")
            a.pin(dict(path="followup-risk-20261110/code/risk_tree_v001.py", sha256=hs["source_sha256"]))
            for name, expected in hs["files"].items():
                a.pin(dict(path=name, sha256=expected), hp.parent)
            a.same(c["predictions"], hs["predictions"], "Component points to exact tree prediction collection")
            hlock = read(hp.parent/"prediction_lock.json")
            a.require(hlock["config_sha256"] == args.config_sha256 and hlock["truth_accessed"] is False,
                      "HGB prediction lock config/no-target-decode")
            a.same(hlock["predictions"], c["predictions"], "HGB locked prediction collection")
        a.require([(p["role"], p["profile"]) for p in c["predictions"]] == expected_coverage,
                  "Exact complete component profile/role coverage " + cid)
        component = {r: {} for r in roles}
        for entry in c["predictions"]:
            path = a.pin(entry)
            arrays = a.arrays(path)
            key = (entry["role"], entry["profile"])
            index_key = entry.get("index_key", "index")
            prediction_key = entry.get("prediction_key", "prediction")
            a.require(np.array_equal(arrays[index_key], source[key]["index"]), "Component index/truth identity " + cid)
            prediction = arrays[prediction_key].astype(np.float64)
            a.require(prediction.shape == source[key]["truth"].shape, "Component scalar prediction shape " + cid)
            if "source_row" in arrays:
                actual_source = arrays["source_row"]
                if len(actual_source) != len(source[key]["source_row"]):
                    actual_source = actual_source[source[key]["index"]]
                a.require(np.array_equal(actual_source, source[key]["source_row"]), "Component raw row lineage " + cid)
            component[key[0]][key[1]] = prediction
            if c["backbone"] == "context":
                a.require(prediction_key in ("prediction__direct9_fitcv_selected", "prediction__positive_rc_insample_boost",
                                             "prediction__linear33_insample_boost"), "Predeclared contextual model identity")
                a.require(np.array_equal(prediction, source[key]["source"][prediction_key]), "Contextual saved predictions unchanged")
        if c["backbone"] == "mlp":
            a.require("audit" in c, "MLP component has independent forward audit " + cid)
            report_path = a.pin(c["audit"])
            report = read(report_path)
            a.require(report.get("status") == "PASS" and report.get("mode") == "mlp"
                      and report.get("config_sha256") == args.config_sha256
                      and report.get("auditor_sha256") == sha(__file__), "Bound MLP forward audit PASS " + cid)
            for entry in c["predictions"]:
                suffix = entry["path"].replace("\\", "/")
                matches = [v for k, v in report["inputs"].items() if k.replace("\\", "/").endswith("/"+suffix)]
                a.require(matches and all(v == entry["sha256"] for v in matches), "MLP audit covers component predictions " + cid)
            a.require(report["results"]["loss"] == c["loss"] and report["results"]["seed"] == c["seed"], "MLP audit loss/seed identity")
            neural_audits[cid] = report
        metadata[cid], components[cid] = c, component
    neural_seeds = sorted({c["seed"] for c in metadata.values() if c["backbone"] == "mlp"})
    a.require(neural_seeds in ([], [2026111701], cfg["training"]["seeds"]),
              "Tree-only control stage, complete initial neural seed, or fixed three-seed panel")
    for seed in neural_seeds:
        group = [c for c in metadata.values() if c["backbone"] == "mlp" and c["seed"] == seed]
        a.require(len(group) == 3 and {c["loss"] for c in group} == {"mse", "asym4", "q90"}, "All three paired MLP losses")
        for field in ("initial_array_sha256", "training_array_sha256", "permutation_digests"):
            values = [neural_audits[c["id"]]["results"][field] for c in group]
            a.require(all(v == values[0] for v in values), "Matched MLP " + field + "/" + str(seed))
    trees = [c for c in metadata.values() if c["backbone"] == "hgb"]
    a.require(len(trees) == 2 and {c["loss"] for c in trees} == {"mse", "q90"}
              and all(c["seed"] == 2026111701 for c in trees), "Both fixed tree controls")
    contextual = [c for c in metadata.values() if c["backbone"] == "context"]
    a.require(len(contextual) == 3 and all(c["loss"] == "context" and c["seed"] is None for c in contextual),
              "Three unselected contextual controls")
    a.require({c["predictions"][0].get("prediction_key") for c in contextual}
              == {"prediction__direct9_fitcv_selected", "prediction__positive_rc_insample_boost", "prediction__linear33_insample_boost"},
              "Exact contextual model panel")
    for c in contextual:
        a.require(len({p.get("prediction_key") for p in c["predictions"]}) == 1, "Context model identity stays fixed across profiles")

    def profiles(cid, role):
        return [(pid, source[(role, pid)]["truth"], components[cid][role][pid], source[(role, pid)]["index"])
                for pid in role_ids[role]]

    settings, metric, groups = [], {}, []
    cal = cfg["calibration_role"]
    interval = cfg["metrics"]["sample_interval_seconds"]

    def add(setting):
        settings.append(setting)
        metric[setting["id"]] = {}
        for role in roles:
            values = []
            for pid, target, base, index in profiles(setting["base"], role):
                if setting["other"] is None:
                    prediction = base + (setting["shift"] or 0.)
                else:
                    prediction = ((1.-setting["weight"])*base
                                  + setting["weight"]*components[setting["other"]][role][pid])
                values.append((pid, target, prediction, index))
            metric[setting["id"]][role] = group_metrics(values, interval, threshold)
            a.budget()

    for c in inputs["components"]:
        add(dict(id=c["id"]+"__raw", family="context" if c["backbone"] == "context" else "unadjusted",
                 base=c["id"], other=None, weight=None, shift=0., backbone=c["backbone"], seed=c["seed"]))
    for cid, c in metadata.items():
        if c["loss"] != "mse" or c["backbone"] == "context":
            continue
        group = dict(backbone=c["backbone"], seed=c["seed"], base_setting_id=cid+"__raw", families={})
        shifts = list(cfg["calibration"]["shift_grid_c"])
        shifts += [max(0., balanced_residual_quantile(profiles(cid, cal), q))
                   for q in cfg["calibration"]["shift_residual_quantiles"]]
        shifts = list(dict.fromkeys(shifts))
        family_ids = []
        for j, shift in enumerate(shifts):
            sid = cid+"__shift_"+str(j); family_ids.append(sid)
            add(dict(id=sid, family="shift", base=cid, other=None, weight=None, shift=float(shift), backbone=c["backbone"], seed=c["seed"]))
        group["families"]["shift"] = family_ids
        for loss, name in (("asym4", "asym_mix"), ("q90", "quantile_mix")):
            matches = [other_id for other_id, other in metadata.items() if other["backbone"] == c["backbone"]
                       and other["seed"] == c["seed"] and other["loss"] == loss]
            if not matches:
                a.require(c["backbone"] == "hgb" and loss == "asym4", "Only missing branch is unregistered asymmetric tree")
                continue
            a.require(len(matches) == 1, "Unique paired component")
            family_ids = []
            for j, weight in enumerate(cfg["calibration"]["mix_weights"]):
                sid = cid+"__"+name+"_"+str(j); family_ids.append(sid)
                add(dict(id=sid, family=name, base=cid, other=matches[0], weight=float(weight), shift=None,
                         backbone=c["backbone"], seed=c["seed"]))
            group["families"][name] = family_ids
        groups.append(group)
    a.same(read(folder/"settings.json"), settings, "All predeclared settings, including calibration residual shifts")
    expected_metrics = dict(dataset=cfg["dataset"], high_temperature_threshold_c=threshold, settings=metric)
    a.same(read(folder/"metrics.json"), expected_metrics, "Independent full risk metrics")
    policies = []
    for group in groups:
        baseline = metric[group["base_setting_id"]][cal]["summary"]
        for rmse_ratio in cfg["calibration"]["rmse_budget_ratios"]:
            for over_ratio in cfg["calibration"]["over_mae_budget_ratios"]:
                for family, names in group["families"].items():
                    eligible = []
                    for pos, sid in enumerate(names):
                        value = metric[sid][cal]["summary"]
                        if (value["macro_rmse"] <= rmse_ratio*baseline["macro_rmse"] + 1e-12
                                and value["macro_over_mae"] <= over_ratio*baseline["macro_over_mae"] + 1e-12):
                            eligible.append((value["macro_under_rmse"], value["macro_rmse"], value["macro_over_mae"], pos, sid))
                    a.require(bool(eligible), "Zero adjustment retains a feasible candidate")
                    chosen = min(eligible)[-1]
                    violations = {}
                    for role in roles:
                        base = metric[group["base_setting_id"]][role]["summary"]
                        selected = metric[chosen][role]["summary"]
                        violations[role] = dict(rmse=selected["macro_rmse"] > rmse_ratio*base["macro_rmse"]+1e-12,
                                                over_mae=selected["macro_over_mae"] > over_ratio*base["macro_over_mae"]+1e-12)
                    policies.append(dict(backbone=group["backbone"], seed=group["seed"], family=family,
                                         rmse_ratio=rmse_ratio, over_mae_ratio=over_ratio,
                                         selected_setting_id=chosen, base_setting_id=group["base_setting_id"],
                                         calibration_role=cal, evaluation_budget_violations=violations))
    a.same(read(folder/"policies.json"), policies, "Calibration-only policy selection and evaluation violations")
    a.require(summary["settings"] == len(settings) and summary["policies"] == len(policies), "All declared settings and policy counts")
    a.results = dict(dataset=cfg["dataset"], settings=len(settings), policies=len(policies),
                     evaluated_roles=roles, neural_seeds=neural_seeds, high_temperature_threshold_c=threshold,
                     panel_scope="tree_controls_only" if not neural_seeds else "single_seed_screen" if len(neural_seeds) == 1 else "three_seed_panel",
                     all_evaluation_previously_exposed=True, independent_confirmation=False,
                     primary_budget_policies=[p for p in policies if p["rmse_ratio"] == cfg["calibration"]["primary_rmse_ratio"]
                                              and p["over_mae_ratio"] == cfg["calibration"]["primary_over_mae_ratio"]],
                     hgb_scope="Saved prediction/lineage/metric checks; no pickle execution, tree replay or retraining.")


def audit_mlp(a, args, cfg, folder):
    a.require(not (folder/"failure.json").exists(), "A failed MLP stage cannot be admitted by its earlier COMPLETE receipt")
    stage_path = a.pin(dict(path="stage_record.json", sha256=args.input_sha256), folder)
    stage = read(stage_path)
    a.require(stage.get("schema") == "c16-risk-mlp-stage-v001" and stage.get("status") == "COMPLETE",
              "Completed immutable neural stage")
    files = stage["files"]
    for name, expected in files.items():
        a.pin(dict(path=name, sha256=expected), folder)
    a.require({"configuration.json", "runtime.json", "summary.json", "model_lock.json", "prediction_lock.json",
               "model.json", "weights_initial.npz", "weights_final.npz", "training_arrays.npz", "epoch_trace.json"} <= set(files),
              "Complete training and prediction artifacts; probes are not comparators")
    summary = read(folder/"summary.json")
    a.same(stage["result"], summary, "Stage receipt result identity")
    configuration = read(folder/"configuration.json")
    a.same(stage["configuration"], configuration, "Stage receipt configuration identity")
    a.require(configuration["mode"] == "train", "Audit complete train+predict stage, not probe or prediction-only stage")
    a.require(configuration["source_sha256"] == args.producer_sha256
              and configuration["config_sha256"] == args.config_sha256, "Registered MLP source and config identities")
    a.require(summary["status"] == "MLP_PREDICTIONS_LOCKED" and summary["completed_comparator"] is True
              and summary["truth_accessed"] is False, "Qualified complete MLP comparator")
    runtime = read(folder/"runtime.json")
    a.require(runtime["tf32_matmul_allowed"] is False and runtime["tf32_cudnn_allowed"] is False
              and runtime["deterministic_algorithms"] is True, "Predeclared floating-point runtime contract")
    a.require(summary["source_sha256"] == args.producer_sha256 and summary["config_sha256"] == args.config_sha256,
              "Summary exact producer/config pins")
    model_path = a.pin(summary["model_lock"])
    a.require(model_path == folder/"model_lock.json", "Model lock belongs to audited training stage")
    model_lock = read(model_path)
    a.require(model_lock["schema"] == "c16-risk-mlp-model-lock-v001" and model_lock["status"] == "MODEL_LOCKED"
              and model_lock["completed_comparator"] is True and model_lock["source_sha256"] == args.producer_sha256
              and model_lock["config_sha256"] == args.config_sha256, "Frozen same-source MLP model lock")
    model_files = {"model.json", "weights_initial.npz", "weights_final.npz", "training_arrays.npz", "epoch_trace.json",
                   "configuration.json", "runtime.json"}
    a.require(set(model_lock["files"]) == model_files, "Complete model lock file set")
    for name, expected in model_lock["files"].items():
        a.require(files[name] == expected, "Stage/model lock hash identity " + name)
    a.require(a.pin(model_lock["model"]) == folder/"model.json", "Model lock points to exact metadata")
    model = read(folder/"model.json")
    a.require(model["schema"] == "c16-risk-mlp-model-v001" and model["dataset"] == cfg["dataset"], "Model dataset/schema")
    a.require(model["completed_comparator"] is True and model["evaluation_truth_accessed"] is False,
              "Model training excludes evaluation target access")
    a.require(model["fit_ids"] == cfg["train_ids"] and model["epochs"] == 200, "Full frozen FIT/epochs")
    a.require(model["loss"] in cfg["training"]["losses"] and model["seed"] in cfg["training"]["seeds"], "Frozen loss/seed membership")
    a.require(model["loss"] == configuration["loss"] == summary["loss"]
              and model["seed"] == configuration["seed"] == summary["seed"], "Loss/seed receipt chain")
    a.require(model["source_sha256"] == args.producer_sha256 and model["config_sha256"] == args.config_sha256,
              "Model source/config chain")
    a.require(model["core_sha256"] == cfg["dependency_sources"]["c15_neural_rc_core_v001.py"]["sha256"]
              and model["shared_worker_sha256"] == cfg["dependency_sources"]["c15_paper_mlp_v001.py"]["sha256"],
              "Exact inherited scientific dependency identities")
    settings = model["settings"]
    for key, value in cfg["training"].items():
        a.same(settings[key], value, "Frozen MLP setting " + key)
    for key, value in dict(parameters=4289, activation="ReLU", optimizer="AdamW", betas=[.9,.999],
                           eps=1e-8, gradient_norm_cap=1., feature_std_floor=1e-8,
                           target_iqr_floor=1., inference_batch_size=4096).items():
        a.same(settings[key], value, "Fixed numerical setting " + key, atol=0., rtol=0.)
    a.same(configuration["settings"], settings, "Configuration/model training settings")

    fit, tr, expected_training, threshold = fit_reference(a, cfg)
    a.same(model["transform"], tr, "Independently reconstructed FIT-only transforms", atol=1e-12, rtol=1e-12)
    a.same(model["high_temperature_threshold_c"], threshold, "FIT-only high-temperature threshold")
    a.same(summary["high_temperature_threshold_c"], threshold, "Summary FIT threshold")
    a.same(summary["transform"], model["transform"], "Summary/model transform identity")
    # Using a separately qualified saved transform preserves explicit FP32 input
    # quantization even at a rounding boundary. The transform itself is checked
    # independently above; no result-dependent prediction tolerance is used.
    saved_tr = model["transform"]
    expected_training["x"] = np.concatenate([
        ((feature_matrix(fit[pid], saved_tr["rc_scale"]["qscale"])[np.arange(300, len(fit[pid]["target"]), 5)]
          - np.asarray(saved_tr["input_mean"]))/np.asarray(saved_tr["input_std"])).astype(np.float32)
        for pid in cfg["train_ids"]])
    expected_training["y"] = np.concatenate([
        ((fit[pid]["target"][300::5]-saved_tr["y_mid"])/saved_tr["y_scale"]).astype(np.float32)
        for pid in cfg["train_ids"]])
    expected_arrays = dict(x=expected_training["x"], y_normalized=expected_training["y"], w=expected_training["w"],
                           profile_ids=expected_training["profile"], index=expected_training["index"])
    arrays = a.arrays(folder/"training_arrays.npz")
    a.require(set(arrays) == set(expected_arrays), "Exact training array keys")
    for key in expected_arrays:
        a.require(arrays[key].dtype == expected_arrays[key].dtype and np.array_equal(arrays[key], expected_arrays[key]),
                  "Independently reconstructed training array " + key)
    n = len(arrays["index"])
    a.require(n == model["training_rows"] == summary["training_rows"], "Exact training endpoint count")
    counts = {str(pid): int(np.sum(arrays["profile_ids"] == pid)) for pid in cfg["train_ids"]}
    a.same(model["training_profile_counts"], counts, "FIT per-profile endpoint counts")
    a.require(array_digest(arrays) == model["training_arrays_sha256"], "Saved training arrays digest")
    a.require(array_digest({k: arrays[k] for k in ("x", "profile_ids", "index")}) == model["feature_array_sha256"],
              "Saved feature arrays digest")
    initial = a.arrays(folder/"weights_initial.npz"); final = a.arrays(folder/"weights_final.npz")
    weight_geometry(a, initial, "initial"); weight_geometry(a, final, "final")
    a.require(array_digest(initial) == model["initial_array_sha256"] and array_digest(final) == model["final_array_sha256"],
              "Initial/final neural parameter digests")
    trace = read(folder/"epoch_trace.json")
    a.require(len(trace) == 200 and [r["epoch"] for r in trace] == list(range(1, 201)), "Complete fixed 200-epoch trace")
    permutations = []
    for row in trace:
        a.budget()
        order = np.random.Generator(np.random.PCG64(model["seed"]+row["epoch"]-1)).permutation(n).astype(np.int64)
        digest = array_digest(dict(order=order))
        a.require(row["permutation_sha256"] == digest, "Independent epoch permutation digest " + str(row["epoch"]))
        permutations.append(digest)
        a.require(all(isinstance(row[k], (int, float)) and math.isfinite(row[k]) and row[k] >= 0
                      for k in ("online_weighted_loss", "max_preclip_gradient_norm", "seconds")), "Finite nonnegative trace diagnostics")
    a.same(model["training_seconds"], sum(r["seconds"] for r in trace), "Training trace seconds sum")
    a.same(summary["training_seconds"], model["training_seconds"], "Summary training duration")

    pl_path = a.pin(summary["prediction_lock"])
    a.require(pl_path == folder/"prediction_lock.json", "Same-stage prediction lock")
    pl = read(pl_path)
    a.require(pl["schema"] == "c16-risk-mlp-prediction-lock-v001" and pl["status"] == "ALL_PREDICTIONS_LOCKED"
              and pl["truth_accessed"] is False and pl["all_evaluation_previously_exposed"] is True,
              "Locked predictions have exploratory exposure scope")
    for key in ("dataset", "loss", "seed", "source_sha256", "config_sha256"):
        a.require(pl[key] == model[key], "Prediction/model identity " + key)
    a.require(a.pin(pl["model_lock"]) == model_path, "Prediction lock binds same final model lock")
    a.same(summary["predictions"], pl["predictions"], "Summary prediction records identity")
    expected = [(role["role"], entry["id"]) for role in cfg["role_inputs"] for entry in role["records"]]
    a.require([(entry["role"], entry["profile"]) for entry in pl["predictions"]] == expected, "Every fixed exploratory/calibration prediction")
    role_config = {r["role"]: r for r in cfg["role_inputs"]}
    sources = existing_role_records(a, cfg)
    metric_profiles = {r["role"]: [] for r in cfg["role_inputs"]}
    for entry in pl["predictions"]:
        key = (entry["role"], entry["profile"])
        c = next(x for x in role_config[key[0]]["records"] if x["id"] == key[1])
        a.same(entry["source_features"], c["features"], "Exact source feature reference")
        a.require(entry["usage"] == role_config[key[0]]["usage"] and entry["rows"] == c["rows"], "Prediction role/row declarations")
        pp = a.pin(entry)
        a.require(pp.is_relative_to(folder/"predictions"), "New predictions contained in this stage")
        saved = a.arrays(pp)
        a.require(set(saved) == {"index", "source_row", "features", "prediction"}, "Exact saved prediction fields")
        record = sources[key]
        a.require(saved["index"].dtype == np.int64 and saved["source_row"].dtype == np.int64,
                  "Explicit portable int64 prediction identifiers")
        a.require(np.array_equal(saved["index"], record["index"]) and np.array_equal(saved["source_row"], record["source_row"]),
                  "Prediction/truth/local/raw row correspondence")
        a.require(entry["evaluated_rows"] == len(record["index"]) and saved["prediction"].dtype == np.float64,
                  "Every evaluation endpoint has FP64 physical-unit prediction")
        xx = ((feature_matrix(record["source"], saved_tr["rc_scale"]["qscale"])[record["index"]]
               - np.asarray(saved_tr["input_mean"]))/np.asarray(saved_tr["input_std"])).astype(np.float32)
        a.require(saved["features"].dtype == np.float32 and np.array_equal(saved["features"], xx), "Exact quantized saved input features")
        a.require(array_digest(saved) == entry["arrays_sha256"], "Prediction arrays digest")
        a.require(array_digest({k: saved[k] for k in ("features", "index", "source_row")}) == entry["feature_array_sha256"],
                  "Prediction feature digest")
        a.require(array_digest({k: saved[k] for k in ("prediction", "index", "source_row")}) == entry["prediction_array_sha256"],
                  "Prediction output digest")
        verify_forward(a, record["source"], saved_tr, final, record["index"], saved["prediction"], key[0]+"/"+str(key[1]))
        metric_profiles[key[0]].append((key[1], record["truth"], saved["prediction"], record["index"]))
    a.results = dict(dataset=cfg["dataset"], loss=model["loss"], seed=model["seed"], epochs=200,
                     initial_array_sha256=model["initial_array_sha256"],
                     training_array_sha256=model["training_arrays_sha256"], permutation_digests=permutations,
                     training_rows=n, high_temperature_threshold_c=threshold,
                     model_lock_sha256=sha(model_path), prediction_lock_sha256=sha(pl_path),
                     metrics={role: group_metrics(rows, cfg["metrics"]["sample_interval_seconds"], threshold)
                              for role, rows in metric_profiles.items()},
                     all_evaluation_previously_exposed=True, independent_confirmation=False)
    a.require(not (folder/"failure.json").exists(), "MLP stage remains free of a terminal failure marker")


def synthetic_self_test():
    """Small independent arithmetic fixtures only; no campaign data access."""
    ix = np.arange(300, 304, dtype=np.int64)
    y = np.array([10., 10., 10., 10.]); p = np.array([4., 4., 10., 4.])
    m = metric_values(y, p, ix, .5)
    assert m["under_mae"] == 4.5 and m["over_mae"] == 0
    assert np.isclose(m["under_rmse"], np.sqrt(27.)) and m["under_gt5_longest_samples"] == 2
    assert m["under_gt5_longest_seconds"] == 1.
    assert metric_values(y, p, ix, None)["under_gt5_longest_seconds"] is None
    high = metric_values(np.array([10., 0., 10.]), np.zeros(3), np.arange(3), .5, np.array([True, False, True]))
    assert high["n"] == 2 and high["under_gt5_longest_samples"] == 1
    assert longest_run(np.array([True, True]), np.array([300, 302])) == 1
    zeros = metric_values(np.zeros(100), np.ones(100), np.arange(100), .5)
    assert zeros["under_q95"] == zeros["under_q99"] == zeros["under_rmse"] == 0.
    empty = group_metrics([(1, np.zeros(3), np.zeros(3), np.arange(3))], None, 1.)
    assert empty["summary"]["high_temperature"]["excluded_profile_ids"] == [1]
    assert empty["summary"]["high_temperature"]["pooled_rmse"] is None
    return {"status": "SYNTHETIC_ARITHMETIC_PASS", "scope": "Constructed arrays only; no scientific evidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mlp", "analysis"))
    parser.add_argument("--config")
    parser.add_argument("--config-sha256")
    parser.add_argument("--input-dir")
    parser.add_argument("--input-sha256", help="MLP stage_record.json or analysis summary.json SHA")
    parser.add_argument("--producer-sha256")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-seconds", type=float, default=900.)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(synthetic_self_test()))
        return
    for name in ("mode", "config", "config_sha256", "input_dir", "input_sha256", "producer_sha256", "output_dir"):
        if not getattr(args, name):
            parser.error("Required argument --" + name.replace("_", "-"))
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 3600:
        parser.error("Audit cap must lie in (0,3600]")
    folder = path_in(ROOT, args.input_dir)
    out = path_in(ROOT, args.output_dir)
    if not folder.is_relative_to(FOLLOWUP/"runs") or not out.is_relative_to(FOLLOWUP/"runs"):
        raise ValueError("Input/output stage must remain below followup runs")
    out.mkdir(parents=True, exist_ok=False)
    a = Audit(args.max_seconds)
    error = None
    try:
        producer = FOLLOWUP/"code"/("risk_mlp_v001.py" if args.mode == "mlp" else "risk_screen_analysis_v001.py")
        a.pin(dict(path=str(producer.relative_to(ROOT)), sha256=args.producer_sha256))
        cfg = load_config(a, args)
        if args.mode == "mlp":
            audit_mlp(a, args, cfg, folder)
        else:
            audit_analysis(a, args, cfg, folder)
        a.finish_pins()
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        a.failures.append(dict(check="Audit execution/required contract", detail=error))
    report = dict(schema="c16-risk-independent-audit-v001", status="HOLD" if a.failures else "PASS", mode=args.mode,
                  config_sha256=args.config_sha256, input_sha256=args.input_sha256,
                  auditor_sha256=sha(__file__), producer_sha256=args.producer_sha256,
                  checks=a.checks, failures=a.failures, inputs=a.inputs, replayed=a.replayed, results=a.results,
                  elapsed_seconds=time.monotonic()-a.start,
                  scope="Saved-artifact lineage, independently reconstructed transforms/metrics and declared calibration rules; "
                        "MLP mode additionally replays saved weights with NumPy. No training, optimizer replay, new hypothesis tests, "
                        "HGB pickle execution or fresh confirmation. Cached causal features reuse prior pinned full-audit lineage.")
    with (out/"report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report["status"], checks=a.checks, failures=len(a.failures), report=str(out/"report.json"))))
    if error or a.failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
