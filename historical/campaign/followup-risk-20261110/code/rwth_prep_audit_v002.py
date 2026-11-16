"""Independent, FIT-only raw-to-cache audit for the frozen RWTH replication.

No scientific producer import, model fit, CAL/CONF member access, or stator parse.
Later auditors may reuse this independently authored raw/feature oracle.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP = ROOT / "followup-risk-20261110"
PROTOCOL_SHA = "f1d201fe4aa6243f8c77c6a8a85a91268d183d0cd24d48bc35ec86b269082b0f"
PARTITION_SHA = "5d5117312eee65bd9811f0ede7326d01d8a0a1aa3d0b0e3fb6ef60ae47959be1"
ARCHIVE_SHA = "a6b219a17acbbc2a9899fa7d750e1b0531756eeefcf16efd56009c43a0c4fce6"
AUDIT_DEP_SHA = "2cfb48953741515e8d9815073075bcc3e46c717c5110837d280acdb93a2bd035"
SPANS = (16, 64, 256, 1024, 4096)
ROLES = {"fit": [38,27,30,23,21,19,29,20,26,37,6,8,35,34,4,15,7,13,18,1],
         "calibration": [10,11,22,9,16,31,28,17],
         "confirmation": [5,32,33,36,24,2,25,14,3,12]}
HEADERS = ["time", "i", "torque", "motor_speed", "coolant", "rotor", "stator", "profile_id"]
DRIVES = ["coolant", "i_squared", "abs_omega", "omega_squared", "abs_mechanical_power"]
FEATURE_NAMES = ["i", "omega", "torque", "coolant", "i_squared", "abs_omega",
                 "omega_squared", "abs_mechanical_power"] + [d+"__span"+str(s) for d in DRIVES for s in SPANS]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def independent_dependency():
    path = Path(__file__).with_name("risk_audit_v001.py")
    if digest(path) != AUDIT_DEP_SHA:
        raise ValueError("Frozen independent audit dependency changed")
    spec = importlib.util.spec_from_file_location("rwth_independent_risk_audit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


I = independent_dependency()


def close_array(a, actual, expected, label, atol=1e-9, rtol=2e-12, exact=False):
    x, y = np.asarray(actual), np.asarray(expected)
    a.require(x.shape == y.shape, label + " shape")
    finite = np.isfinite(x).all() and np.isfinite(y).all()
    equal = np.array_equal(x, y) if exact else np.allclose(x, y, atol=atol, rtol=rtol)
    a.check(finite and equal, label, {"max_absolute_error": float(np.max(np.abs(x-y))) if x.size else 0.})


def raw_profile(a, archive, member, pid, include_target):
    """Only the caller-admitted member is decoded; rotor is optional, stator never."""
    a.budget()
    payload = archive.read(member)
    reader = csv.reader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
    a.require(next(reader, None) == HEADERS, "Raw CSV exact column schema")
    rows = []
    targets = []
    for offset, row in enumerate(reader):
        if offset % 2000 == 0:
            a.budget()
        a.require(len(row) == 8, "Raw CSV row width")
        value = float(row[7])
        a.require(math.isfinite(value) and value == pid, "Raw profile identity")
        # time,current,rpm,torque,coolant. Neither temperature column enters X.
        rows.append([float(row[j]) for j in (0, 1, 3, 2, 4)])
        if include_target:
            targets.append(float(row[5]))
    values = np.asarray(rows, dtype=np.float64)
    a.require(values.ndim == 2 and values.shape[1] == 5 and len(values) > 0,
              "Raw nonempty profile retained without warmup")
    a.require(np.isfinite(values).all(), "Raw finite observed inputs/time")
    a.require(np.all(np.diff(values[:, 0]) > 0), "Raw timestamps strictly increasing")
    result = {"time": values[:, 0], "operating": values[:, 1:],
              "csv_sha256": hashlib.sha256(payload).hexdigest(), "rows": len(values)}
    if include_target:
        result["target"] = np.asarray(targets, dtype=np.float64)
        a.require(np.isfinite(result["target"]).all(), "Finite FIT rotor")
    return result


def causal_features(time_values, operating, dt_ref):
    """Independent channel/span recursion: no access to labels or profile IDs."""
    t, op = np.asarray(time_values, dtype=np.float64), np.asarray(operating, dtype=np.float64)
    if not (t.ndim == 1 and op.shape == (len(t), 4) and len(t) > 0
            and np.isfinite(t).all() and np.isfinite(op).all()
            and np.all(np.diff(t) > 0) and math.isfinite(dt_ref) and dt_ref > 0):
        raise ValueError("Invalid causal-feature input")
    current, rpm, torque, coolant = op.T
    omega = rpm * (2. * math.pi / 60.)
    instantaneous = np.column_stack((current, omega, torque, coolant,
                                    current**2, np.abs(omega), omega**2,
                                    np.abs(torque * omega)))
    drives = np.column_stack((coolant, instantaneous[:, 4:]))
    states = np.empty((len(t), 25), dtype=np.float64)
    elapsed = np.diff(t) / dt_ref
    for channel in range(5):
        for si, span in enumerate(SPANS):
            column = 5*channel + si
            states[0, column] = coolant[0] if channel == 0 else 0.
            alphas = -np.expm1(np.log1p(-2./(span+1.)) * elapsed)
            for row, alpha in enumerate(alphas, start=1):
                states[row, column] = (1.-alpha)*states[row-1, column] + alpha*drives[row, channel]
    result = np.column_stack((instantaneous, states))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite reconstructed feature")
    return result


def fit_statistics(records, ids):
    """Prospective shared oracle: pooled FIT0::5 scales and profile weights."""
    xx, yy, profile, index, stamp = [], [], [], [], []
    for pid in ids:
        rec = records[pid]
        ix = np.arange(0, len(rec["time"]), 5, dtype=np.int64)
        xx.append(rec["X33"][ix]); yy.append(rec["target"][ix])
        profile.append(np.full(len(ix), pid, dtype=np.int64)); index.append(ix)
        stamp.append(rec["time"][ix])
    x, y = np.concatenate(xx), np.concatenate(yy)
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-8)
    mid = float(np.median(y)); scale = max(float(np.quantile(y, .75)-np.quantile(y, .25)), 1.)
    weights = np.concatenate([np.full(len(ix), len(y)/(len(ids)*len(ix)), dtype=np.float64) for ix in index])
    q = np.maximum(np.sqrt(np.mean(x[:, 4:8]**2, axis=0)), 1e-8)
    return {"x": x, "y": y, "input_mean": mean, "input_std": std,
            "y_mid": mid, "y_scale": scale, "qscale": q, "w": weights,
            "profile_ids": np.concatenate(profile), "index": np.concatenate(index),
            "time": np.concatenate(stamp), "fit_ids": list(ids)}


def load_fit_manifest(a, manifest_ref, producer_sha=None):
    mp = a.pin(manifest_ref)
    manifest = I.read(mp)
    a.require(manifest.get("schema") == "rwth-cache-manifest-v002", "RWTH cache schema")
    a.require(manifest.get("status") == "READY_FIT_ONLY", "Initial audit only FIT cache")
    a.require(manifest.get("dataset") == "rwth48v", "RWTH dataset identity")
    a.same(manifest["roles"], ROLES, "Frozen complete partition")
    a.require(manifest["protocol"]["sha256"] == PROTOCOL_SHA, "Frozen replication protocol")
    a.pin(manifest["protocol"])
    a.require(manifest["partition"]["sha256"] == PARTITION_SHA, "Frozen ID partition")
    partition = I.read(a.pin(manifest["partition"]))
    a.same(partition["roles"], ROLES, "Partition unchanged")
    a.require(manifest["archive"]["sha256"] == ARCHIVE_SHA, "Exact archive identity")
    archive_path = a.pin(manifest["archive"])
    a.require(manifest["archive"] == partition["archive"], "Same partition archive reference")
    if producer_sha is not None:
        a.pin({"path": str(Path(__file__).with_name("rwth_prepare_v002.py")), "sha256": producer_sha})
        a.require(manifest["source_sha256"] == producer_sha, "Manifest exact producer identity")
    a.same(manifest["feature_names"], FEATURE_NAMES, "Exact named feature order")
    a.same(manifest["operating_columns"], ["i", "motor_speed", "torque", "coolant"], "Exact observed input order")
    a.same(manifest["spans"], list(SPANS), "Frozen spans")
    for key, expected in (("warmup", 0), ("train_stride", 5), ("evaluation_stride", 1),
                          ("calibration_member_contents_opened", False), ("confirmation_member_contents_opened", False),
                          ("stator_numeric_values_parsed", False), ("confirmation_attempt_consumed", False),
                          ("time_unit_independently_verified", False), ("fitted_model_count", 0)):
        a.same(manifest[key], expected, "Preparation contract " + key)
    entries = manifest["records"]
    a.require([x["id"] for x in entries] == ROLES["fit"], "Exactly FIT20 in fixed order")
    members = {r["id"]: r["member"] for r in partition["ordered_records"]}
    source = {}
    with zipfile.ZipFile(archive_path) as archive:
        a.require(len(archive.namelist()) == len(set(archive.namelist())), "Unique archive member names")
        for entry in entries:
            pid = entry["id"]
            a.require(entry["role"] == "fit" and entry["member"] == members[pid],
                      "Only admitted FIT member")
            a.require(entry.get("truth") is None and entry.get("target_key") == "target",
                      "FIT target cache schema")
            source[pid] = raw_profile(a, archive, members[pid], pid, True)
            a.same(entry["rows"], source[pid]["rows"], "Unmodified source row count")
            a.same(entry["csv_sha256"], source[pid]["csv_sha256"], "Unmodified raw CSV member")
    a.same(manifest["opened_csv_members"], [members[p] for p in ROLES["fit"]], "Only FIT member access declared")
    dt_ref = float(np.median(np.concatenate([np.diff(source[p]["time"]) for p in ROLES["fit"]])))
    a.require(math.isfinite(dt_ref) and dt_ref > 0, "Positive FIT-pooled timestamp reference")
    a.same(manifest["dt_ref"], dt_ref, "FIT-only pooled timestamp median", atol=1e-13, rtol=1e-13)
    records = {}
    for entry in entries:
        pid = entry["id"]; raw = source[pid]; n = raw["rows"]
        cache = a.arrays(a.pin(entry["features"]))
        a.require(set(cache) == {"profile_id", "index", "source_row", "time", "operating", "X33", "target"},
                  "Exact FIT cache keys; no extra channels")
        a.require(cache["profile_id"].dtype == np.int64 and cache["profile_id"].shape == ()
                  and int(cache["profile_id"]) == pid, "Scalar int64 profile identity")
        for key in ("index", "source_row"):
            a.require(cache[key].dtype == np.int64, "Explicit int64 source index")
            close_array(a, cache[key], np.arange(n, dtype=np.int64), "All chronological rows " + key, exact=True)
        for key in ("time", "operating", "X33", "target"):
            a.require(cache[key].dtype == np.float64, "FP64 raw cache " + key)
        for key in ("time", "operating", "target"):
            close_array(a, cache[key], raw[key], "Raw lineage " + key, exact=True)
        features = causal_features(raw["time"], raw["operating"], dt_ref)
        close_array(a, cache["X33"], features, "Independent causal X33 order/initialization/recurrence")
        records[pid] = dict(raw, X33=features)
    a.results.update(dataset="rwth48v", stage="prep_fit", fit_profiles=ROLES["fit"],
                     fit_rows=sum(x["rows"] for x in source.values()), dt_ref=dt_ref,
                     raw_members_decoded=[members[p] for p in ROLES["fit"]],
                     calibration_members_decoded=[], confirmation_members_decoded=[],
                     stator_numerically_parsed=False, timestamp_unit="not independently verified",
                     scope="FIT raw lineage and causal feature reconstruction; no model or held-target audit")
    return manifest, records


def self_test():
    t = np.array([0., .5, 2., 2.75]); op = np.array([[2., -60., 3., 25.], [3., 0., -2., 26.],
                                                 [4., 120., -4., 27.], [5., 60., 2., 28.]])
    x = causal_features(t, op, 1.)
    assert x.shape == (4, 33) and np.array_equal(x[0, 8:13], np.repeat(25., 5))
    assert np.count_nonzero(x[0, 13:]) == 0
    assert x[0, 1] < 0 and x[0, 2] == 3.
    for end in (1, 2, 3):
        assert np.array_equal(causal_features(t[:end], op[:end], 1.), x[:end])
    alpha = 1.-(1.-2./17.)**.5
    assert abs(x[1, 13]-alpha*9.) < 1e-13
    assert np.allclose(causal_features(t*1000., op, 1000.), x, atol=1e-12, rtol=1e-14)
    changed = op.copy(); changed[-1] *= 100.
    assert np.array_equal(causal_features(t, changed, 1.)[:-1], x[:-1])
    print(json.dumps({"status": "PASS_SYNTHETIC_ONLY", "checks": ["initialization", "signed_inputs", "prefix_causality",
          "manual_one_step", "timestamp_unit_cancellation", "future_perturbation"]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir"); p.add_argument("--input-sha256")
    p.add_argument("--producer-sha256"); p.add_argument("--output-dir")
    p.add_argument("--max-seconds", type=float, default=600.)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test(); return 0
    if not all((args.input_dir, args.input_sha256, args.producer_sha256, args.output_dir)):
        p.error("input directory/summary hash, producer hash and new output directory required")
    if not 0 < args.max_seconds <= 1200:
        p.error("max-seconds must be in (0,1200]")
    output = I.path_in(ROOT, args.output_dir)
    if output.exists():
        raise FileExistsError("Audit output must be new")
    output.mkdir(parents=True)
    a = I.Audit(args.max_seconds)
    status = "HOLD"
    try:
        a.pin({"path": str(Path(__file__)), "sha256": digest(__file__)})
        a.pin({"path": str(Path(__file__).with_name("risk_audit_v001.py")), "sha256": AUDIT_DEP_SHA})
        folder = I.path_in(ROOT, args.input_dir)
        a.require(not (folder/"failure.json").exists(), "No producer failure marker")
        summary = I.read(a.pin({"path": str(folder/"summary.json"), "sha256": args.input_sha256}))
        a.require(summary.get("schema") == "rwth-preparation-summary-v002"
                  and summary.get("status") == "RWTH_FIT_CACHE_READY", "Completed preparation summary")
        a.require(summary["source_sha256"] == args.producer_sha256, "Summary producer identity")
        a.require(I.path_in(ROOT, summary["manifest"]["path"]) == folder/"manifest.json", "Canonical manifest location")
        for name, expected in summary["files"].items():
            path = I.path_in(folder, name)
            a.pin({"path": str(path), "sha256": expected})
        a.require(summary["files"].get("manifest.json") == summary["manifest"]["sha256"], "Summary manifest binding")
        manifest, records = load_fit_manifest(a, summary["manifest"], args.producer_sha256)
        for entry in manifest["records"]:
            path = I.path_in(ROOT, entry["features"]["path"])
            a.require(path.is_relative_to(folder) and summary["files"].get(path.relative_to(folder).as_posix())
                      == entry["features"]["sha256"], "Every feature cache bound by completion summary")
        a.same(summary["roles_prepared"], ["fit"], "Only FIT role completed")
        a.same(summary["fit_profiles"], 20, "Twenty FIT profiles")
        a.same(summary["fit_rows"], sum(len(x["time"]) for x in records.values()), "Full FIT row total")
        a.same(summary["dt_ref"], manifest["dt_ref"], "Summary timestamp reference")
        for key, expected in (("calibration_member_contents_opened", False), ("confirmation_member_contents_opened", False),
                              ("stator_numeric_values_parsed", False), ("confirmation_attempt_consumed", False),
                              ("features_generated", True), ("models_fitted", False)):
            a.same(summary[key], expected, "Summary preparation contract " + key)
        pins = I.read(folder/"input_pins.json")
        a.same(pins, manifest["inputs"], "Producer input receipt")
        for name, expected in pins.items():
            a.pin({"path": name, "sha256": expected})
        a.finish_pins()
        a.require(not (folder/"failure.json").exists(), "No final producer failure marker")
        status = "PASS" if not a.failures else "HOLD"
    except Exception as exc:
        a.failures.append({"check": "Audit completed without exception", "detail": repr(exc)})
    report = {"status": status, "stage": "prep_fit", "auditor_sha256": digest(__file__),
              "producer_sha256": args.producer_sha256, "checks": a.checks, "failures": a.failures,
              "inputs": a.inputs, "results": a.results, "elapsed_seconds": time.monotonic()-a.start,
              "scope": "Independent FIT source/cache audit only; no training, CAL/CONF decode or model evaluation"}
    (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": status, "checks": a.checks, "failures": len(a.failures)}))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
