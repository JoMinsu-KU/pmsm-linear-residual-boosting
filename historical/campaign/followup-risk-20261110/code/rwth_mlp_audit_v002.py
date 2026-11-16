"""Independent saved-artifact audit of full-FIT RWTH MLP training.

Reconstructs scales, sample geometry, weights and epoch permutations from the
independently audited FIT cache. Does not replay AdamW or claim convergence.
No CAL/CONF arrays, scientific producer imports, GPU, or model fitting.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREP_AUDITOR_SHA = "e3cb0bc015fbe59f3914b6fa74438626b6456634d3b8dd2fd181459ed33d5249"
SHARED_CUDA_SHA = "4ecd405a408a5ec34a20103390f99893ac3264ab28600de0bbde491b58919dbb"
SEEDS = [2026111701, 2026111702, 2026111703]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def dependency():
    p = Path(__file__).with_name("rwth_prep_audit_v002.py")
    if sha(p) != PREP_AUDITOR_SHA:
        raise ValueError("Independent preparation audit source changed")
    spec = importlib.util.spec_from_file_location("rwth_independent_prep", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


P = dependency()
I = P.I


def local(value):
    value = str(value)
    if os.name == "nt" and re.match(r"^/mnt/[a-z]/", value):
        value = value[5].upper()+":/"+value[7:]
    elif os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", value):
        value = "/mnt/"+value[0].lower()+"/"+value[3:].replace("\\", "/")
    return I.path_in(ROOT, value)


def pin(a, entry):
    return a.pin({"path": str(local(entry["path"])), "sha256": entry["sha256"]})


def binding(a, report, path, expected, label):
    matches = [value for key, value in report["inputs"].items() if local(key) == path]
    a.require(bool(matches) and all(x == expected for x in matches), label)


def completed_stage(a, folder, summary_sha, producer_sha):
    a.require(not (folder/"failure.json").exists(), "No producer failure marker")
    summary = I.read(pin(a, {"path": str(folder/"summary.json"), "sha256": summary_sha}))
    stage = I.read(pin(a, {"path": str(folder/"stage_record.json"), "sha256": sha(folder/"stage_record.json")}))
    a.require(stage.get("schema") == "rwth-mlp-stage-v002" and stage.get("status") == "COMPLETE", "Completed MLP stage")
    a.require(stage["source_sha256"] == summary["source_sha256"] == producer_sha, "Exact producer source identities")
    a.same(stage["result"], summary, "Stage exact summary copy")
    a.require(stage["files"].get("summary.json") == summary_sha, "Stage summary digest")
    for name, expected in stage["files"].items():
        path = I.path_in(folder, name)
        pin(a, {"path": str(path), "sha256": expected})
    return summary, stage


def fit_cache(a, entry, prep_audit_entry):
    manifest_path = pin(a, entry); manifest = I.read(manifest_path)
    report = I.read(pin(a, prep_audit_entry))
    a.require(report.get("status") == "PASS" and report.get("stage") == "prep_fit"
              and report.get("auditor_sha256") == PREP_AUDITOR_SHA and not report.get("failures"),
              "Full independent FIT preparation PASS")
    binding(a, report, manifest_path, entry["sha256"], "Prep PASS binds the exact FIT manifest")
    a.require(manifest["schema"] == "rwth-cache-manifest-v002" and manifest["status"] == "READY_FIT_ONLY"
              and manifest["dataset"] == "rwth48v", "Original FIT-only cache identity")
    a.same(manifest["roles"], P.ROLES, "Frozen partition")
    a.same(manifest["feature_names"], P.FEATURE_NAMES, "Common raw33 feature order")
    a.require(manifest["protocol"]["sha256"] == P.PROTOCOL_SHA
              and manifest["partition"]["sha256"] == P.PARTITION_SHA, "Frozen source protocol and partition")
    pin(a, manifest["protocol"]); pin(a, manifest["partition"])
    a.require([r["id"] for r in manifest["records"]] == P.ROLES["fit"], "All20 FIT caches in frozen order")
    records = {}
    for entry in manifest["records"]:
        pid = entry["id"]
        a.require(entry["role"] == "fit" and entry["truth"] is None and entry["target_key"] == "target", "FIT-only record")
        path = pin(a, entry["features"])
        binding(a, report, path, entry["features"]["sha256"], "Each FIT feature cache independently audited")
        value = a.arrays(path); n = entry["rows"]
        a.require(set(value) == {"profile_id", "index", "source_row", "time", "operating", "X33", "target"}, "Exact FIT cache keys")
        a.require(value["X33"].shape == (n, 33) and value["target"].shape == (n,), "Full FIT geometry")
        for name in ("index", "source_row"):
            a.require(value[name].dtype == np.int64 and np.array_equal(value[name], np.arange(n, dtype=np.int64)), "Original complete row indexing")
        records[pid] = value
    return manifest, records


def expected_settings():
    return dict(architecture=[33,64,32,1], parameters=4289, activation="ReLU", loss="mse", epochs=200,
        seeds=SEEDS, optimizer="AdamW", learning_rate=.001, betas=[.9,.999], eps=1e-8, weight_decay=.0001,
        batch_size=4096, gradient_norm_cap=None, warmup=0, label_stride=5, feature_std_floor=1e-8,
        target_iqr_floor=1., permutation="NumPy PCG64(seed+epoch-1); explicit int64",
        profile_weight="N/(P*n_profile)", precision="FP64 cache/scalers, FP32 network/weights, FP64 physical-C output",
        seed_aggregation="arithmetic mean of seed metrics; no seed selection or prediction ensemble")


def audit_train(a, args):
    folder = local(args.input_dir)
    pin(a, {"path": str(Path(__file__).with_name("rwth_mlp_v002.py")), "sha256": args.producer_sha256})
    summary, stage = completed_stage(a, folder, args.input_sha256, args.producer_sha256)
    a.require(summary["status"] == "RWTH_MLP_FIT_COMPLETE" and summary["completed_comparator"] is True
              and summary["epochs"] == 200 and summary["seed"] in SEEDS, "Full200 epoch fixed comparator completed")
    seed = summary["seed"]
    a.require(summary["nonfit_arrays_accessed"] is False, "No nonFIT arrays in training")
    manifest, records = fit_cache(a, summary["cache_manifest"], {"path": args.prep_audit, "sha256": args.prep_audit_sha256})
    ref = P.fit_statistics(records, P.ROLES["fit"])
    lock_path = pin(a, summary["model_lock"])
    a.require(lock_path == folder/"model_lock.json", "Exact training lock path")
    lock = I.read(lock_path)
    a.require(lock["schema"] == "rwth-mlp-model-lock-v002" and lock["status"] == "MODEL_LOCKED"
              and lock["source_sha256"] == args.producer_sha256 and lock["seed"] == seed
              and lock["completed_comparator"] is True, "Immutable fixed-seed model lock")
    required = {"configuration.json", "runtime.json", "training_arrays.npz", "weights_initial.npz",
                "weights_final.npz", "epoch_trace.json", "model.json"}
    a.require(set(lock["files"]) == required, "Complete seven-file training model lock")
    for name, expected in lock["files"].items():
        a.require(stage["files"].get(name) == expected, "Stage/model lock common digest " + name)
        pin(a, {"path": str(folder/name), "sha256": expected})
    a.require(pin(a, lock["model"]) == folder/"model.json", "Canonical model metadata path")
    a.same(lock["cache_manifest"], summary["cache_manifest"], "One immutable FIT manifest")
    model = I.read(folder/"model.json")
    a.require(model["schema"] == "rwth-mlp-model-v002" and model["dataset"] == "rwth48v"
              and model["loss"] == "mse" and model["seed"] == seed and model["epochs"] == 200
              and model["source_sha256"] == args.producer_sha256 and model["completed_comparator"] is True,
              "Fixed full MSE comparator identity")
    a.same(model["fit_ids"], P.ROLES["fit"], "Full fixed FIT20")
    settings = expected_settings()
    a.same(model["settings"], settings, "Frozen training settings")
    configuration = I.read(folder/"configuration.json")
    a.require(configuration["mode"] == "train" and configuration["seed"] == seed
              and configuration["source_sha256"] == args.producer_sha256, "Actual training invocation")
    a.same(configuration["settings"], settings, "Invocation settings")
    for value, tag in ((model, "model"), (lock, "lock"), (summary, "summary"), (configuration, "configuration")):
        a.same(value["protocol"], manifest["protocol"], "Frozen protocol " + tag)
        a.same(value["partition"], manifest["partition"], "Frozen partition " + tag)
    for key in ("calibration_features_accessed", "calibration_target_accessed", "confirmation_accessed"):
        a.require(model[key] is False, "Training role separation " + key)
    a.same(model["cache_manifest"], summary["cache_manifest"], "Model FIT manifest")
    a.require(model["shared_cuda_source"]["sha256"] == SHARED_CUDA_SHA, "Frozen CUDA primitives")
    pin(a, model["shared_cuda_source"])
    tr = dict(input_mean=ref["input_mean"].tolist(), input_std=ref["input_std"].tolist(),
              y_mid=ref["y_mid"], y_scale=ref["y_scale"], fit_ids=P.ROLES["fit"], warmup=0,
              label_stride=5, feature_names=P.FEATURE_NAMES, dt_ref=manifest["dt_ref"],
              high_temperature_threshold_c=float(np.quantile(ref["y"], .95)))
    a.same(model["transform"], tr, "Independent pooled FIT-only scaling", atol=1e-10, rtol=2e-12)
    expected = dict(x=((ref["x"]-ref["input_mean"])/ref["input_std"]).astype(np.float32),
                    y_normalized=((ref["y"]-ref["y_mid"])/ref["y_scale"]).astype(np.float32),
                    w=ref["w"].astype(np.float32), profile_ids=ref["profile_ids"], index=ref["index"],
                    source_row=ref["index"], time=ref["time"])
    arrays = a.arrays(folder/"training_arrays.npz")
    a.require(set(arrays) == set(expected), "Exact training array fields")
    for name in expected:
        a.require(arrays[name].dtype == expected[name].dtype, "Explicit training dtype " + name)
        P.close_array(a, arrays[name], expected[name], "Independent exact training array " + name, exact=True)
    a.require(model["training_arrays_sha256"] == I.array_digest(expected), "Independent full training digest")
    a.require(model["feature_array_sha256"] == I.array_digest({k: expected[k] for k in ("x", "profile_ids", "index")}),
              "Independent feature digest")
    n = len(ref["y"])
    a.same(model["training_rows"], n, "Training row count")
    a.same(summary["training_rows"], n, "Summary training count")
    a.same(model["training_profile_counts"], {str(p): int(np.sum(ref["profile_ids"] == p)) for p in P.ROLES["fit"]},
           "Every FIT profile contributes stride5 rows")
    for label in ("initial", "final"):
        values = a.arrays(folder/("weights_"+label+".npz"))
        I.weight_geometry(a, values, label)
        a.require(model[label+"_array_sha256"] == I.array_digest(values), "Saved " + label + " parameter digest")
    runtime = I.read(folder/"runtime.json")
    a.require(runtime["tf32_matmul_allowed"] is False and runtime["tf32_cudnn_allowed"] is False
              and runtime["deterministic_algorithms"] is True and runtime["torch_num_threads"] == 1,
              "Declared deterministic FP32/single-thread runtime")
    trace = I.read(folder/"epoch_trace.json")
    a.require([r["epoch"] for r in trace] == list(range(1, 201)), "Exactly200 ordered epochs")
    permutations = []
    for row in trace:
        order = np.random.Generator(np.random.PCG64(seed+row["epoch"]-1)).permutation(n).astype(np.int64)
        value = I.array_digest(dict(order=order)); permutations.append(value)
        a.require(row["permutation_sha256"] == value, "Independent explicit-int64 epoch permutation")
        for field in ("online_weighted_loss", "max_gradient_norm", "seconds"):
            a.require(isinstance(row[field], (int, float)) and math.isfinite(row[field]) and row[field] >= 0,
                      "Finite logged epoch diagnostic " + field)
    a.same(model["training_seconds"], sum(r["seconds"] for r in trace), "Epoch training-time sum")
    inputs = I.read(folder/"input_pins.json")
    a.same(inputs, lock["input_hashes_before_nonfit"], "All training inputs locked before any nonFIT access")
    allowed = {local(summary["cache_manifest"]["path"]), local(manifest["protocol"]["path"]),
               local(manifest["partition"]["path"]), local(model["shared_cuda_source"]["path"])}
    allowed.update(local(r["features"]["path"]) for r in manifest["records"])
    a.require({local(path) for path in inputs} == allowed, "Exact FIT-only numerical/source input set")
    for path, expected_sha in inputs.items():
        pin(a, {"path": path, "sha256": expected_sha})
    a.results.update(stage="mlp_train", dataset="rwth48v", seed=seed, model_lock=summary["model_lock"],
        cache_manifest=summary["cache_manifest"], epochs=200, training_rows=n, fit_ids=P.ROLES["fit"],
        training_arrays_sha256=I.array_digest(expected), permutation_digests=permutations,
        initial_array_sha256=model["initial_array_sha256"], final_array_sha256=model["final_array_sha256"],
        independent_optimizer_replay=False, forward_prediction_audit="Deferred until separately locked role predictions exist",
        calibration_arrays_decoded=False, confirmation_arrays_decoded=False)
    return folder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("input-dir", "input-sha256", "producer-sha256", "prep-audit", "prep-audit-sha256", "output-dir"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--max-seconds", type=float, default=600.)
    args = p.parse_args()
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 1200:
        raise ValueError("Audit cap must be finite and <=1200s")
    out = local(args.output_dir); out.mkdir(parents=True, exist_ok=False)
    a = I.Audit(args.max_seconds); status = "HOLD"
    try:
        for path, expected in ((Path(__file__), sha(__file__)), (Path(__file__).with_name("rwth_prep_audit_v002.py"), PREP_AUDITOR_SHA),
                               (Path(__file__).with_name("risk_audit_v001.py"), P.AUDIT_DEP_SHA)):
            pin(a, {"path": str(path), "sha256": expected})
        folder = audit_train(a, args); a.finish_pins()
        a.require(not (folder/"failure.json").exists(), "No producer failure on exit")
        status = "PASS" if not a.failures else "HOLD"
    except Exception as exc:
        a.failures.append({"check": "Audit completed without exception", "detail": repr(exc)})
    report = dict(status=status, stage="mlp_train", auditor_sha256=sha(__file__), producer_sha256=args.producer_sha256,
                  checks=a.checks, failures=a.failures, inputs=a.inputs, results=a.results,
                  elapsed_seconds=time.monotonic()-a.start,
                  scope="Independent FIT sample/scaler/weight/permutation and saved-model integrity audit; no optimizer or held-prediction replay")
    (out/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(status=status, checks=a.checks, failures=len(a.failures))))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
