"""Weighted ridge and in-sample residual boosting for the current study.

Arithmetic is adapted from c15_paper_core_v001, c18_own_controls_v001 and
rwth_cpu_core_v002; operational guards belong to the new runner, not this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .features import balanced_weights, boost_features, scaling, training, train_index

BASE = dict(learning_rate=.05, min_samples_leaf=20, l2_regularization=.1,
            max_bins=255, loss="squared_error", early_stopping=False, random_state=2026111201)
METHODS = ("selected_direct", "matched_direct", "capacity_direct", "linear_prior",
           "linear_residual", "linear_half", "prior_half", "matched_half",
           "history_prior", "history_residual", "history_half")
ALIASES = {"d9": "selected_direct", "direct200": "matched_direct",
           "direct200_half": "matched_half", "free_prior": "history_prior",
           "free_residual": "history_residual", "free_half": "history_half"}


def ridge_design(record, scale, family="linear"):
    ds = record["dataset"]
    if family == "linear":
        x = boost_features(record, scale)
        if ds != "rwth":
            x = (x - np.asarray(scale["linear33_mean"])) / np.asarray(scale["linear33_scale"])
        return np.column_stack((np.ones(len(x)), x))
    if family != "history":
        raise ValueError("Expected linear or history prior")
    if ds == "rwth":
        x = record["X33"]
        # RWTH history prior has no independent intercept: 25 penalized coefficients.
        return np.column_stack(((x[:, 8:13] - scale["y_mid"]) / scale["y_scale"],
                                x[:, 13:33] / np.repeat(np.asarray(scale["drive_rms"]), 5)))
    # Legacy history prior has 26 coefficients including an unpenalized intercept.
    return np.column_stack((np.ones(len(record["instant"])),
                            (record["boundary"] - scale["y_mid"]) / scale["y_scale"],
                            record["heat"] / np.asarray(scale["qscale"])))


def fit_prior(records, ids, family, lam, scale=None, check=None):
    """Fit only the supplied training profiles; no held target is accessed."""
    if check is not None:
        check()
    started = time.perf_counter()
    ids = list(ids)
    scale = scaling(records, ids) if scale is None else scale
    ds = records[ids[0]]["dataset"]
    ix = {p: train_index(records[p]) for p in ids}
    x = np.concatenate([ridge_design(records[p], scale, family)[ix[p]] for p in ids])
    y = np.concatenate([(records[p]["y"][ix[p]] - scale["y_mid"]) / scale["y_scale"] for p in ids])
    counts = [len(ix[p]) for p in ids]
    w = balanced_weights(counts, mean_one=True) / len(y) if ds == "rwth" else balanced_weights(counts)
    penalty = np.ones(x.shape[1])
    if family == "linear" or ds != "rwth":
        penalty[0] = 0.
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("Positive finite ridge penalty required")
    # Preserve the original RWTH doubled normal equations and multiplication order.
    if ds == "rwth":
        q = 2 * (x.T @ (w[:, None] * x) + lam * np.diag(penalty))
        rhs = 2 * x.T @ (w * y)
    else:
        q = x.T @ (w[:, None] * x) + lam * np.diag(penalty)
        rhs = x.T @ (w * y)
    np.linalg.cholesky(q)
    coef = np.linalg.solve(q, rhs)
    residual = q @ coef - rhs
    gap = float(.5 * residual @ np.linalg.solve(q, residual))
    if not np.isfinite(coef).all() or gap > 1e-8:
        raise ArithmeticError("Ridge numerical qualification failed")
    return dict(family=family, dataset=ds, lam=float(lam), fit_ids=ids, scale=scale,
                coef=coef.tolist(), penalty=penalty.tolist(),
                certificate=dict(stationarity=float(np.max(np.abs(residual))), stable_gap=gap),
                fit_seconds=time.perf_counter() - started)


def predict_prior(model, record):
    return model["scale"]["y_mid"] + model["scale"]["y_scale"] * (
        ridge_design(record, model["scale"], model["family"]) @ np.asarray(model["coef"]))


def select_lambda(records, fit_ids, family="linear", lambdas=(.0001, .01, 1.), check=None):
    """Optional leave-one-FIT-profile-out replay; no external evaluation roles.

    Historical RWTH folds score every row; legacy folds score 300::5.
    Initialization interventions reuse the original selected lambda instead.
    """
    if len(fit_ids) < 2:
        raise ValueError("At least two FIT profiles required for leave-profile-out selection")
    rows = []
    for lam in lambdas:
        errors = []
        for held in fit_ids:
            fit = [p for p in fit_ids if p != held]
            model = fit_prior(records, fit, family, lam, check=check)
            ix = np.arange(len(records[held]["y"])) if records[held]["dataset"] == "rwth" else train_index(records[held])
            error = predict_prior(model, records[held])[ix] - records[held]["y"][ix]
            errors.append(float(np.sqrt(np.mean(error * error))))
        rows.append(dict(lam=float(lam), profile_rmse=errors, macro_rmse=float(np.mean(errors))))
    selected = min(range(len(rows)), key=lambda j: (rows[j]["macro_rmse"], j))
    return rows[selected]["lam"], rows


def _configuration(settings, key, defaults, smoke):
    value = {**BASE, **defaults, **settings.get(key, {})}
    if smoke:
        value["max_iter"] = min(3, value["max_iter"])
    if value["early_stopping"] or value["loss"] != "squared_error":
        raise ValueError("Fixed-iteration squared-error boosters required")
    return value


def fit_cpu(records, fit_ids, dataset, settings, smoke=False, check=None):
    """Fit current frozen configurations, without repeating hyperparameter search.

    ``smoke`` caps boosting at three trees and always labels the artifact non-paper.
    The runner controls profile selection, resource limits and endpoint admission.
    """
    ids = list(fit_ids)
    if any(records[p]["dataset"] != dataset for p in ids):
        raise ValueError("Training dataset mismatch")
    settings = settings.get("cpu", settings)
    lam = settings.get("lambda", .0001 if dataset == "paderborn" else .01)
    history_lam = settings.get("history_lambda", lam)
    with threadpool_limits(limits=1):
        x, y, w, scale, indices = training(records, ids)
        components = settings.get("fit_components", ["selected_direct", "matched_direct", "capacity_direct", "residual", "history_residual"])
        if not components or len(components) != len(set(components)) or any(k not in ("selected_direct", "matched_direct", "capacity_direct", "residual", "history_residual") for k in components):
            raise ValueError("Nonempty distinct known booster components required")
        prior_specs = [("linear", lam)]
        if "history_residual" in components:
            prior_specs.append(("history", history_lam))
        priors = {family: fit_prior(records, ids, family, regularization, scale, check)
                  for family, regularization in prior_specs}
        prior_y = {family: np.concatenate([predict_prior(model, records[p])[indices[p]] for p in ids])
                   for family, model in priors.items()}
        selected = dict(max_leaf_nodes=15, max_iter=1000) if dataset == "paderborn" else dict(max_leaf_nodes=7, max_iter=500)
        capacity = dict(max_leaf_nodes=15, max_iter=1200) if dataset == "paderborn" else dict(max_leaf_nodes=7, max_iter=967)
        configs = {"selected_direct": _configuration(settings, "selected_direct", selected, smoke),
                   "matched_direct": _configuration(settings, "matched_direct", dict(max_leaf_nodes=15, max_iter=200), smoke),
                   "capacity_direct": _configuration(settings, "capacity_direct", capacity, smoke)}
        configs["residual"] = configs["matched_direct"].copy()
        configs["history_residual"] = configs["matched_direct"].copy()
        boosters, elapsed = {}, {}
        configs = {name: configs[name] for name in components}
        for name, config in configs.items():
            if check is not None:
                check()
            target = y - prior_y["linear"] if name == "residual" else y - prior_y["history"] if name == "history_residual" else y
            start = time.perf_counter()
            model = HistGradientBoostingRegressor(**config).fit(x, target, sample_weight=w)
            elapsed[name] = time.perf_counter() - start
            if model.n_iter_ != config["max_iter"]:
                raise ArithmeticError("Fixed boosting iteration count changed")
            boosters[name] = model
    if check is not None:
        check()
    return dict(schema="pmsm-cpu-v2", dataset=dataset, scale=scale, priors=priors, boosters=boosters,
                metadata=dict(fit_ids=ids, arm=records[ids[0]].get("arm", "original"),
                              training_rows=len(y), configurations=configs, fit_seconds=elapsed,
                              numpy=np.__version__, sklearn=sklearn.__version__, smoke=bool(smoke),
                              fitted_components=list(components),
                              paper_configuration=not smoke, residual_targets="in-sample degrees Celsius",
                              new_independent_confirmation=False, cpu_threads=1))


def predict_cpu(bundle, record, methods=None):
    """Evaluate only components required by requested methods, once per call.

    Full-length rows are returned; the caller applies the historical warmup.
    Passing a scalar-length feature record therefore performs no history reset.
    """
    requested = list(available_methods(bundle) if methods is None else ([methods] if isinstance(methods, str) else methods))
    if any(ALIASES.get(m, m) not in METHODS for m in requested):
        raise ValueError("Unknown CPU method")
    if record["dataset"] != bundle["dataset"]:
        raise ValueError("Prediction dataset mismatch")
    cache = {}
    def compute(name):
        if name in cache:
            return cache[name]
        if name in ("linear_prior", "history_prior"):
            value = predict_prior(bundle["priors"]["linear" if name == "linear_prior" else "history"], record)
        elif name in ("linear_half", "prior_half", "matched_half", "history_half"):
            right = {"linear_half": "linear_residual", "prior_half": "linear_prior",
                     "matched_half": "matched_direct", "history_half": "history_residual"}[name]
            value = .5 * (compute("selected_direct") + compute(right))
        else:
            if "x" not in cache:
                cache["x"] = boost_features(record, bundle["scale"])
            key = {"linear_residual": "residual", "history_residual": "history_residual"}.get(name, name)
            value = bundle["boosters"][key].predict(cache["x"])
            if name == "linear_residual":
                value = compute("linear_prior") + value
            elif name == "history_residual":
                value = compute("history_prior") + value
        value = np.asarray(value, dtype=np.float64)
        if not np.isfinite(value).all():
            raise FloatingPointError("Nonfinite estimator prediction")
        cache[name] = value
        return value
    return {m: compute(ALIASES.get(m, m)) for m in requested}


def available_methods(bundle):
    required = {"selected_direct": ({"selected_direct"}, set()),
                "matched_direct": ({"matched_direct"}, set()),
                "capacity_direct": ({"capacity_direct"}, set()),
                "linear_prior": (set(), {"linear"}),
                "linear_residual": ({"residual"}, {"linear"}),
                "linear_half": ({"selected_direct", "residual"}, {"linear"}),
                "prior_half": ({"selected_direct"}, {"linear"}),
                "matched_half": ({"selected_direct", "matched_direct"}, set()),
                "history_prior": (set(), {"history"}),
                "history_residual": ({"history_residual"}, {"history"}),
                "history_half": ({"selected_direct", "history_residual"}, {"history"})}
    return [key for key in METHODS if required[key][0] <= bundle["boosters"].keys() and required[key][1] <= bundle["priors"].keys()]


def predict_method(bundle, record, method):
    return predict_cpu(bundle, record, [method])[method]


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_cpu(bundle, folder):
    """Write a new directory only. Pickles are for trusted local reproduction."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    meta = {k: v for k, v in bundle.items() if k != "boosters"}
    (folder / "model.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, model in bundle["boosters"].items():
        with (folder / (name + ".pkl")).open("xb") as stream:
            pickle.dump(model, stream, protocol=5)
        # Portable, non-executable prediction structure retained alongside sklearn.
        arrays = {"baseline": model._baseline_prediction.copy(), "n_iter": np.asarray([model.n_iter_], dtype=np.int64)}
        for j, stage in enumerate(model._predictors):
            arrays[f"tree_{j:04d}"] = stage[0].nodes.copy()
        np.savez_compressed(folder / (name + ".npz"), **arrays)
    manifest = dict(schema="pmsm-cpu-artifacts-v2", files={p.name: _sha(p) for p in sorted(folder.iterdir()) if p.is_file()},
                    trust="Hashes detect changes, not malicious provenance. Load only self-generated/trusted models.")
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_cpu(folder):
    """Verify before deserializing. Only use models generated by a trusted runner.

    A matching manifest is integrity evidence, not a sandbox for hostile pickle.
    """
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "pmsm-cpu-artifacts-v2" or "model.json" not in manifest.get("files", {}):
        raise ValueError("CPU artifact manifest must bind model metadata before reading")
    for name, digest in manifest["files"].items():
        p = folder / name
        if p.is_symlink() or p.resolve().parent != folder or _sha(p) != digest:
            raise ValueError("CPU artifact hash/path mismatch: " + name)
    bundle = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    if bundle.get("schema") != "pmsm-cpu-v2":
        raise ValueError("Unknown CPU model schema")
    if bundle["metadata"]["sklearn"] != sklearn.__version__:
        raise ValueError("Restore with the recorded scikit-learn version")
    components = bundle["metadata"]["fitted_components"]
    if not components or len(components) != len(set(components)) or any(k not in ("selected_direct", "matched_direct", "capacity_direct", "residual", "history_residual") for k in components):
        raise ValueError("Invalid fitted component list")
    required = {"model.json"} | {name + ext for name in components for ext in (".pkl", ".npz")}
    if set(manifest["files"]) != required:
        raise ValueError("Complete CPU artifact manifest required")
    bundle["boosters"] = {}
    for name in components:
        with (folder / (name + ".pkl")).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, HistGradientBoostingRegressor):
            raise TypeError("Expected fitted sklearn histogram regressor")
        bundle["boosters"][name] = model
    return bundle
