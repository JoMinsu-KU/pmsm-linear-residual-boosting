"""Causal features adapted from the archived C12/C18 and RWTH numerical cores.

No campaign runner is imported. See provenance in the repository documentation.
Legacy operating order: id, iq, ud, uq, speed, coolant, ambient (recorded units).
RWTH operating order: current, rpm, torque, coolant; timestamps retain raw units.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

SPANS = (16, 64, 256, 1024, 4096)
DATASETS = ("ipmsm", "paderborn", "rwth")
ARMS = ("original", "own_boundary", "first_load")


def filtered(u, span, initial):
    """Original scipy recurrence: row zero is the initial state, not an update."""
    u = np.asarray(u, dtype=np.float64)
    initial = np.asarray(initial, dtype=np.float64)
    if u.ndim != 2 or not len(u) or initial.shape != (u.shape[1],):
        raise ValueError("Nonempty two-dimensional drive and one initial state per channel required")
    a = 1. - 2. / (span + 1.)
    out = np.empty_like(u, dtype=np.float64)
    out[0] = initial
    if len(u) > 1:
        out[1:] = lfilter([1. - a], [1., -a], u[1:], axis=0,
                         zi=(a * initial)[None, :])[0]
    return out


def reference_timestep(records, fit_ids):
    """Pool every consecutive timestamp increment from FIT profiles only."""
    increments = []
    for pid in fit_ids:
        t = np.asarray(records[pid]["time"], dtype=np.float64)
        if len(t) < 2 or not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
            raise ValueError("Strictly increasing finite FIT timestamps required")
        increments.append(np.diff(t))
    value = float(np.median(np.concatenate(increments)))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Positive pooled FIT reference timestep required")
    return value


def make_record(dataset, raw, arm="original", dt_ref=None):
    if dataset not in DATASETS or arm not in ARMS:
        raise ValueError("Unknown dataset or initialization arm")
    if dataset == "rwth" and arm != "original":
        raise ValueError("RWTH was evaluated only with original initialization")
    op = np.asarray(raw["operating"], dtype=np.float64)
    columns = 4 if dataset == "rwth" else 7
    if op.ndim != 2 or op.shape[1] != columns or not len(op) or not np.isfinite(op).all():
        raise ValueError("Finite original operating-column geometry required")
    record = {"id": int(raw.get("id", raw.get("profile_id", -1))),
              "dataset": dataset, "arm": arm, "operating": op}
    target = raw.get("y", raw.get("target"))
    if target is not None:
        y = np.asarray(target, dtype=np.float64)
        if y.shape != (len(op),) or not np.isfinite(y).all():
            raise ValueError("Finite target aligned with every original row required")
        record.update(y=y, target=y)
    if dataset == "rwth":
        t = np.asarray(raw["time"], dtype=np.float64)
        if (t.shape != (len(op),) or not np.isfinite(t).all() or
                not np.all(np.diff(t) > 0) or dt_ref is None or
                not np.isfinite(dt_ref) or dt_ref <= 0):
            raise ValueError("Strictly increasing raw timestamps and FIT dt_ref required")
        current, rpm, torque, coolant = op.T
        omega = 2. * np.pi * rpm / 60.
        instant = np.column_stack((current, omega, torque, coolant, current ** 2,
                                   np.abs(omega), omega ** 2, np.abs(torque * omega)))
        drive = np.column_stack((coolant, current ** 2, np.abs(omega), omega ** 2,
                                 np.abs(torque * omega)))
        state = np.zeros((5, 5), dtype=np.float64)
        state[0] = coolant[0]
        history = np.empty((len(t), 25), dtype=np.float64)
        history[0] = state.reshape(-1)
        log_decay = np.log1p(-2. / (np.asarray(SPANS, dtype=np.float64) + 1.))
        for row in range(1, len(t)):
            alpha = -np.expm1(log_decay * ((t[row] - t[row - 1]) / dt_ref))
            state = (1. - alpha[None, :]) * state + alpha[None, :] * drive[row, :, None]
            history[row] = state.reshape(-1)
        record.update(time=t, dt_ref=float(dt_ref), X33=np.column_stack((instant, history)))
    else:
        current = np.hypot(op[:, 0], op[:, 1])
        speed = np.abs(op[:, 4])
        voltage = np.hypot(op[:, 2], op[:, 3])
        instant = np.column_stack((current, current ** 2, voltage, speed, speed ** 2,
                                   current * speed, op[:, 5], op[:, 6]))
        q = instant[:, [1, 4, 5]]
        boundary_input = op[:, [6, 5]]
        b0 = boundary_input[0] if arm == "own_boundary" else np.repeat(op[0, 6], 2)
        q0 = q[0] if arm == "first_load" else np.zeros(3)
        record.update(instant=instant,
                      boundary=np.column_stack([filtered(boundary_input, s, b0) for s in SPANS]),
                      heat=np.column_stack([filtered(q, s, q0) for s in SPANS]),
                      ewma=np.column_stack([instant] + [filtered(q, s, q[0]) for s in SPANS]))
    return record


def _length(record):
    return len(record["X33"] if record["dataset"] == "rwth" else record["instant"])


def train_index(record):
    return np.arange(0 if record["dataset"] == "rwth" else 300, _length(record), 5, dtype=np.int64)


def evaluation_index(record):
    return np.arange(0 if record["dataset"] == "rwth" else 300, _length(record), dtype=np.int64)


def balanced_weights(lengths, mean_one=False):
    """Ridge sum-one weights and HGB mean-one weights retain dataset arithmetic."""
    lengths = list(lengths)
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError("Every training profile must contribute an endpoint")
    if mean_one:
        n = sum(lengths)
        return np.concatenate([np.full(k, n / (len(lengths) * k), dtype=np.float64) for k in lengths])
    return np.concatenate([np.full(k, 1. / (len(lengths) * k), dtype=np.float64) for k in lengths])


def scaling(records, ids):
    ids = list(ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Nonempty distinct FIT IDs required")
    ds = records[ids[0]]["dataset"]
    if any(records[p]["dataset"] != ds for p in ids):
        raise ValueError("One dataset per estimator")
    indices = {p: train_index(records[p]) for p in ids}
    if any(not len(ix) for ix in indices.values()):
        raise ValueError("Every FIT profile requires an endpoint after warmup")
    ys = np.concatenate([records[p]["y"][indices[p]] for p in ids])
    scale = dict(dataset=ds, ids=ids, fit_ids=ids, y_mid=float(np.median(ys)),
                 y_scale=max(float(np.percentile(ys, 75) - np.percentile(ys, 25)), 1.))
    if ds == "rwth":
        x = np.concatenate([records[p]["X33"][indices[p]] for p in ids])
        scale.update(x_mean=x.mean(0).tolist(), x_std=np.maximum(x.std(0), 1e-8).tolist(),
                     drive_rms=np.maximum(np.sqrt(np.mean(x[:, 4:8] ** 2, axis=0)), 1e-8).tolist())
    else:
        instant = np.concatenate([records[p]["instant"][indices[p]] for p in ids])
        rms_i = max(float(np.sqrt(np.mean(instant[:, 0] ** 2))), 1e-8)
        rms_w = max(float(np.sqrt(np.mean(instant[:, 3] ** 2))), 1e-8)
        scale["qscale"] = np.tile([rms_i ** 2, rms_w ** 2, rms_i * rms_w], 5).tolist()
        x = np.concatenate([boost_features(records[p], scale)[indices[p]] for p in ids])
        scale.update(linear33_mean=x.mean(0).tolist(), linear33_scale=np.maximum(x.std(0), 1e-8).tolist())
    return scale


def boost_features(record, scale):
    if record["dataset"] == "rwth":
        return (record["X33"] - np.asarray(scale["x_mean"])) / np.asarray(scale["x_std"])
    return np.column_stack((record["boundary"], record["heat"] / np.asarray(scale["qscale"]), record["instant"]))


def neural_features(record, scale):
    return record["X33"] if record["dataset"] == "rwth" else boost_features(record, scale)


def training(records, ids, scale=None):
    ids = list(ids)
    scale = scaling(records, ids) if scale is None else scale
    indices = {p: train_index(records[p]) for p in ids}
    x = np.concatenate([boost_features(records[p], scale)[indices[p]] for p in ids])
    y = np.concatenate([records[p]["y"][indices[p]] for p in ids])
    counts = [len(indices[p]) for p in ids]
    # Original legacy routine multiplies rounded sum-one weights; RWTH constructs directly.
    w = balanced_weights(counts, mean_one=True) if records[ids[0]]["dataset"] == "rwth" else len(y) * balanced_weights(counts)
    return x, y, w, scale, indices


train_arrays = training
