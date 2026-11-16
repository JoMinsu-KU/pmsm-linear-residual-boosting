"""C12 FIT-only positive thermal response and cross-fitted residual boosting.

Source-only preparation. Root must register before execution. No public physics
coefficients, target initialization, CAL model selection, or confirmation IDs.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from scipy.signal import lfilter
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits, threadpool_info

FIT = [0, 2, 3, 5, 7, 8, 11, 14]
CAL = [1, 6, 12]
PILOT = [4, 9, 13]
OPERATING = ['IdRef', 'IqRef', 'Ud', 'Uq', 'speed', 'water_outlet', 'ambient']
SPANS = [16, 64, 256, 1024, 4096]
LAMBDAS = [0.0001, 0.01, 1.]
WARMUP, STRIDE = 300, 5
MANIFEST_SHA = '76dc4eaf66e6e60cf8edc9f3f0cff6591a68e71aad2098a131f2b25cccfbcfba'
PRIMARY = 'positive_rc_oof_boost'
METHODS = ['positive_rc', 'unconstrained_rc_ridge', 'direct_boost', PRIMARY,
           'unconstrained_rc_oof_boost', 'mean_A_B', 'ewma_ridge',
           'positive_rc_insample_boost']
BOOST = dict(max_iter=200, max_leaf_nodes=15, learning_rate=.05,
             min_samples_leaf=20, l2_regularization=.1, loss='squared_error',
             early_stopping=False, random_state=2026111201)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                   allow_nan=False), encoding='utf-8')


class Clock:
    def __init__(self, limit, output):
        self.start, self.limit = time.perf_counter(), limit
        self.output = Path(output)

    def check(self):
        if time.perf_counter() - self.start > self.limit:
            raise TimeoutError('C12 admitted wall-clock limit exceeded')
        if self.output.exists():
            written = sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())
            if written > 100_000_000:
                raise ValueError('C12 100000000-byte output ceiling exceeded')


def read_profile(assets, item, allowed, target):
    if item['id'] not in allowed or item['id'] in [10, 15]:
        raise ValueError('C12 role access violation')
    root = Path(assets).resolve()
    path = (root / item['path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Asset escapes admitted root')
    raw = path.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    if len(raw) != item['bytes'] or blob != item['git_blob_sha1']:
        raise ValueError('Pinned source byte mismatch')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    if set(OPERATING + ['rotor']) - set(reader.fieldnames or []):
        raise ValueError('Missing source columns')
    rows = list(reader)
    op = np.array([[float(r[k]) for k in OPERATING] for r in rows])
    y = np.array([float(r['rotor']) for r in rows]) if target else None
    if len(op) <= WARMUP + 32 or not np.isfinite(op).all():
        raise ValueError('Invalid operating record')
    if y is not None and not np.isfinite(y).all():
        raise ValueError('Invalid target record')
    return op, y, dict(id=item['id'], sha256=hashlib.sha256(raw).hexdigest(),
                      git_blob_sha1=blob, rows=len(op), target_decoded=target)


def filtered(u, span, initial):
    a = 1. - 2. / (span + 1.)
    out = np.empty_like(u, dtype=np.float64)
    out[0] = initial
    if len(u) > 1:
        out[1:] = lfilter([1. - a], [1., -a], u[1:],
                         axis=0, zi=(a * np.asarray(initial))[None, :])[0]
    return out


def raw_features(op):
    current = np.hypot(op[:, 0], op[:, 1])
    speed = np.abs(op[:, 4])
    voltage = np.hypot(op[:, 2], op[:, 3])
    instant = np.column_stack([current, current ** 2, voltage, speed,
                              speed ** 2, current * speed, op[:, 5], op[:, 6]])
    q = instant[:, [1, 4, 5]]
    # Boundary column order: each span, ambient then water.
    boundary = np.column_stack([filtered(op[:, [6, 5]], s,
                              np.repeat(op[0, 6], 2)) for s in SPANS])
    heat = np.column_stack([filtered(q, s, np.zeros(3)) for s in SPANS])
    # Established nonphysical EWMA control initializes heating at first input.
    ewma = np.column_stack([instant] + [filtered(q, s, q[0]) for s in SPANS])
    if boundary.shape[1] != 10 or heat.shape[1] != 15 or ewma.shape[1] != 23:
        raise ValueError('Feature geometry changed')
    return dict(boundary=boundary, heat=heat, instant=instant, ewma=ewma)


def train_index(record):
    return np.arange(WARMUP, len(record['y']), STRIDE)


def balanced_weights(lengths):
    return np.concatenate([np.full(n, 1. / (len(lengths) * n)) for n in lengths])


def scaling(records, ids):
    indices = {p: train_index(records[p]) for p in ids}
    instant = np.concatenate([records[p]['instant'][indices[p]] for p in ids])
    ys = np.concatenate([records[p]['y'][indices[p]] for p in ids])
    rms_i = max(float(np.sqrt(np.mean(instant[:, 0] ** 2))), 1e-8)
    rms_w = max(float(np.sqrt(np.mean(instant[:, 3] ** 2))), 1e-8)
    qscale = np.tile([rms_i ** 2, rms_w ** 2, rms_i * rms_w], 5)
    ewma = np.concatenate([records[p]['ewma'][indices[p]] for p in ids])
    return dict(ids=list(ids), qscale=qscale.tolist(), y_mid=float(np.median(ys)),
                y_scale=max(float(np.percentile(ys, 75) - np.percentile(ys, 25)), 1.),
                ewma_mean=ewma.mean(0).tolist(),
                ewma_scale=np.maximum(ewma.std(0), 1e-8).tolist())


def design(record, scale, family):
    if family == 'ewma':
        return np.column_stack([np.ones(len(record['ewma'])),
            (record['ewma'] - scale['ewma_mean']) / scale['ewma_scale']])
    x = np.column_stack([(record['boundary'] - scale['y_mid']) / scale['y_scale'],
                         record['heat'] / scale['qscale']])
    # Free control includes an unpenalized intercept; positive model has exactly25.
    return np.column_stack([np.ones(len(x)), x]) if family == 'free' else x


def positive_qp(q, r, constant):
    """Strictly convex nonnegative QP with sum first10 =1, active-set solve."""
    n = len(r)
    np.linalg.cholesky(q)
    e = np.r_[np.ones(10), np.zeros(n - 10)]
    theta = np.zeros(n); theta[0] = 1.
    free = [0]
    tol = 2e-10 * (1. + float(np.max(np.abs(r))))
    for iteration in range(1, 1001):
        f = np.array(sorted(free), dtype=int)
        qf, ef, rf = q[np.ix_(f, f)], e[f], r[f]
        solved = np.linalg.solve(qf, np.column_stack([rf, ef]))
        denom = float(ef @ solved[:, 1])
        if denom <= 0:
            raise ValueError('QP face lost feasible boundary coefficient')
        nu = float((ef @ solved[:, 0] - 1.) / denom)
        candidate = np.zeros(n); candidate[f] = solved[:, 0] - nu * solved[:, 1]
        negative = [j for j in f if candidate[j] < -1e-12]
        if negative:
            ratios = [(theta[j] / (theta[j] - candidate[j]), int(j)) for j in negative]
            alpha, hit = min(ratios)
            theta += alpha * (candidate - theta)
            theta[hit] = 0.; free.remove(hit)
            continue
        theta = candidate
        # Tiny negatives are not clipped: retain equality and qualify explicitly.
        grad = q @ theta - r + nu * e
        inactive = [j for j in range(n) if j not in free]
        if inactive:
            add = min(inactive, key=lambda j: (grad[j], j))
            if grad[add] < -tol:
                free.append(add)
                continue
        break
    else:
        raise ValueError('QP active-set iteration limit')
    grad = q @ theta - r + nu * e
    mu = np.maximum(grad, 0.)
    dual_argument = r - nu * e + mu
    z = np.linalg.solve(q, dual_argument)
    primal = float(.5 * theta @ q @ theta - r @ theta + constant)
    dual = float(constant - nu - .5 * dual_argument @ z)
    stable_gap = float(.5 * (theta - z) @ q @ (theta - z) + mu @ theta + nu * (1. - e @ theta))
    equality = float(abs(e @ theta - 1.))
    gap_tolerance = 1e-8 * (1. + abs(primal))
    cert = dict(iterations=iteration, primal=primal, dual=dual,
                direct_gap=primal - dual, stable_gap=stable_gap,
                gap_tolerance=gap_tolerance, equality_residual=equality,
                min_coefficient=float(theta.min()), nu=nu, mu=mu.tolist(),
                stationarity=float(np.max(np.abs(grad - mu))),
                complementarity=float(np.max(np.abs(mu * theta))),
                active_coefficients=[int(j) for j in np.flatnonzero(theta > 1e-10)])
    if (equality > 1e-10 or theta.min() < -1e-10 or
            stable_gap < -1e-10 or stable_gap > gap_tolerance or
            abs((primal - dual) - stable_gap) > 1e-8 * (1. + abs(primal))):
        raise ValueError('QP numerical qualification failed: ' + json.dumps(cert))
    return theta, cert


def fit_linear(records, ids, family, lam, clock):
    clock.check(); started = time.perf_counter()
    scale = scaling(records, ids)
    xs = [design(records[p], scale, family)[train_index(records[p])] for p in ids]
    ys = [(records[p]['y'][train_index(records[p])] - scale['y_mid']) / scale['y_scale'] for p in ids]
    x, y = np.concatenate(xs), np.concatenate(ys)
    weights = balanced_weights([len(v) for v in ys])
    penalty = np.ones(x.shape[1])
    if family != 'positive':
        penalty[0] = 0.
    q = x.T @ (weights[:, None] * x) + lam * np.diag(penalty)
    r = x.T @ (weights * y)
    constant = float(.5 * np.dot(weights, y * y))
    if family == 'positive':
        coef, cert = positive_qp(q, r, constant)
    else:
        np.linalg.cholesky(q)
        coef = np.linalg.solve(q, r)
        residual = q @ coef - r
        gap = float(.5 * residual @ np.linalg.solve(q, residual))
        cert = dict(stationarity=float(np.max(np.abs(residual))), stable_gap=gap,
                    gap_tolerance=1e-8, iterations=1)
        if gap > 1e-8 or not np.isfinite(coef).all():
            raise ValueError('Unconstrained ridge qualification failed')
    clock.check()
    return dict(family=family, lam=lam, scale=scale, coef=coef.tolist(),
                certificate=cert, seconds=time.perf_counter() - started)


def predict_linear(model, record):
    value = design(record, model['scale'], model['family']) @ model['coef']
    return model['scale']['y_mid'] + model['scale']['y_scale'] * value


def select_family(records, ids, family, clock):
    folds, score_table = {}, []
    for lam in LAMBDAS:
        errors = []
        for held in ids:
            model = fit_linear(records, [p for p in ids if p != held], family, lam, clock)
            index = train_index(records[held])
            prediction = predict_linear(model, records[held])[index]
            error = float(np.sqrt(np.mean((prediction - records[held]['y'][index]) ** 2)))
            folds[(lam, held)] = model
            errors.append(error)
        score_table.append(dict(lam=lam, profile_rmse=errors, macro_rmse=float(np.mean(errors))))
    best = min(range(len(LAMBDAS)), key=lambda j: (score_table[j]['macro_rmse'], j))
    lam = LAMBDAS[best]
    final = fit_linear(records, ids, family, lam, clock)
    oof = {p: predict_linear(folds[(lam, p)], records[p])[train_index(records[p])] for p in ids}
    return final, oof, dict(family=family, selected_lambda=lam, cv=score_table,
        folds=[dict(lam=l, held_out=p, model=folds[(l, p)]) for l in LAMBDAS for p in ids])


def boost_features(record, scale):
    return np.column_stack([record['boundary'], record['heat'] / scale['qscale'], record['instant']])


def export_booster(model, directory, name):
    arrays = {'baseline': model._baseline_prediction.copy(),
              'n_iter': np.array([model.n_iter_], dtype=np.int64)}
    for j, stage in enumerate(model._predictors):
        if len(stage) != 1:
            raise ValueError('Unexpected multivariate booster')
        arrays['tree_%03d' % j] = stage[0].nodes.copy()
    for j, thresholds in enumerate(model._bin_mapper.bin_thresholds_):
        arrays['bin_thresholds_%02d' % j] = thresholds.copy()
    path = directory / (name + '.npz')
    np.savez_compressed(path, **arrays)
    return dict(path=path.name, sha256=sha(path), bytes=path.stat().st_size,
                n_iter=model.n_iter_, baseline=model._baseline_prediction.tolist())


def train_all(records, ids, output, clock):
    started = time.perf_counter()
    linear, oof, cv = {}, {}, {}
    for family in ['positive', 'free', 'ewma']:
        linear[family], oof[family], cv[family] = select_family(records, ids, family, clock)
    scale = linear['positive']['scale']
    xparts = [boost_features(records[p], scale)[train_index(records[p])] for p in ids]
    yparts = [records[p]['y'][train_index(records[p])] for p in ids]
    x, y = np.concatenate(xparts), np.concatenate(yparts)
    sample_weight = len(y) * balanced_weights([len(v) for v in yparts])
    pos_oof, free_oof = np.concatenate([oof['positive'][p] for p in ids]), np.concatenate([oof['free'][p] for p in ids])
    pos_in = np.concatenate([predict_linear(linear['positive'], records[p])[train_index(records[p])] for p in ids])
    responses = dict(direct_boost=y, positive_rc_oof_boost=y - pos_oof,
                     unconstrained_rc_oof_boost=y - free_oof,
                     positive_rc_insample_boost=y - pos_in)
    boosters, metadata = {}, {}
    for name, target in responses.items():
        clock.check(); fit_started = time.perf_counter()
        model = HistGradientBoostingRegressor(**BOOST).fit(x, target, sample_weight=sample_weight)
        clock.check()
        boosters[name] = model
        metadata[name] = export_booster(model, output, name)
        metadata[name]['fit_seconds'] = time.perf_counter() - fit_started
    np.savez_compressed(output / 'training.npz', features=x, target=y,
        sample_weight=sample_weight, positive_oof=pos_oof, free_oof=free_oof,
        positive_insample=pos_in, profile_ids=np.concatenate([np.full(len(v), p) for p, v in zip(ids, yparts)]))
    dump(output / 'linear_models.json', linear)
    dump(output / 'cv_records.json', cv)
    dump(output / 'booster_records.json', metadata)
    return linear, boosters, dict(seconds=time.perf_counter() - started,
            fit_rows=len(y), booster_records=metadata, cv_models=3 * len(ids) * len(LAMBDAS),
            all_linear_fits=3 * (len(ids) * len(LAMBDAS) + 1))


def predictions(record, linear, boosters):
    x = boost_features(record, linear['positive']['scale'])
    a = predict_linear(linear['positive'], record)
    free = predict_linear(linear['free'], record)
    b = boosters['direct_boost'].predict(x)
    answer = dict(positive_rc=a, unconstrained_rc_ridge=free, direct_boost=b,
        positive_rc_oof_boost=a + boosters[PRIMARY].predict(x),
        unconstrained_rc_oof_boost=free + boosters['unconstrained_rc_oof_boost'].predict(x),
        mean_A_B=.5 * (a + b), ewma_ridge=predict_linear(linear['ewma'], record),
        positive_rc_insample_boost=a + boosters['positive_rc_insample_boost'].predict(x))
    if list(answer) != METHODS or any(not np.isfinite(v).all() for v in answer.values()):
        raise ValueError('Prediction coverage/finite qualification failed')
    return answer


def fixtures():
    # Analytic positive simplex case, and a heating component forced to zero.
    q = np.eye(25); r = np.r_[np.repeat(.2, 10), [-1., 2.], np.zeros(13)]
    coef, cert = positive_qp(q, r, 3.)
    expected = np.r_[np.repeat(.1, 10), [0., 2.], np.zeros(13)]
    if not np.allclose(coef, expected, atol=1e-10, rtol=0):
        raise ValueError('QP analytic fixture failed')
    u = np.column_stack([np.arange(12.), np.repeat(4., 12)])
    got = filtered(u, 16, np.array([3., 4.]))
    manual = np.empty_like(u); manual[0] = [3., 4.]
    for t in range(1, len(u)):
        manual[t] = (15. / 17.) * manual[t - 1] + (2. / 17.) * u[t]
    if not np.allclose(got, manual, atol=1e-12, rtol=0):
        raise ValueError('Thermal recurrence fixture failed')
    # Units/inverse transform under boundary simplex constraint.
    b = np.arange(30.).reshape(3, 10) + 10.
    h = np.ones((3, 15)); theta = np.r_[np.repeat(.1, 10), np.repeat(.2, 15)]
    physical = b @ theta[:10] + 7. * h @ theta[10:]
    reconstructed = 20. + 7. * (np.c_[(b - 20.) / 7., h] @ theta)
    if not np.allclose(physical, reconstructed, atol=1e-12, rtol=0):
        raise ValueError('Physical degree-unit fixture failed')
    return dict(status='PASS', analytic_qp_certificate=cert, thermal_recurrence=True,
                degree_unit_inverse=True)


def run(args):
    cap = 300. if args.mode == 'costprobe' else 1200.
    if not np.isfinite(args.max_seconds) or not 0 < args.max_seconds <= cap:
        raise ValueError('C12 mode-specific wall-clock cap exceeded')
    clock = Clock(args.max_seconds, args.output_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if sha(args.manifest) != MANIFEST_SHA:
        raise ValueError('Manifest pin changed')
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8-sig'))
    items = {int(v['id']): v for v in manifest['files']}
    expected = {**{p: 'fit' for p in FIT}, **{p: 'calibration' for p in CAL}, **{p: 'pilot' for p in PILOT}}
    if set(items) != set(expected) or any(items[p]['role'] != role for p, role in expected.items()):
        raise ValueError('Manifest role contract changed')
    dump(output / 'fixtures.json', fixtures())
    ids = FIT[:2] if args.mode == 'costprobe' else FIT
    records, source_records = {}, []
    for p in ids:
        clock.check()
        op, y, receipt = read_profile(args.assets_dir, items[p], ids, True)
        record = raw_features(op); record['y'] = y
        records[p] = record; source_records.append(receipt)
        np.savez_compressed(output / ('fit_%02d.npz' % p), operating=op, target=y, **{k: v for k, v in record.items() if k != 'y'})
    linear, boosters, train_timing = train_all(records, ids, output, clock)
    model_files = sorted(p for p in output.iterdir() if p.is_file())
    model_lock = dict(schema='c12-model-lock-v001', mode=args.mode,
        source_sha256=sha(__file__), manifest_sha256=MANIFEST_SHA, fit_ids=ids,
        config=dict(methods=METHODS, primary=PRIMARY, spans=SPANS, lambdas=LAMBDAS,
                    warmup=WARMUP, fit_stride=STRIDE, booster=BOOST),
        files={p.name: sha(p) for p in model_files}, sources=source_records,
        locked_elapsed=time.perf_counter() - clock.start,
        training=train_timing, native_threads=threadpool_info())
    dump(output / 'model_lock.json', model_lock)
    clock.check()
    if args.mode == 'costprobe':
        summary = dict(status='COSTPROBE_COMPLETE_NO_SCIENTIFIC_TEST', fit_ids=ids,
            elapsed_seconds=time.perf_counter() - clock.start, training=train_timing,
            no_cal_or_pilot_access=True, projection_is_not_measured_full_runtime=True,
            source_sha256=sha(__file__))
        dump(output / 'summary.json', summary)
        clock.check()
        return summary
    eval_records, pred_receipts = {}, []
    for p in CAL + PILOT:
        clock.check()
        op, _, receipt = read_profile(args.assets_dir, items[p], CAL + PILOT, False)
        record = raw_features(op)
        started = time.perf_counter(); pred = predictions(record, linear, boosters)
        prediction_seconds = time.perf_counter() - started
        index = np.arange(WARMUP, len(op))
        pred_file = output / ('profile_%02d_predictions.npz' % p)
        np.savez_compressed(pred_file, operating=op, index=index,
            **record, **{'prediction__' + k: v[index] for k, v in pred.items()})
        pred_receipts.append(dict(id=p, role=items[p]['role'], path=pred_file.name,
            sha256=sha(pred_file), rows=len(op), evaluated_rows=len(index),
            prediction_seconds_shared_all_methods=prediction_seconds, source=receipt))
        eval_records[p] = (index, pred)
    dump(output / 'prediction_lock.json', dict(schema='c12-prediction-lock-v001',
        model_lock_sha256=sha(output / 'model_lock.json'), records=pred_receipts,
        locked_elapsed=time.perf_counter() - clock.start))
    rows = []
    for p in CAL + PILOT:
        clock.check()
        _, y, _ = read_profile(args.assets_dir, items[p], CAL + PILOT, True)
        index, pred = eval_records[p]
        np.savez_compressed(output / ('profile_%02d_truth.npz' % p), target=y[index], index=index)
        for name in METHODS:
            residual = pred[name][index] - y[index]
            rows.append(dict(id=p, role=items[p]['role'], method=name,
                rmse=float(np.sqrt(np.mean(residual ** 2))), mae=float(np.mean(np.abs(residual))),
                max_absolute_error=float(np.max(np.abs(residual))),
                max_underprediction=float(max(0., np.max(-residual)))))
    macro = {role: {name: float(np.mean([r['rmse'] for r in rows if r['role'] == role and r['method'] == name]))
                   for name in METHODS} for role in ['calibration', 'pilot']}
    best = min([m for m in METHODS if m != PRIMARY], key=lambda m: (macro['pilot'][m], METHODS.index(m)))
    reduction = 1. - macro['pilot'][PRIMARY] / macro['pilot'][best]
    positive_profiles = sum(next(r['rmse'] for r in rows if r['id'] == p and r['method'] == PRIMARY) <
                            next(r['rmse'] for r in rows if r['id'] == p and r['method'] == best) for p in PILOT)
    summary = dict(status='COMPLETE_PENDING_INDEPENDENT_AUDIT', source_sha256=sha(__file__),
        primary=PRIMARY, macro_rmse_C=macro, best_pilot_control=best,
        relative_error_reduction_vs_best_control=reduction, positive_profiles_vs_fixed_best_control=positive_profiles,
        prospective_clean_screen_pass=bool(reduction >= .03 and positive_profiles >= 2),
        calibration_is_diagnostic_only=True, confirmation_not_accessed=True,
        training=train_timing, elapsed_seconds=time.perf_counter() - clock.start)
    dump(output / 'metrics.json', rows)
    dump(output / 'summary.json', summary)
    clock.check()
    # Final source/locks retained; no post-PILOT fitting or model selection permitted.
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets-dir', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mode', choices=['costprobe', 'pilot'], required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
