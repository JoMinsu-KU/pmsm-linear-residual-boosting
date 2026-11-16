"""C17 fixed seven-method observable gate; root-registered execution only.

fit-predict reads base FIT and gate CAL targets, never held evaluation targets.
evaluate requires immutable model/prediction locks. All cohorts are exploratory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import minimize_scalar
import scipy

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP = ROOT / 'followup-risk-20261110'
PROTOCOL_SHA = '2a1aa6f2c815840a3419b6ae6bb05b0a68a8fa97f8780966cce4bd8614b3f227'
RISK_METRICS_SHA = '7d514ccd7f9471a4e5d8dcc29604d8782c6c0dad300cdc830a253d4e1c759282'
METHODS = ['direct9', 'C15', 'fixed_half', 'fixed_cal_mse', 'fixed_cal_macro_rmse',
           'load_threshold', 'observable_simplex_gate']
DIRECT = 'prediction__direct9_fitcv_selected'
HYBRID = 'prediction__positive_rc_insample_boost'
RULES = ['direct', 'hybrid', 'rising_load', 'falling_load']


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def local(value):
    p = Path(value)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('Path outside original campaign')
    return p


def ref(path):
    p = local(path)
    return dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


class Clock:
    def __init__(self, seconds, output):
        self.start, self.seconds, self.output = time.monotonic(), seconds, output
        self.inputs = {}

    def check(self, training=False):
        if time.monotonic() - self.start >= self.seconds:
            raise TimeoutError('Registered internal stage cap reached')
        now = datetime.now(timezone.utc)
        if now >= datetime.fromisoformat('2026-11-11T00:23:59+00:00'):
            raise TimeoutError('Campaign deadline reached')
        if training and now >= datetime.fromisoformat('2026-11-10T20:23:59+00:00'):
            raise TimeoutError('New-training cutoff reached')
        if sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file()) > 100_000_000:
            raise RuntimeError('Stage100MB output cap exceeded')

    def checked(self, entry):
        self.check()
        p = local(entry['path'])
        observed = sha(p)
        if observed != entry['sha256']:
            raise ValueError('Pinned input hash mismatch: ' + str(p))
        if str(p) in self.inputs and self.inputs[str(p)] != observed:
            raise ValueError('Conflicting input identity')
        self.inputs[str(p)] = observed
        return p


def load_config(args, clock):
    cfg = read(clock.checked(dict(path=args.config, sha256=args.config_sha256)))
    if cfg['schema'] != 'c17-gate-dataset-v001' or cfg['scientific_execution_admitted'] is not True:
        raise ValueError('Explicit C17 admission required')
    if cfg['protocol'] != dict(path='followup-risk-20261110/protocols/c17-gate-v001-frozen.md', sha256=PROTOCOL_SHA):
        raise ValueError('Frozen protocol identity changed')
    clock.checked(cfg['protocol'])
    fixed = dict(direct_key=DIRECT, hybrid_key=HYBRID, ridge_lambda=.01,
                 scalar_xatol=1e-12, scalar_maxiter=500, simplex_feasibility_tolerance=1e-10,
                 negative_face_tolerance=1e-12, fw_gap_relative_tolerance=1e-8,
                 methods=METHODS, max_output_bytes=100_000_000,
                 deadline_utc='2026-11-11T00:23:59Z', new_training_cutoff_utc='2026-11-10T20:23:59Z')
    if any(cfg[k] != v for k, v in fixed.items()):
        raise ValueError('C17 frozen numerical choice changed')
    risk = read(clock.checked(cfg['risk_config']))
    clock.checked(risk['protocol'])
    old = read(clock.checked(risk['original_manifest']))
    if (risk['schema'] != 'c16-risk-dataset-v001' or risk['all_evaluation_previously_exposed'] is not True
            or risk['dataset'] != cfg['dataset'] or old['dataset'] != cfg['dataset']
            or old['roles']['fit'] != risk['train_ids'] or old['fit_cache'] != risk['fit_cache']):
        raise ValueError('Original dataset/FIT identity changed')
    expected_roles = [r for r in old['roles'] if r != 'fit']
    if [r['role'] for r in risk['role_inputs']] != expected_roles:
        raise ValueError('Complete original role coverage required')
    used, cal = set(risk['train_ids']), []
    for role in risk['role_inputs']:
        if role['ids'] != old['roles'][role['role']] or role['ids'] != [e['id'] for e in role['records']]:
            raise ValueError('Role/profile coverage changed')
        if used.intersection(role['ids']) or len(set(role['ids'])) != len(role['ids']):
            raise ValueError('Overlapping profile roles')
        used.update(role['ids'])
        if role['usage'] == 'calibration':
            cal.append(role['role'])
        elif role['usage'] != 'exploratory_evaluation':
            raise ValueError('Only exposed exploratory evaluation admitted')
    if cal != [risk['calibration_role']]:
        raise ValueError('Exactly one gate calibration role required')
    return cfg, risk


def variables(instant, heat, transform):
    if instant.ndim != 2 or instant.shape[1] != 8 or heat.shape != (len(instant), 15):
        raise ValueError('Inherited feature geometry changed')
    if not np.isfinite(instant).all() or not np.isfinite(heat).all():
        raise ValueError('Nonfinite inherited features')
    current, speed = instant[:, 0], instant[:, 3]
    if np.any(current < 0) or np.any(speed < 0):
        raise ValueError('Magnitude features cannot be negative')
    return np.column_stack((np.log1p(current / transform['I_rms']),
        np.log1p(speed / transform['omega_rms']),
        (heat[:, 0] - heat[:, 9]) / transform['I_rms'] ** 2))


def basis(v, transform):
    z = (v - np.asarray(transform['mean'])) / np.asarray(transform['std'])
    ramps = np.clip(.5 + .25 * z, 0., 1.)
    return np.column_stack((np.zeros(len(v)), np.ones(len(v)), ramps[:, 0], 1. - ramps[:, 0],
                            ramps[:, 1], 1. - ramps[:, 1], ramps[:, 2], 1. - ramps[:, 2]))


def fit_transform(risk, clock):
    instants, heats, targets, counts = [], [], [], {}
    if [e['id'] for e in risk['fit_cache']] != risk['train_ids']:
        raise ValueError('Ordered FIT caches required')
    for entry in risk['fit_cache']:
        clock.check(training=True)
        ix = np.arange(300, entry['rows'], 5, dtype=np.int64)
        if not len(ix):
            raise ValueError('Empty FIT endpoint profile')
        with np.load(clock.checked(entry), allow_pickle=False) as saved:
            instant = saved['instant']; heat = saved['heat']; target = saved[entry['target_key']]
            if instant.shape != (entry['rows'], 8) or heat.shape != (entry['rows'], 15) or target.shape != (entry['rows'],):
                raise ValueError('FIT cache shape changed')
            instants.append(instant[ix].astype(np.float64)); heats.append(heat[ix].astype(np.float64))
            targets.append(target[ix].astype(np.float64))
        counts[str(entry['id'])] = len(ix)
    instant, heat, y = np.concatenate(instants), np.concatenate(heats), np.concatenate(targets)
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite FIT target')
    result = dict(I_rms=max(float(np.sqrt(np.mean(instant[:, 0] ** 2))), 1e-8),
                  omega_rms=max(float(np.sqrt(np.mean(instant[:, 3] ** 2))), 1e-8),
                  y_scale=max(float(np.percentile(y, 75) - np.percentile(y, 25)), 1.),
                  high_temperature_threshold_c=float(np.quantile(y, .95)),
                  fit_ids=risk['train_ids'], fit_endpoint_counts=counts)
    v = variables(instant, heat, result)
    result.update(mean=v.mean(0).tolist(), std=np.maximum(v.std(0), 1e-8).tolist())
    return result


def prior_role_lineage(role, clock):
    lock_path = clock.checked(role['prediction_lock'])
    lock = read(lock_path)
    audit = read(clock.checked(role['prior_evaluation_audit']))
    if audit.get('status') != 'PASS' or audit.get('stage') != 'evaluation' or lock['role'] != role['role']:
        raise ValueError('Prior independently audited evaluation lineage required')
    if [e['id'] for e in lock['records']] != role['ids']:
        raise ValueError('Original prediction lock role coverage')
    for entry, previous in zip(role['records'], lock['records']):
        if (entry['features']['sha256'] != previous['sha256'] or entry['rows'] != previous['rows']
                or local(entry['features']['path']) != (lock_path.parent / previous['path']).resolve()):
            raise ValueError('Frozen old prediction source mapping changed')
        for pin in (entry['features'], entry['truth'], role['prediction_lock']):
            suffix = pin['path'].replace('\\', '/')
            hits = [d for p, d in audit['inputs'].items() if p.replace('\\', '/').endswith('/' + suffix)]
            if not hits or any(d != pin['sha256'] for d in hits):
                raise ValueError('Original full audit does not bind reused input')
    # This checks metadata only; no held target NPZ is opened or hashed here.


def role_records(role, transform, clock, calibration=False):
    prior_role_lineage(role, clock)
    records = []
    for entry in role['records']:
        clock.check(training=calibration)
        ix = np.arange(300, entry['rows'], dtype=np.int64)
        with np.load(clock.checked(entry['features']), allow_pickle=False) as saved:
            if not np.array_equal(saved['index'], ix):
                raise ValueError('Every300: endpoint required')
            instant, heat = saved['instant'], saved['heat']
            source_row = saved['source_row']
            d, h = saved[DIRECT].astype(np.float64), saved[HYBRID].astype(np.float64)
            if (instant.shape != (entry['rows'], 8) or heat.shape != (entry['rows'], 15)
                    or source_row.shape != (entry['rows'],)
                    or not np.issubdtype(source_row.dtype, np.integer)
                    or not np.all(np.diff(source_row) > 0)):
                raise ValueError('Whole-profile inherited shape/source-row mismatch')
            v = variables(instant[ix].astype(np.float64), heat[ix].astype(np.float64), transform)
            source_row = source_row[ix].astype(np.int64)
        if d.shape != ix.shape or h.shape != ix.shape or not np.isfinite(d).all() or not np.isfinite(h).all():
            raise ValueError('Complete finite component predictions required')
        row = dict(id=entry['id'], index=ix, source_row=source_row, d=d, h=h, v=v,
                   phi=basis(v, transform), source_features=entry['features'], truth_ref=entry['truth'])
        if calibration:
            with np.load(clock.checked(entry['truth']), allow_pickle=False) as saved:
                y = saved['target'].astype(np.float64)
                if not np.array_equal(saved['index'], ix) or y.shape != ix.shape or not np.isfinite(y).all():
                    raise ValueError('Aligned finite CAL truth required')
                row['y'] = y
        records.append(row)
    return records


def sufficient_statistics(records, transform):
    values = {k: [] for k in ('profile_ids', 'counts', 'gram', 'linear', 'constant', 'mse_cross', 'mse_delta2')}
    for row in records:
        delta, residual = row['h'] - row['d'], row['y'] - row['d']
        a = delta[:, None] * row['phi'] / transform['y_scale']
        b = residual / transform['y_scale']
        n = len(b)
        for k, value in dict(profile_ids=row['id'], counts=n, gram=(a.T @ a) / n,
            linear=(a.T @ b) / n, constant=float(np.mean(b * b)),
            mse_cross=float(np.mean(delta * residual)), mse_delta2=float(np.mean(delta * delta))).items():
            values[k].append(value)
    return {k: np.asarray(v, dtype=np.int64 if k in ('profile_ids', 'counts') else np.float64)
            for k, v in values.items()}


def macro_rmse(records, weight):
    return float(np.mean([np.sqrt(np.mean((r['d'] + weight * (r['h'] - r['d']) - r['y']) ** 2)) for r in records]))


def scalar_blend(records, cfg, clock):
    def objective(weight):
        clock.check(training=True)
        return macro_rmse(records, float(weight))
    result = minimize_scalar(objective, bounds=(0., 1.), method='bounded',
                             options={'xatol': cfg['scalar_xatol'], 'maxiter': cfg['scalar_maxiter']})
    if not result.success or not math.isfinite(float(result.fun)):
        raise ArithmeticError('Bounded scalar blend optimizer failed')
    endpoints = [dict(weight=w, objective=objective(w)) for w in (0., 1.)]
    candidates = endpoints + [dict(weight=float(result.x), objective=objective(result.x))]
    chosen = min(candidates, key=lambda x: (x['objective'], x['weight']))
    lo, hi = [], []
    for row in records:
        delta = row['h'] - row['d']; error = row['d'] + chosen['weight'] * delta - row['y']
        norm = float(np.sqrt(np.mean(error * error)))
        if norm == 0.:
            slope = float(np.sqrt(np.mean(delta * delta))); lo.append(-slope); hi.append(slope)
        else:
            slope = float(np.mean(delta * error)) / norm; lo.append(slope); hi.append(slope)
    left, right = float(np.mean(lo)), float(np.mean(hi))
    subgradient = float(np.clip(0., left, right))
    gap = subgradient * chosen['weight'] - min(0., subgradient)
    return dict(**chosen, optimizer=dict(success=bool(result.success), x=float(result.x),
        objective=float(result.fun), nfev=int(result.nfev), message=str(result.message),
        xatol=cfg['scalar_xatol'], maxiter=cfg['scalar_maxiter']), endpoint_objectives=endpoints,
        subgradient_interval=[left, right], convex_gap=max(0., float(gap)),
        gap_scope='Convex objective-gap upper bound reported; no extra result-chosen tolerance')


def simplex_solver(q, r, constant, cfg, clock):
    np.linalg.cholesky(q)
    best, faces = None, []
    for mask in range(1, 256):
        clock.check(training=True)
        active = np.array([j for j in range(8) if mask & (1 << j)], dtype=np.int64)
        face = q[np.ix_(active, active)]
        solved = np.linalg.solve(face, np.column_stack((r[active], np.ones(len(active)))))
        denominator = float(np.sum(solved[:, 1]))
        if denominator <= 0.:
            raise ArithmeticError('SPD simplex face lost positive equality denominator')
        nu = (float(np.sum(solved[:, 0])) - 1.) / denominator
        trial = solved[:, 0] - nu * solved[:, 1]
        feasible = float(trial.min()) >= -cfg['negative_face_tolerance']
        record = dict(mask=mask, feasible=feasible, minimum_unclipped_coordinate=float(trial.min()))
        if feasible:
            theta = np.zeros(8, dtype=np.float64)
            theta[active] = np.maximum(trial, 0.)
            theta /= theta.sum()
            objective = float(.5 * theta @ q @ theta - r @ theta + constant)
            record['objective'] = objective
            if best is None or (objective, mask) < (best[0], best[1]):
                best = objective, mask, theta
        faces.append(record)
    if best is None:
        raise ArithmeticError('No feasible simplex face')
    objective, mask, theta = best
    gradient = q @ theta - r
    raw_gap = float(gradient @ theta - np.min(gradient))
    gap = max(0., raw_gap)
    tolerance = cfg['fw_gap_relative_tolerance'] * (1. + abs(objective))
    equality = abs(float(theta.sum()) - 1.)
    minimum = float(theta.min())
    if (not np.isfinite(theta).all() or equality > cfg['simplex_feasibility_tolerance']
            or minimum < -cfg['simplex_feasibility_tolerance'] or raw_gap < -tolerance or gap > tolerance):
        raise ArithmeticError('Simplex solution numerical qualification failed')
    certificate = dict(spd_cholesky_pass=True, equality_residual=equality,
        minimum_coefficient=minimum, fw_gap=gap, raw_fw_gap=raw_gap, gap_tolerance=tolerance,
        feasibility_tolerance=cfg['simplex_feasibility_tolerance'], qualified=True)
    return dict(theta=theta.tolist(), Q=q.tolist(), r=r.tolist(), constant=float(constant),
                objective=objective, selected_support_mask=mask, faces=faces, certificate=certificate)


def threshold_weight(row, rule):
    if rule == 0:
        return np.zeros(len(row['d']))
    if rule == 1:
        return np.ones(len(row['d']))
    if rule == 2:
        return (row['v'][:, 2] >= 0.).astype(np.float64)
    if rule == 3:
        return (row['v'][:, 2] < 0.).astype(np.float64)
    raise ValueError('Unknown fixed load rule')


def fit_model(records, stats, ids, cfg, clock):
    clock.check(training=True)
    rows = [r for r in records if r['id'] in ids]
    positions = [list(stats['profile_ids']).index(pid) for pid in ids]
    numerator = float(np.mean(stats['mse_cross'][positions]))
    denominator = float(np.mean(stats['mse_delta2'][positions]))
    mse_weight = float(np.clip(numerator / denominator, 0., 1.)) if denominator != 0. else 0.
    fixed_mse = dict(weight=mse_weight, numerator=numerator, denominator=denominator)
    scalar = scalar_blend(rows, cfg, clock)
    rule_scores = []
    for rule in range(4):
        rule_scores.append(float(np.mean([np.sqrt(np.mean((r['d'] + threshold_weight(r, rule)
            * (r['h'] - r['d']) - r['y']) ** 2)) for r in rows])))
    selected_rule = min(range(4), key=lambda j: (rule_scores[j], j))
    theta_ref = np.r_[1. - mse_weight, mse_weight, np.zeros(6)]
    lam = cfg['ridge_lambda']
    q = 2. * (np.mean(stats['gram'][positions], axis=0) + lam * np.eye(8))
    r = 2. * (np.mean(stats['linear'][positions], axis=0) + lam * theta_ref)
    constant = float(np.mean(stats['constant'][positions]) + lam * (theta_ref @ theta_ref))
    gate = simplex_solver(q, r, constant, cfg, clock)
    gate.update(theta_ref=theta_ref.tolist(), ridge_lambda=lam)
    return dict(fit_ids=list(ids), fixed_cal_mse=fixed_mse, fixed_cal_macro_rmse=scalar,
        load_threshold=dict(rule=selected_rule, rule_name=RULES[selected_rule], cal_macro_rmse=rule_scores), gate=gate)


def predictions(row, model):
    d, h = row['d'], row['h']; delta = h - d
    weight = row['phi'] @ np.asarray(model['gate']['theta'])
    if not np.isfinite(weight).all() or weight.min() < -1e-12 or weight.max() > 1. + 1e-12:
        raise ArithmeticError('Convex gate weight range violated')
    # Only floating-point endpoint residue is clipped, never scientific values.
    weight = np.clip(weight, 0., 1.)
    values = dict(direct9=d, C15=h, fixed_half=(d + h) / 2.,
        fixed_cal_mse=d + model['fixed_cal_mse']['weight'] * delta,
        fixed_cal_macro_rmse=d + model['fixed_cal_macro_rmse']['weight'] * delta,
        load_threshold=d + threshold_weight(row, model['load_threshold']['rule']) * delta,
        observable_simplex_gate=d + weight * delta)
    if list(values) != METHODS or any(v.shape != d.shape or not np.isfinite(v).all() for v in values.values()):
        raise ArithmeticError('Prediction shape/finite qualification')
    gap = values['observable_simplex_gate'] - values['fixed_cal_mse']
    diagnostic = dict(weight_min=float(weight.min()), weight_mean=float(weight.mean()),
        weight_max=float(weight.max()), weight_std=float(weight.std()),
        versus_fixed_mse_rms_difference=float(np.sqrt(np.mean(gap * gap))),
        versus_fixed_mse_max_absolute_difference=float(np.max(np.abs(gap))),
        label_free=True)
    return values, weight, diagnostic


def fit_predict(args, cfg, risk, output, clock):
    transform = fit_transform(risk, clock)
    dump(output / 'transform.json', transform)
    cal_role = next(r for r in risk['role_inputs'] if r['role'] == risk['calibration_role'])
    calibration = role_records(cal_role, transform, clock, calibration=True)
    stats = sufficient_statistics(calibration, transform)
    np.savez_compressed(output / 'calibration_statistics.npz', **stats)
    ids = cal_role['ids']
    if len(ids) < 2:
        raise ValueError('LOPO needs at least two CAL profiles')
    lopo = [dict(held_out=pid, model=fit_model(calibration, stats, [q for q in ids if q != pid], cfg, clock)) for pid in ids]
    final = fit_model(calibration, stats, ids, cfg, clock)
    models = dict(schema='c17-gate-models-v001', dataset=cfg['dataset'], methods=METHODS,
        calibration_role=cal_role['role'], calibration_ids=ids, final=final, lopo=lopo)
    dump(output / 'models.json', models)
    model_files = {name: sha(output / name) for name in
                   ['configuration.json', 'transform.json', 'calibration_statistics.npz', 'models.json']}
    model_lock = dict(schema='c17-gate-model-lock-v001', status='MODELS_LOCKED',
        dataset=cfg['dataset'], methods=METHODS, config_sha256=args.config_sha256,
        source_sha256=sha(__file__), files=model_files, calibration_role=cal_role['role'],
        calibration_ids=ids, calibration_truth_accessed=True, evaluation_truth_accessed=False,
        input_hashes_before_evaluation_features=dict(clock.inputs), locked_at_utc=utc())
    dump(output / 'model_lock.json', model_lock)
    entries, diagnostics = [], []

    def save_role(name, source_role, rows, models_by_id, kind):
        folder = output / 'predictions' / name
        folder.mkdir(parents=True, exist_ok=False)
        for row in rows:
            clock.check()
            values, weight, diagnostic = predictions(row, models_by_id[row['id']])
            filename = folder / ('profile_' + str(row['id']) + '.npz')
            np.savez_compressed(filename, index=row['index'], source_row=row['source_row'], gate_weight=weight,
                                **{'prediction__' + key: value for key, value in values.items()})
            entries.append(dict(role=name, source_role=source_role, profile=row['id'], **ref(filename),
                model_kind=kind, held_out=row['id'] if kind == 'lopo' else None,
                source_features=row['source_features'], truth_ref=row['truth_ref'],
                evaluated_rows=len(row['index'])))
            diagnostics.append(dict(role=name, profile=row['id'], **diagnostic))
    save_role('calibration_lopo', cal_role['role'], calibration,
              {r['held_out']: r['model'] for r in lopo}, 'lopo')
    for role in risk['role_inputs']:
        rows = calibration if role['role'] == cal_role['role'] else role_records(role, transform, clock)
        save_role(role['role'], role['role'], rows, {r['id']: final for r in rows}, 'final')
    dump(output / 'weight_diagnostics.json', diagnostics)
    dump(output / 'input_pins.json', clock.inputs)
    prediction_lock = dict(schema='c17-gate-prediction-lock-v001', status='ALL_PREDICTIONS_LOCKED',
        dataset=cfg['dataset'], methods=METHODS, config_sha256=args.config_sha256, source_sha256=sha(__file__),
        model_lock=ref(output / 'model_lock.json'), predictions=entries,
        weight_diagnostics=ref(output / 'weight_diagnostics.json'), calibration_truth_accessed=True,
        evaluation_truth_accessed=False, all_evaluation_previously_exposed=True,
        independent_confirmation=False, locked_at_utc=utc())
    dump(output / 'prediction_lock.json', prediction_lock)
    return dict(status='GATE_MODELS_PREDICTIONS_LOCKED', model_lock=ref(output / 'model_lock.json'),
        prediction_lock=ref(output / 'prediction_lock.json'), models=1 + len(lopo),
        predictions=len(entries), methods=METHODS, calibration_role=cal_role['role'],
        high_temperature_threshold_c=transform['high_temperature_threshold_c'],
        calibration_truth_accessed=True, evaluation_truth_accessed=False, independent_confirmation=False)


def metric_module(clock):
    path = clock.checked(dict(path='followup-risk-20261110/code/risk_screen_analysis_v001.py', sha256=RISK_METRICS_SHA))
    spec = importlib.util.spec_from_file_location('c17_shared_risk_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(args, cfg, risk, output, clock):
    if not args.prediction_lock or not args.prediction_lock_sha256:
        raise ValueError('Evaluation requires exact prediction lock')
    lockpath = clock.checked(dict(path=args.prediction_lock, sha256=args.prediction_lock_sha256))
    lock = read(lockpath)
    if (lock.get('schema') != 'c17-gate-prediction-lock-v001' or lock.get('status') != 'ALL_PREDICTIONS_LOCKED'
            or lock['dataset'] != cfg['dataset'] or lock['config_sha256'] != args.config_sha256
            or lock['source_sha256'] != sha(__file__) or lock['methods'] != METHODS
            or lock['evaluation_truth_accessed'] is not False or lock['independent_confirmation'] is not False
            or (lockpath.parent / 'failure.json').exists()):
        raise ValueError('Qualified same-source exploratory prediction lock required')
    modelpath = clock.checked(lock['model_lock'])
    if modelpath != lockpath.parent / 'model_lock.json':
        raise ValueError('Model/prediction lock directory mismatch')
    model_lock = read(modelpath)
    if (model_lock['schema'] != 'c17-gate-model-lock-v001' or model_lock['status'] != 'MODELS_LOCKED'
            or model_lock['config_sha256'] != args.config_sha256 or model_lock['source_sha256'] != sha(__file__)
            or model_lock['evaluation_truth_accessed'] is not False):
        raise ValueError('Frozen model lock identity changed')
    for name, digest in model_lock['files'].items():
        path = local(modelpath.parent / name)
        if path.parent != modelpath.parent:
            raise ValueError('Model artifact path escape')
        clock.checked(dict(path=str(path), sha256=digest))
    transform = read(modelpath.parent / 'transform.json')
    roles = {r['role']: r for r in risk['role_inputs']}
    calibration = risk['calibration_role']
    expected = [('calibration_lopo', calibration, p) for p in roles[calibration]['ids']]
    expected += [(r['role'], r['role'], p) for r in risk['role_inputs'] for p in r['ids']]
    if [(e['role'], e['source_role'], e['profile']) for e in lock['predictions']] != expected:
        raise ValueError('Exact LOPO and full final prediction coverage required')
    saved_predictions = []
    for entry in lock['predictions']:
        source_entry = next(e for e in roles[entry['source_role']]['records'] if e['id'] == entry['profile'])
        if entry['truth_ref'] != source_entry['truth'] or entry['source_features'] != source_entry['features']:
            raise ValueError('Evaluation inputs differ from frozen original role')
        with np.load(clock.checked(entry), allow_pickle=False) as saved:
            fields = set(saved.files)
            if fields != {'index', 'source_row', 'gate_weight'} | {'prediction__' + m for m in METHODS}:
                raise ValueError('Exactly seven saved methods required')
            arrays = {k: saved[k].copy() for k in saved.files}
        ix = np.arange(300, source_entry['rows'], dtype=np.int64)
        if (not np.array_equal(arrays['index'], ix) or arrays['source_row'].shape != ix.shape
                or any(arrays['prediction__' + m].shape != ix.shape for m in METHODS)
                or any(not np.isfinite(x).all() for x in arrays.values())):
            raise ValueError('Complete finite prediction coverage required')
        saved_predictions.append((entry, arrays))
    # All seven predictions and their immutable models have been validated first.
    metrics_helper = metric_module(clock)
    truth, predictions_by_method = {}, {m: {} for m in METHODS}
    truth_cache = {}
    for entry, arrays in saved_predictions:
        clock.check()
        key = (entry['source_role'], entry['profile'])
        if key not in truth_cache:
            with np.load(clock.checked(entry['truth_ref']), allow_pickle=False) as saved:
                y, ix = saved['target'].astype(np.float64), saved['index']
                if not np.array_equal(ix, arrays['index']) or y.shape != ix.shape or not np.isfinite(y).all():
                    raise ValueError('Exact aligned evaluation truth required')
                truth_cache[key] = y
        role, pid = entry['role'], entry['profile']
        truth.setdefault(role, {})[pid] = truth_cache[key]
        for m in METHODS:
            predictions_by_method[m].setdefault(role, {})[pid] = arrays['prediction__' + m]
    threshold = transform['high_temperature_threshold_c']
    dt = risk['metrics']['sample_interval_seconds']
    metrics = {m: {} for m in METHODS}
    for m in METHODS:
        for role in truth:
            clock.check()
            metrics[m][role] = metrics_helper.role_metrics(truth[role], predictions_by_method[m][role], threshold, dt)
    comparisons = []
    for role in truth:
        gate = metrics['observable_simplex_gate'][role]['summary']
        for comparator in ('direct9', 'fixed_cal_macro_rmse'):
            base = metrics[comparator][role]['summary']
            def ratio(key):
                return gate[key] / base[key] if base[key] != 0. else None
            differences = [g['rmse'] - b['rmse'] for g, b in zip(
                metrics['observable_simplex_gate'][role]['per_profile'], metrics[comparator][role]['per_profile'])]
            comparisons.append(dict(role=role, comparator=comparator,
                delta_macro_rmse=gate['macro_rmse'] - base['macro_rmse'], rmse_ratio=ratio('macro_rmse'),
                under_rmse_ratio=ratio('macro_under_rmse'), over_mae_ratio=ratio('macro_over_mae'),
                under_rmse_diagnostic_flag=gate['macro_under_rmse'] > 1.05 * base['macro_under_rmse'] + 1e-12,
                over_mae_diagnostic_flag=gate['macro_over_mae'] > 1.25 * base['macro_over_mae'] + 1e-12,
                per_profile_rmse_differences=differences, profile_ids=list(truth[role]),
                wins=sum(d < 0. for d in differences), ties=sum(d == 0. for d in differences),
                losses=sum(d > 0. for d in differences),
                zero_baseline_ratio_policy='null ratio; exact inequality flag without denominator floor'))
    dump(output / 'metrics.json', dict(dataset=cfg['dataset'], high_temperature_threshold_c=threshold,
                                      methods=metrics))
    dump(output / 'comparison.json', comparisons)
    dump(output / 'input_pins.json', clock.inputs)
    return dict(status='EXPLORATORY_GATE_EVALUATION_COMPLETE', prediction_lock=ref(lockpath),
        methods=METHODS, roles=list(truth), high_temperature_threshold_c=threshold,
        metrics_source_sha256=RISK_METRICS_SHA, independent_confirmation=False,
        all_evaluation_previously_exposed=True, model_selection_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['fit-predict', 'evaluate'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--prediction-lock')
    parser.add_argument('--prediction-lock-sha256')
    args = parser.parse_args()
    cap = 300. if args.mode == 'fit-predict' else 120.
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= cap:
        raise ValueError('Finite registered mode-specific cap required')
    output = local(args.output_dir)
    if FOLLOWUP / 'runs' not in output.parents:
        raise ValueError('Output must be a new followup run directory')
    output.mkdir(parents=True, exist_ok=False)
    clock = Clock(args.max_seconds, output)
    dump(output / 'configuration.json', dict(vars(args), source=str(Path(__file__).resolve()),
        source_sha256=sha(__file__), output_dir=str(output), pid=os.getpid(), argv=sys.argv,
        started_utc=utc(), numpy=np.__version__, scipy=scipy.__version__))
    try:
        cfg, risk = load_config(args, clock)
        result = fit_predict(args, cfg, risk, output, clock) if args.mode == 'fit-predict' else evaluate(args, cfg, risk, output, clock)
        clock.check()
        result.update(dataset=cfg['dataset'], source_sha256=sha(__file__), config_sha256=args.config_sha256,
            elapsed_seconds=time.monotonic() - clock.start,
            files={p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
        dump(output / 'summary.json', result)
        clock.check()
        print(json.dumps(dict(status=result['status'], dataset=cfg['dataset'],
                              elapsed_seconds=time.monotonic() - clock.start)))
    except Exception as error:
        dump(output / 'failure.json', dict(status='FAILED', error_type=type(error).__name__,
            message=str(error), elapsed_seconds=time.monotonic() - clock.start))
        raise


if __name__ == '__main__':
    main()
