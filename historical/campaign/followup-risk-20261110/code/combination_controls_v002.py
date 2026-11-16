"""Fixed C17 five-pair/five-rule exploratory controls; root execution only.

No base-model refit, gate, shift, held-out pair selection or confirmation claim.
Only the exact pinned v001 scalar helper and scientific metrics are imported.
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
import scipy

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP = ROOT / 'followup-risk-20261110'
PROTOCOL = dict(path='followup-risk-20261110/protocols/combination-controls-v002.md',
                sha256='eb06ca878a603ba954bc5f53613f5d16749da87ba041798ab22241aa794fffab')
SCALAR_SOURCE = dict(path='followup-risk-20261110/code/gate_worker_v001.py',
    sha256='d73b6eb70db5c6e57f79a5334c4000d373189d3e49aaaf7371daee9b9a08e25b')
METRIC_SOURCE = dict(path='followup-risk-20261110/code/risk_screen_analysis_v001.py',
    sha256='7d514ccd7f9471a4e5d8dcc29604d8782c6c0dad300cdc830a253d4e1c759282')
DIRECT = 'prediction__direct9_fitcv_selected'
PAIRS = dict(c15='prediction__positive_rc_insample_boost',
             direct200='prediction__direct_boost',
             linear33='prediction__linear33_insample_boost',
             free25='prediction__unconstrained_rc_insample_boost',
             positive_prior='prediction__positive_rc')
RULES = ['left', 'right', 'half', 'cal_mse', 'cal_macro_rmse']
METHODS = [pair + '__' + rule for pair in PAIRS for rule in RULES]
PRIMARY = 'c15__cal_macro_rmse'
CAP_BYTES = 200_000_000


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def local(value):
    p = Path(value)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('Path outside campaign')
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
            raise TimeoutError('Internal mode cap reached')
        now = datetime.now(timezone.utc)
        if now >= datetime.fromisoformat('2026-11-11T00:23:59+00:00'):
            raise TimeoutError('Campaign deadline reached')
        if training and now >= datetime.fromisoformat('2026-11-10T20:23:59+00:00'):
            raise TimeoutError('New-training cutoff reached')
        if sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file()) > CAP_BYTES:
            raise RuntimeError('Stage200MB output cap exceeded')

    def checked(self, entry):
        self.check()
        p = local(entry['path'])
        observed = sha(p)
        if observed != entry['sha256'] or (str(p) in self.inputs and self.inputs[str(p)] != observed):
            raise ValueError('Pinned input changed: ' + str(p))
        self.inputs[str(p)] = observed
        return p


def load_module(entry, name, clock):
    path = clock.checked(entry)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(args, clock):
    cfg = read(clock.checked(dict(path=args.config, sha256=args.config_sha256)))
    clock.checked(PROTOCOL)
    if (cfg['schema'] != 'c17-gate-dataset-v001' or cfg['scientific_execution_admitted'] is not True
            or cfg['direct_key'] != DIRECT or cfg['hybrid_key'] != PAIRS['c15']
            or cfg['scalar_xatol'] != 1e-12 or cfg['scalar_maxiter'] != 500):
        raise ValueError('Exact admitted v001 scalar/dataset configuration required')
    if cfg['protocol'] != dict(path='followup-risk-20261110/protocols/c17-gate-v001-frozen.md',
            sha256='2a1aa6f2c815840a3419b6ae6bb05b0a68a8fa97f8780966cce4bd8614b3f227'):
        raise ValueError('Inherited protocol changed')
    clock.checked(cfg['protocol'])
    risk = read(clock.checked(cfg['risk_config']))
    clock.checked(risk['protocol'])
    old = read(clock.checked(risk['original_manifest']))
    if (risk['schema'] != 'c16-risk-dataset-v001' or risk['scientific_execution_admitted'] is not True
            or risk['all_evaluation_previously_exposed'] is not True
            or risk['dataset'] != cfg['dataset'] or old['dataset'] != cfg['dataset']
            or old['roles']['fit'] != risk['train_ids'] or old['fit_cache'] != risk['fit_cache']):
        raise ValueError('Original exposed dataset/FIT identity changed')
    if [r['role'] for r in risk['role_inputs']] != [r for r in old['roles'] if r != 'fit']:
        raise ValueError('Original full role coverage required')
    used, calibration = set(risk['train_ids']), []
    for role in risk['role_inputs']:
        if (role['ids'] != old['roles'][role['role']] or role['ids'] != [e['id'] for e in role['records']]
                or len(set(role['ids'])) != len(role['ids']) or used.intersection(role['ids'])):
            raise ValueError('Disjoint complete source roles required')
        used.update(role['ids'])
        if role['usage'] == 'calibration':
            calibration.append(role['role'])
        elif role['usage'] != 'exploratory_evaluation':
            raise ValueError('Only previously exposed evaluation admitted')
    if calibration != [risk['calibration_role']]:
        raise ValueError('Exactly one original CAL role required')
    return cfg, risk


def fit_threshold(risk, clock):
    targets, counts = [], {}
    if [e['id'] for e in risk['fit_cache']] != risk['train_ids']:
        raise ValueError('Original ordered FIT cache required')
    for entry in risk['fit_cache']:
        with np.load(clock.checked(entry), allow_pickle=False) as saved:
            y = saved[entry['target_key']].astype(np.float64)
            if y.shape != (entry['rows'],) or not np.isfinite(y).all():
                raise ValueError('Finite full FIT targets required')
            sampled = y[300::5]
        if not len(sampled):
            raise ValueError('Empty FIT profile')
        targets.append(sampled)
        counts[str(entry['id'])] = len(sampled)
    return dict(high_temperature_threshold_c=float(np.quantile(np.concatenate(targets), .95)),
                fit_ids=risk['train_ids'], fit_endpoint_counts=counts, sample_scope='300::5')


def prior_lineage(role, clock):
    lockpath = clock.checked(role['prediction_lock'])
    lock = read(lockpath)
    audit = read(clock.checked(role['prior_evaluation_audit']))
    if (audit.get('status') != 'PASS' or audit.get('stage') != 'evaluation'
            or lock['role'] != role['role'] or [e['id'] for e in lock['records']] != role['ids']):
        raise ValueError('Prior full independent audit and role lock required')
    for entry, previous in zip(role['records'], lock['records']):
        if (entry['features']['sha256'] != previous['sha256'] or entry['rows'] != previous['rows']
                or local(entry['features']['path']) != (lockpath.parent / previous['path']).resolve()):
            raise ValueError('Prior source mapping changed')
        for pin in (entry['features'], entry['truth'], role['prediction_lock']):
            suffix = pin['path'].replace('\\', '/')
            hits = [d for p, d in audit['inputs'].items() if p.replace('\\', '/').endswith('/' + suffix)]
            if not hits or any(d != pin['sha256'] for d in hits):
                raise ValueError('Prior full audit does not bind reused component artifact')
    # Reading audit metadata does not open or hash any held target file.


def role_records(role, clock, calibration=False):
    prior_lineage(role, clock)
    records = []
    for entry in role['records']:
        clock.check(training=calibration)
        ix = np.arange(300, entry['rows'], dtype=np.int64)
        if not len(ix):
            raise ValueError('Empty evaluation profile')
        with np.load(clock.checked(entry['features']), allow_pickle=False) as saved:
            sr = saved['source_row']
            if (not np.array_equal(saved['index'], ix) or sr.shape != (entry['rows'],)
                    or not np.issubdtype(sr.dtype, np.integer) or not np.all(np.diff(sr) > 0)):
                raise ValueError('Every300: endpoint and exact monotonic source rows required')
            source_row = sr[ix].astype(np.int64)
            d = saved[DIRECT].astype(np.float64)
            rights = {pair: saved[key].astype(np.float64) for pair, key in PAIRS.items()}
        if any(x.shape != ix.shape or not np.isfinite(x).all() for x in [d] + list(rights.values())):
            raise ValueError('All six complete finite base component predictions required')
        row = dict(id=entry['id'], index=ix, source_row=source_row, d=d, rights=rights,
                   source_features=entry['features'], truth_ref=entry['truth'])
        if calibration:
            with np.load(clock.checked(entry['truth']), allow_pickle=False) as saved:
                y = saved['target'].astype(np.float64)
                if not np.array_equal(saved['index'], ix) or y.shape != ix.shape or not np.isfinite(y).all():
                    raise ValueError('Aligned finite CAL target required')
                row['y'] = y
        records.append(row)
    return records


def statistics(records):
    result = dict(profile_ids=np.asarray([r['id'] for r in records], dtype=np.int64),
                  counts=np.asarray([len(r['index']) for r in records], dtype=np.int64))
    for pair in PAIRS:
        result[pair + '__mse_cross'] = np.asarray([float(np.mean((r['rights'][pair] - r['d']) *
                                                    (r['y'] - r['d']))) for r in records])
        result[pair + '__mse_delta2'] = np.asarray([float(np.mean((r['rights'][pair] - r['d']) ** 2))
                                                   for r in records])
    return result


def fit_model(pair, records, stats, ids, cfg, scalar, clock):
    clock.check(training=True)
    positions = [list(stats['profile_ids']).index(pid) for pid in ids]
    numerator = float(np.mean(stats[pair + '__mse_cross'][positions]))
    denominator = float(np.mean(stats[pair + '__mse_delta2'][positions]))
    weight = float(np.clip(numerator / denominator, 0., 1.)) if denominator != 0. else 0.
    rows = [dict(id=r['id'], d=r['d'], h=r['rights'][pair], y=r['y']) for r in records if r['id'] in ids]
    return dict(fit_ids=list(ids), cal_mse=dict(weight=weight, numerator=numerator, denominator=denominator),
                cal_macro_rmse=scalar.scalar_blend(rows, cfg, clock))


def predictions(row, models_by_pair):
    values = {}
    for pair in PAIRS:
        d, h, model = row['d'], row['rights'][pair], models_by_pair[pair]
        delta = h - d
        by_rule = dict(left=d, right=h, half=(d + h) / 2.,
                       cal_mse=d + model['cal_mse']['weight'] * delta,
                       cal_macro_rmse=d + model['cal_macro_rmse']['weight'] * delta)
        for rule, value in by_rule.items():
            if value.shape != d.shape or not np.isfinite(value).all():
                raise ArithmeticError('Finite full mixture prediction qualification failed')
            values[pair + '__' + rule] = value
    if list(values) != METHODS:
        raise ValueError('Fixed twenty-five prediction columns required')
    return values


def fit_predict(args, cfg, risk, output, clock):
    scalar = load_module(SCALAR_SOURCE, 'c17_v002_pinned_scalar_only', clock)
    threshold = fit_threshold(risk, clock)
    dump(output / 'fit_threshold.json', threshold)
    cal_role = next(r for r in risk['role_inputs'] if r['role'] == risk['calibration_role'])
    rows = role_records(cal_role, clock, calibration=True)
    ids = cal_role['ids']
    if len(ids) < 2:
        raise ValueError('CAL-LOPO requires at least two profiles')
    stats = statistics(rows)
    np.savez_compressed(output / 'calibration_statistics.npz', **stats)
    pairs = {}
    for pair, right_key in PAIRS.items():
        lopo = [dict(held_out=pid, model=fit_model(pair, rows, stats, [p for p in ids if p != pid], cfg, scalar, clock)) for pid in ids]
        final = fit_model(pair, rows, stats, ids, cfg, scalar, clock)
        pairs[pair] = dict(right_key=right_key, final=final, lopo=lopo)
    models = dict(schema='c17-combination-models-v002', dataset=cfg['dataset'], left_key=DIRECT,
                  pair_order=list(PAIRS), rules=RULES, methods=METHODS, primary=PRIMARY,
                  calibration_role=cal_role['role'], calibration_ids=ids, pairs=pairs)
    dump(output / 'models.json', models)
    model_files = {name: sha(output / name) for name in
                   ('configuration.json', 'fit_threshold.json', 'calibration_statistics.npz', 'models.json')}
    dump(output / 'model_lock.json', dict(schema='c17-combination-model-lock-v002', status='MODELS_LOCKED',
        dataset=cfg['dataset'], methods=METHODS, primary=PRIMARY, protocol=PROTOCOL,
        config_sha256=args.config_sha256, source_sha256=sha(__file__), scalar_dependency=SCALAR_SOURCE,
        files=model_files, calibration_role=cal_role['role'], calibration_ids=ids,
        calibration_truth_accessed=True, evaluation_truth_accessed=False,
        input_hashes_before_evaluation_features=dict(clock.inputs), locked_at_utc=utc()))
    entries = []

    def save_role(name, source_role, records, lopo):
        folder = output / 'predictions' / name
        folder.mkdir(parents=True, exist_ok=False)
        for row in records:
            clock.check()
            selected = {pair: next(f['model'] for f in value['lopo'] if f['held_out'] == row['id'])
                        if lopo else value['final'] for pair, value in pairs.items()}
            values = predictions(row, selected)
            path = folder / ('profile_' + str(row['id']) + '.npz')
            np.savez_compressed(path, index=row['index'], source_row=row['source_row'],
                                **{'prediction__' + key: value for key, value in values.items()})
            entries.append(dict(role=name, source_role=source_role, profile=row['id'], **ref(path),
                model_kind='lopo' if lopo else 'final', held_out=row['id'] if lopo else None,
                source_features=row['source_features'], truth_ref=row['truth_ref'], evaluated_rows=len(row['index'])))
    save_role('calibration_lopo', cal_role['role'], rows, True)
    for role in risk['role_inputs']:
        records = rows if role['role'] == cal_role['role'] else role_records(role, clock)
        save_role(role['role'], role['role'], records, False)
    dump(output / 'input_pins.json', clock.inputs)
    dump(output / 'prediction_lock.json', dict(schema='c17-combination-prediction-lock-v002',
        status='ALL_PREDICTIONS_LOCKED', dataset=cfg['dataset'], methods=METHODS, primary=PRIMARY,
        protocol=PROTOCOL, config_sha256=args.config_sha256, source_sha256=sha(__file__),
        model_lock=ref(output / 'model_lock.json'), predictions=entries,
        calibration_truth_accessed=True, evaluation_truth_accessed=False,
        all_evaluation_previously_exposed=True, independent_confirmation=False, locked_at_utc=utc()))
    return dict(status='COMBINATION_MODELS_PREDICTIONS_LOCKED', model_lock=ref(output / 'model_lock.json'),
        prediction_lock=ref(output / 'prediction_lock.json'), pair_count=5, rules=RULES, methods=METHODS,
        models=5 * (1 + len(ids)), predictions=len(entries), primary=PRIMARY,
        calibration_role=cal_role['role'], high_temperature_threshold_c=threshold['high_temperature_threshold_c'],
        calibration_truth_accessed=True, evaluation_truth_accessed=False, independent_confirmation=False)


def comparison(metrics, role, candidate, comparator):
    cand, base = metrics[candidate][role], metrics[comparator][role]
    g, b = cand['summary'], base['summary']
    if [r['id'] for r in cand['per_profile']] != [r['id'] for r in base['per_profile']]:
        raise ValueError('Aligned comparison profiles required')
    differences = [x['rmse'] - y['rmse'] for x, y in zip(cand['per_profile'], base['per_profile'])]
    def ratio(key):
        return g[key] / b[key] if b[key] != 0. else None
    return dict(role=role, candidate=candidate, comparator=comparator,
        delta_macro_rmse=g['macro_rmse'] - b['macro_rmse'], rmse_ratio=ratio('macro_rmse'),
        under_rmse_ratio=ratio('macro_under_rmse'), over_mae_ratio=ratio('macro_over_mae'),
        under_rmse_diagnostic_flag=g['macro_under_rmse'] > 1.05 * b['macro_under_rmse'] + 1e-12,
        over_mae_diagnostic_flag=g['macro_over_mae'] > 1.25 * b['macro_over_mae'] + 1e-12,
        per_profile_rmse_differences=differences, profile_ids=[r['id'] for r in cand['per_profile']],
        wins=sum(d < 0. for d in differences), ties=sum(d == 0. for d in differences),
        losses=sum(d > 0. for d in differences),
        zero_baseline_ratio_policy='null ratio; exact inequality flag without denominator floor')


def evaluate(args, cfg, risk, output, clock):
    if not args.prediction_lock or not args.prediction_lock_sha256:
        raise ValueError('Exact prediction lock required for evaluation')
    lockpath = clock.checked(dict(path=args.prediction_lock, sha256=args.prediction_lock_sha256))
    lock = read(lockpath)
    if (lock.get('schema') != 'c17-combination-prediction-lock-v002'
            or lock.get('status') != 'ALL_PREDICTIONS_LOCKED' or lock['dataset'] != cfg['dataset']
            or lock['config_sha256'] != args.config_sha256 or lock['source_sha256'] != sha(__file__)
            or lock['methods'] != METHODS or lock['primary'] != PRIMARY or lock['protocol'] != PROTOCOL
            or lock['evaluation_truth_accessed'] is not False or lock['independent_confirmation'] is not False
            or lock['all_evaluation_previously_exposed'] is not True or (lockpath.parent / 'failure.json').exists()):
        raise ValueError('Qualified same-source exploratory prediction lock required')
    modelpath = clock.checked(lock['model_lock'])
    mlock = read(modelpath)
    if (modelpath != lockpath.parent / 'model_lock.json' or mlock['schema'] != 'c17-combination-model-lock-v002'
            or mlock['status'] != 'MODELS_LOCKED' or mlock['config_sha256'] != args.config_sha256
            or mlock['source_sha256'] != sha(__file__) or mlock['methods'] != METHODS
            or mlock['evaluation_truth_accessed'] is not False or mlock['protocol'] != PROTOCOL):
        raise ValueError('Frozen scalar-model lock identity changed')
    expected_files = {'configuration.json', 'fit_threshold.json', 'calibration_statistics.npz', 'models.json'}
    if set(mlock['files']) != expected_files:
        raise ValueError('Exact four scalar-model artifacts required')
    for name, digest in mlock['files'].items():
        clock.checked(dict(path=str(modelpath.parent / name), sha256=digest))
    threshold = read(modelpath.parent / 'fit_threshold.json')['high_temperature_threshold_c']
    roles = {r['role']: r for r in risk['role_inputs']}
    cal = risk['calibration_role']
    expected = [('calibration_lopo', cal, pid) for pid in roles[cal]['ids']]
    expected += [(r['role'], r['role'], pid) for r in risk['role_inputs'] for pid in r['ids']]
    if [(e['role'], e['source_role'], e['profile']) for e in lock['predictions']] != expected:
        raise ValueError('Complete ordered CAL-LOPO and original-role coverage required')
    saved_predictions = []
    for entry in lock['predictions']:
        source = next(e for e in roles[entry['source_role']]['records'] if e['id'] == entry['profile'])
        if entry['truth_ref'] != source['truth'] or entry['source_features'] != source['features']:
            raise ValueError('Evaluation source mapping differs from frozen role')
        with np.load(clock.checked(entry), allow_pickle=False) as saved:
            if set(saved.files) != {'index', 'source_row'} | {'prediction__' + m for m in METHODS}:
                raise ValueError('Exactly twenty-five saved predictions required')
            arrays = {k: saved[k].copy() for k in saved.files}
        ix = np.arange(300, source['rows'], dtype=np.int64)
        if (arrays['index'].dtype != np.int64 or arrays['source_row'].dtype != np.int64
                or not np.array_equal(arrays['index'], ix) or arrays['source_row'].shape != ix.shape
                or any(arrays['prediction__' + m].dtype != np.float64 or arrays['prediction__' + m].shape != ix.shape for m in METHODS)
                or any(not np.isfinite(value).all() for value in arrays.values())):
            raise ValueError('Exact finite prediction dtype/geometry required')
        saved_predictions.append((entry, arrays))
    # All pair/rule predictions and scalar models are locked before any held truth.
    metric = load_module(METRIC_SOURCE, 'c17_v002_pinned_metrics', clock)
    truth, values, truth_cache, complementarity = {}, {m: {} for m in METHODS}, {}, []
    for entry, arrays in saved_predictions:
        clock.check()
        key = entry['source_role'], entry['profile']
        if key not in truth_cache:
            with np.load(clock.checked(entry['truth_ref']), allow_pickle=False) as saved:
                y, index = saved['target'].astype(np.float64), saved['index']
                if not np.array_equal(index, arrays['index']) or y.shape != index.shape or not np.isfinite(y).all():
                    raise ValueError('Aligned finite evaluation truth required')
                truth_cache[key] = y
        role, pid = entry['role'], entry['profile']
        y = truth_cache[key]
        truth.setdefault(role, {})[pid] = y
        for method in METHODS:
            values[method].setdefault(role, {})[pid] = arrays['prediction__' + method]
        left_error = arrays['prediction__c15__left'] - y
        right_error = arrays['prediction__c15__right'] - y
        lc, rc = left_error - np.mean(left_error), right_error - np.mean(right_error)
        lv, rv = float(np.mean(lc * lc)), float(np.mean(rc * rc))
        covariance = float(np.mean(lc * rc))
        denominator = math.sqrt(lv * rv)
        complementarity.append(dict(role=role, profile=pid, evaluated_rows=len(y),
            error_sign='prediction_minus_truth', left_mean_error=float(np.mean(left_error)),
            right_mean_error=float(np.mean(right_error)), left_centered_variance=lv,
            right_centered_variance=rv, centered_covariance=covariance,
            centered_error_correlation=covariance / denominator if denominator != 0. else None,
            correlation_defined=denominator != 0., uncentered_mean_error_product=float(np.mean(left_error * right_error)),
            scope='descriptive complementarity; not a causal explanation'))
    metrics = {m: {} for m in METHODS}
    for method in METHODS:
        for role in truth:
            clock.check()
            metrics[method][role] = metric.role_metrics(truth[role], values[method][role], threshold,
                                                        risk['metrics']['sample_interval_seconds'])
    comparisons = []
    for role in truth:
        comparisons.extend(comparison(metrics, role, method, 'c15__left') for method in METHODS)
        comparisons.extend(comparison(metrics, role, PRIMARY, pair + '__cal_macro_rmse') for pair in PAIRS if pair != 'c15')
    dump(output / 'metrics.json', dict(dataset=cfg['dataset'], high_temperature_threshold_c=threshold, methods=metrics))
    dump(output / 'comparison.json', comparisons)
    dump(output / 'complementarity.json', dict(primary_pair='c15', records=complementarity,
        scope='Previously exposed data, descriptive only; no pair selection or causal attribution'))
    dump(output / 'input_pins.json', clock.inputs)
    return dict(status='EXPLORATORY_COMBINATION_EVALUATION_COMPLETE', prediction_lock=ref(lockpath),
        methods=METHODS, primary=PRIMARY, roles=list(truth), high_temperature_threshold_c=threshold,
        metric_dependency=METRIC_SOURCE, independent_confirmation=False, all_evaluation_previously_exposed=True,
        model_selection_performed=False, pair_selection_performed=False, new_base_fit_performed=False)


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
        raise ValueError('Finite registered internal mode cap required')
    output = local(args.output_dir)
    if FOLLOWUP / 'runs' not in output.parents:
        raise ValueError('New followup run output directory required')
    output.mkdir(parents=True, exist_ok=False)
    clock = Clock(args.max_seconds, output)
    dump(output / 'configuration.json', dict(vars(args), source=str(Path(__file__).resolve()),
        source_sha256=sha(__file__), output_dir=str(output), pid=os.getpid(), argv=sys.argv,
        started_utc=utc(), numpy=np.__version__, scipy=scipy.__version__, protocol=PROTOCOL,
        scalar_dependency=SCALAR_SOURCE, metric_dependency=METRIC_SOURCE, max_output_bytes=CAP_BYTES))
    try:
        cfg, risk = load_config(args, clock)
        result = fit_predict(args, cfg, risk, output, clock) if args.mode == 'fit-predict' else evaluate(args, cfg, risk, output, clock)
        clock.check()
        result.update(dataset=cfg['dataset'], source_sha256=sha(__file__), config_sha256=args.config_sha256,
            protocol=PROTOCOL, elapsed_seconds=time.monotonic() - clock.start,
            files={p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
        dump(output / 'summary.json', result)
        clock.check()
        print(json.dumps(dict(status=result['status'], dataset=cfg['dataset'], elapsed_seconds=result['elapsed_seconds'])))
    except Exception as error:
        dump(output / 'failure.json', dict(status='FAILED', error_type=type(error).__name__,
            message=str(error), elapsed_seconds=time.monotonic() - clock.start))
        raise


if __name__ == '__main__':
    main()
