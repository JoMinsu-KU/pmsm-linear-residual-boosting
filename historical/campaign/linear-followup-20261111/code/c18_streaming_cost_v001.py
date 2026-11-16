"""Bounded, label-free CPU batch-one benchmark of pinned C18 models.

Root registers and runs this only while training is stopped. A config defines
one dataset/initialization/profile panel and all methods before timing. It is
not a training, accuracy-selection, or confirmation program. No worker imports.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np
import sklearn
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'linear-followup-20261111'
PROTOCOL = dict(path='linear-followup-20261111/protocols/c18-review-followup-v001-frozen.md',
                sha256='00aee40005399af1df692671e757d998f937eeb939817a5251b6ff478a92de0d')
SPANS = np.asarray([16, 64, 256, 1024, 4096], dtype=np.float64)
DEADLINE = datetime.fromisoformat('2026-11-12T00:23:59+00:00')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def local(value):
    p = Path(value)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('Campaign-local paths required')
    return p


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


class Budget:
    def __init__(self, out, seconds, attempt):
        self.out, self.seconds, self.attempt = out, seconds, attempt
        self.started = time.perf_counter()
        self.inputs = {}

    def check(self):
        if time.perf_counter() - self.started > self.seconds:
            raise TimeoutError('Registered benchmark stage exceeded its cap')
        if datetime.now(timezone.utc) >= DEADLINE:
            raise TimeoutError('Existing campaign deadline')

    def quiescence(self):
        self.check()
        active = read(ROOT / 'manifest.json')['active_attempts']
        ids = [str(v.get('attempt_id', v.get('id', ''))) if isinstance(v, dict) else str(v)
               for v in active]
        if any(v != self.attempt for v in ids):
            raise RuntimeError('Another campaign attempt is active; no concurrent training/benchmark')
        return ids

    def pin(self, entry):
        self.check()
        p = local(entry['path'])
        if sha(p) != entry['sha256']:
            raise ValueError('Changed pinned input: ' + str(p))
        self.inputs[str(p)] = entry['sha256']
        return p

    def arrays(self, entry, keys):
        with np.load(self.pin(entry), allow_pickle=False) as archive:
            return {k: archive[k].copy() for k in keys}

    def close(self):
        self.quiescence()
        for name, digest in self.inputs.items():
            if sha(name) != digest:
                raise ValueError('Input mutated during measurement')
        if sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file()) > 100_000_000:
            raise ValueError('100MB output cap')


def selected_document(entry, budget):
    obj = read(budget.pin(entry))
    for key in entry.get('selector', []):
        obj = obj[key]
    return obj


class FeatureState:
    """Only required causal33 features; no unused 23-feature auxiliary builder."""
    def __init__(self, kind, arm, dt_ref=None):
        self.kind, self.arm, self.dt_ref = kind, arm, dt_ref
        self.state = None
        self.previous_time = None
        self.decay = 1. - 2. / (SPANS + 1.)
        self.log_decay = np.log1p(-2. / (SPANS + 1.))
        if kind == 'rwth' and (arm != 'original' or dt_ref is None or dt_ref <= 0):
            raise ValueError('RWTH original time-aware feature definition required')
        if kind == 'legacy' and arm not in ('original', 'own_boundary'):
            raise ValueError('Frozen original/own_boundary timing panels')

    def snapshot(self):
        return (None if self.state is None else self.state.copy(), self.previous_time)

    def restore(self, snapshot):
        self.state = None if snapshot[0] is None else snapshot[0].copy()
        self.previous_time = snapshot[1]

    def step(self, op, timestamp=None):
        if self.kind == 'legacy':
            current = np.hypot(op[0], op[1])
            speed = np.abs(op[4])
            voltage = np.hypot(op[2], op[3])
            instant = np.asarray([current, current ** 2, voltage, speed, speed ** 2,
                                  current * speed, op[5], op[6]], dtype=np.float64)
            drive = np.asarray([op[6], op[5], instant[1], instant[4], instant[5]])
            if self.state is None:
                self.state = np.zeros((5, 5), dtype=np.float64)  # span-major
                self.state[:, :2] = ([op[6], op[5]] if self.arm == 'own_boundary'
                                     else [op[6], op[6]])
            else:
                # Same multiplication/addition order as the frozen scipy lfilter.
                self.state = ((1. - self.decay)[:, None] * drive[None, :]
                              + self.decay[:, None] * self.state)
            return (self.state[:, :2].reshape(-1), self.state[:, 2:].reshape(-1), instant)
        current, rpm, torque, coolant = op
        omega = 2. * np.pi * rpm / 60.
        instant = np.asarray([current, omega, torque, coolant, current ** 2,
                              np.abs(omega), omega ** 2, np.abs(torque * omega)])
        drive = np.asarray([coolant, instant[4], instant[5], instant[6], instant[7]])
        if self.state is None:
            self.state = np.zeros((5, 5), dtype=np.float64)  # drive-major
            self.state[0] = coolant
        else:
            if timestamp <= self.previous_time:
                raise ValueError('Strictly increasing original timestamps required')
            alpha = -np.expm1(self.log_decay * ((timestamp - self.previous_time) / self.dt_ref))
            self.state = (1. - alpha[None, :]) * self.state + alpha[None, :] * drive[:, None]
        self.previous_time = timestamp
        return np.concatenate((instant, self.state.reshape(-1)))


def legacy_x(packet, scale):
    b, heat, instant = packet
    return np.concatenate((b, heat / np.asarray(scale['qscale']), instant))


class Component:
    def __init__(self, entry, budget):
        self.entry = entry
        self.kind = entry['kind']
        self.artifacts = []
        for name in ('metadata', 'scale', 'pickle', 'weights'):
            if name in entry:
                self.artifacts.append(budget.pin(entry[name]))
        self.meta = selected_document(entry['metadata'], budget)
        self.parameter_bytes = 0
        if self.kind == 'tree':
            # Only exact local model files pinned by the root's reviewed config.
            with budget.pin(entry['pickle']).open('rb') as stream:
                self.model = pickle.load(stream)
            if type(self.model).__name__ != 'HistGradientBoostingRegressor':
                raise ValueError('Only locked HistGradientBoostingRegressor pickles')
            self.scale = selected_document(entry['scale'], budget)
            self.parameter_bytes = self.model._baseline_prediction.nbytes + sum(
                p.nodes.nbytes for row in self.model._predictors for p in row)
            self.tree_count = len(self.model._predictors)
        elif self.kind == 'linear':
            self.coef = np.asarray(self.meta['coef'], dtype=np.float64)
            self.scale = self.meta['scale']
            self.parameter_bytes = self.coef.nbytes
        elif self.kind == 'mlp':
            self.transform = self.meta['transform'] if 'transform' in self.meta else self.meta
            names = [str(i) + suffix for i in (0, 2, 4) for suffix in ('.weight', '.bias')]
            self.weights = budget.arrays(entry['weights'], names)
            expected = [(64, 33), (64,), (32, 64), (32,), (1, 32), (1,)]
            if [self.weights[k].shape for k in names] != expected:
                raise ValueError('Fixed 33-64-32-1 MLP geometry')
            if any(self.weights[k].dtype != np.float32 for k in names):
                raise ValueError('Original float32 MLP weights required')
            self.parameter_bytes = sum(v.nbytes for v in self.weights.values())
        else:
            raise ValueError('Unsupported component adapter')

    def predict(self, packet, feature_kind):
        if self.kind == 'tree':
            x = (legacy_x(packet, self.scale) if feature_kind == 'legacy' else
                 (packet - np.asarray(self.scale['x_mean'])) / np.asarray(self.scale['x_std']))
            return float(self.model.predict(x[None, :])[0])
        if self.kind == 'linear':
            s, family = self.scale, self.meta['family']
            if feature_kind == 'legacy':
                if family == 'linear33':
                    x = (legacy_x(packet, s) - np.asarray(s['linear33_mean'])) / np.asarray(s['linear33_scale'])
                elif family == 'free':
                    x = np.concatenate(((packet[0] - s['y_mid']) / s['y_scale'],
                                        packet[1] / np.asarray(s['qscale'])))
                else:
                    raise ValueError('Only linear33/free legacy priors in timing panel')
                design = np.r_[1., x]
            elif family == 'linear33':
                design = np.r_[1., (packet - np.asarray(s['x_mean'])) / np.asarray(s['x_std'])]
            elif family == 'free25':
                design = np.r_[(packet[8:13] - s['y_mid']) / s['y_scale'],
                               packet[13:33] / np.repeat(np.asarray(s['drive_rms']), 5)]
            else:
                raise ValueError('Unexpected RWTH linear family')
            return float(s['y_mid'] + s['y_scale'] * (design @ self.coef))
        tr = self.transform
        x = legacy_x(packet, tr['rc_scale']) if feature_kind == 'legacy' else packet
        x = ((x - np.asarray(tr['input_mean'])) / np.asarray(tr['input_std'])).astype(np.float32)
        for layer in (0, 2, 4):
            x = self.weights[str(layer) + '.weight'] @ x + self.weights[str(layer) + '.bias']
            if layer != 4:
                x = np.maximum(x, np.float32(0.))
        return float(tr['y_mid'] + tr['y_scale'] * float(x[0]))


def prediction(method, components, packet, feature_kind):
    # Shared features are generated once. Each required model is called once;
    # no cached prediction from another competing method is reused.
    total = 0.
    for name, weight in method['terms']:
        total += weight * components[name].predict(packet, feature_kind)
    return total


def endpoints(n, warmup, mode):
    retained = n - warmup
    if retained <= 0:
        raise ValueError('No evaluated endpoints')
    width = min(32 if mode == 'probe' else 256, retained)
    if mode == 'probe':
        return np.arange(warmup, warmup + width, dtype=np.int64)
    starts = (warmup, warmup + max(0, (retained - width) // 2), n - width)
    return np.unique(np.concatenate([np.arange(s, s + width, dtype=np.int64) for s in starts]))


def expected_values(method, ix, budget):
    result = np.zeros(len(ix))
    for term in method['expected']:
        z = budget.arrays(term['artifact'], [term['index_key'], term['key']])
        index = z[term['index_key']]
        if index.ndim != 1 or not np.all(np.diff(index) > 0):
            raise ValueError('Expected prediction index alignment')
        positions = np.searchsorted(index, ix)
        if np.any(positions >= len(index)) or not np.array_equal(index[positions], ix):
            raise ValueError('Every timed endpoint must have an audited saved prediction')
        value = z[term['key']]
        if value.shape != index.shape:
            raise ValueError('One saved scalar prediction per retained endpoint')
        result += term['weight'] * value[positions]
    return result


def describe_ns(values):
    value = np.asarray(values, dtype=np.float64)
    return dict(calls=len(value), median_us=float(np.median(value)) / 1000.,
                p95_us=float(np.quantile(value, .95)) / 1000.,
                min_us=float(value.min()) / 1000., max_us=float(value.max()) / 1000.)


def run(args, out, budget):
    cfg = read(budget.pin(dict(path=args.config, sha256=args.config_sha256)))
    if cfg['schema'] != 'c18-streaming-cost-config-v001' or cfg['protocol'] != PROTOCOL:
        raise ValueError('Root-frozen streaming config and scientific protocol required')
    budget.pin(PROTOCOL)
    if cfg['producer_sha256'] != sha(__file__) or cfg['no_concurrent_training_required'] is not True:
        raise ValueError('Exact benchmark source and no-training contract')
    if cfg['mode'] != args.mode or cfg['warmups'] != 1 or cfg['repetitions'] != (2 if args.mode == 'probe' else 5):
        raise ValueError('Frozen probe/main repetitions')
    budget.quiescence()
    # Root includes model/prediction locks and current independent PASS reports.
    for entry in cfg['lineage']:
        budget.pin(entry)
    if not cfg['audits']:
        raise ValueError('Independent artifact audit evidence required')
    for entry in cfg['audits']:
        report = read(budget.pin(entry))
        if report.get('status') != 'PASS' or report.get('failures'):
            raise ValueError('Only completed independently checked model/prediction artifacts')
    case = cfg['case']
    if case['dataset'] not in ('ipmsm', 'paderborn', 'rwth') or case['arm'] not in ('original', 'own_boundary'):
        raise ValueError('Frozen dataset/arm panels')
    kind = 'rwth' if case['dataset'] == 'rwth' else 'legacy'
    warmup = 0 if kind == 'rwth' else 300
    keys = [case['operating_key']] + ([case['time_key']] if kind == 'rwth' else [])
    raw = budget.arrays(case['operating'], keys)
    op = raw[case['operating_key']]
    timestamps = raw[case['time_key']] if kind == 'rwth' else np.arange(len(op), dtype=np.float64)
    if op.ndim != 2 or op.shape[1] != (4 if kind == 'rwth' else 7) or not np.isfinite(op).all():
        raise ValueError('Finite original operating channels')
    if len(timestamps) != len(op) or not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0):
        raise ValueError('Time/index geometry')
    ix = endpoints(len(op), warmup, args.mode)
    components = {name: Component(entry, budget) for name, entry in cfg['components'].items()}
    methods = cfg['methods']
    names = [m['name'] for m in methods]
    if len(names) != len(set(names)) or not {'D', 'L', 'H'}.issubset(names):
        raise ValueError('Unique, prospectively specified D/L/H and comparator list')
    for method in methods:
        if len({v[0] for v in method['terms']}) != len(method['terms']):
            raise ValueError('Each component is called once within a method')
        if any(name not in components or not math.isfinite(weight) for name, weight in method['terms']):
            raise ValueError('Fixed finite method coefficients')
    definitions = {m['name']: dict(m['terms']) for m in methods}
    expected_half = {}
    for name in ('D', 'L'):
        for component, weight in definitions[name].items():
            expected_half[component] = expected_half.get(component, 0.) + .5 * weight
    if definitions['H'] != expected_half:
        raise ValueError('H must be exactly half of D plus L, using the same components')
    state = FeatureState(kind, case['arm'], case.get('dt_ref'))
    snapshots, packets = {}, {}
    selected = set(ix.tolist())
    for row in range(int(ix[-1]) + 1):
        if row in selected:
            snapshots[row] = state.snapshot()
        packet = state.step(op[row], timestamps[row])
        if row in selected:
            packets[row] = tuple(v.copy() for v in packet) if kind == 'legacy' else packet.copy()
        if row % 2048 == 0:
            budget.check()
    feature_keys = ['X33'] if kind == 'rwth' else ['boundary', 'heat', 'instant']
    reference = budget.arrays(case['feature_reference'], feature_keys)
    feature_errors = {}
    for j, key in enumerate(feature_keys):
        value = np.asarray([packets[int(i)] if kind == 'rwth' else packets[int(i)][j] for i in ix])
        saved = reference[key]
        if len(saved) != len(op) or not np.isfinite(saved).all():
            raise ValueError('Full-history, finite saved feature reference')
        if not np.allclose(value, saved[ix], atol=1e-8, rtol=2e-12):
            raise ArithmeticError('Independent streaming feature parity failed: ' + key)
        feature_errors[key] = float(np.max(np.abs(value - saved[ix])))
    # Every timed endpoint is parity-qualified before any timing observations.
    parity = {}
    for method in methods:
        actual = np.asarray([prediction(method, components, packets[int(i)], kind) for i in ix])
        expected = expected_values(method, ix, budget)
        has_mlp = any(components[name].kind == 'mlp' for name, _ in method['terms'])
        atol = 1e-3 if has_mlp else 1e-7
        error = float(np.max(np.abs(actual - expected)))
        if not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=atol, rtol=0.):
            raise ArithmeticError('Scalar pipeline parity failed; no timing result: ' + method['name'])
        parity[method['name']] = dict(max_absolute_difference_c=error, atol_c=atol,
                                    reason='float32 CPU NumPy versus stored neural backend' if has_mlp else 'float64 scalar versus batch arithmetic')
        budget.check()
    timings = {name: {'e2e_ns': [], 'model_only_ns': []} for name in names}
    trace = {}
    for repeat in range(-1, cfg['repetitions']):
        shift = max(repeat, 0) % len(methods)
        order = methods[shift:] + methods[:shift]
        budget.quiescence()
        for method in order:
            e2e, model_only, outputs = [], [], []
            state = FeatureState(kind, case['arm'], case.get('dt_ref'))
            previous = None
            for index in ix:
                index = int(index)
                if previous is None or index != previous + 1:
                    state.restore(snapshots[index])  # Untimed full-history state before block.
                if repeat % 2 == 0:
                    tick_model = time.perf_counter_ns()
                    second = prediction(method, components, packets[index], kind)
                    end_model = time.perf_counter_ns()
                tick = time.perf_counter_ns()
                packet = state.step(op[index], timestamps[index])
                value = prediction(method, components, packet, kind)
                end = time.perf_counter_ns()
                if repeat % 2 != 0:
                    tick_model = time.perf_counter_ns()
                    second = prediction(method, components, packet, kind)
                    end_model = time.perf_counter_ns()
                if value != second or not math.isfinite(value):
                    raise ArithmeticError('Nondeterministic or nonfinite inference')
                e2e.append(end - tick)
                model_only.append(end_model - tick_model)
                outputs.append(value)
                previous = index
            if repeat >= 0:
                timings[method['name']]['e2e_ns'].append(e2e)
                timings[method['name']]['model_only_ns'].append(model_only)
                trace[method['name'] + '__repeat' + str(repeat)] = np.asarray(outputs)
            budget.check()
    files = {}
    for name, modes in timings.items():
        for mode, values in modes.items():
            files[name + '__' + mode] = np.asarray(values, dtype=np.int64)
    np.savez_compressed(out / 'timings.npz', index=ix, **files)
    np.savez_compressed(out / 'prediction_trace.npz', index=ix, **trace)
    results = {}
    for method in methods:
        name = method['name']
        unique = {str(p): p for component, _ in method['terms'] for p in components[component].artifacts}
        results[name] = dict(
            timings={mode: dict(per_repetition=[describe_ns(v) for v in values],
                                pooled_calls=describe_ns(np.asarray(values).reshape(-1)))
                     for mode, values in timings[name].items()},
            pinned_deployment_artifact_bytes=sum(p.stat().st_size for p in unique.values()),
            deployment_artifacts={p.relative_to(ROOT).as_posix(): p.stat().st_size for p in unique.values()},
            numeric_model_payload_bytes=sum(components[c].parameter_bytes for c, _ in method['terms']),
            numeric_model_payload_scope='Tree node structs + baseline, linear coefficients, or MLP arrays; excludes scalers/metadata/object allocator',
            feature_state_numeric_bytes=200)
    return dict(status='C18_STREAMING_COST_COMPLETE', mode=args.mode, dataset=case['dataset'],
                arm=case['arm'], role=case['role'], profile_id=case['profile_id'], raw_rows=len(op),
                endpoint_count=len(ix), endpoint_blocks='first/middle/last256 union' if args.mode == 'main' else 'first32',
                repetitions=cfg['repetitions'], warmups=1, parity=parity,
                feature_parity_max_absolute_difference=feature_errors, methods=results,
                scope='resident in-memory operating row -> causal required33 features -> transforms -> sklearn HGB/NumPy FP32 MLP -> blend',
                excluded='file/network/sensor IO, model loading, prefix-history replay before each sampled block, training; no embedded or hard-real-time claim',
                model_only_scope='prebuilt current feature packet -> model-specific transforms, predictions and blend; no feature-state update',
                mlp_backend='NumPy float32 CPU, not PyTorch CUDA/CPU runtime; each fixed seed is an individual model, never a three-model ensemble',
                target_values_read=False, performance_metrics_computed=False,
                training_concurrency_contract='Root scheduler stopped training; manifest checked before timing cycles. This is not an OS-wide process exclusion lock.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['probe', 'main'], required=True)
    for name in ('config', 'config-sha256', 'output-dir', 'attempt-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-seconds', type=float, default=600.)
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 3600:
        raise ValueError('Finite cap <=3600 seconds')
    out = local(args.output_dir)
    if WORK / 'runs' not in out.parents:
        raise ValueError('New C18 run destination')
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(out, args.max_seconds, args.attempt_id)
    dump(out / 'configuration.json', dict(vars(args), source_sha256=sha(__file__), pid=os.getpid()))
    try:
        with threadpool_limits(limits=1):
            runtime = dict(platform=platform.platform(), python=sys.version, numpy=np.__version__,
                           sklearn=sklearn.__version__, cpu=platform.processor(), logical_cpus=os.cpu_count(),
                           threadpools=threadpool_info(), clock=vars(time.get_clock_info('perf_counter')),
                           environment={k: os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
            if any(v['num_threads'] != 1 for v in runtime['threadpools']):
                raise ValueError('Every detected native numerical pool must use one thread')
            dump(out / 'runtime.json', runtime)
            result = run(args, out, budget)
            budget.close()
        dump(out / 'input_pins.json', budget.inputs)
        result.update(source_sha256=sha(__file__), config_sha256=args.config_sha256,
                      elapsed_seconds=time.perf_counter()-budget.started,
                      files={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
        dump(out / 'summary.json', result)
        budget.close()
        print(json.dumps({k: result[k] for k in ('status','mode','dataset','arm','elapsed_seconds')}))
    except Exception as exc:
        dump(out / 'failure.json', dict(status='FAILED', exception=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__':
    main()
