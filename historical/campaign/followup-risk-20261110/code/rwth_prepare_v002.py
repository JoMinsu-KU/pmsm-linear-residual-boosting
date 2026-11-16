"""Root-registered RWTH FIT-only cache preparation under frozen v002 protocol.

Opens only the twenty fixed FIT CSV members. CAL/CONF and stator are untouched.
No model fitting, new downloads, role changes or confirmation consumption.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP = ROOT / 'followup-risk-20261110'
PROTOCOL = dict(path='followup-risk-20261110/protocols/rwth-replication-v002-frozen.md',
    sha256='f1d201fe4aa6243f8c77c6a8a85a91268d183d0cd24d48bc35ec86b269082b0f')
PARTITION = dict(path='followup-risk-20261110/configs/rwth-partition-v001.json',
    sha256='5d5117312eee65bd9811f0ede7326d01d8a0a1aa3d0b0e3fb6ef60ae47959be1')
ARCHIVE_SHA = 'a6b219a17acbbc2a9899fa7d750e1b0531756eeefcf16efd56009c43a0c4fce6'
HEADERS = ['time', 'i', 'torque', 'motor_speed', 'coolant', 'rotor', 'stator', 'profile_id']
ROLES = dict(fit=[38,27,30,23,21,19,29,20,26,37,6,8,35,34,4,15,7,13,18,1],
    calibration=[10,11,22,9,16,31,28,17], confirmation=[5,32,33,36,24,2,25,14,3,12])
SPANS = [16, 64, 256, 1024, 4096]
DRIVES = ['coolant', 'i_squared', 'abs_omega', 'omega_squared', 'abs_mechanical_power']
FEATURE_NAMES = ['i', 'omega', 'torque', 'coolant', 'i_squared', 'abs_omega',
                 'omega_squared', 'abs_mechanical_power'] + [d + '__span' + str(s) for d in DRIVES for s in SPANS]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def local(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Path outside campaign')
    return path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def ref(path):
    p = local(path)
    return dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p))


class Clock:
    def __init__(self, seconds, output):
        self.start, self.seconds, self.output = time.monotonic(), seconds, output
        self.inputs = {}

    def check(self, training=False):
        if time.monotonic() - self.start >= self.seconds:
            raise TimeoutError('Internal stage cap')
        now = datetime.now(timezone.utc)
        if now >= datetime.fromisoformat('2026-11-11T00:23:59+00:00'):
            raise TimeoutError('Campaign deadline')
        if training and now >= datetime.fromisoformat('2026-11-10T20:23:59+00:00'):
            raise TimeoutError('New training cutoff')
        if sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file()) > 1_000_000_000:
            raise RuntimeError('Per-stage1GB ceiling; root additionally enforces aggregate1GB')

    def pin(self, entry):
        self.check()
        path = local(entry['path'])
        observed = sha(path)
        if observed != entry['sha256'] or (str(path) in self.inputs and self.inputs[str(path)] != observed):
            raise ValueError('Pinned input changed: ' + str(path))
        self.inputs[str(path)] = observed
        return path


def parse_member(archive, pid, clock, include_target):
    name = 'final/' + str(pid).zfill(3) + '.csv'
    info = archive.getinfo(name)
    if info.is_dir() or info.file_size > 32_000_000:
        raise ValueError('Unexpected/big CSV member')
    with archive.open(info) as handle:
        raw = handle.read(info.file_size + 1)
    if len(raw) != info.file_size:
        raise ValueError('CSV member exact byte length')
    reader = csv.reader(io.StringIO(raw.decode('utf-8-sig')))
    if next(reader) != HEADERS:
        raise ValueError('Exact frozen CSV header required')
    rows, targets = [], []
    for line, fields in enumerate(reader):
        if line % 2048 == 0:
            clock.check()
        if len(fields) != 8:
            raise ValueError('Eight fields required in every data row')
        identity = float(fields[7])
        if not math.isfinite(identity) or identity != pid:
            raise ValueError('Profile identity mismatch')
        # Stator (field6) is never converted. Held rotor is never converted.
        rows.append([float(fields[j]) for j in (0, 1, 3, 2, 4)])
        if include_target:
            targets.append(float(fields[5]))
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Nonempty finite time/four observed inputs required')
    times, operating = values[:, 0].copy(), values[:, 1:].copy()
    if not np.all(np.diff(times) > 0):
        raise ValueError('Strictly increasing original timestamps required')
    target = np.asarray(targets, dtype=np.float64) if include_target else None
    if include_target and (target.shape != times.shape or not np.isfinite(target).all()):
        raise ValueError('Finite exact FIT target coverage required')
    return dict(id=pid, member=name, csv_sha256=hashlib.sha256(raw).hexdigest(),
                uncompressed_bytes=len(raw), time=times, operating=operating, target=target)


def features(operating, timestamps, dt_ref, clock=None):
    if (not math.isfinite(dt_ref) or dt_ref <= 0 or operating.shape != (len(timestamps), 4)
            or not len(timestamps) or not np.isfinite(operating).all() or not np.isfinite(timestamps).all()
            or not np.all(np.diff(timestamps) > 0)):
        raise ValueError('Finite original geometry and positive time increments required')
    current, rpm, torque, coolant = operating.T
    omega = 2. * np.pi * rpm / 60.
    instant = np.column_stack((current, omega, torque, coolant, current ** 2,
                               np.abs(omega), omega ** 2, np.abs(torque * omega)))
    drive = np.column_stack((coolant, current ** 2, np.abs(omega), omega ** 2, np.abs(torque * omega)))
    state = np.zeros((5, 5), dtype=np.float64)
    state[0] = coolant[0]
    history = np.empty((len(timestamps), 25), dtype=np.float64)
    history[0] = state.reshape(-1)
    log_decay = np.log1p(-2. / (np.asarray(SPANS, dtype=np.float64) + 1.))
    for t in range(1, len(timestamps)):
        if clock is not None and t % 2048 == 0:
            clock.check()
        alpha = -np.expm1(log_decay * ((timestamps[t] - timestamps[t - 1]) / dt_ref))
        state = (1. - alpha[None, :]) * state + alpha[None, :] * drive[t, :, None]
        history[t] = state.reshape(-1)
    result = np.column_stack((instant, history))
    if result.shape != (len(timestamps), 33) or not np.isfinite(result).all():
        raise FloatingPointError('Finite exact33 feature geometry required')
    return result


def synthetic_fixtures():
    operating = np.asarray([[2., 60., 3., 10.]] * 4, dtype=np.float64)
    t = np.arange(4, dtype=np.float64)
    x = features(operating, t, 1.)
    a = 2. / (np.asarray(SPANS) + 1.)
    expected = 4. * (1. - (1. - a) ** 3)
    if (not np.array_equal(x[0, 8:13], np.full(5, 10.)) or not np.array_equal(x[0, 13:], np.zeros(20))
            or not np.allclose(x[3, 13:18], expected, atol=1e-13, rtol=1e-13)):
        raise ArithmeticError('No-advance initialization/constant-step closed-form fixture failed')
    if not np.allclose(x, features(operating, t * 17., 17.), atol=1e-13, rtol=1e-13):
        raise ArithmeticError('Time-unit scale invariance fixture failed')
    if not np.array_equal(x[:2], features(operating[:2], t[:2], 1.)):
        raise ArithmeticError('Causal prefix fixture failed')
    return dict(status='PASS', checks=['t0_no_advance', 'constant_dt_closed_form', 'time_unit_invariance', 'causal_prefix'])


def prepare(args, output, clock):
    if dict(path=args.partition, sha256=args.partition_sha256) != PARTITION:
        if local(args.partition) != local(PARTITION['path']) or args.partition_sha256 != PARTITION['sha256']:
            raise ValueError('Exact fixed partition required')
    partition = read(clock.pin(PARTITION))
    clock.pin(PROTOCOL)
    if partition['schema'] != 'rwth-partition-v001' or partition['roles'] != ROLES or partition['headers'] != HEADERS:
        raise ValueError('Frozen role/header partition required')
    if partition['archive']['sha256'] != ARCHIVE_SHA:
        raise ValueError('Exact original archive required')
    archive_path = clock.pin(partition['archive'])
    fixtures = synthetic_fixtures()
    dump(output / 'synthetic_fixtures.json', fixtures)
    parsed = []
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive member names')
        for pid in ROLES['fit']:
            parsed.append(parse_member(archive, pid, clock, include_target=True))
    increments = np.concatenate([np.diff(r['time']) for r in parsed])
    if not len(increments):
        raise ValueError('FIT timestamps cannot identify a reference increment')
    dt_ref = float(np.median(increments))
    if not math.isfinite(dt_ref) or dt_ref <= 0:
        raise ValueError('Finite positive FIT-only reference increment required')
    records = []
    folder = output / 'features' / 'fit'
    folder.mkdir(parents=True, exist_ok=False)
    for record in parsed:
        clock.check()
        pid, n = record['id'], len(record['time'])
        x = features(record['operating'], record['time'], dt_ref, clock)
        index = np.arange(n, dtype=np.int64)
        path = folder / ('profile_' + str(pid).zfill(3) + '.npz')
        np.savez_compressed(path, profile_id=np.asarray(pid, dtype=np.int64), index=index, source_row=index,
                            time=record['time'], operating=record['operating'], X33=x, target=record['target'])
        records.append(dict(id=pid, role='fit', rows=n, member=record['member'], csv_sha256=record['csv_sha256'],
                            features=ref(path), truth=None, target_key='target'))
    dump(output / 'input_pins.json', clock.inputs)
    manifest = dict(schema='rwth-cache-manifest-v002', status='READY_FIT_ONLY', dataset='rwth48v',
        source_sha256=sha(__file__), protocol=PROTOCOL, partition=PARTITION, archive=partition['archive'],
        dt_ref=dt_ref, dt_ref_scope='pooled all FIT20 consecutive timestamp increments; no CAL/CONF',
        feature_names=FEATURE_NAMES, operating_columns=['i', 'motor_speed', 'torque', 'coolant'], spans=SPANS,
        roles=ROLES, records=records, warmup=0, train_stride=5, evaluation_stride=1,
        calibration_member_contents_opened=False, confirmation_member_contents_opened=False,
        stator_numeric_values_parsed=False, confirmation_attempt_consumed=False,
        time_unit_independently_verified=False, fitted_model_count=0,
        opened_csv_members=[r['member'] for r in parsed],
        source_bytes=sum(r['uncompressed_bytes'] for r in parsed),
        inputs=dict(clock.inputs), generated_at_utc=datetime.now(timezone.utc).isoformat())
    dump(output / 'manifest.json', manifest)
    return dict(status='RWTH_FIT_CACHE_READY', manifest=ref(output / 'manifest.json'),
        dataset='rwth48v', roles_prepared=['fit'], fit_profiles=len(records), fit_rows=sum(r['rows'] for r in records),
        calibration_member_contents_opened=False, confirmation_member_contents_opened=False,
        stator_numeric_values_parsed=False, confirmation_attempt_consumed=False, features_generated=True,
        models_fitted=False, dt_ref=dt_ref, source_bytes=manifest['source_bytes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition', required=True)
    parser.add_argument('--partition-sha256', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-seconds', type=float, default=300.)
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 300:
        raise ValueError('Finite preparation cap in(0,300] required')
    output = local(args.output_dir)
    if FOLLOWUP / 'runs' not in output.parents:
        raise ValueError('Fresh followup run output required')
    output.mkdir(parents=True, exist_ok=False)
    clock = Clock(args.max_seconds, output)
    dump(output / 'configuration.json', dict(vars(args), source_sha256=sha(__file__), protocol=PROTOCOL,
        source=str(Path(__file__).resolve()), output_dir=str(output), pid=os.getpid(), argv=sys.argv,
        numpy=np.__version__, started_at_utc=datetime.now(timezone.utc).isoformat()))
    try:
        summary = prepare(args, output, clock)
        clock.check()
        summary.update(schema='rwth-preparation-summary-v002', source_sha256=sha(__file__), protocol=PROTOCOL,
            partition=PARTITION, elapsed_seconds=time.monotonic() - clock.start,
            files={p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
        dump(output / 'summary.json', summary)
        clock.check()
        print(json.dumps(dict(status=summary['status'], fit_profiles=summary['fit_profiles'],
                              fit_rows=summary['fit_rows'], elapsed_seconds=summary['elapsed_seconds'])))
    except Exception as exc:
        dump(output / 'failure.json', dict(status='FAILED', error_type=type(exc).__name__, message=str(exc),
                                          elapsed_seconds=time.monotonic() - clock.start))
        raise


if __name__ == '__main__':
    main()
