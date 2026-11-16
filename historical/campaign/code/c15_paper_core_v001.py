"""Numerical core for the prospectively registered C15 paper suite.

This source contains no execution at import beyond imports. It preserves the
audited C12 feature/prior implementation through an exact source dependency.
Scientific execution and every evaluation-role admission belong to the runner.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

C12_SHA = 'd4aa942ca9db8d8e9319197632cf376730456f3ea5cdc48b9d1c1fd51fc59412'
LAMBDAS = [0.0001, 0.01, 1.0]
FAMILIES = ['positive', 'free', 'ewma', 'linear33', 'nn_only', 'eq_only', 'unconstrained25']
ABLATIONS = ['positive', 'nn_only', 'eq_only', 'unconstrained25']
BASE = dict(learning_rate=.05, min_samples_leaf=20, l2_regularization=.1,
            loss='squared_error', early_stopping=False, random_state=2026111201)
GRID = [dict(max_leaf_nodes=leaves, max_iter=iterations, l2_regularization=penalty)
        for leaves, iterations, penalty in [(15, 200, .1), (15, 500, .1),
        (15, 1000, .1), (31, 200, .1), (31, 500, .1), (31, 1000, .1),
        (7, 200, .1), (7, 500, .1), (15, 100, 1.)]]
DIRECT = ['direct_boost', 'direct_boost_15x500', 'direct_boost_15x1000',
          'direct_boost_31x200', 'direct_boost_31x500', 'direct_boost_31x1000',
          'direct_boost_7x200', 'direct_boost_7x500', 'direct_boost_15x100_l2_1']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode('ascii'))
    h.update(json.dumps(list(value.shape)).encode('ascii'))
    h.update(value.tobytes())
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def helper():
    path = Path(__file__).with_name('combination_c12_fit_rc_v001.py')
    if sha(path) != C12_SHA:
        raise ValueError('Audited C12 numerical dependency changed')
    spec = importlib.util.spec_from_file_location('c15_paper_audited_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    def __init__(self, seconds, root, cap):
        self.start, self.seconds, self.root, self.cap = time.monotonic(), seconds, Path(root), cap

    def check(self):
        if time.monotonic() - self.start > self.seconds:
            raise TimeoutError('Registered stage elapsed-time cap; preserve incomplete artifacts')
        if sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file()) > self.cap:
            raise RuntimeError('Registered stage storage cap; preserve incomplete artifacts')


def quadratic(q, r, constant, nonnegative, equality):
    """Strictly convex QP with optional theta>=0 and sum(theta[:10])=1.

    Active-set arithmetic for the NN+EQ case is the exact audited C12 solver.
    Other cases use the same KKT signs and an independently replayable feasible
    dual certificate. No coefficient clipping or outcome-driven fallback occurs.
    """
    np.linalg.cholesky(q)
    n = len(r)
    e = np.r_[np.ones(10), np.zeros(n - 10)] if equality else np.zeros(n)
    if nonnegative and equality:
        return helper().positive_qp(q, r, constant)
    nu = 0.
    iteration = 1
    if not nonnegative:
        if equality:
            solved = np.linalg.solve(q, np.column_stack([r, e]))
            nu = float((e @ solved[:, 0] - 1.) / (e @ solved[:, 1]))
            theta = solved[:, 0] - nu * solved[:, 1]
        else:
            theta = np.linalg.solve(q, r)
    else:
        # NN only: begin at the feasible origin, add negative-gradient entries.
        theta = np.zeros(n)
        free = []
        tolerance = 2e-10 * (1. + float(np.max(np.abs(r))))
        for iteration in range(1, 1001):
            grad = q @ theta - r
            inactive = [j for j in range(n) if j not in free]
            if inactive:
                add = min(inactive, key=lambda j: (grad[j], j))
                if grad[add] < -tolerance:
                    free.append(add)
                elif not free or np.max(np.abs(grad[free])) <= tolerance:
                    break
            elif np.max(np.abs(grad)) <= tolerance:
                break
            while free:
                f = np.array(sorted(free), dtype=int)
                candidate = np.zeros(n)
                candidate[f] = np.linalg.solve(q[np.ix_(f, f)], r[f])
                negative = [j for j in f if candidate[j] < -1e-12]
                if not negative:
                    theta = candidate
                    break
                alpha, hit = min((theta[j] / (theta[j] - candidate[j]), int(j)) for j in negative)
                theta += alpha * (candidate - theta)
                theta[hit] = 0.
                free.remove(hit)
        else:
            raise ValueError('NN-only active-set iteration cap')
    grad = q @ theta - r + nu * e
    mu = np.maximum(grad, 0.) if nonnegative else np.zeros(n)
    argument = r - nu * e + mu
    z = np.linalg.solve(q, argument)
    primal = float(.5 * theta @ q @ theta - r @ theta + constant)
    dual = float(constant - nu - .5 * argument @ z)
    gap = float(.5 * (theta - z) @ q @ (theta - z) + mu @ theta + nu * (1. - e @ theta))
    eq_error = float(abs(e @ theta - 1.)) if equality else 0.
    cert = dict(iterations=iteration, primal=primal, dual=dual, direct_gap=primal-dual,
        stable_gap=gap, gap_tolerance=1e-8*(1.+abs(primal)), equality_residual=eq_error,
        min_coefficient=float(theta.min()), nu=nu, mu=mu.tolist(),
        stationarity=float(np.max(np.abs(grad-mu))),
        complementarity=float(np.max(np.abs(mu*theta))),
        nonnegative=nonnegative, equality=equality)
    if (not np.isfinite(theta).all() or eq_error > 1e-10 or
        (nonnegative and theta.min() < -1e-10) or gap < -1e-10 or
        gap > cert['gap_tolerance'] or abs(primal-dual-gap) > cert['gap_tolerance']):
        raise ValueError('New QP numerical qualification failed: '+json.dumps(cert))
    return theta, cert


def fit_linear(h, records, ids, family, lam, clock):
    if family in ('positive', 'free', 'ewma'):
        model = h.fit_linear(records, ids, family, lam, clock)
        model['constraint_flags'] = dict(nonnegative=family=='positive', equality=family=='positive',
                                        intercept=family!='positive', intercept_penalized=False)
        return model
    clock.check()
    started = time.monotonic()
    scale = h.scaling(records, ids)
    indices = {p: h.train_index(records[p]) for p in ids}
    if family == 'linear33':
        blocks = [h.boost_features(records[p], scale)[indices[p]] for p in ids]
        feature = np.concatenate(blocks)
        scale['linear33_mean'] = feature.mean(0).tolist()
        scale['linear33_scale'] = np.maximum(feature.std(0), 1e-8).tolist()
        xs = [np.column_stack([np.ones(len(v)), (v-scale['linear33_mean'])/scale['linear33_scale']]) for v in blocks]
    else:
        xs = [h.design(records[p], scale, 'positive')[indices[p]] for p in ids]
    ys = [(records[p]['y'][indices[p]]-scale['y_mid'])/scale['y_scale'] for p in ids]
    x, y = np.concatenate(xs), np.concatenate(ys)
    weights = h.balanced_weights([len(v) for v in ys])
    penalty = np.ones(x.shape[1])
    if family == 'linear33':
        penalty[0] = 0.
    q = x.T @ (weights[:, None]*x)+lam*np.diag(penalty)
    r = x.T @ (weights*y)
    constant = float(.5*np.dot(weights, y*y))
    nn, eq = family=='nn_only', family=='eq_only'
    coefficient, cert = quadratic(q, r, constant, nn, eq)
    clock.check()
    return dict(family=family, lam=lam, scale=scale, coef=coefficient.tolist(),
        certificate=cert, seconds=time.monotonic()-started,
        constraint_flags=dict(nonnegative=nn, equality=eq, intercept=family=='linear33', intercept_penalized=False),
        penalty=penalty.tolist())


def predict_linear(h, model, record):
    if model['family'] in ('positive', 'free', 'ewma'):
        return h.predict_linear(model, record)
    scale = model['scale']
    if model['family'] == 'linear33':
        feature = h.boost_features(record, scale)
        design = np.column_stack([np.ones(len(feature)),
            (feature-scale['linear33_mean'])/scale['linear33_scale']])
    else:
        design = h.design(record, scale, 'positive')
    return scale['y_mid']+scale['y_scale']*(design@model['coef'])


def training(h, records, ids):
    scale = h.scaling(records, ids)
    indices = {p: h.train_index(records[p]) for p in ids}
    xs = [h.boost_features(records[p], scale)[indices[p]] for p in ids]
    ys = [records[p]['y'][indices[p]] for p in ids]
    y = np.concatenate(ys)
    weights = len(y)*h.balanced_weights([len(v) for v in ys])
    return np.concatenate(xs), y, weights, scale, indices


def fit_booster(h, x, target, weights, grid, folder, clock):
    configuration = dict(BASE, **grid)
    clock.check()
    started = time.monotonic()
    model = HistGradientBoostingRegressor(**configuration).fit(x, target, sample_weight=weights)
    fit_seconds = time.monotonic()-started
    clock.check()
    if model.n_iter_ != configuration['max_iter']:
        raise ValueError('Frozen iteration count changed')
    exported = h.export_booster(model, folder, 'booster')
    seed, limit = int(model._bin_mapper.random_state), model._bin_mapper.subsample
    rows = np.random.RandomState(seed).choice(len(x), limit, replace=False) if limit is not None and len(x)>limit else np.arange(len(x), dtype=np.int64)
    info = dict(configuration=configuration, model=exported, fit_seconds=fit_seconds,
        fit_rows=len(x), binning_random_state=seed, binning_subsample=limit,
        binning_subsample_active=len(rows)<len(x), features_sha256=array_sha(x),
        target_sha256=array_sha(target), sample_weight_sha256=array_sha(weights))
    return model, info, rows


def metrics(prediction, target):
    error = np.asarray(prediction)-np.asarray(target)
    if error.ndim != 1 or not len(error) or not np.isfinite(error).all():
        raise ValueError('Metric target/prediction geometry or finite violation')
    return dict(n=len(error), squared_error_sum=float(np.dot(error, error)),
        rmse=float(np.sqrt(np.mean(error*error))), mae=float(np.mean(np.abs(error))),
        max_absolute_error=float(np.max(np.abs(error))),
        max_underprediction=float(max(0., np.max(-error))))


def fixtures():
    """Analytic, data-free qualifications; root explicitly executes this stage."""
    checks=[]
    q=np.eye(25); r=np.linspace(-1.,1.,25); constant=3.
    for nn,eq in [(False,False),(True,False),(False,True),(True,True)]:
        candidate_r=r.copy()
        if nn and eq:
            candidate_r[:10]=.2;candidate_r[10:]=-1.
            expected=np.r_[np.repeat(.1,10),np.zeros(15)]
        elif eq:
            e=np.r_[np.ones(10),np.zeros(15)]
            expected=r-((e@r-1.)/10.)*e
        elif nn:
            expected=np.maximum(r,0.)
        else:
            expected=r
        value,certificate=quadratic(q,candidate_r,constant,nn,eq)
        error=float(np.max(np.abs(value-expected)))
        if error>1e-9:
            raise ValueError('Analytic constraint fixture failed')
        checks.append(dict(nonnegative=nn,equality=eq,max_absolute_error=error,certificate=certificate))
    # This is the exact offset identity for squared-error residual learning;
    # it is not a newly fitted booster or a claim of bitwise training replay.
    y=np.array([1.,3.,-2.,8.]);p=np.array([.5,2.,-4.,9.]);g=np.array([.2,-.3,.7,1.])
    left=(y-p-g)**2;right=(y-(p+g))**2
    if not np.allclose(left,right,atol=1e-14,rtol=0):
        raise ValueError('Squared-error offset identity fixture failed')
    return dict(status='DATA_FREE_ANALYTIC_FIXTURES_PASS',checks=checks,
        no_dataset_or_fitted_model_access=True)
