"""Independent C12 raw audit. No scientific worker import or model pickle.

Rebuild raw features, all FIT folds/scales/certificates and OOF responses; replay
saved HGB trees on raw features and recalculate the whole fixed metric panel.
Tree training/split search is not rerun. Root alone registers actual execution.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import time
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from threadpoolctl import threadpool_limits

WORKER = 'd4aa942ca9db8d8e9319197632cf376730456f3ea5cdc48b9d1c1fd51fc59412'
MANIFEST = '76dc4eaf66e6e60cf8edc9f3f0cff6591a68e71aad2098a131f2b25cccfbcfba'
FIT, CAL, TEST = [0, 2, 3, 5, 7, 8, 11, 14], [1, 6, 12], [4, 9, 13]
OPERATING = ['IdRef', 'IqRef', 'Ud', 'Uq', 'speed', 'water_outlet', 'ambient']
SPANS, LAMBDAS = [16, 64, 256, 1024, 4096], [.0001, .01, 1.]
PRIMARY = 'positive_rc_oof_boost'
METHODS = ['positive_rc', 'unconstrained_rc_ridge', 'direct_boost', PRIMARY,
           'unconstrained_rc_oof_boost', 'mean_A_B', 'ewma_ridge', 'positive_rc_insample_boost']
BOOSTERS = ['direct_boost', PRIMARY, 'unconstrained_rc_oof_boost', 'positive_rc_insample_boost']
BOOST = dict(max_iter=200, max_leaf_nodes=15, learning_rate=.05, min_samples_leaf=20,
             l2_regularization=.1, loss='squared_error', early_stopping=False, random_state=2026111201)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Audit:
    def __init__(self, seconds):
        self.start, self.seconds = time.monotonic(), seconds
        self.checks, self.failures, self.files, self.errors, self.bounds = 0, [], {}, {}, []
        self.bin_diagnostics = []

    def budget(self):
        if time.monotonic() - self.start > self.seconds:
            raise TimeoutError('C12 independent raw audit cap')

    def check(self, good, name):
        self.budget(); self.checks += 1
        if not bool(good): self.failures.append(name)

    def close(self, actual, expected, name, atol=2e-9, rtol=2e-10):
        a, b = np.asarray(actual), np.asarray(expected)
        good = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        if good: self.errors[name] = float(np.max(np.abs(a-b))) if a.size else 0.
        self.check(good and np.allclose(a, b, atol=atol, rtol=rtol), name)

    def equal(self, actual, expected, name):
        self.check(np.array_equal(np.asarray(actual), np.asarray(expected)), name)

    def hash(self, path):
        self.budget(); value = sha(path); self.files[str(Path(path).resolve())] = value
        return value

    def json(self, path):
        self.hash(path); return json.loads(Path(path).read_text(encoding='utf-8-sig'))

    def arrays(self, path):
        self.hash(path)
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}


def quantile(value, fraction):
    value = np.sort(value); t = fraction*(len(value)-1); lo, hi = int(np.floor(t)), int(np.ceil(t))
    return float(value[lo] + (t-lo)*(value[hi]-value[lo]))


def raw_profile(audit, assets, item):
    if item['id'] in [10, 15] or item['id'] not in FIT+CAL+TEST:
        raise ValueError('No confirmation or unknown profile access')
    path = (assets/item['path']).resolve()
    if not path.is_relative_to(assets): raise ValueError('Asset path escapes root')
    raw = path.read_bytes(); digest = audit.hash(path)
    git = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    audit.check(len(raw) == item['bytes'] and git == item['git_blob_sha1'], 'Raw byte/git pin '+str(item['id']))
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    audit.check(set(OPERATING+['rotor']) <= set(reader.fieldnames or []), 'Raw header '+str(item['id']))
    rows = list(reader)
    op = np.asarray([[float(row[key]) for key in OPERATING] for row in rows], dtype=np.float64)
    y = np.asarray([float(row['rotor']) for row in rows], dtype=np.float64)
    audit.check(len(y) > 332 and np.isfinite(op).all() and np.isfinite(y).all(), 'Raw finite envelope '+str(item['id']))
    return op, y, {'id': item['id'], 'sha256': digest, 'git_blob_sha1': git, 'rows': len(y)}


def features(op, audit):
    # Preserve the specified stable norm primitive. Independent construction is
    # not a reason to introduce avoidable last-bit changes at tree boundaries.
    current = np.hypot(op[:, 0], op[:, 1])
    speed = np.abs(op[:, 4]); voltage = np.hypot(op[:, 2], op[:, 3])
    instant = np.column_stack([current, current**2, voltage, speed, speed**2,
                              current*speed, op[:, 5], op[:, 6]])
    q = instant[:, [1, 4, 5]]
    signals = np.column_stack([np.tile(op[:, [6, 5]], (1, 5)), np.tile(q, (1, 5)), np.tile(q, (1, 5))])
    poles = np.r_[np.repeat(1-2/(np.asarray(SPANS)+1), 2),
                  np.repeat(1-2/(np.asarray(SPANS)+1), 3), np.repeat(1-2/(np.asarray(SPANS)+1), 3)]
    history = np.empty_like(signals)
    history[0] = np.r_[np.repeat(op[0, 6], 10), np.zeros(15), np.tile(q[0], 5)]
    for t in range(1, len(op)):
        if t % 4096 == 0: audit.budget()
        history[t] = history[t-1]*poles + signals[t]*(1-poles)
    return {'instant': instant, 'boundary': history[:, :10], 'heat': history[:, 10:25],
            'ewma': np.column_stack([instant, history[:, 25:]])}


def indices(record):
    return np.arange(300, len(record['y']), 5)


def scaler(records, ids):
    raw = np.concatenate([records[p]['instant'][indices(records[p])] for p in ids])
    y = np.concatenate([records[p]['y'][indices(records[p])] for p in ids])
    ewma = np.concatenate([records[p]['ewma'][indices(records[p])] for p in ids])
    ri = max(float(np.sqrt(np.dot(raw[:, 0], raw[:, 0])/len(raw))), 1e-8)
    rw = max(float(np.sqrt(np.dot(raw[:, 3], raw[:, 3])/len(raw))), 1e-8)
    return {'ids': ids, 'qscale': np.tile([ri*ri, rw*rw, ri*rw], 5),
            'y_mid': quantile(y, .5), 'y_scale': max(quantile(y, .75)-quantile(y, .25), 1.),
            'ewma_mean': ewma.mean(0), 'ewma_scale': np.maximum(ewma.std(0), 1e-8)}


def design(record, scale, family):
    if family == 'ewma':
        return np.column_stack([np.ones(len(record['ewma'])),
                               (record['ewma']-scale['ewma_mean'])/scale['ewma_scale']])
    response = np.column_stack([(record['boundary']-scale['y_mid'])/scale['y_scale'],
                                record['heat']/scale['qscale']])
    return np.column_stack([np.ones(len(response)), response]) if family == 'free' else response


def prediction(model, record):
    return model['scale']['y_mid'] + model['scale']['y_scale']*(design(record, model['scale'], model['family']) @ model['coef'])


def verify_linear(audit, model, records, ids, family, lam, tag):
    audit.check(model['family'] == family and model['lam'] == lam, tag+': family/lambda')
    scale = scaler(records, ids)
    for key, value in scale.items():
        if key == 'ids': audit.equal(model['scale'][key], value, tag+': fold-only fit IDs')
        else: audit.close(model['scale'][key], value, tag+': fold-only scale '+key)
    parts = [design(records[p], scale, family)[indices(records[p])] for p in ids]
    responses = [(records[p]['y'][indices(records[p])]-scale['y_mid'])/scale['y_scale'] for p in ids]
    x, y = np.concatenate(parts), np.concatenate(responses)
    w = np.concatenate([np.ones(len(v))/(len(ids)*len(v)) for v in responses])
    beta = np.asarray(model['coef']); dimension = {'positive': 25, 'free': 26, 'ewma': 24}[family]
    audit.check(beta.shape == (dimension,) and np.isfinite(beta).all(), tag+': coefficients')
    regularizer = np.ones(dimension)
    if family != 'positive': regularizer[0] = 0
    hessian = x.T @ (w[:, None]*x) + lam*np.diag(regularizer)
    rhs = x.T @ (w*y); constant = .5*np.dot(w, y*y)
    np.linalg.cholesky(hessian)
    residual = y-x@beta
    objective = float(.5*np.dot(w, residual*residual)+.5*lam*np.dot(regularizer*beta, beta))
    cert = model['certificate']
    if family == 'positive':
        e = np.r_[np.ones(10), np.zeros(15)]; nu = float(cert['nu']); mu = np.asarray(cert['mu'])
        audit.check(mu.shape == beta.shape and np.isfinite(mu).all() and np.all(mu >= 0), tag+': feasible nonnegative dual multiplier')
        equality = abs(float(e@beta-1)); stationarity = hessian@beta-rhs+nu*e-mu
        argument = rhs-nu*e+mu; inverse_argument = np.linalg.solve(hessian, argument)
        dual = float(constant-nu-.5*np.dot(argument, inverse_argument))
        stable_gap = float(.5*(beta-inverse_argument)@hessian@(beta-inverse_argument)+mu@beta+nu*(1-e@beta))
        allowed = 1e-8*(1+abs(float(cert['primal'])))
        audit.close(cert['primal'], objective, tag+': direct weighted residual objective', atol=2e-8)
        audit.close(cert['dual'], dual, tag+': independent dual objective', atol=2e-8)
        audit.close(cert['direct_gap'], cert['primal']-cert['dual'], tag+': recorded direct gap')
        audit.close(cert['stable_gap'], stable_gap, tag+': stable gap', atol=2e-9)
        audit.close(cert['gap_tolerance'], allowed, tag+': unchanged qualification tolerance')
        audit.close(cert['equality_residual'], equality, tag+': equality residual')
        audit.close(cert['min_coefficient'], beta.min(), tag+': smallest coefficient')
        audit.close(cert['stationarity'], np.max(np.abs(stationarity)), tag+': stationarity')
        audit.close(cert['complementarity'], np.max(np.abs(mu*beta)), tag+': complementarity')
        audit.equal(cert['active_coefficients'], np.flatnonzero(beta > 1e-10), tag+': active set')
        audit.check(equality <= 1e-10 and beta.min() >= -1e-10 and -1e-10 <= cert['stable_gap'] <= allowed
                    and abs(cert['direct_gap']-cert['stable_gap']) <= allowed, tag+': original worker qualification')
        audit.check(-3e-9 <= stable_gap <= allowed+2e-9 and abs(objective-dual-stable_gap) <= allowed+4e-8,
                    tag+': independent certificate numerical bound')
        audit.check(1 <= cert['iterations'] <= 1000, tag+': bounded active-set iterations')
    else:
        gradient = hessian@beta-rhs
        stable_gap = float(.5*gradient@np.linalg.solve(hessian, gradient))
        dual = objective-stable_gap
        audit.close(cert['stationarity'], np.max(np.abs(gradient)), tag+': unconstrained stationarity')
        audit.close(cert['stable_gap'], stable_gap, tag+': unconstrained objective gap')
        audit.check(cert['gap_tolerance'] == 1e-8 and cert['iterations'] == 1 and 0 <= cert['stable_gap'] <= 1e-8,
                    tag+': recorded ridge qualification')
        audit.check(stable_gap <= 1e-8+2e-9, tag+': independent ridge precision')
    audit.bounds.append({'fit': tag, 'objective_normalized': objective, 'dual_normalized': dual,
                         'stable_gap_normalized': stable_gap, 'y_scale_C': scale['y_scale'],
                         'approximate_floating_point_certificate': True})
    return model


def boost_x(record, scale):
    return np.column_stack([record['boundary'], record['heat']/scale['qscale'], record['instant']])


def replay_tree(tree, x, audit, tag, verify_counts=False):
    fields = set(tree.dtype.names or [])
    needed = {'value', 'count', 'feature_idx', 'num_threshold', 'left', 'right', 'is_leaf', 'is_categorical'}
    if not needed <= fields: raise ValueError('Unexpected HGB node schema')
    audit.check(np.isfinite(tree['value']).all() and np.isfinite(tree['num_threshold']).all()
                and np.all(tree['is_categorical'] == 0), tag+': finite numeric tree')
    leaf = tree['is_leaf'].astype(bool); active = np.zeros(len(x), dtype=np.int64)
    visits = np.zeros(len(tree), dtype=np.int64)
    for depth in range(len(tree)+1):
        audit.budget()
        moving = np.flatnonzero(~leaf[active])
        if depth == 0 and verify_counts: visits[0] = len(x)
        if not len(moving): break
        node = active[moving]; feature = tree['feature_idx'][node].astype(np.int64)
        if np.any(feature >= x.shape[1]): raise ValueError('Feature index outside33')
        left = x[moving, feature] <= tree['num_threshold'][node]
        child = np.where(left, tree['left'][node], tree['right'][node]).astype(np.int64)
        if np.any(child <= node) or np.any(child >= len(tree)): raise ValueError('Invalid/cyclic child index')
        active[moving] = child
        if verify_counts: visits += np.bincount(child, minlength=len(tree))
    else: raise ValueError('Tree traversal did not terminate')
    if verify_counts: audit.equal(tree['count'], visits, tag+': all FIT node sample counts')
    return tree['value'][active]


def replay_booster(data, x, audit, tag, verify_counts=False):
    audit.equal(data['n_iter'], [200], tag+': fixed200 iterations')
    baseline = np.asarray(data['baseline'])
    audit.check(baseline.shape == (1, 1) and np.isfinite(baseline).all(), tag+': one-output finite baseline')
    expected_keys = {'baseline', 'n_iter'} | {'tree_%03d'%j for j in range(200)} | {'bin_thresholds_%02d'%j for j in range(33)}
    audit.check(set(data) == expected_keys, tag+': exact tree/bin archive')
    for j in range(33):
        bins = data['bin_thresholds_%02d'%j]
        audit.check(bins.ndim == 1 and len(bins) <= 254 and np.isfinite(bins).all() and np.all(np.diff(bins) >= 0),
                    tag+': nondecreasing bin thresholds '+str(j))
        if verify_counts:
            # sklearn1.5.1 binning.py leaves repeated midpoint percentiles intact.
            # Reconstruct every threshold from the independently validated FIT
            # features, requiring the cohort to be below the200000 subsampling cap.
            if len(x) > 200000:
                raise ValueError('Threshold reconstruction requires no bin subsampling')
            ordered = np.sort(x[:, j]); distinct = np.unique(ordered)
            if np.max(np.abs(ordered)) >= 1e290:
                raise ValueError('Unexpected feature scale near bin clipping boundary')
            if len(distinct) <= 255:
                expected = .5*(distinct[:-1]+distinct[1:])
            else:
                fractions = np.linspace(0.,100.,256)[1:-1]/100.
                locations = fractions*(len(ordered)-1)
                lo, hi = np.floor(locations).astype(np.int64), np.ceil(locations).astype(np.int64)
                expected = .5*(ordered[lo]+ordered[hi])
            audit.close(bins,expected,tag+': independent FIT bin thresholds '+str(j))
            audit.bin_diagnostics.append({'booster':tag,'feature':j,'threshold_count':len(bins),
                'repeated_adjacent_thresholds':int(np.count_nonzero(np.diff(bins)==0)),
                'fit_rows':len(x),'independent_midpoint_reconstruction':True})
    result = np.full(len(x), baseline.item(), dtype=np.float64)
    for j in range(200):
        tree = data['tree_%03d'%j]
        audit.check(np.count_nonzero(tree['is_leaf']) <= 15, tag+': leaf cap '+str(j))
        result += replay_tree(tree, x, audit, tag+'/'+str(j), verify_counts)
    return result


def metric(pred, truth):
    error = pred-truth
    return {'rmse': float(np.sqrt(np.dot(error, error)/len(error))),
            'mae': float(np.abs(error).mean()), 'max_absolute_error': float(np.abs(error).max()),
            'max_underprediction': float(max(0., -error.min()))}


def main():
    parser = argparse.ArgumentParser()
    for name in ('run-dir', 'assets-dir', 'manifest', 'report'): parser.add_argument('--'+name, required=True)
    parser.add_argument('--max-seconds', type=float, default=600.)
    args = parser.parse_args(); root, assets = Path(args.run_dir).resolve(), Path(args.assets_dir).resolve()
    destination = Path(args.report)
    if destination.exists(): raise FileExistsError('Preserve prior raw audit')
    if not np.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 600: raise ValueError('Audit cap0..600s')
    audit = Audit(args.max_seconds); report = {'auditor_source_sha256': sha(__file__), 'worker_source_sha256': WORKER}
    try:
        with threadpool_limits(limits=1):
            audit.check(audit.hash(Path(__file__).with_name('combination_c12_fit_rc_v001.py')) == WORKER, 'Worker final source pin')
            audit.check(audit.hash(args.manifest) == MANIFEST, 'Raw manifest pin')
            manifest = audit.json(args.manifest); items = {int(v['id']): v for v in manifest['files']}
            expected_roles = {**{p:'fit' for p in FIT}, **{p:'calibration' for p in CAL}, **{p:'pilot' for p in TEST}}
            audit.check(set(items) == set(expected_roles) and all(items[p]['role'] == v for p,v in expected_roles.items()), 'Exact14 admitted roles')
            lock = audit.json(root/'model_lock.json'); mode = lock['mode']
            audit.check(mode in ('costprobe', 'pilot'), 'Declared run mode'); ids = FIT[:2] if mode == 'costprobe' else FIT
            audit.equal(lock['fit_ids'], ids, 'Fixed FIT identifiers')
            audit.check(lock['source_sha256'] == WORKER and lock['manifest_sha256'] == MANIFEST, 'Model lock source provenance')
            expected_config = dict(methods=METHODS, primary=PRIMARY, spans=SPANS, lambdas=LAMBDAS, warmup=300, fit_stride=5, booster=BOOST)
            audit.check(lock['config'] == expected_config, 'Exact frozen model configuration')
            expected_files = {'fixtures.json', 'linear_models.json', 'cv_records.json', 'booster_records.json', 'training.npz'}
            expected_files |= {'fit_%02d.npz'%p for p in ids} | {m+'.npz' for m in BOOSTERS}
            audit.check(set(lock['files']) == expected_files, 'All fitted artifacts locked')
            for name, digest in lock['files'].items(): audit.check(audit.hash(root/name) == digest, 'Locked model artifact '+name)
            fixture = audit.json(root/'fixtures.json')
            audit.check(fixture['status'] == 'PASS' and fixture['thermal_recurrence'] is True and fixture['degree_unit_inverse'] is True, 'Recorded fixed fixtures')
            records, receipts = {}, {}
            for p in ids:
                op, y, receipt = raw_profile(audit, assets, items[p]); receipts[p] = receipt
                record = features(op, audit); record['y'] = y; records[p] = record
                saved = audit.arrays(root/('fit_%02d.npz'%p))
                audit.equal(saved['operating'], op, str(p)+': exact raw operating rows'); audit.equal(saved['target'], y, str(p)+': exact FIT rotor labels')
                for name in ('instant','boundary','heat','ewma'): audit.close(saved[name], record[name], str(p)+': independently recursive '+name)
                logged = next(v for v in lock['sources'] if v['id'] == p)
                audit.check(all(logged[k] == v for k,v in receipt.items()) and logged['target_decoded'] is True, str(p)+': raw FIT receipt')
            models = audit.json(root/'linear_models.json'); cv = audit.json(root/'cv_records.json')
            audit.check(set(models) == set(cv) == {'positive','free','ewma'}, 'All3 linear families')
            oof, extrapolation = {}, {}
            for family in ('positive','free','ewma'):
                rows = cv[family]; folds = rows['folds']; lookup = {(row['lam'], row['held_out']):row['model'] for row in folds}
                audit.check(len(folds) == len(ids)*3 and set(lookup) == {(lam,p) for lam in LAMBDAS for p in ids}, family+': complete LOO grid')
                scores = []
                for lam in LAMBDAS:
                    errors = []
                    for held in ids:
                        model = lookup[(lam,held)]
                        verify_linear(audit, model, records, [p for p in ids if p != held], family, lam, family+'/'+str(lam)+'/'+str(held))
                        ix = indices(records[held]); pred = prediction(model, records[held])[ix]
                        errors.append(metric(pred,records[held]['y'][ix])['rmse'])
                    logged = next(v for v in rows['cv'] if v['lam'] == lam)
                    audit.close(logged['profile_rmse'], errors, family+': held-out losses '+str(lam))
                    scores.append(float(np.mean(errors))); audit.close(logged['macro_rmse'], scores[-1], family+': equal-profile CV '+str(lam))
                selected = LAMBDAS[min(range(3),key=lambda i:(scores[i],i))]
                audit.check(rows['selected_lambda'] == selected and rows['family'] == family, family+': FIT-only selected lambda')
                verify_linear(audit, models[family], records, ids, family, selected, family+'/final')
                oof[family] = {p:prediction(lookup[(selected,p)],records[p])[indices(records[p])] for p in ids}
                extrapolation[family] = {str(p):{'minimum_oof_prediction_C':float(oof[family][p].min()),
                    'maximum_oof_prediction_C':float(oof[family][p].max()),
                    'oof_rmse_C':metric(oof[family][p],records[p]['y'][indices(records[p])])['rmse']} for p in ids}
            scale = models['positive']['scale']; xparts = [boost_x(records[p],scale)[indices(records[p])] for p in ids]
            ys = [records[p]['y'][indices(records[p])] for p in ids]; x,y = np.concatenate(xparts),np.concatenate(ys)
            weights = np.concatenate([np.full(len(v),len(y)/(len(ids)*len(v))) for v in ys])
            poof,foof = [np.concatenate([oof[family][p] for p in ids]) for family in ('positive','free')]
            pins = np.concatenate([prediction(models['positive'],records[p])[indices(records[p])] for p in ids])
            training = audit.arrays(root/'training.npz')
            for key,value in [('features',x),('target',y),('sample_weight',weights),('positive_oof',poof),('free_oof',foof),('positive_insample',pins)]:
                audit.close(training[key],value,'Exact FIT training construction '+key)
            audit.equal(training['profile_ids'],np.concatenate([np.full(len(v),p) for p,v in zip(ids,ys)]),'Training profile order')
            targets = dict(direct_boost=y,positive_rc_oof_boost=y-poof,unconstrained_rc_oof_boost=y-foof,positive_rc_insample_boost=y-pins)
            boost_meta = audit.json(root/'booster_records.json'); boosters = {}
            audit.check(set(boost_meta) == set(BOOSTERS),'All4 boosters')
            for name in BOOSTERS:
                info = boost_meta[name]; path = root/(name+'.npz')
                audit.check(info['path'] == path.name and info['sha256'] == audit.hash(path) and info['bytes'] == path.stat().st_size and info['n_iter'] == 200,name+': fitted tree receipt')
                boosters[name] = audit.arrays(path)
                audit.close(boosters[name]['baseline'].item(),np.dot(weights,targets[name])/weights.sum(),name+': independent initial weighted response mean',atol=2e-8)
                audit.close(info['baseline'],boosters[name]['baseline'],name+': baseline record')
                replay_booster(boosters[name],x,audit,name+'/FIT',True)
            audit.check(lock['training']['cv_models'] == 3*len(ids)*3 and lock['training']['all_linear_fits'] == 3*(len(ids)*3+1)
                        and lock['training']['fit_rows'] == len(y),'Exact linear-fit counts and training cohort')
            summary = audit.json(root/'summary.json')
            if mode == 'costprobe':
                audit.check(summary['status'] == 'COSTPROBE_COMPLETE_NO_SCIENTIFIC_TEST' and summary['no_cal_or_pilot_access'] is True,'Costprobe not scientific evidence')
                report.update(scientific_status='COSTPROBE_ONLY_NO_CAL_OR_PILOT_AUDIT')
            else:
                predlock = audit.json(root/'prediction_lock.json'); saved_metrics = audit.json(root/'metrics.json')
                audit.check(predlock['schema'] == 'c12-prediction-lock-v001' and predlock['model_lock_sha256'] == audit.hash(root/'model_lock.json'),'Prediction lock provenance')
                audit.equal([row['id'] for row in predlock['records']],CAL+TEST,'All6 predictions before evaluation')
                audit.check(len(saved_metrics) == 48 and len({(r['id'],r['method']) for r in saved_metrics}) == 48,'All48 method/profile rows')
                results, ptimes, ttimes = [],[],[]
                for p in CAL+TEST:
                    op,truth,receipt = raw_profile(audit,assets,items[p]); record = features(op,audit); ix=np.arange(300,len(op))
                    path=root/('profile_%02d_predictions.npz'%p); saved=audit.arrays(path)
                    recorded=next(v for v in predlock['records'] if v['id']==p)
                    audit.check(recorded['path']==path.name and recorded['sha256']==audit.hash(path)
                        and recorded['role']==items[p]['role'] and recorded['rows']==len(op) and recorded['evaluated_rows']==len(ix)
                        and all(recorded['source'][k]==v for k,v in receipt.items()) and recorded['source']['target_decoded'] is False,str(p)+': prediction/input receipt')
                    audit.equal(saved['operating'],op,str(p)+': raw operating rows'); audit.equal(saved['index'],ix,str(p)+': original evaluation indices')
                    for key,value in record.items(): audit.close(saved[key],value,str(p)+': causal '+key)
                    bx=boost_x(record,scale); ap=prediction(models['positive'],record); fp=prediction(models['free'],record)
                    tree_pred={name:replay_booster(data,bx,audit,str(p)+'/'+name) for name,data in boosters.items()}
                    predictions={'positive_rc':ap,'unconstrained_rc_ridge':fp,'direct_boost':tree_pred['direct_boost'],
                        PRIMARY:ap+tree_pred[PRIMARY],'unconstrained_rc_oof_boost':fp+tree_pred['unconstrained_rc_oof_boost'],
                        'mean_A_B':(ap+tree_pred['direct_boost'])/2,'ewma_ridge':prediction(models['ewma'],record),
                        'positive_rc_insample_boost':ap+tree_pred['positive_rc_insample_boost']}
                    truth_path=root/('profile_%02d_truth.npz'%p); logged_truth=audit.arrays(truth_path)
                    audit.equal(logged_truth['index'],ix,str(p)+': exact truth indices'); audit.equal(logged_truth['target'],truth[ix],str(p)+': raw truth')
                    for name,values in predictions.items():
                        audit.close(saved['prediction__'+name],values[ix],str(p)+': raw-feature/tree replay '+name,atol=2e-8,rtol=2e-10)
                        # Metrics use the fully verified saved predictions, retaining exact arithmetic values.
                        calculated=metric(saved['prediction__'+name],truth[ix]); row=next(r for r in saved_metrics if r['id']==p and r['method']==name)
                        audit.check(row['role']==items[p]['role'],str(p)+': metric role '+name)
                        for key,value in calculated.items(): audit.close(row[key],value,str(p)+'/'+name+': metric '+key)
                        results.append(dict(id=p,role=items[p]['role'],method=name,**calculated))
                    ptimes.append(path.stat().st_mtime_ns); ttimes.append(truth_path.stat().st_mtime_ns)
                audit.check((root/'model_lock.json').stat().st_mtime_ns <= min(ptimes)
                    and max(ptimes) <= (root/'prediction_lock.json').stat().st_mtime_ns <= min(ttimes),'Timestamps corroborate model/prediction/target order')
                macro={role:{name:float(np.mean([r['rmse'] for r in results if r['role']==role and r['method']==name])) for name in METHODS} for role in ('calibration','pilot')}
                best=min((m for m in METHODS if m!=PRIMARY),key=lambda m:(macro['pilot'][m],METHODS.index(m)))
                gain=1-macro['pilot'][PRIMARY]/macro['pilot'][best]
                positive=sum(next(r['rmse'] for r in results if r['id']==p and r['method']==PRIMARY)<next(r['rmse'] for r in results if r['id']==p and r['method']==best) for p in TEST)
                for role in macro:
                    for name,value in macro[role].items(): audit.close(summary['macro_rmse_C'][role][name],value,'Macro '+role+'/'+name)
                audit.check(summary['status']=='COMPLETE_PENDING_INDEPENDENT_AUDIT' and summary['primary']==PRIMARY and summary['best_pilot_control']==best
                    and summary['positive_profiles_vs_fixed_best_control']==positive and summary['prospective_clean_screen_pass']==bool(gain>=.03 and positive>=2)
                    and summary['calibration_is_diagnostic_only'] is True and summary['confirmation_not_accessed'] is True,'Fixed primary decision and diagnostic-only CAL')
                audit.close(summary['relative_error_reduction_vs_best_control'],gain,'Primary relative reduction')
                report.update(scientific_status='ADAPTIVE_DEVELOPMENT_SIGNAL' if gain>=.03 and positive>=2 else 'PRIMARY_SCREEN_CRITERIA_NOT_MET',
                    case_results=results,macro_rmse_C=macro,strongest_control=best,relative_reduction=gain)
            for name,digest in lock['files'].items(): audit.check(audit.hash(root/name)==digest,'Final unchanged fitted artifact '+name)
            report.update(status='PASS' if not audit.failures else 'INTEGRITY_FAILURE',mode=mode,oof_extrapolation_diagnostics=extrapolation)
    except BaseException as error:
        report.update(status='AUDIT_INCOMPLETE',exception_type=type(error).__name__,exception=str(error))
    report.update(checks=audit.checks,failures=audit.failures,input_file_sha256=audit.files,maximum_absolute_errors=audit.errors,
        linear_objective_bounds=audit.bounds,bin_threshold_diagnostics=audit.bin_diagnostics,elapsed_seconds=time.monotonic()-audit.start,fresh_confirmation=False,
        tolerances={'default_atol':2e-9,'default_rtol':2e-10,'prediction_atol':2e-8,'certificate_roundoff_allowance':2e-9},
        limitations=['No worker import, model pickle or new profile download; exact source/asset pins',
            'All linear FIT/CV objectives and dual qualifications reconstructed; active-set path not rerun',
            'v001 strict bin-threshold monotonicity was invalid; v002 allows legitimate repeated quantiles and reconstructs all FIT thresholds',
            'Reference sklearn1.5.1 binning.py SHA256 ca07067868dcab018ec8bc814070ef46b4bb481af67f4133c9293b0cc4785339',
            'Saved HGB trees traversed on raw features, including all FIT node counts; tree split optimization/leaf Newton fitting not rerun',
            'OOF predictions use selected all-FIT CV lambda and are not nested independent risk estimates',
            'Extreme OOF extrapolation is diagnosed, never clipped or repaired from evaluation labels',
            'File times corroborate ordering but are not tamper-proof attestation',
            'One motor and already exposed development profiles; no final confirmation'])
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('status','checks','elapsed_seconds')}))
    if report['status']!='PASS': raise SystemExit(2)


if __name__ == '__main__': main()
