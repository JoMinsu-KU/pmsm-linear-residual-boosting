"""Independent C18 diagnostic artifact audit, without scientific worker imports.

Rebuilds every saved diagnostic from already independently audited predictions,
truth and feature caches. This is not a raw CSV/feature-generation or training
audit; those predecessor audits remain explicit pinned dependencies.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'linear-followup-20261111'
METHODS = ['d9', 'linear_prior', 'linear_residual', 'linear_half', 'linear_cal_mix',
           'direct200', 'direct200_half', 'free25_residual', 'free25_half', 'prior_half']
PAIRS = ['linear_residual', 'direct200', 'free25_residual', 'linear_prior']
COHORTS = [('ipmsm', 'calibration'), ('ipmsm', 'pilot'), ('ipmsm', 'confirmation'),
           ('paderborn', 'development'), ('paderborn', 'confirmation'),
           ('rwth48v', 'calibration'), ('rwth48v', 'confirmation')]
MAPPING = dict(d9='c15__left', linear_residual='linear33__right', linear_half='linear33__half',
               linear_cal_mix='linear33__cal_macro_rmse', direct200='direct200__right',
               direct200_half='direct200__half', free25_residual='free25__right', free25_half='free25__half')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def local(value):
    value = str(value)
    if os.name != 'nt' and re.match(r'^[A-Za-z]:[\\/]', value):
        value = '/mnt/' + value[0].lower() + '/' + value[3:].replace('\\', '/')
    elif os.name == 'nt' and re.match(r'^/mnt/[a-z]/', value):
        value = value[5].upper() + ':/' + value[7:]
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Audit input must remain in campaign')
    return path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class Audit:
    def __init__(self, cap):
        self.start = time.monotonic()
        self.cap = cap
        self.inputs = {}
        self.reports = {}
        self.checks = 0
        self.failures = []

    def budget(self):
        if time.monotonic() - self.start > self.cap:
            raise TimeoutError('Independent audit time cap')
        if datetime.now(timezone.utc) >= datetime.fromisoformat('2026-11-12T00:23:59+00:00'):
            raise TimeoutError('Campaign deadline')

    def require(self, condition, label):
        self.budget()
        self.checks += 1
        if not condition:
            raise AssertionError(label)

    def pin(self, entry):
        self.budget()
        path = local(entry['path'])
        actual = sha(path)
        self.require(actual == entry['sha256'], 'Pinned bytes unchanged: ' + str(path))
        if str(path) in self.inputs:
            self.require(self.inputs[str(path)] == actual, 'No contradictory input identity')
        self.inputs[str(path)] = actual
        return path

    def previous(self, entry):
        path = self.pin(entry)
        key = (str(path), entry['sha256'])
        if key not in self.reports:
            value = read(path)
            self.require(value.get('status') == 'PASS' and not value.get('failures'), 'Prior complete independent PASS')
            identities = [value[k] for k in ('auditor_sha256', 'source_sha256') if k in value]
            self.require(bool(identities) and all(v == entry['auditor_sha256'] for v in identities), 'Exact predecessor auditor identity')
            self.pin(dict(path=entry['auditor_path'], sha256=entry['auditor_sha256']))
            index = {}
            for raw, digest in value['inputs'].items():
                index.setdefault(str(local(raw)), []).append(digest)
            self.reports[key] = (value, index)
        return self.reports[key]

    def audited(self, entry, report):
        path = self.pin(entry)
        _, index = self.previous(report)
        matches = index.get(str(path), [])
        self.require(bool(matches) and all(d == entry['sha256'] for d in matches), 'Prior PASS binds exact source artifact: ' + str(path))
        return path

    def arrays(self, entry, report=None, keys=None):
        path = self.audited(entry, report) if report else self.pin(entry)
        with np.load(path, allow_pickle=False) as z:
            return {key: z[key].copy() for key in (z.files if keys is None else keys)}

    def close(self, actual, expected, label, atol=1e-8, rtol=1e-10):
        x, y = np.asarray(actual), np.asarray(expected)
        self.require(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
                     and np.allclose(x, y, atol=atol, rtol=rtol), label)

    def exact(self, actual, expected, label):
        self.require(np.asarray(actual).shape == np.asarray(expected).shape
                     and np.array_equal(actual, expected), label)

    def same(self, actual, expected, label):
        if isinstance(expected, dict):
            self.require(isinstance(actual, dict) and set(actual) == set(expected), label + '/keys')
            for key in expected:
                self.same(actual[key], expected[key], label + '/' + key)
        elif isinstance(expected, (list, tuple)):
            self.require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label + '/length')
            for i, value in enumerate(expected):
                self.same(actual[i], value, label + '/' + str(i))
        elif expected is None or isinstance(expected, (str, bool, int)):
            self.require(actual == expected, label)
        else:
            self.close(actual, expected, label)

    def finish(self):
        for name, digest in list(self.inputs.items()):
            self.require(sha(name) == digest, 'Input remains unchanged at audit finish')


def average(x):
    return float(np.sum(x, dtype=np.float64) / len(x))


def product(x, y):
    return float(np.dot(x, y) / len(x))


def metrics(y, p):
    error = p - y
    mu = average(error)
    centered = error - mu
    under = np.where(error < 0., -error, 0.)
    over = np.where(error > 0., error, 0.)
    mse = product(error, error)
    return dict(n=len(y), mse=mse, rmse=math.sqrt(mse), mae=average(np.abs(error)), bias=mu,
                error_variance=product(centered, centered), under_rmse=math.sqrt(product(under, under)),
                over_mae=average(over), max_under=float(np.max(under)), max_over=float(np.max(over)))


def half_accounting(y, d, r):
    e, f = d-y, r-y
    mean_e, mean_f = average(e), average(f)
    ec, fc = e-mean_e, f-mean_f
    md, mr = product(e, e), product(f, f)
    # Calculate the half error independently from producer prediction arithmetic.
    half_error = (d+r)*0.5-y
    mh = product(half_error, half_error)
    disagreement = product(e-f, e-f)
    quality, diversity = 0.5*(mr-md), -0.25*disagreement
    vd, vr, cov = product(ec, ec), product(fc, fc), product(ec, fc)
    bd = ((mean_e+mean_f)*0.5)**2-mean_e**2
    variance_delta = (vd+vr+2.*cov)*0.25-vd
    denom = math.sqrt(mh)+math.sqrt(md)
    def normalize(v): return v/denom if denom else 0.
    return dict(mse_d9=md, mse_expert=mr, mse_half=mh, delta_mse=mh-md,
                expert_quality_term=quality, disagreement=disagreement, diversity_term=diversity,
                mu_d9=mean_e, mu_expert=mean_f, variance_d9=vd, variance_expert=vr,
                error_covariance=cov, centered_error_correlation=cov/math.sqrt(vd*vr) if vd>0 and vr>0 else None,
                bias_squared_delta=bd, variance_delta=variance_delta, rmse_denominator=denom,
                delta_rmse=math.sqrt(mh)-math.sqrt(md), rmse_expert_quality_term=normalize(quality),
                rmse_diversity_term=normalize(diversity), rmse_bias_squared_term=normalize(bd),
                rmse_variance_term=normalize(variance_delta))


def residual_accounting(y, prior, hybrid):
    e, r = prior-y, hybrid-prior
    base = product(e, e)
    final = product(hybrid-y, hybrid-y)
    energy = product(r, r)
    return dict(prior_mse=base, hybrid_mse=final, delta_mse=final-base,
                error_correction_cross_term=2.*product(e, r), correction_energy=energy,
                correction_mean=average(r), correction_rms=math.sqrt(energy))


def subset(y, predictions, indices):
    if not len(indices):
        return dict(n=0, methods=None, decomposition=None, residual_effect=None)
    yy = y[indices]
    pp = {k: v[indices] for k, v in predictions.items()}
    return dict(n=len(indices), methods={k: metrics(yy, v) for k, v in pp.items()},
                decomposition={k: half_accounting(yy, pp['d9'], pp[k]) for k in PAIRS},
                residual_effect=residual_accounting(yy, pp['linear_prior'], pp['linear_residual']))


def view(data, kind):
    if kind == 'legacy':
        x = np.concatenate((data['boundary'], data['heat'], data['instant']), axis=1)
        obs = np.stack((data['instant'][:,0], data['instant'][:,3], data['instant'][:,6]), axis=1)
    else:
        x = data['X33']
        obs = np.stack((np.abs(x[:,0]), np.abs(x[:,1]), x[:,3]), axis=1)
    return np.asarray(x, dtype=np.float64), np.asarray(obs, dtype=np.float64)


def quantile_linear(values, probability):
    ordered = np.sort(values)
    location = (len(ordered)-1)*probability
    lo, hi = math.floor(location), math.ceil(location)
    weight = location-lo
    return float((1.-weight)*ordered[lo]+weight*ordered[hi])


def fit_reference(a, item):
    source = read(a.pin(item['reference_config']))
    expected_ids = source['train_ids'] if item['kind']=='legacy' else source['roles']['fit']
    a.same(item['fit_ids'], expected_ids, 'Original frozen FIT role IDs')
    a.same([v['id'] for v in item['fit_cache']], expected_ids, 'All FIT cache IDs')
    xx, oo, yy = [], [], []
    for entry in item['fit_cache']:
        fields = ['boundary','heat','instant','target'] if item['kind']=='legacy' else ['X33','target']
        data = a.arrays(entry, item['fit_audit'], fields)
        x, o = view(data, item['kind'])
        a.require(x.shape==(entry['rows'],33) and o.shape==(entry['rows'],3)
                  and data['target'].shape==(entry['rows'],), 'Source FIT cache geometry')
        ix = np.arange(item['warmup'], entry['rows'], 5, dtype=np.int64)
        a.require(len(ix)>0 and np.isfinite(x).all() and np.isfinite(data['target']).all(), 'Complete finite FIT endpoints')
        xx.append(x[ix]); oo.append(o[ix]); yy.append(data['target'][ix])
    x, o, y = np.vstack(xx), np.vstack(oo), np.concatenate(yy)
    conditions = {}
    for name, values in zip(['current','abs_speed','coolant','target_temperature'], [o[:,0],o[:,1],o[:,2],y]):
        conditions[name] = dict(min=float(np.min(values)), q05=quantile_linear(values,.05),
            median=quantile_linear(values,.5), q95=quantile_linear(values,.95), max=float(np.max(values)))
    return dict(dataset=item['dataset'], kind=item['kind'], fit_ids=expected_ids, training_endpoints=len(y),
        warmup=item['warmup'], stride=5, feature_min=np.min(x,axis=0).tolist(), feature_max=np.max(x,axis=0).tolist(),
        conditions=conditions, scope='Pooled actual FIT training endpoints only; coordinate ranges are descriptive, not a deployment OOD guarantee')


def source_profile(a, item, entry, reference, rwth_prior, cpu_rows):
    names = ['index','source_row']+['prediction__'+v for v in MAPPING.values()]
    raw = a.arrays(entry, item['audit'], names)
    index = raw['index']
    a.require(index.dtype.kind in 'iu' and index.ndim==1 and len(index)>0, 'Original integer evaluation rows')
    truth_ref = item['truths'][str(entry['profile'])] if item['kind']=='rwth' and item['role']=='confirmation' else entry['truth_ref']
    truth = a.arrays(truth_ref, item['audit'], ['index','target'])
    a.exact(truth['index'], index, 'Exact target-to-prediction row alignment')
    a.exact(index, np.arange(300 if item['kind']=='legacy' else 0, int(index[-1])+1, dtype=np.int64), 'All original contiguous rows, no omission')
    target = truth['target'].astype(np.float64)
    if item['kind']=='legacy':
        data = a.arrays(entry['source_features'], item['audit'], ['boundary','heat','instant','source_row','prediction__linear33_ridge'])
        all_x, all_o = view(data, 'legacy')
        a.require(int(index[-1])==len(all_x)-1, 'Complete retained original legacy profile')
        x, observed = all_x[index], all_o[index]
        prior = data['prediction__linear33_ridge'].astype(np.float64)
        a.exact(raw['source_row'], data['source_row'][index], 'Legacy source-row identity')
    else:
        data = a.arrays(entry['source_features'], item['audit'], ['X33','index','source_row'])
        x, observed = view(data, 'rwth')
        a.exact(data['index'], index, 'RWTH cache local-row identity')
        a.exact(raw['source_row'], data['source_row'], 'RWTH original source-row identity')
        scale, coef = rwth_prior['scale'], np.asarray(rwth_prior['coef'], dtype=np.float64)
        a.require(rwth_prior['family']=='linear33' and coef.shape==(34,), 'Original frozen unconstrained linear33 prior')
        xx = (x-np.asarray(scale['x_mean']))/np.asarray(scale['x_std'])
        prior = scale['y_mid']+scale['y_scale']*(coef[0]+np.einsum('ij,j->i', xx, coef[1:]))
        saved_prior = a.arrays(cpu_rows[entry['profile']], item['audit'], ['prediction__linear33_prior'])
        a.close(prior, saved_prior['prediction__linear33_prior'], 'RWTH34-coefficient independent affine prediction')
    pp = {key: raw['prediction__'+value].astype(np.float64) for key,value in MAPPING.items()}
    pp['linear_prior'] = prior
    pp['prior_half'] = (pp['d9']+prior)*0.5
    pp = {key:pp[key] for key in METHODS}
    for key, prediction in pp.items():
        a.require(prediction.shape==target.shape==(len(index),) and np.isfinite(prediction).all(), 'Finite aligned method/'+key)
    a.require(np.isfinite(target).all() and x.shape==(len(index),33), 'Finite source truth and33 features')
    for name, right in [('linear_half','linear_residual'),('direct200_half','direct200'),('free25_half','free25_residual')]:
        a.exact(pp[name], (pp['d9']+pp[right])*.5, 'Exact existing fixed-half prediction/'+name)
    outside = np.logical_or(x<np.asarray(reference['feature_min']), x>np.asarray(reference['feature_max']))
    arrays = dict(index=index.astype(np.int64), source_row=raw['source_row'].astype(np.int64), target=target,
        observed_current=observed[:,0], observed_abs_speed=observed[:,1], observed_coolant=observed[:,2], feature_outside_fit=outside)
    arrays.update({'prediction__'+k:v for k,v in pp.items()})
    return arrays, pp, observed, outside


def profile_result(item, entry, reference, arrays, pp, observed, outside):
    y, ix = arrays['target'], arrays['index']
    n = len(y)
    whole = subset(y, pp, np.arange(n))
    blocks = []
    denominator = whole['decomposition']['linear_residual']['rmse_denominator']
    for b in range(5):
        lo, hi = n*b//5, n*(b+1)//5
        current = subset(y, pp, np.arange(lo,hi))
        current.update(bin=b, index_start=int(ix[lo]) if hi>lo else None,
            index_end_inclusive=int(ix[hi-1]) if hi>lo else None,
            evaluated_fraction_start=b/5, evaluated_fraction_end=(b+1)/5)
        # Use a sum of raw squared-error differences / full n and denominator.
        d, h = pp['d9'][lo:hi]-y[lo:hi], pp['linear_half'][lo:hi]-y[lo:hi]
        current['primary_delta_rmse_contribution'] = float(np.sum(h*h-d*d)/n/denominator) if denominator else 0.
        blocks.append(current)
    conditions, ranges = {}, {}
    for name, values in zip(['current','abs_speed','coolant','target_temperature'], [observed[:,0],observed[:,1],observed[:,2],y]):
        limits = reference['conditions'][name]
        masks = dict(below_fit_q05=values<limits['q05'], central_fit_q05_q95=(values>=limits['q05'])&(values<=limits['q95']), above_fit_q95=values>limits['q95'])
        conditions[name] = {key:subset(y,pp,np.flatnonzero(mask)) for key,mask in masks.items()}
        ranges[name] = dict(below_fit_min_fraction=float(np.count_nonzero(values<limits['min'])/n),
                            above_fit_max_fraction=float(np.count_nonzero(values>limits['max'])/n))
    return dict(dataset=item['dataset'], role=item['role'], profile=entry['profile'], original_index_start=int(ix[0]),
        original_index_end=int(ix[-1]), whole=whole, progress_blocks=blocks, condition_strata=conditions,
        outside_fit_ranges=ranges, feature_range_diagnostic=dict(any_coordinate_outside_fraction=float(np.count_nonzero(outside.any(axis=1))/n),
            mean_coordinate_outside_fraction=float(np.count_nonzero(outside)/outside.size),
            per_coordinate_outside_fraction=(np.count_nonzero(outside,axis=0)/n).tolist()), all_original_rows_retained=True)


def cohort_result(rows):
    summary = {}
    for method in METHODS:
        measures = [row['whole']['methods'][method] for row in rows]
        total = sum(v['n'] for v in measures)
        summary[method] = dict(profiles=len(measures), n=total,
            **{'macro_'+key:math.fsum(v[key] for v in measures)/len(measures) for key in ['rmse','mae','bias','under_rmse','over_mae']},
            pooled_rmse=math.sqrt(math.fsum(v['n']*v['mse'] for v in measures)/total))
    names = ['delta_rmse','rmse_expert_quality_term','rmse_diversity_term','rmse_bias_squared_term','rmse_variance_term']
    attribution = {pair:{key:math.fsum(r['whole']['decomposition'][pair][key] for r in rows)/len(rows) for key in names} for pair in PAIRS}
    differences = [r['whole']['decomposition']['linear_residual']['delta_rmse'] for r in rows]
    base, candidate = summary['d9']['macro_rmse'], summary['linear_half']['macro_rmse']
    return dict(methods=summary, macro_rmse_attribution=attribution, primary=dict(delta_macro_rmse=candidate-base,
        relative_reduction=1.-candidate/base if base else None,
        per_profile_deltas=[dict(profile=r['profile'],delta_rmse=d) for r,d in zip(rows,differences)],
        wins=sum(d<0 for d in differences), ties=sum(d==0 for d in differences), losses=sum(d>0 for d in differences)),
        formal_test_performed=False, independent_confirmation=False)


def run(a, args):
    folder = local(args.input_dir)
    a.require(not (folder/'failure.json').exists(), 'No diagnostic producer failure marker')
    config = read(a.pin(dict(path=args.config,sha256=args.config_sha256)))
    summary = read(a.pin(dict(path=str(folder/'summary.json'),sha256=args.input_sha256)))
    a.require(summary['status']=='C18_FIXED_HALF_DIAGNOSTIC_COMPLETE' and summary['source_sha256']==args.producer_sha256,
              'Complete same-source diagnostic stage')
    a.require(config['producer']['sha256']==args.producer_sha256, 'Config pins exact producer')
    a.pin(config['producer']); a.pin(config['protocol']); a.pin(config['limits'])
    a.require(config['schema']=='c18-linear-diagnostic-v001' and config['primary']=='linear_half' and config['weight']==.5,
              'Fixed exploratory C18 method')
    a.require(config['progress_blocks']==5 and config['condition_quantiles']==[.05,.95] and config['all_data_exposed'] is True
              and config['formal_confirmation_allowed'] is False, 'Frozen diagnostic scopes')
    a.same([(v['dataset'],v['role']) for v in config['cohorts']], COHORTS, 'All seven roles')
    a.same(summary['methods'],METHODS,'Exact10 methods'); a.same(summary['profiles'],54,'All54 profile-role records')
    a.require(summary['new_training_fits']==0 and summary['new_weight_optimization'] is False and summary['new_formal_tests'] is False,
              'No training, reselection or new formal tests')
    a.require(summary['all_data_previously_exposed'] is True, 'Exploratory-only interpretation')
    for name,digest in summary['files'].items():
        path=(folder/name).resolve()
        a.require(path.is_relative_to(folder), 'Safe stage-relative artifact')
        a.pin(dict(path=str(path),sha256=digest))
    invocation=read(folder/'configuration.json')
    a.require(invocation['config_sha256']==args.config_sha256 and invocation['source_sha256']==args.producer_sha256,
              'Actual invocation source/config identities')
    a.require(local(summary['config']['path'])==local(args.config) and summary['config']['sha256']==args.config_sha256,
              'Summary exact configuration binding')
    saved_references=read(folder/'fit_reference.json')
    references={x['dataset']:fit_reference(a,x) for x in config['datasets']}
    a.same(saved_references,references,'Independent sorted-order FIT thresholds and feature ranges')
    # Exact bin decisions use the now independently validated serialized
    # thresholds. A different FP64 interpolation order must not flip points
    # lying exactly on the producer's floating-point threshold.
    references=saved_references
    output_records=[]; results=[]
    for item in config['cohorts']:
        reference=references[item['dataset']]
        a.require(not set(item['ids']).intersection(reference['fit_ids']), 'FIT/evaluation profile disjointness')
        lock=read(a.audited(item['prediction_lock'],item['audit']))
        entries=[v for v in lock['predictions'] if v['role']==item['role']]
        a.same([v['profile'] for v in entries],item['ids'],'Every fixed role profile in original order')
        old_metrics=read(a.audited(item['metrics'],item['audit']))['methods']
        prior=None; cpu_rows={}
        if item['kind']=='rwth':
            cpu_lock=read(a.audited(item['cpu_prediction_lock'],item['audit']))
            a.require(cpu_lock['model_lock']==item['model_lock'],'Original RWTH CPU model-lock binding')
            model=read(a.audited(item['model_lock'],item['model_audit']))
            a.require(model['priors']==item['priors'],'Original RWTH frozen prior artifact')
            prior=read(a.audited(item['priors'],item['model_audit']))['linear33']
            a.same(prior['fit_ids'],reference['fit_ids'],'RWTH prior uses original FIT20 only')
            cpu_rows={v['id']:v for v in cpu_lock['records']}
            a.same(list(cpu_rows),item['ids'],'RWTH base predictions complete role coverage')
        rows=[]
        for entry in entries:
            arrays, pp, observed, outside=source_profile(a,item,entry,reference,prior,cpu_rows)
            expected=profile_result(item,entry,reference,arrays,pp,observed,outside)
            stem=folder/(item['dataset']+'__'+item['role'])/('profile_'+str(entry['profile']))
            saved=read(stem.with_suffix('.json'))
            expected['arrays']=saved['arrays']
            a.require(local(saved['arrays']['path'])==stem.with_suffix('.npz'),'Own diagnostic array path')
            a.require(summary['files'][stem.with_suffix('.npz').relative_to(folder).as_posix()]==saved['arrays']['sha256'],
                      'Array summary/record hash identity')
            actual=a.arrays(saved['arrays'])
            a.require(set(actual)==set(arrays),'Exact saved diagnostic array keys')
            for name,value in arrays.items():
                a.require(actual[name].dtype==value.dtype,'Array dtype/'+name)
                if value.dtype.kind in 'biu' or name!='prediction__linear_prior' and name!='prediction__prior_half':
                    a.exact(actual[name],value,'All saved raw array values/'+name)
                else:
                    a.close(actual[name],value,'Independent affine-derived array/'+name)
            a.same(saved,expected,'All profile diagnostics/'+item['dataset']+'/'+item['role']+'/'+str(entry['profile']))
            for method,old_name in MAPPING.items():
                old=next(v for v in old_metrics[old_name][item['role']]['per_profile'] if v['id']==entry['profile'])
                for key in ('rmse','under_rmse','over_mae'):
                    a.close(expected['whole']['methods'][method][key],old[key],'Earlier audited metric preserved/'+method+'/'+key)
            d=expected['whole']['decomposition']['linear_residual']
            a.close(sum(v['primary_delta_rmse_contribution'] for v in expected['progress_blocks']),d['delta_rmse'],
                    'All five blocks attribute original profile RMSE difference')
            rows.append(expected)
            output_records.append(dict(dataset=item['dataset'],role=item['role'],profile=entry['profile'],arrays=saved['arrays']))
        result=dict(dataset=item['dataset'],role=item['role'],ids=item['ids'],summary=cohort_result(rows))
        a.same(read(folder/(item['dataset']+'__'+item['role'])/'summary.json'),result,'Every cohort macro/pooled metric and attribution')
        results.append(result)
    a.require(len(output_records)==54,'All54 profile-role reconstructions completed')
    a.same(read(folder/'diagnostic_arrays_lock.json'),dict(primary='linear_half',methods=METHODS,records=output_records),'Complete output array lock')
    a.same(read(folder/'results.json'),results,'Complete seven-cohort result table')
    for path,digest in read(folder/'input_pins.json').items():
        a.pin(dict(path=path,sha256=digest))
    a.require(not (folder/'failure.json').exists(),'No late producer failure')
    a.finish()
    return dict(cohorts=7,profiles=54,methods=10,progress_blocks_per_profile=5,
        condition_variables=4,condition_strata_per_variable=3,all_source_predictions_checked=True,
        all_prior_report_bindings_checked=True,rwth_linear_prior_independently_replayed=True,
        no_new_training=True,no_scientific_worker_import=True,no_new_formal_test=True,
        raw_feature_generation_replayed=False,predecessor_audit_dependencies=list({v[0] for v in a.reports}),
        scope='Full diagnostic recomputation from exact predecessor-audited artifacts; no raw CSV regeneration or training replay')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','config-sha256','input-dir','input-sha256','producer-sha256','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--max-seconds',type=float,default=600.)
    args=p.parse_args()
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=1200:
        raise ValueError('Finite bounded audit cap in (0,1200]')
    out=local(args.output_dir)
    if WORK/'runs' not in out.parents:
        raise ValueError('New C18 run output directory only')
    out.mkdir(parents=True,exist_ok=False)
    a=Audit(args.max_seconds); result={}; error=None
    try:
        result=run(a,args)
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        a.failures.append(dict(check='All mandatory audit operations completed',detail=error))
    report=dict(schema='c18-diagnostic-independent-audit-v001',status='HOLD' if a.failures else 'PASS',
        auditor_sha256=sha(__file__),producer_sha256=args.producer_sha256,config_sha256=args.config_sha256,
        input_summary_sha256=args.input_sha256,checks=a.checks,failures=a.failures,inputs=a.inputs,
        results=result,error=error,elapsed_seconds=time.monotonic()-a.start)
    payload=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)
    if len(payload.encode('utf-8'))>100000000:
        raise RuntimeError('Audit report100MB ceiling')
    with (out/'report.json').open('x',encoding='utf-8') as stream:
        stream.write(payload)
    print(json.dumps(dict(status=report['status'],checks=a.checks,failures=len(a.failures))))
    return 0 if not a.failures else 2


if __name__=='__main__':
    raise SystemExit(main())
