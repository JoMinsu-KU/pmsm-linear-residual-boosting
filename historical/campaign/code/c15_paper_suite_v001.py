"""Portable immutable-job CLI for the C15 paper experiment suite.

Stages: plan, fit, assemble, predict, evaluate, nested-hybrid.
Only root-admitted manifests may be used. FIT and evaluation access are separate
stages; evaluation targets cannot be decoded by plan/fit/assemble/predict.
Completed artifacts are never overwritten. Each invocation has its own output.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

# Works under python -I, which deliberately omits the script directory from
# sys.path. Both this source and the sibling are pinned in the root manifest.
_core_path = Path(__file__).with_name('c15_paper_core_v001.py')
_core_spec = importlib.util.spec_from_file_location('c15_paper_core_v001', _core_path)
c = importlib.util.module_from_spec(_core_spec)
_core_spec.loader.exec_module(c)

OLD15 = ['positive_rc', 'unconstrained_rc_ridge', 'direct_boost',
    'positive_rc_oof_boost', 'unconstrained_rc_oof_boost', 'mean_A_B',
    'ewma_ridge', 'positive_rc_insample_boost', 'unconstrained_rc_insample_boost'] + c.DIRECT[1:6] + ['direct_boost_fitcv_selected']
PRIMARY = 'positive_rc_insample_boost'
MAIN_COMPARATOR = 'direct9_fitcv_selected'


def method_specs():
    result = {
        'positive_rc': dict(kind='linear', prior='positive__tuned'),
        'unconstrained_rc_ridge': dict(kind='linear', prior='free__tuned'),
        'ewma_ridge': dict(kind='linear', prior='ewma__tuned'),
        'linear33_ridge': dict(kind='linear', prior='linear33__tuned'),
        'positive_rc_oof_boost': dict(kind='residual', prior='positive__tuned', target='oof'),
        'unconstrained_rc_oof_boost': dict(kind='residual', prior='free__tuned', target='oof'),
        PRIMARY: dict(kind='residual', prior='positive__tuned', target='insample'),
        'unconstrained_rc_insample_boost': dict(kind='residual', prior='free__tuned', target='insample'),
        'linear33_insample_boost': dict(kind='residual', prior='linear33__tuned', target='insample'),
        'mean_A_B': dict(kind='mean', components=['positive_rc', 'direct_boost']),
        'direct_boost_fitcv_selected': dict(kind='selected', grid_count=6),
        MAIN_COMPARATOR: dict(kind='selected', grid_count=9)}
    for gi, name in enumerate(c.DIRECT):
        result[name] = dict(kind='direct', grid_index=gi)
    for family in c.ABLATIONS:
        for mode in ('common', 'tuned'):
            if family == 'positive' and mode == 'tuned':
                continue
            name = 'ablation_'+family+'_'+mode
            result[name] = dict(kind='linear', prior=family+'__'+mode)
            result[name+'_residual'] = dict(kind='residual', prior=family+'__'+mode, target='insample')
    names = OLD15 + [name for name in result if name not in OLD15]
    if len(names) != 35 or len(set(names)) != 35:
        raise ValueError('Fixed35-method panel changed')
    return {name: result[name] for name in names}


def plan(ids):
    jobs = []
    for family in c.FAMILIES:
        for li, lam in enumerate(c.LAMBDAS):
            for p in ids:
                jobs.append(dict(job_id=f'linear_cv__{family}__l{li}__p{p}', kind='linear_cv',
                    family=family, lambda_index=li, lam=lam, held=p))
    for family in c.FAMILIES:
        jobs.append(dict(job_id='linear_final__'+family+'__tuned', kind='linear_final', family=family, mode='tuned'))
    for family in c.ABLATIONS:
        jobs.append(dict(job_id='linear_final__'+family+'__common', kind='linear_final', family=family, mode='common'))
    for gi in range(9):
        for p in ids:
            jobs.append(dict(job_id=f'direct_cv__g{gi}__p{p}', kind='direct_cv', grid_index=gi, held=p))
    specs = method_specs()
    ordered = sorted(specs, key=lambda name: bool(specs[name].get('prior','').endswith('__common')))
    for name in ordered:
        spec = specs[name]
        if spec['kind'] in ('direct', 'residual'):
            jobs.append(dict(job_id='full__'+name, kind='full', method=name, spec=spec))
    return jobs


def resolve(project, name):
    path = Path(name)
    return path.resolve() if path.is_absolute() else (project/path).resolve()


def verify_entry(project, entry, clock):
    clock.check()
    path = resolve(project, entry['path'])
    if not path.is_file() or c.sha(path) != entry['sha256']:
        raise ValueError('Registered source artifact changed: '+str(path))
    return path


def manifest(args, clock):
    project = Path(args.project_root).resolve()
    source = resolve(project, args.manifest)
    data = c.read(source)
    if data.get('schema') != 'c15-paper-dataset-v001' or data.get('dataset') != args.dataset:
        raise ValueError('Explicit frozen dataset manifest identity required')
    if data.get('scientific_execution_admitted') is not True:
        raise ValueError('Root source/protocol admission is required before numerical access')
    if data.get('core_sha256') != c.sha(Path(c.__file__)) or data.get('worker_sha256') != c.sha(__file__):
        raise ValueError('Manifest numerical sources are not pinned to this version')
    roles = data['roles']
    ids = [p for group in roles.values() for p in group]
    if (len(set(ids)) != len(ids) or len(roles['fit']) < 3 or
        any(type(p) is not int or p < 0 for p in ids)):
        raise ValueError('Profile-role overlap/type/minimumFIT violation')
    if set(v['id'] for v in data['fit_cache']) != set(roles['fit']):
        raise ValueError('FIT cache does not cover the exact frozen FIT role')
    for entry in data.get('source_files', {}).values():
        verify_entry(project, entry, clock)
    # Hash-only access is distinguished from numeric decoding for all roles.
    for entry in data['fit_cache']:
        verify_entry(project, entry, clock)
    if data['evaluation_source']['format'] == 'profile_csv':
        if set(v['id'] for v in data['evaluation_source']['files']) != set(ids)-set(roles['fit']):
            raise ValueError('Evaluation CSV list must cover every nonFIT role')
        for entry in data['evaluation_source']['files']:
            verify_entry(project, entry, clock)
    elif data['evaluation_source']['format'] != 'paderborn_csv':
        raise ValueError('Unknown explicit dataset parser')
    return project, source, data


def fit_records(project, data, clock):
    records = {}
    for item in data['fit_cache']:
        path = verify_entry(project, item, clock)
        saved = c.arrays(path)
        record = {k: saved[k] for k in ('boundary', 'heat', 'instant', 'ewma')}
        record['y'] = saved[item.get('target_key', 'target')]
        n = item['rows']
        if (record['y'].shape != (n,) or record['boundary'].shape != (n,10) or
            record['heat'].shape != (n,15) or record['instant'].shape != (n,8) or
            record['ewma'].shape != (n,23) or n <= 332 or
            any(not np.isfinite(v).all() for v in record.values())):
            raise ValueError('FIT cache numerical/geometry mismatch')
        records[item['id']] = record
    return records


def receipts(args, project, manifest_path, data, clock):
    lookup = {}
    if not args.dependencies:
        return lookup
    dependency = c.read(resolve(project, args.dependencies))
    if dependency.get('schema') != 'c15-paper-dependencies-v001' or dependency.get('manifest_sha256') != c.sha(manifest_path):
        raise ValueError('Dependency manifest identity changed')
    valid = {job['job_id']: job for job in plan(data['roles']['fit'])}
    for item in dependency['jobs']:
        folder = resolve(project, item['path'])
        receipt_path = folder/'receipt.json'
        receipt = c.read(receipt_path)
        key = receipt['job']['job_id']
        if (key not in valid or key in lookup or receipt['job'] != valid[key] or
            receipt.get('status') != 'FIT_JOB_COMPLETE' or (folder/'failure.json').exists() or
            receipt.get('manifest_sha256') != c.sha(manifest_path) or
            receipt.get('worker_sha256') != c.sha(__file__) or
            receipt.get('core_sha256') != c.sha(c.__file__) or
            item['receipt_sha256'] != c.sha(receipt_path)):
            raise ValueError('Immutable completed job lineage mismatch')
        for name, digest in receipt['files'].items():
            clock.check()
            path = (folder/name).resolve()
            if folder not in path.parents or c.sha(path) != digest:
                raise ValueError('Job artifact changed/escaped its directory')
        lookup[key] = dict(root=folder, model=c.read(folder/'model.json'), receipt=receipt)
    return lookup


def linear_choice(lookup, ids, family):
    rows = []
    for li, lam in enumerate(c.LAMBDAS):
        values = []
        for p in ids:
            entry = lookup[f'linear_cv__{family}__l{li}__p{p}']
            values.append(entry['model']['held_metrics']['rmse'])
        rows.append(dict(lam=lam, profile_rmse=values, macro_rmse=float(np.mean(values))))
    at = min(range(3), key=lambda j:(rows[j]['macro_rmse'], j))
    return at, rows


def direct_choice(lookup, ids, count):
    rows = []
    for gi in range(count):
        values = [lookup[f'direct_cv__g{gi}__p{p}']['model']['held_metrics']['rmse'] for p in ids]
        rows.append(dict(method=c.DIRECT[gi], grid_index=gi, profile_rmse=values, macro_rmse=float(np.mean(values))))
    at = min(range(count), key=lambda j:(rows[j]['macro_rmse'], j))
    return c.DIRECT[at], rows


def prior_model(lookup, key):
    return lookup['linear_final__'+key]['model']['linear_model']


def resolved_booster(lookup, key):
    seen = set()
    while True:
        if key in seen:
            raise ValueError('Cyclic model alias')
        seen.add(key)
        entry = lookup[key]
        alias = entry['model'].get('alias_job_id')
        if alias is None:
            return entry
        key = alias


def train_job(job, output, h, records, ids, lookup, manifest_digest, clock):
    output.mkdir()
    started = time.monotonic()
    key, kind = job['job_id'], job['kind']
    info = dict(job=job, fit_role_ids=ids, manifest_sha256=manifest_digest,
                sklearn=sklearn.__version__, numpy=np.__version__)
    if kind == 'linear_cv':
        train = [p for p in ids if p != job['held']]
        model = c.fit_linear(h, records, train, job['family'], job['lam'], clock)
        p = job['held']; ix = h.train_index(records[p])
        pred = c.predict_linear(h, model, records[p])[ix]
        c.dump(output/'linear_model.json', model)
        np.savez_compressed(output/'held.npz', index=ix, prediction=pred)
        info.update(train_ids=train, linear_model=model, held_metrics=c.metrics(pred, records[p]['y'][ix]))
    elif kind == 'linear_final':
        at, rows = linear_choice(lookup, ids, job['family'])
        li = at if job['mode'] == 'tuned' else 0
        alias = 'linear_final__'+job['family']+'__tuned'
        if job['mode'] == 'common' and at == 0:
            model = lookup[alias]['model']['linear_model']
            info['alias_job_id'] = alias
            info['actual_linear_fits'] = 0
        else:
            model = c.fit_linear(h, records, ids, job['family'], c.LAMBDAS[li], clock)
            info['actual_linear_fits'] = 1
        info.update(train_ids=ids, linear_model=model, selected_lambda_index=li,
            selection_mode=job['mode'], global_tuned_lambda_index=at, cv=rows,
            selection_performance_is_not_unbiased_procedure_estimate=True)
    else:
        train = [p for p in ids if p != job['held']] if kind == 'direct_cv' else ids
        x, y, weights, scale, ix = c.training(h, records, train)
        target, prior_pred = y, None
        spec = job.get('spec', {})
        gi = job['grid_index'] if kind == 'direct_cv' else spec.get('grid_index', 0)
        target_rule = 'raw_target'
        if kind == 'full' and spec['kind'] == 'residual':
            prior = prior_model(lookup, spec['prior'])
            family, mode = spec['prior'].split('__')
            if mode == 'common' and prior['lam'] == prior_model(lookup, family+'__tuned')['lam']:
                matching = [(name, value) for name, value in method_specs().items()
                    if value.get('kind') == 'residual' and value.get('prior') == family+'__tuned' and value.get('target') == spec['target']]
                if len(matching) != 1:
                    raise ValueError('Common/tuned residual alias is ambiguous')
                alias = 'full__'+matching[0][0]
                if alias not in lookup:
                    raise ValueError('Tuned residual must finish before its common-lambda alias')
                info.update(alias_job_id=alias, train_ids=ids, prior=prior,
                    target_construction='identical_common_and_tuned_lambda', actual_booster_fits=0)
                return finish_job(job, output, info, manifest_digest, clock, started)
            if spec['target'] == 'oof':
                li, _ = linear_choice(lookup, ids, family)
                blocks = []
                for p in ids:
                    saved = c.arrays(lookup[f'linear_cv__{family}__l{li}__p{p}']['root']/'held.npz')
                    if not np.array_equal(saved['index'], ix[p]):
                        raise ValueError('OOF endpoint/order mismatch')
                    blocks.append(saved['prediction'])
                prior_pred = np.concatenate(blocks)
                target_rule = 'selected_lambda_nonnested_OOF_residual'
            else:
                prior_pred = np.concatenate([c.predict_linear(h, prior, records[p])[ix[p]] for p in train])
                target_rule = 'fullFIT_insample_residual'
            target = y-prior_pred
            info['prior'] = prior
        model, metadata, bin_rows = c.fit_booster(h, x, target, weights, c.GRID[gi], output, clock)
        index_data = dict(profile_ids=np.concatenate([np.full(len(ix[p]), p, dtype=np.int64) for p in train]),
            local_indices=np.concatenate([ix[p] for p in train]), sample_weight=weights,
            binning_row_indices=bin_rows, fit_target=target)
        if prior_pred is not None:
            index_data['prior_prediction'] = prior_pred
        np.savez_compressed(output/'training_index.npz', **index_data)
        info.update(train_ids=train, scale=scale, **metadata, target_construction=target_rule, actual_booster_fits=1)
        if kind == 'direct_cv':
            p = job['held']; held_ix = h.train_index(records[p])
            pred = model.predict(h.boost_features(records[p], scale)[held_ix])
            np.savez_compressed(output/'held.npz', index=held_ix, prediction=pred)
            info['held_metrics'] = c.metrics(pred, records[p]['y'][held_ix])
        else:
            path = output/'model.pickle'
            with path.open('xb') as stream:
                pickle.dump(model, stream, protocol=5)
            info['pickle'] = dict(path=path.name, sha256=c.sha(path), bytes=path.stat().st_size,
                trust_scope='this worker hash-pinned local artifact only; auditor replays exported nodes')
    return finish_job(job, output, info, manifest_digest, clock, started)


def finish_job(job, output, info, manifest_digest, clock, started):
    info['elapsed_seconds'] = time.monotonic()-started
    c.dump(output/'model.json', info)
    clock.check()
    receipt = dict(status='FIT_JOB_COMPLETE', job=job, manifest_sha256=manifest_digest,
        worker_sha256=c.sha(__file__), core_sha256=c.sha(c.__file__),
        files={p.name:c.sha(p) for p in output.iterdir() if p.is_file()},
        elapsed_seconds=time.monotonic()-started)
    c.dump(output/'receipt.json', receipt)
    clock.check()
    return dict(root=output, model=info, receipt=receipt)


def assemble(lookup, ids, output, manifest_digest, clock):
    required = {job['job_id'] for job in plan(ids)}
    if set(lookup) != required:
        raise ValueError('Every predeclared CV/fullfit job must finish before assembly: '+str(sorted(required-set(lookup))[:10]))
    selected6, cv6 = direct_choice(lookup, ids, 6)
    selected9, cv9 = direct_choice(lookup, ids, 9)
    entries = {}
    for key, entry in lookup.items():
        entries[key] = dict(path=str(entry['root']), receipt_sha256=c.sha(entry['root']/'receipt.json'))
    lock = dict(schema='c15-paper-model-lock-v001', status='ALL_FIT_MODELS_LOCKED_NO_EVALUATION',
        manifest_sha256=manifest_digest, worker_sha256=c.sha(__file__), core_sha256=c.sha(c.__file__),
        fit_ids=ids, methods=method_specs(), primary=PRIMARY, main_comparator=MAIN_COMPARATOR,
        selected_direct6=selected6, selected_direct9=selected9, direct_cv6=cv6, direct_cv9=cv9,
        entries=entries, selection_uses_FIT_only=True, CV_selection_scores_are_not_unbiased=True,
        actual_booster_fits=sum(v['model'].get('actual_booster_fits',0) for v in lookup.values()),
        actual_linear_fits=sum(1 if v['model']['job']['kind']=='linear_cv' else v['model'].get('actual_linear_fits',0) for v in lookup.values()),
        evaluation_operating_decoded=False, evaluation_targets_decoded=False)
    c.dump(output/'model_lock.json', lock)
    clock.check()
    return lock


def read_role(project, data, role, targets, clock, confirmation_admitted=False):
    if role == 'fit' or role not in data['roles']:
        raise ValueError('Explicit nonFIT role required')
    if role == 'confirmation' and not confirmation_admitted:
        raise ValueError('Fresh confirmation requires separate prospective root admission')
    ids = data['roles'][role]
    source = data['evaluation_source']
    columns = source['operating_columns']; target = source['target_column']
    if len(columns) != 7:
        raise ValueError('Same seven operating signals required')
    raw = {}
    if source['format'] == 'profile_csv':
        entries = {v['id']:v for v in source['files']}
        for p in ids:
            path = verify_entry(project, entries[p], clock)
            reader = csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig')))
            values, labels = [], []
            for row in reader:
                if targets:
                    labels.append(float(row[target]))
                else:
                    values.append([float(row[k]) for k in columns])
            n = len(labels) if targets else len(values)
            if n != entries[p]['rows']:
                raise ValueError('Registered source profile length changed')
            raw[p] = dict(target=np.asarray(labels, np.float64)) if targets else dict(operating=np.asarray(values,np.float64), source_row=np.arange(n,dtype=np.int64))
    else:
        path = verify_entry(project, data['source_files'][source['csv_source_key']], clock)
        inventory = c.read(verify_entry(project, data['source_files'][source['inventory_source_key']], clock))
        selected = {p:[] for p in ids}; row_ids = {p:[] for p in ids}; counts = {p:0 for p in inventory['profile_ids']}
        with path.open('rb') as stream:
            header = next(csv.reader(io.StringIO(stream.readline(65537).decode('utf-8-sig'))))
            if header != inventory['header']:
                raise ValueError('Official header changed')
            pid = header.index('profile_id')
            take = [header.index(target)] if targets else [header.index(k) for k in columns]
            count = 0
            for line in stream:
                if count % 4096 == 0:
                    clock.check()
                if len(line)>65536 or b'"' in line or b'\0' in line:
                    raise ValueError('Unsupported source row encoding')
                tokens = line.rstrip(b'\r\n').split(b',')
                if len(tokens) != len(header) or not re.fullmatch(rb'(0|[1-9][0-9]*)(?:\.0+)?', tokens[pid].strip()):
                    raise ValueError('CSV/profile token contract')
                p = int(tokens[pid].split(b'.',1)[0])
                if p not in counts:
                    raise ValueError('Unregistered profile ID')
                counts[p] += 1
                if p in selected:
                    selected[p].append([float(tokens[j]) for j in take]); row_ids[p].append(count)
                count += 1
        if count != inventory['row_count'] or any(counts[p]!=inventory['profile_row_counts'][str(p)] for p in counts):
            raise ValueError('Frozen profile counts changed')
        for p in ids:
            value = np.asarray(selected[p],np.float64)
            raw[p] = dict(target=value[:,0]) if targets else dict(operating=value, source_row=np.asarray(row_ids[p],np.int64))
    for p, value in raw.items():
        if any(not np.isfinite(v).all() for v in value.values()) or len(next(iter(value.values()))) <= 332:
            raise ValueError('Nonfinite/short role record; no dropping or repair')
    return raw


def confirmation_admission(args, project, manifest_path, model_lock_sha):
    if args.role != 'confirmation':
        return False
    if not args.confirmation_admission:
        raise ValueError('Fresh confirmation requires --confirmation-admission')
    gate = c.read(resolve(project,args.confirmation_admission))
    expected = dict(schema='c15-paper-confirmation-admission-v001',
        manifest_sha256=c.sha(manifest_path),model_lock_sha256=model_lock_sha,
        worker_sha256=c.sha(__file__),core_sha256=c.sha(c.__file__),
        primary=PRIMARY,main_comparator=MAIN_COMPARATOR,
        numeric_operating_and_target_access_admitted=True)
    if any(gate.get(k)!=v for k,v in expected.items()):
        raise ValueError('Confirmation admission is not linked to the frozen models/protocol')
    protocol=gate.get('confirmation_protocol')
    if not isinstance(protocol,dict) or c.sha(resolve(project,protocol['path']))!=protocol['sha256']:
        raise ValueError('Confirmation protocol must be pinned before numerical opening')
    return True


def load_model_lock(args, project, manifest_path, data, clock):
    if not args.model_lock:
        raise ValueError('--model-lock required')
    path = resolve(project, args.model_lock)
    lock = c.read(path)
    if (lock.get('schema') != 'c15-paper-model-lock-v001' or lock['manifest_sha256'] != c.sha(manifest_path)
        or lock['worker_sha256'] != c.sha(__file__) or lock['core_sha256'] != c.sha(c.__file__)
        or lock['fit_ids'] != data['roles']['fit'] or lock['methods'] != method_specs()):
        raise ValueError('Model lock identity changed')
    lookup = {}
    for key, item in lock['entries'].items():
        folder = resolve(project,item['path']); receipt = c.read(folder/'receipt.json')
        if c.sha(folder/'receipt.json') != item['receipt_sha256'] or receipt['status'] != 'FIT_JOB_COMPLETE':
            raise ValueError('Completed model receipt changed')
        for name,digest in receipt['files'].items():
            clock.check(); artifact=(folder/name).resolve()
            if folder not in artifact.parents or c.sha(artifact)!=digest:
                raise ValueError('Model artifact changed/escaped before prediction')
        lookup[key] = dict(root=folder, model=c.read(folder/'model.json'), receipt=receipt)
    if set(lookup) != {j['job_id'] for j in plan(data['roles']['fit'])}:
        raise ValueError('Model lock omits registered jobs')
    return path,lock,lookup


def prediction_stage(args, project, manifest_path, data, output, h, clock):
    model_path, lock, lookup = load_model_lock(args,project,manifest_path,data,clock)
    admitted=confirmation_admission(args,project,manifest_path,c.sha(model_path))
    raw = read_role(project,data,args.role,False,clock,admitted)
    models = {}
    for name,spec in method_specs().items():
        if spec['kind'] in ('direct','residual'):
            entry = resolved_booster(lookup,'full__'+name)
            info=entry['model']; path=entry['root']/info['pickle']['path']
            if info['sklearn'] != sklearn.__version__ or c.sha(path)!=info['pickle']['sha256']:
                raise ValueError('Local fitted estimator trust/version mismatch')
            with path.open('rb') as stream:
                models[name] = (pickle.load(stream), info)
    records=[]
    for p in data['roles'][args.role]:
        clock.check(); feature_start=time.monotonic(); record=h.raw_features(raw[p]['operating'])
        feature_seconds=time.monotonic()-feature_start; ix=np.arange(300,len(raw[p]['operating']),dtype=np.int64)
        values={}; times={}
        for name,spec in method_specs().items():
            started=time.monotonic()
            if spec['kind']=='linear':
                value=c.predict_linear(h,prior_model(lookup,spec['prior']),record)[ix]
            elif spec['kind'] in ('direct','residual'):
                estimator,info=models[name]
                value=estimator.predict(h.boost_features(record,info['scale'])[ix])
                if spec['kind']=='residual':
                    value=value+c.predict_linear(h,prior_model(lookup,spec['prior']),record)[ix]
            else:
                continue
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite prediction')
            values[name]=value; times[name]=time.monotonic()-started
        values['mean_A_B']=.5*(values['positive_rc']+values['direct_boost'])
        values['direct_boost_fitcv_selected']=values[lock['selected_direct6']].copy()
        values[MAIN_COMPARATOR]=values[lock['selected_direct9']].copy()
        if set(values)!=set(method_specs()):
            raise ValueError('Prediction method coverage changed')
        path=output/f'profile_{p}_predictions.npz'
        np.savez_compressed(path,**raw[p],**record,index=ix,
                            **{'prediction__'+name:values[name] for name in method_specs()})
        records.append(dict(id=p,role=args.role,path=path.name,sha256=c.sha(path),rows=len(raw[p]['operating']),
            evaluated_rows=len(ix),feature_seconds=feature_seconds,prediction_seconds=times))
    pred_lock=dict(schema='c15-paper-prediction-lock-v001',status='ALL_ROLE_PREDICTIONS_LOCKED_BEFORE_TARGETS',manifest_sha256=c.sha(manifest_path),
        model_lock_path=str(model_path),model_lock_sha256=c.sha(model_path),role=args.role,
        records=records,methods=list(method_specs()),all_role_predictions_before_target_decode=True,
        target_values_decoded=False,elapsed_seconds=time.monotonic()-clock.start)
    c.dump(output/'prediction_lock.json',pred_lock)
    clock.check()
    return pred_lock


def evaluation_stage(args, project, manifest_path, data, output, clock):
    if not args.prediction_lock:
        raise ValueError('--prediction-lock required')
    path=resolve(project,args.prediction_lock); lock=c.read(path)
    if (lock.get('schema')!='c15-paper-prediction-lock-v001' or lock['manifest_sha256']!=c.sha(manifest_path)
        or lock['role']!=args.role or lock['methods']!=list(method_specs())
        or lock['all_role_predictions_before_target_decode'] is not True
        or [v['id'] for v in lock['records']]!=data['roles'][args.role]):
        raise ValueError('Complete predictions must be locked before target decode')
    if c.sha(resolve(project,lock['model_lock_path']))!=lock['model_lock_sha256']:
        raise ValueError('Model lock changed before evaluation')
    saved={}
    for item in lock['records']:
        artifact=(path.parent/item['path']).resolve()
        if path.parent not in artifact.parents or c.sha(artifact)!=item['sha256']:
            raise ValueError('Locked prediction artifact changed')
        saved[item['id']]=c.arrays(artifact)
    # Only this boundary decodes the requested role's targets.
    admitted=confirmation_admission(args,project,manifest_path,lock['model_lock_sha256'])
    truth=read_role(project,data,args.role,True,clock,admitted); rows=[]
    for p in data['roles'][args.role]:
        ix=saved[p]['index']; target=truth[p]['target']
        if not np.array_equal(ix,np.arange(300,len(target),dtype=np.int64)):
            raise ValueError('Evaluation endpoint alignment changed')
        np.savez_compressed(output/f'profile_{p}_truth.npz',index=ix,target=target[ix])
        for name in method_specs():
            rows.append(dict(id=p,role=args.role,method=name,**c.metrics(saved[p]['prediction__'+name],target[ix])))
    summary={}
    for name in method_specs():
        block=[v for v in rows if v['method']==name]; n=sum(v['n'] for v in block)
        summary[name]=dict(profile_count=len(block),n=n,macro_rmse=float(np.mean([v['rmse'] for v in block])),
            pooled_rmse=float(np.sqrt(sum(v['squared_error_sum'] for v in block)/n)),
            macro_mae=float(np.mean([v['mae'] for v in block])),
            max_absolute_error=max(v['max_absolute_error'] for v in block),
            max_underprediction=max(v['max_underprediction'] for v in block))
    paired={str(p):next(v['rmse'] for v in rows if v['id']==p and v['method']==MAIN_COMPARATOR)
                  -next(v['rmse'] for v in rows if v['id']==p and v['method']==PRIMARY) for p in data['roles'][args.role]}
    c.dump(output/'metrics.json',rows)
    result=dict(status='ROLE_EVALUATION_COMPLETE_REQUIRES_INDEPENDENT_AUDIT',role=args.role,
        primary=PRIMARY,main_comparator=MAIN_COMPARATOR,summary=summary,paired_rmse_gain=paired,
        relative_macro_reduction=1-summary[PRIMARY]['macro_rmse']/summary[MAIN_COMPARATOR]['macro_rmse'],
        prediction_lock_sha256=c.sha(path),profile_unit='sessions of one motor, not independent motors',
        no_automatic_three_percent_rejection=True,topic_or_paper_status_not_changed=True)
    c.dump(output/'metrics_summary.json',result)
    clock.check()
    return result


def nested_hybrid(args, h, records, ids, output, clock):
    """Optional oldFIT diagnostic; no external41 nested-CV requirement.

    Each outer held profile is excluded from inner lambda selection. Fixed HGB
    15x200 is compared only as a fixed algorithm; no selected-direct9 procedure
    claim is inferred from the nonnested direct capacity-selection scores.
    """
    if args.dataset!='ipmsm' or args.held not in ids or args.family not in ('positive','free','linear33'):
        raise ValueError('Nested diagnostic is restricted to oldFIT and three fixed priors')
    train=[p for p in ids if p!=args.held]; folder=output/'inner';folder.mkdir(); table=[]
    for li,lam in enumerate(c.LAMBDAS):
        scores=[]
        for p in train:
            fit_ids=[v for v in train if v!=p]
            model=c.fit_linear(h,records,fit_ids,args.family,lam,clock)
            ix=h.train_index(records[p]);pred=c.predict_linear(h,model,records[p])[ix]
            stem=f'l{li}_held{p}'
            c.dump(folder/(stem+'.json'),dict(fit_ids=fit_ids,held=p,linear_model=model,metrics=c.metrics(pred,records[p]['y'][ix])))
            np.savez_compressed(folder/(stem+'.npz'),index=ix,prediction=pred)
            scores.append(c.metrics(pred,records[p]['y'][ix])['rmse'])
        table.append(dict(lam=lam,profile_rmse=scores,macro_rmse=float(np.mean(scores))))
    at=min(range(3),key=lambda j:(table[j]['macro_rmse'],j))
    prior=c.fit_linear(h,records,train,args.family,c.LAMBDAS[at],clock)
    x,y,weights,scale,ix=c.training(h,records,train)
    prior_pred=np.concatenate([c.predict_linear(h,prior,records[p])[ix[p]] for p in train])
    model,metadata,bin_rows=c.fit_booster(h,x,y-prior_pred,weights,c.GRID[0],output,clock)
    p=args.held;held_ix=h.train_index(records[p])
    prediction=c.predict_linear(h,prior,records[p])[held_ix]+model.predict(h.boost_features(records[p],scale)[held_ix])
    np.savez_compressed(output/'held.npz',index=held_ix,prediction=prediction)
    np.savez_compressed(output/'training_index.npz',profile_ids=np.concatenate([np.full(len(ix[v]),v,dtype=np.int64) for v in train]),
        local_indices=np.concatenate([ix[v] for v in train]),sample_weight=weights,fit_target=y-prior_pred,
        prior_prediction=prior_pred,binning_row_indices=bin_rows)
    result=dict(status='NESTED_OLD_FIT_DIAGNOSTIC_COMPLETE',held=p,train_ids=train,family=args.family,
        selected_lambda_index=at,inner_cv=table,prior=prior,scale=scale,**metadata,
        held_metrics=c.metrics(prediction,records[p]['y'][held_ix]),
        old_FIT_is_not_fresh_confirmation=True,direct_capacity_selection_not_nested_by_this_job=True)
    c.dump(output/'model.json',result)
    clock.check()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',required=True)
    parser.add_argument('--dataset',choices=['ipmsm','paderborn'],required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--stage',choices=['plan','fixtures','fit','assemble','predict','evaluate','nested-hybrid'],required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--dependencies')
    parser.add_argument('--job-ids',nargs='+')
    parser.add_argument('--model-lock')
    parser.add_argument('--prediction-lock')
    parser.add_argument('--confirmation-admission')
    parser.add_argument('--role',choices=['calibration','pilot','development','confirmation'])
    parser.add_argument('--family',choices=['positive','free','linear33'])
    parser.add_argument('--held',type=int)
    parser.add_argument('--max-seconds',type=float,default=3600.)
    parser.add_argument('--max-output-bytes',type=int,default=2_000_000_000)
    args=parser.parse_args()
    if not np.isfinite(args.max_seconds) or not 0<args.max_seconds<=3600 or not 0<args.max_output_bytes<=10_000_000_000:
        raise ValueError('Stage cap must be <=3600seconds and <=10GB')
    if sklearn.__version__!='1.5.1':
        raise ValueError('Frozen sklearn1.5.1 required; no automatic dependency installation')
    output=resolve(Path(args.project_root).resolve(),args.output_dir)
    output.mkdir(parents=True,exist_ok=False)
    clock=c.Clock(args.max_seconds,output,args.max_output_bytes)
    try:
        project,manifest_path,data=manifest(args,clock);ids=data['roles']['fit']
        c.dump(output/'configuration.json',dict(worker_sha256=c.sha(__file__),core_sha256=c.sha(c.__file__),
            C12_source_sha256=c.C12_SHA,manifest_path=str(manifest_path),manifest_sha256=c.sha(manifest_path),
            dataset=args.dataset,stage=args.stage,role=args.role,methods=method_specs(),
            direct_grid=c.GRID,base_booster=c.BASE,lambdas=c.LAMBDAS,fit_ids=ids,
            warmup=300,fit_stride=5,evaluation_stride=1,cpu_threads=1,gpu_count=0,
            numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
            max_seconds=args.max_seconds,max_output_bytes=args.max_output_bytes))
        with threadpool_limits(limits=1):
            h=c.helper()
            if args.stage=='plan':
                result=dict(status='METADATA_PLAN_ONLY',jobs=plan(ids),methods=method_specs(),
                    numeric_data_decoded=False,total_jobs=len(plan(ids)))
                c.dump(output/'plan.json',result)
            elif args.stage=='fixtures':
                result=c.fixtures()
                c.dump(output/'fixtures.json',result)
            elif args.stage in ('fit','assemble'):
                lookup=receipts(args,project,manifest_path,data,clock)
                if args.stage=='assemble':
                    result=assemble(lookup,ids,output,c.sha(manifest_path),clock)
                else:
                    if not args.job_ids or len(args.job_ids)!=len(set(args.job_ids)):
                        raise ValueError('Explicit unique admitted job IDs required')
                    jobs={j['job_id']:j for j in plan(ids)}
                    if any(key not in jobs or key in lookup for key in args.job_ids):
                        raise ValueError('Unknown or already-completed job; no overwrite')
                    records=fit_records(project,data,clock); completed=[]
                    for key in args.job_ids:
                        clock.check();entry=train_job(jobs[key],output/key,h,records,ids,lookup,c.sha(manifest_path),clock)
                        lookup[key]=entry;completed.append(dict(job_id=key,path=str(entry['root']),receipt_sha256=c.sha(entry['root']/'receipt.json')))
                        c.dump(output/'progress.json',dict(completed=completed,elapsed_seconds=time.monotonic()-clock.start))
                    result=dict(status='FIT_BATCH_COMPLETE',completed=completed,evaluation_roles_decoded=False)
            elif args.stage=='predict':
                result=prediction_stage(args,project,manifest_path,data,output,h,clock)
            elif args.stage=='evaluate':
                result=evaluation_stage(args,project,manifest_path,data,output,clock)
            else:
                result=nested_hybrid(args,h,fit_records(project,data,clock),ids,output,clock)
        clock.check()
        c.dump(output/'summary.json',dict(status='STAGE_COMPLETE_REQUIRES_AUDIT',stage=args.stage,
            scientific_status=result.get('status'),elapsed_seconds=time.monotonic()-clock.start,
            worker_sha256=c.sha(__file__),core_sha256=c.sha(c.__file__),manifest_sha256=c.sha(manifest_path),
            artifact_sha256={str(p.relative_to(output)):c.sha(p) for p in output.rglob('*') if p.is_file()},
            topic_or_paper_status_not_changed=True))
        clock.check()
    except BaseException as error:
        c.dump(output/'failure.json',dict(status='INCOMPLETE_OPERATIONAL_OR_NUMERICAL_HOLD',
            type=type(error).__name__,message=str(error),elapsed_seconds=time.monotonic()-clock.start))
        raise


if __name__=='__main__':
    main()
