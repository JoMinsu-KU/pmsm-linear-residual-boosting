"""Frozen matched MLP comparator, immutable supervised stages; NumPy/PyTorch only.

No job launcher, network access, installations, automatic retries or training on
evaluation data. The scientific RC helper is shared code, not an independent
audit. Plan/select are metadata-only and do not import PyTorch. Root registers
every invocation and separately admits confirmation access.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.dont_write_bytecode = True
import numpy as np

CORE_SHA = '52d897657163391de6e1e42a51dad04550d5ca6f1dfc2a60f2e0041ec3f68a86'
C12_SHA = 'd4aa942ca9db8d8e9319197632cf376730456f3ea5cdc48b9d1c1fd51fc59412'
FIT = [0, 2, 3, 5, 7, 8, 11, 14]
EXTERNAL_FIT = [61,79,14,9,27,60,46,72,75,63,71,6,21,43,54,59,29,17,31,42,64,32,70,23,76,7,10,41,47,74,44,69,56,45,50,73,4,49,51,16,65]
EXTERNAL_DEV = [18,53,78,20,12,11,81,8,3,30,13,24,48,66]
EXTERNAL_CONF = [26,19,80,55,67,52,62,57,36,2,15,68,5,58]
ROLES = {'ipmsm': {'fit': FIT, 'calibration': [1,6,12], 'pilot': [4,9,13], 'confirmation': [10,15]},
         'paderborn': {'fit': EXTERNAL_FIT, 'development': EXTERNAL_DEV, 'confirmation': EXTERNAL_CONF}}
KINDS = ['pure', 'rc_residual']
EPOCHS = [50,100,200]
SEEDS = [2026111501,2026111502,2026111503]
WORKSPACE = Path(__file__).resolve().parents[1]
SETTINGS = dict(architecture=[33,64,32,1], activation='ReLU', parameters=4289,
    optimizer='AdamW', learning_rate=.001, betas=[.9,.999], eps=1e-8,
    weight_decay=.0001, batch_size=4096, gradient_norm_cap=1.,
    epoch_candidates=EPOCHS, seeds=SEEDS, warmup=300, fit_stride=5,
    feature_std_floor=1e-8, target_iqr_floor=1.,
    profile_loss='mean(N/(P*n_profile) * squared_normalized_error)',
    permutation='NumPy PCG64(seed + epoch - 1), fresh full permutation each epoch',
    inference_batch_size=4096, precision='network float32; transforms and physical outputs float64',
    final_seed_aggregation='mean of seed metrics, not best seed and not an ensemble')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Immutable artifact already exists: ' + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_arrays(path, **values):
    if Path(path).exists():
        raise FileExistsError('Immutable array artifact already exists')
    np.savez_compressed(path, **values)


def local_path(value, base=None):
    value = str(value)
    if os.name != 'nt' and re.match(r'^[A-Za-z]:[\\/]', value):
        value = '/mnt/' + value[0].lower() + '/' + value[3:].replace('\\', '/')
    elif os.name == 'nt' and re.match(r'^/mnt/[a-z]/', value):
        value = value[5].upper() + ':/' + value[7:]
    path = Path(value)
    if not path.is_absolute():
        path = (base or WORKSPACE) / path
    path = path.resolve()
    if path != WORKSPACE and WORKSPACE not in path.parents:
        raise ValueError('Path outside campaign workspace: ' + str(path))
    return path


def checked_path(path, expected):
    path = local_path(path)
    if not isinstance(expected, str) or len(expected) != 64 or sha(path) != expected:
        raise ValueError('Artifact hash mismatch: ' + str(path))
    return path


def array_digest(values):
    digest = hashlib.sha256()
    for key in sorted(values):
        value = np.ascontiguousarray(values[key])
        digest.update(json.dumps([key, value.dtype.str, list(value.shape)], separators=(',', ':')).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class Clock:
    def __init__(self, seconds, output):
        self.started, self.seconds, self.output = time.monotonic(), seconds, output

    def check(self):
        if time.monotonic() - self.started > self.seconds:
            raise TimeoutError('MLP stage wall cap exceeded')
        if sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file()) > 2_000_000_000:
            raise RuntimeError('MLP stage 2 GB output cap exceeded')


def core_module():
    path = Path(__file__).with_name('c15_neural_rc_core_v001.py')
    if sha(path) != CORE_SHA or sha(path.with_name('combination_c12_fit_rc_v001.py')) != C12_SHA:
        raise ValueError('Pinned original/shared scientific source changed')
    spec = importlib.util.spec_from_file_location('c15_paper_mlp_shared_rc', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(args, clock):
    path = checked_path(args.manifest, args.manifest_sha256)
    value = read(path)
    if value.get('schema') != 'c15-paper-dataset-v001' or value.get('dataset') != args.dataset:
        raise ValueError('Exact CPU dataset manifest required')
    if value.get('scientific_execution_admitted') is not True or value.get('roles') != ROLES[args.dataset]:
        raise ValueError('Frozen dataset roles/admission changed')
    entries = value['fit_cache']
    if len(entries) != len(ROLES[args.dataset]['fit']) or set(x['id'] for x in entries) != set(ROLES[args.dataset]['fit']):
        raise ValueError('All and only frozen FIT caches required')
    for entry in entries:
        clock.check()
        checked_path(entry['path'], entry['sha256'])
        if entry.get('target_key', 'target') != 'target' or entry['rows'] <= 300:
            raise ValueError('Unexpected FIT target/row metadata')
    for entry in value['source_files'].values():
        clock.check()
        checked_path(entry['path'], entry['sha256'])
    protocol = value['source_files'].get('protocol')
    if not protocol or local_path(protocol['path']) != WORKSPACE/'protocols/c15-paper-v001-frozen.md':
        raise ValueError('Frozen paper protocol missing')
    return value


def fit_records(data, ids, clock):
    if not set(ids) <= set(data['roles']['fit']):
        raise ValueError('Only FIT numerical data may be decoded for training')
    entries = {x['id']: x for x in data['fit_cache']}
    records = {}
    for profile in ids:
        clock.check()
        entry = entries[profile]
        with np.load(local_path(entry['path']), allow_pickle=False) as saved:
            record = {k: saved[k].astype(np.float64) for k in ('instant','boundary','heat','ewma')}
            record['y'] = saved['target'].astype(np.float64)
        n = entry['rows']
        if record['y'].shape != (n,) or any(record[k].shape != (n,d) for k,d in [('instant',8),('boundary',10),('heat',15),('ewma',23)]):
            raise ValueError('FIT record geometry mismatch')
        if any(not np.isfinite(v).all() for v in record.values()):
            raise ValueError('Nonfinite FIT data')
        records[profile] = record
    return records


def transforms(core, records, ids):
    scale = core.scaling(records, ids)
    x = np.concatenate([core.features(records[p], scale)[core.indices(records[p])] for p in ids])
    return dict(rc_scale=scale, input_mean=x.mean(0).tolist(), input_std=np.maximum(x.std(0), 1e-8).tolist(),
                y_mid=scale['y_mid'], y_scale=scale['y_scale'], fit_ids=list(ids), warmup=300, label_stride=5)


def verify_prior(prior, ids, transform):
    coef = np.asarray(prior['coef'])
    if prior.get('family') != 'positive' or prior['lam'] not in [.0001,.01,1.] or prior['scale']['ids'] != ids:
        raise ValueError('Prior family, lambda or FIT cohort changed')
    if coef.shape != (25,) or not np.isfinite(coef).all() or coef.min() < -1e-10 or abs(coef[:10].sum()-1.) > 1e-10:
        raise ValueError('Prior coefficient constraints failed')
    for key in ('qscale','y_mid','y_scale','ewma_mean','ewma_scale'):
        if not np.allclose(prior['scale'][key], transform['rc_scale'][key], rtol=1e-12, atol=1e-12):
            raise ValueError('Prior scale differs from actual training cohort: ' + key)


def obtain_prior(args, core, records, ids, transform, clock):
    if args.kind == 'pure':
        if args.prior_receipt:
            raise ValueError('Pure model cannot consume a prior receipt')
        return None, None
    if args.prior_receipt:
        if args.mode != 'final':
            raise ValueError('Fold prior must be selected inside outer training profiles')
        receipt_path = checked_path(args.prior_receipt, args.prior_receipt_sha256)
        receipt = read(receipt_path)
        if receipt.get('status') != 'FIT_JOB_COMPLETE' or receipt.get('manifest_sha256') != args.manifest_sha256:
            raise ValueError('CPU prior receipt is incomplete or dataset-mismatched')
        source_manifest = read(local_path(args.manifest))
        if (receipt.get('worker_sha256') != source_manifest['worker_sha256'] or
                receipt.get('core_sha256') != source_manifest['core_sha256'] or 'model.json' not in receipt['files']):
            raise ValueError('CPU prior receipt lacks exact source/model pins')
        if receipt['job']['job_id'] != 'linear_final__positive__tuned':
            raise ValueError('Only canonical full-FIT positive tuned prior is permitted')
        for name, expected in receipt['files'].items():
            candidate = local_path(name, receipt_path.parent)
            if receipt_path.parent not in candidate.parents:
                raise ValueError('CPU receipt artifact path escapes its directory')
            checked_path(candidate, expected)
        metadata = read(receipt_path.parent/'model.json')
        if metadata['train_ids'] != ids or metadata['fit_role_ids'] != ids or metadata['job'] != receipt['job']:
            raise ValueError('CPU prior training profiles differ')
        prior = metadata['linear_model']
        selection = dict(reused_cpu_receipt=str(receipt_path), receipt_sha256=sha(receipt_path),
                         model_sha256=sha(receipt_path.parent/'model.json'), selected_lambda=prior['lam'], fit_ids=ids)
    else:
        prior, selection = core.select(records, ids, clock)
    verify_prior(prior, ids, transform)
    return prior, selection


def cuda_runtime():
    global torch
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Registered local CUDA GPU required; no silent CPU fallback')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return dict(torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
                device=torch.cuda.get_device_name(0), device_count=torch.cuda.device_count())


def new_model(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = torch.nn.Sequential(torch.nn.Linear(33,64), torch.nn.ReLU(), torch.nn.Linear(64,32),
                                torch.nn.ReLU(), torch.nn.Linear(32,1)).cuda()
    if sum(x.numel() for x in model.parameters()) != 4289:
        raise AssertionError('MLP parameter count changed')
    return model


def weights(model):
    return {key: value.detach().cpu().numpy().copy() for key, value in model.state_dict().items()}


def arithmetic_fixtures():
    """Constructed inputs only, executed with a registered cost probe."""
    model = new_model(SEEDS[0])
    x = torch.arange(396, dtype=torch.float32, device='cuda').reshape(12,33)/100.
    model.eval()
    with torch.no_grad():
        first = model(x)
        changed = x.clone(); changed[7:] += 5.
        second = model(changed)
        if not torch.equal(first[:7], second[:7]):
            raise AssertionError('Pointwise MLP is not causal with respect to feature rows')
        model[-1].weight.zero_(); model[-1].bias.zero_()
        zero = model(x).reshape(-1)
        if not torch.equal(zero,torch.zeros_like(zero)):
            raise AssertionError('Zero-head correction must be zero')
    counts = [3,9]
    w = torch.as_tensor(np.r_[np.full(3,12/(2*3)),np.full(9,12/(2*9))], dtype=torch.float64,device='cuda')
    error = torch.arange(12,dtype=torch.float64,device='cuda',requires_grad=True)
    loss = (w*error.square()).mean()
    expected = .5*(error[:3].square().mean()+error[3:].square().mean())
    gradient = torch.autograd.grad(loss,error)[0]
    if not torch.allclose(loss,expected,rtol=1e-14,atol=1e-14) or not torch.allclose(gradient,2*w*error/12,rtol=1e-14,atol=1e-14):
        raise AssertionError('Profile-balanced MSE value/gradient identity failed')
    return dict(status='DATA_FREE_MLP_INVARIANTS_PASS',parameters=4289,zero_head=True,
                causal_feature_row_mapping=True,profile_balanced_value_and_gradient=True,profile_counts=counts)


def load_weights(model, path):
    with np.load(path, allow_pickle=False) as saved:
        values = {key: torch.as_tensor(saved[key], device='cuda') for key in saved.files}
    model.load_state_dict(values, strict=True)


def transformed(core, record, transform, index):
    value = (core.features(record, transform['rc_scale'])[index] - np.asarray(transform['input_mean'])) / np.asarray(transform['input_std'])
    if value.shape[1] != 33 or not np.isfinite(value).all():
        raise ValueError('Invalid common-feature standardized array')
    return value.astype(np.float32)


def predict(core, model, record, transform, prior, index, clock):
    x = transformed(core, record, transform, index)
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            clock.check()
            y = model(torch.as_tensor(x[start:start+4096], device='cuda')).reshape(-1)
            result.append(y.cpu().numpy().astype(np.float64))
    base = transform['y_mid'] if prior is None else core.predict(prior, record)[index]
    output = base + transform['y_scale'] * np.concatenate(result)
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite physical-unit MLP prediction')
    return output


def training_arrays(core, records, ids, transform, prior):
    xs, ys, ps, indices, counts = [], [], [], [], []
    for profile in ids:
        index = core.indices(records[profile])
        xs.append(transformed(core, records[profile], transform, index))
        base = transform['y_mid'] if prior is None else core.predict(prior, records[profile])[index]
        ys.append(((records[profile]['y'][index] - base)/transform['y_scale']).astype(np.float32))
        ps.append(np.full(len(index), profile, dtype=np.int64)); indices.append(index); counts.append(len(index))
    n = sum(counts)
    w = np.concatenate([np.full(count, n/(len(ids)*count), dtype=np.float64) for count in counts])
    if not np.isclose(w.sum(), n) or any(not np.isclose(w[np.concatenate(ps)==p].sum(), n/len(ids)) for p in ids):
        raise AssertionError('Profile weight identity failed')
    return np.concatenate(xs), np.concatenate(ys), w.astype(np.float32), np.concatenate(ps), np.concatenate(indices)


def train(args, core, records, ids, epochs, checkpoint_epochs, output, clock, held=None):
    transform = transforms(core, records, ids)
    prior, prior_selection = obtain_prior(args, core, records, ids, transform, clock)
    model = new_model(args.seed)
    initial = weights(model)
    save_arrays(output/'initial_weights.npz', **initial)
    x, y, w, profiles, index = training_arrays(core, records, ids, transform, prior)
    save_arrays(output/'training_index.npz', profile_ids=profiles, local_indices=index, sample_weight=w,
                normalized_target=y)
    metadata = dict(kind=args.kind, seed=args.seed, dataset=args.dataset, fit_ids=ids,
        held_out=held, epochs=epochs, transform=transform, prior=prior, prior_selection=prior_selection,
        settings=SETTINGS, initial_array_sha256=array_digest(initial),
        training_array_sha256=array_digest(dict(x=x,y=y,w=w,profile=profiles,index=index)),
        feature_array_sha256=array_digest(dict(x=x,profile=profiles,index=index)),
        weight_array_sha256=array_digest(dict(w=w,profile=profiles,index=index)),
        training_rows=len(x), training_profile_counts={str(p):int((profiles==p).sum()) for p in ids})
    dump(output/'model.json', metadata)
    xt = torch.as_tensor(x, device='cuda'); yt = torch.as_tensor(y, device='cuda'); wt = torch.as_tensor(w, device='cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=.0001)
    trace, checkpoints = [], []
    for epoch in range(1, epochs+1):
        clock.check(); started = time.monotonic(); model.train()
        permutation = np.random.Generator(np.random.PCG64(args.seed + epoch - 1)).permutation(len(x))
        loss_sum, max_grad = 0., 0.
        for start in range(0, len(x), 4096):
            clock.check()
            batch = torch.as_tensor(permutation[start:start+4096], dtype=torch.int64, device='cuda')
            optimizer.zero_grad(set_to_none=True)
            residual = model(xt[batch]).reshape(-1)-yt[batch]
            loss = (wt[batch]*residual.square()).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach().cpu())*len(batch)
            max_grad = max(max_grad, float(grad.detach().cpu()))
        torch.cuda.synchronize()
        trace.append(dict(epoch=epoch, online_weighted_loss=loss_sum/len(x), max_preclip_gradient_norm=max_grad,
                          seconds=time.monotonic()-started, permutation_sha256=array_digest({'order':permutation})))
        if epoch in checkpoint_epochs:
            name = 'epoch_%03d_weights.npz' % epoch
            save_arrays(output/name, **weights(model))
            result = dict(epoch=epoch, weights=name, weights_sha256=sha(output/name))
            if held is not None:
                ix = core.indices(records[held])
                prediction = predict(core,model,records[held],transform,prior,ix,clock)
                save_arrays(output/('epoch_%03d_held.npz'%epoch), index=ix, prediction=prediction)
                result['held_rmse'] = float(np.sqrt(np.mean((prediction-records[held]['y'][ix])**2)))
            checkpoints.append(result)
    save_arrays(output/'final_weights.npz', **weights(model))
    dump(output/'epoch_trace.json', trace)
    dump(output/'checkpoints.json', checkpoints)
    return dict(kind=args.kind, seed=args.seed, dataset=args.dataset, held_out=held, epochs=epochs,
                fit_ids=ids, checkpoints=checkpoints, initial_array_sha256=metadata['initial_array_sha256'],
                final_array_sha256=array_digest(weights(model)), training_rows=len(x),
                feature_array_sha256=metadata['feature_array_sha256'],weight_array_sha256=metadata['weight_array_sha256'],
                training_seconds=sum(row['seconds'] for row in trace))


def load_stages(args, expected_mode, clock):
    lock_path = checked_path(args.stage_lock, args.stage_lock_sha256)
    lock = read(lock_path)
    if lock.get('schema') != 'c15-paper-mlp-stage-lock-v001' or lock.get('mode') != expected_mode:
        raise ValueError('Expected explicit MLP stage dependency lock')
    results = []
    for item in lock['stages']:
        clock.check(); path = checked_path(item['path'], item['sha256']); stage = read(path)
        if stage.get('schema') != 'c15-paper-mlp-stage-record-v001' or stage.get('status') != 'COMPLETE':
            raise ValueError('Incomplete or foreign stage receipt')
        if stage.get('source_sha256') != sha(__file__) or stage.get('core_sha256') != CORE_SHA or stage['configuration']['mode'] != expected_mode:
            raise ValueError('MLP source/mode differs from dependency')
        if not {'summary.json','model.json','epoch_trace.json','checkpoints.json','initial_weights.npz','final_weights.npz','training_index.npz'} <= set(stage['files']):
            raise ValueError('Stage receipt does not cover required training artifacts')
        for name, digest in stage['files'].items():
            candidate = local_path(name, path.parent)
            if path.parent not in candidate.parents:
                raise ValueError('Stage artifact escapes immutable directory')
            checked_path(candidate, digest)
        if (path.parent/'failure.json').exists():
            raise ValueError('Failed stage is not a valid dependency')
        if read(path.parent/'summary.json') != stage['result']:
            raise ValueError('Stage result and hashed summary differ')
        results.append((path.parent, stage))
    return results


def select_epochs(args, clock):
    stages = load_stages(args, 'fold', clock)
    rows = {}
    for folder, stage in stages:
        result = stage['result']; key = (result['kind'], result['held_out'])
        if key in rows or key[0] not in KINDS or key[1] not in FIT or result['seed'] != SEEDS[0] or result['dataset'] != 'ipmsm':
            raise ValueError('Duplicate/nonselection fold')
        if result['fit_ids'] != [p for p in FIT if p != key[1]] or result['epochs'] != 200:
            raise ValueError('Outer fold has wrong FIT cohort or training length')
        if stage['configuration']['manifest_sha256'] != args.manifest_sha256:
            raise ValueError('Fold manifest differs')
        if [x['epoch'] for x in result['checkpoints']] != EPOCHS:
            raise ValueError('Incomplete epoch checkpoint grid')
        rows[key] = result
    if set(rows) != {(kind, held) for kind in KINDS for held in FIT}:
        raise ValueError('Complete matched 16-fold selection required')
    for held in FIT:
        for key in ('initial_array_sha256','feature_array_sha256','weight_array_sha256'):
            if rows[('pure',held)][key] != rows[('rc_residual',held)][key]:
                raise ValueError('Paired initialization/features/weights differ: '+key)
    table, chosen = {}, {}
    for kind in KINDS:
        table[kind] = [dict(epoch=epoch, profile_rmse={str(p):rows[(kind,p)]['checkpoints'][j]['held_rmse'] for p in FIT},
                           macro_rmse=float(np.mean([rows[(kind,p)]['checkpoints'][j]['held_rmse'] for p in FIT])))
                       for j,epoch in enumerate(EPOCHS)]
        chosen[kind] = min(table[kind], key=lambda x:(x['macro_rmse'],x['epoch']))['epoch']
    return dict(schema='c15-paper-mlp-epoch-selection-v001', status='FROZEN_OLD_FIT_LOPO_EPOCH_SELECTION',
                source_sha256=sha(__file__), core_sha256=CORE_SHA, manifest_sha256=args.manifest_sha256,
                stage_lock_sha256=args.stage_lock_sha256, selection_dataset='ipmsm', fit_ids=FIT,
                seed=SEEDS[0], table=table, selected_epochs=chosen,
                limitation='Selection folds are internal validation; external epoch counts are transferred unchanged.')


def selection_file(args):
    path = checked_path(args.selection_file, args.selection_sha256)
    value = read(path)
    if (value.get('schema') != 'c15-paper-mlp-epoch-selection-v001' or value.get('status') != 'FROZEN_OLD_FIT_LOPO_EPOCH_SELECTION'
            or value.get('source_sha256') != sha(__file__) or value.get('core_sha256') != CORE_SHA
            or value.get('fit_ids') != FIT or value.get('seed') != SEEDS[0]
            or set(value['selected_epochs']) != set(KINDS) or any(x not in EPOCHS for x in value['selected_epochs'].values())):
        raise ValueError('Invalid frozen epoch selection')
    if args.dataset == 'ipmsm' and value['manifest_sha256'] != args.manifest_sha256:
        raise ValueError('Original final FIT manifest differs from epoch selection')
    for kind in KINDS:
        rows = value['table'][kind]
        if [row['epoch'] for row in rows] != EPOCHS:
            raise ValueError('Epoch selection table is incomplete')
        for row in rows:
            if set(row['profile_rmse']) != {str(p) for p in FIT}:
                raise ValueError('Epoch selection lacks exact old FIT profiles')
            values = np.asarray([row['profile_rmse'][str(p)] for p in FIT])
            if not np.isfinite(values).all() or (values<0).any() or not np.isclose(values.mean(),row['macro_rmse'],atol=1e-12,rtol=1e-12):
                raise ValueError('Epoch selection macro score is inconsistent')
        if min(rows,key=lambda row:(row['macro_rmse'],row['epoch']))['epoch'] != value['selected_epochs'][kind]:
            raise ValueError('Epoch selection violates frozen choice/tie rule')
    return value


def metrics(prediction, truth):
    difference = prediction-truth
    return dict(n=len(truth), sse=float(difference@difference), sae=float(np.abs(difference).sum()),
                rmse=float(np.sqrt(np.mean(difference**2))), mae=float(np.abs(difference).mean()),
                max_abs_error=float(np.abs(difference).max()), max_underprediction=float(np.maximum(-difference,0).max()))


def aggregate(rows):
    n = sum(x['n'] for x in rows)
    return dict(profile_count=len(rows), n=n, macro_rmse=float(np.mean([x['rmse'] for x in rows])),
                macro_mae=float(np.mean([x['mae'] for x in rows])), pooled_rmse=float(np.sqrt(sum(x['sse'] for x in rows)/n)),
                pooled_mae=sum(x['sae'] for x in rows)/n, max_abs_error=max(x['max_abs_error'] for x in rows),
                max_underprediction=max(x['max_underprediction'] for x in rows))


def evaluate(args, data, core, output, clock):
    stages = load_stages(args, 'final', clock)
    selection = selection_file(args)
    by_key = {}
    for folder, stage in stages:
        result = stage['result']; key = (result['kind'], result['seed'])
        if key in by_key or key[0] not in KINDS or key[1] not in SEEDS or result['dataset'] != args.dataset or result['fit_ids'] != data['roles']['fit']:
            raise ValueError('Final model stage mismatch/duplicate')
        conf = stage['configuration']
        if conf['manifest_sha256'] != args.manifest_sha256 or conf['selection_sha256'] != args.selection_sha256 or result['epochs'] != selection['selected_epochs'][key[0]]:
            raise ValueError('Final model has different FIT/epoch selection')
        by_key[key] = (folder,stage)
    if set(by_key) != {(kind, seed) for kind in KINDS for seed in SEEDS}:
        raise ValueError('All six final paired models required before evaluation')
    for seed in SEEDS:
        for key in ('initial_array_sha256','feature_array_sha256','weight_array_sha256'):
            if by_key[('pure',seed)][1]['result'][key] != by_key[('rc_residual',seed)][1]['result'][key]:
                raise ValueError('Final paired initialization/features/weights differ: '+key)
    evaluation_path = checked_path(args.evaluation_input, args.evaluation_input_sha256)
    evaluation = read(evaluation_path)
    if evaluation.get('schema') != 'c15-paper-mlp-evaluation-input-v001' or evaluation.get('dataset') != args.dataset:
        raise ValueError('Wrong operating/truth evaluation manifest')
    role = evaluation['evaluation_role']
    if role not in data['roles'] or role == 'fit':
        raise ValueError('Explicit frozen nonFIT role required')
    records = evaluation['records']
    if len(records) != len(data['roles'][role]) or set(x['id'] for x in records) != set(data['roles'][role]) or any(x['role'] != role for x in records):
        raise ValueError('Complete role with no FIT contamination required')
    if role == 'confirmation':
        admission_path = checked_path(args.confirmation_admission, args.confirmation_admission_sha256)
        admission = read(admission_path)
        expected = dict(schema='c15-paper-mlp-confirmation-admission-v001', dataset=args.dataset,
            manifest_sha256=args.manifest_sha256, stage_lock_sha256=args.stage_lock_sha256,
            evaluation_input_manifest_sha256=args.evaluation_input_sha256, worker_sha256=sha(__file__),
            protocol_sha256=data['source_files']['protocol']['sha256'], confirmation_attempt_consumed=True,
            numeric_operating_and_target_access_admitted=True)
        if any(admission.get(k) != v for k,v in expected.items()) or not admission.get('confirmation_attempt_id'):
            raise ValueError('Separate frozen MLP confirmation admission is required')
    model_lock = dict(schema='c15-paper-mlp-model-lock-v001', status='ALL_SIX_MODELS_LOCKED_BEFORE_EVALUATION',
        dataset=args.dataset, role=role, source_sha256=sha(__file__), core_sha256=CORE_SHA,
        manifest_sha256=args.manifest_sha256, stage_lock_sha256=args.stage_lock_sha256,
        selection_sha256=args.selection_sha256, evaluation_input_sha256=args.evaluation_input_sha256,
        entries=[dict(kind=k[0],seed=k[1],stage_record=str(v[0]/'stage_record.json'),stage_record_sha256=sha(v[0]/'stage_record.json'),
                      weights_sha256=sha(v[0]/'final_weights.npz')) for k,v in sorted(by_key.items())])
    dump(output/'model_lock.json', model_lock)
    models = {}
    for key,(folder,stage) in by_key.items():
        model = new_model(key[1]); load_weights(model,folder/'final_weights.npz')
        models[key] = (model,read(folder/'model.json'))
    prediction_records = []
    for entry in records:
        clock.check()
        input_path = checked_path(entry['input_path'],entry['input_sha256'])
        with np.load(input_path,allow_pickle=False) as saved:
            if {'target','y','rotor','pm'} & set(saved.files):
                raise ValueError('Operating-only cache contains forbidden target key')
            record = {k:saved[k].astype(np.float64) for k in ('boundary','heat','instant')}
            ix = saved['index'].astype(np.int64)
            source_row = saved['source_row'].astype(np.int64)
        n = len(record['instant'])
        if not np.array_equal(ix,np.arange(300,n)) or source_row.shape != (n,) or any(record[k].shape != (n,d) for k,d in [('boundary',10),('heat',15),('instant',8)]):
            raise ValueError('Evaluation feature/index geometry mismatch')
        if any(not np.isfinite(v).all() for v in record.values()):
            raise ValueError('Nonfinite evaluation features')
        predictions = dict(index=ix,source_row=source_row[ix])
        for key,(model,metadata) in models.items():
            predictions['prediction__%s__%d'%key] = predict(core,model,record,metadata['transform'],metadata['prior'],ix,clock)
        name = 'profile_%s_predictions.npz'%entry['id']
        save_arrays(output/name,**predictions)
        prediction_records.append(dict(id=entry['id'],role=role,path=name,sha256=sha(output/name),input_path=str(input_path),
                                       input_sha256=entry['input_sha256'],evaluated_rows=len(ix)))
    dump(output/'prediction_lock.json',dict(schema='c15-paper-mlp-prediction-lock-v001',
         status='ALL_ROLE_ALL_SEED_PREDICTIONS_LOCKED_BEFORE_TRUTH',model_lock_sha256=sha(output/'model_lock.json'),records=prediction_records))
    # No truth array is loaded above this completed lock boundary.
    per_profile, grouped = [], {}
    for entry in records:
        clock.check()
        truth_path = checked_path(entry['truth_path'],entry['truth_sha256'])
        with np.load(truth_path,allow_pickle=False) as saved:
            truth,index = saved['target'].astype(np.float64),saved['index'].astype(np.int64)
        with np.load(output/('profile_%s_predictions.npz'%entry['id']),allow_pickle=False) as saved:
            if truth.shape != index.shape or not np.array_equal(index,saved['index']) or not np.isfinite(truth).all():
                raise ValueError('Locked prediction and separate truth do not align')
            for kind in KINDS:
                for seed in SEEDS:
                    key = '%s__%d'%(kind,seed)
                    values = metrics(saved['prediction__'+key],truth)
                    per_profile.append(dict(id=entry['id'],role=role,kind=kind,seed=seed,**values))
                    grouped.setdefault(key,[]).append(values)
        save_arrays(output/('profile_%s_truth.npz'%entry['id']),index=index,target=truth)
    by_seed = {key:aggregate(value) for key,value in grouped.items()}
    means = {kind:{metric:float(np.mean([by_seed['%s__%d'%(kind,seed)][metric] for seed in SEEDS]))
                  for metric in ('macro_rmse','macro_mae','pooled_rmse','pooled_mae','max_abs_error','max_underprediction')} for kind in KINDS}
    summary = dict(dataset=args.dataset,role=role,per_profile=per_profile,by_seed=by_seed,mean_of_seed_metrics=means,
                   statistical_role='Secondary descriptive matched MLP comparison; no primary confirmation claim.',
                   prediction_lock_sha256=sha(output/'prediction_lock.json'))
    dump(output/'metrics.json',summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['plan','costprobe','fold','select','final','evaluate'])
    parser.add_argument('--dataset',required=True,choices=['ipmsm','paderborn'])
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--max-seconds',type=float,default=600.)
    parser.add_argument('--kind',choices=KINDS)
    parser.add_argument('--held-out',type=int)
    parser.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0])
    for stem in ('stage-lock','selection-file','prior-receipt','evaluation-input','confirmation-admission'):
        parser.add_argument('--'+stem)
    for stem in ('stage-lock','selection','prior-receipt','evaluation-input','confirmation-admission'):
        parser.add_argument('--'+stem+'-sha256')
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 3600:
        raise ValueError('Stage wall cap must be in (0,3600] seconds')
    if args.mode in ('costprobe','fold','final') and args.kind is None:
        raise ValueError('Explicit pure or rc_residual training kind required')
    if args.mode in ('costprobe','fold','select') and args.dataset != 'ipmsm':
        raise ValueError('Cost/epoch selection uses original FIT only')
    if args.mode == 'fold' and (args.held_out not in FIT or args.seed != SEEDS[0]):
        raise ValueError('Frozen outer fold or selection seed mismatch')
    output = local_path(args.output_dir)
    output.mkdir(parents=True,exist_ok=False)
    clock = Clock(args.max_seconds,output)
    configuration = dict(vars(args),source=str(Path(__file__).resolve()),source_sha256=sha(__file__),core_sha256=CORE_SHA,
                         output_dir=str(output),pid=os.getpid(),process_group=os.getpgrp() if hasattr(os,'getpgrp') else None,
                         argv=list(sys.argv),settings=SETTINGS,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    dump(output/'configuration.json',configuration)
    try:
        data = manifest(args,clock)
        if args.mode == 'plan':
            result = dict(mode='metadata_only_plan',dataset=args.dataset,fit_ids=data['roles']['fit'],settings=SETTINGS,
                folds=[dict(kind=kind,held_out=p,seed=SEEDS[0],epochs=200) for kind in KINDS for p in FIT] if args.dataset=='ipmsm' else [],
                final=[dict(kind=kind,seed=seed,epochs='frozen old FIT LOPO selection') for kind in KINDS for seed in SEEDS],
                numerical_arrays_decoded=False,training_executed=False)
        elif args.mode == 'select':
            result = select_epochs(args,clock)
            dump(output/'selection.json',result)
        else:
            runtime = cuda_runtime();dump(output/'runtime.json',runtime);core = core_module()
            if args.mode == 'costprobe':
                dump(output/'fixtures.json',arithmetic_fixtures())
            if args.mode == 'evaluate':
                result = evaluate(args,data,core,output,clock)
            else:
                ids = FIT[:2] if args.mode=='costprobe' else data['roles']['fit']
                records = fit_records(data,ids,clock)
                held = args.held_out if args.mode=='fold' else None
                train_ids = [p for p in ids if p!=held]
                epochs = 2 if args.mode=='costprobe' else 200 if args.mode=='fold' else selection_file(args)['selected_epochs'][args.kind]
                result = train(args,core,records,train_ids,epochs,EPOCHS if args.mode=='fold' else [],output,clock,held)
                if args.mode == 'costprobe':
                    result['status'] = 'TWO_EPOCH_COST_PROBE_NOT_A_COMPLETED_COMPARATOR'
        clock.check()
        dump(output/'summary.json',result)
        files = {str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
        dump(output/'stage_record.json',dict(schema='c15-paper-mlp-stage-record-v001',status='COMPLETE',
            source_sha256=sha(__file__),core_sha256=CORE_SHA,configuration=configuration,result=result,files=files,
            elapsed_seconds=time.monotonic()-clock.started))
        print(json.dumps(dict(status='COMPLETE',mode=args.mode,output_dir=str(output),elapsed_seconds=time.monotonic()-clock.started)))
    except Exception as error:
        dump(output/'failure.json',dict(status='FAILED',error=type(error).__name__+': '+str(error),elapsed_seconds=time.monotonic()-clock.started))
        raise


if __name__ == '__main__':
    main()
