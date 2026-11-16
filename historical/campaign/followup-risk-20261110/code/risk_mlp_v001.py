"""C16 fixed risk losses; one immutable CUDA training/prediction stage per call.

All old evaluation profiles are previously exposed exploratory data. This worker
never loads their target files, calibrates a policy, or computes evaluation loss.
Shared, pinned C15 transforms are scientific dependencies, not an independent audit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP = ROOT / 'followup-risk-20261110'
PROTOCOL_SHA = '10d816a74e763f64f4d47a0a485133d88f5189313e2d9138fd7f52f0cf80edac'
MLP_SHA = '4ecd405a408a5ec34a20103390f99893ac3264ab28600de0bbde491b58919dbb'
CORE_SHA = '52d897657163391de6e1e42a51dad04550d5ca6f1dfc2a60f2e0041ec3f68a86'
C12_SHA = 'd4aa942ca9db8d8e9319197632cf376730456f3ea5cdc48b9d1c1fd51fc59412'
SEEDS = [2026111701, 2026111702, 2026111703]
LOSSES = ['mse', 'asym4', 'q90']
TRAINING = dict(architecture=[33,64,32,1], epochs=200, seeds=SEEDS,
    losses=LOSSES, batch_size=4096, learning_rate=.001, weight_decay=.0001,
    warmup=300, stride=5, alpha=4., quantile=.9)
SETTINGS = dict(**TRAINING, parameters=4289, activation='ReLU', optimizer='AdamW',
    betas=[.9,.999], eps=1e-8, gradient_norm_cap=1., feature_std_floor=1e-8,
    target_iqr_floor=1., inference_batch_size=4096,
    permutation='NumPy PCG64(seed + epoch - 1), full permutation, explicit int64',
    precision='FP64 transforms, FP32 network and row weights, FP64 physical outputs',
    profile_weight='N/(P*n_profile); weighted minibatch mean including final partial batch')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def path(value):
    value = str(value)
    if os.name != 'nt' and re.match(r'^[A-Za-z]:[\\/]', value):
        value = '/mnt/' + value[0].lower() + '/' + value[3:].replace('\\', '/')
    elif os.name == 'nt' and re.match(r'^/mnt/[a-z]/', value):
        value = value[5].upper() + ':/' + value[7:]
    result = Path(value)
    result = (result if result.is_absolute() else ROOT/result).resolve()
    if ROOT not in result.parents and result != ROOT:
        raise ValueError('Path outside original campaign')
    return result


def rel(value):
    return path(value).relative_to(ROOT).as_posix()


def read(value):
    return Path(value).read_text(encoding='utf-8-sig')


def doc(value):
    return json.loads(read(value))


def dump(value, item):
    with Path(value).open('x', encoding='utf-8') as stream:
        json.dump(item, stream, ensure_ascii=False, indent=2, allow_nan=False)


def checked(entry):
    result = path(entry['path'])
    if sha(result) != entry['sha256']:
        raise ValueError('Pinned artifact changed: ' + str(result))
    return result


def ref(value):
    return dict(path=rel(value), sha256=sha(value))


class Clock:
    def __init__(self, seconds, output):
        self.started, self.seconds, self.output = time.monotonic(), seconds, output
        self.training_cutoff = datetime.fromisoformat('2026-11-10T20:23:59+00:00')

    def check(self, training=False):
        if time.monotonic()-self.started > self.seconds:
            raise TimeoutError('Registered stage wall cap exceeded')
        now = datetime.now(timezone.utc)
        if now >= datetime.fromisoformat('2026-11-11T00:23:59+00:00'):
            raise TimeoutError('Campaign deadline reached')
        if training and now >= self.training_cutoff:
            raise TimeoutError('New-training cutoff reached')
        if sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file()) > 2_000_000_000:
            raise RuntimeError('Stage output cap exceeded')


def dependency_modules(cfg):
    expected = {'c15_paper_mlp_v001.py': MLP_SHA, 'c15_neural_rc_core_v001.py': CORE_SHA}
    if set(cfg['dependency_sources']) != set(expected):
        raise ValueError('Shared scientific dependency set changed')
    for name, digest in expected.items():
        entry = cfg['dependency_sources'][name]
        if entry != dict(path='code/'+name, sha256=digest):
            raise ValueError('Scientific dependency identity changed')
        checked(entry)
    checked(dict(path='code/combination_c12_fit_rc_v001.py', sha256=C12_SHA))
    spec = importlib.util.spec_from_file_location('c16_pinned_mlp', ROOT/'code/c15_paper_mlp_v001.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker, worker.core_module()


def configuration(args, clock):
    cfg = doc(checked(dict(path=args.config, sha256=args.config_sha256)))
    if (cfg.get('schema') != 'c16-risk-dataset-v001' or
        cfg.get('scientific_execution_admitted') is not True or
        cfg.get('all_evaluation_previously_exposed') is not True):
        raise ValueError('Frozen exploratory admission required')
    if cfg['training'] != TRAINING:
        raise ValueError('Fixed training panel changed')
    if cfg['protocol'] != dict(path='followup-risk-20261110/protocols/risk-screen-v001.md', sha256=PROTOCOL_SHA):
        raise ValueError('Frozen protocol changed')
    checked(cfg['protocol'])
    expected_limits = dict(deadline_utc='2026-11-11T00:23:59Z',
        new_training_cutoff_utc='2026-11-10T20:23:59Z', max_stage_seconds=3600,
        max_stage_output_bytes=2_000_000_000)
    if cfg['limits'] != expected_limits:
        raise ValueError('Resource/deadline contract changed')
    worker, core = dependency_modules(cfg)
    base = doc(checked(cfg['original_manifest']))
    if (base.get('schema') != 'c15-paper-dataset-v001' or base.get('scientific_execution_admitted') is not True or
        base['dataset'] != cfg['dataset'] or base['roles'] != worker.ROLES[cfg['dataset']] or
        cfg['train_ids'] != base['roles']['fit'] or cfg['fit_cache'] != base['fit_cache']):
        raise ValueError('FIT roles/source lineage changed')
    if [e['id'] for e in cfg['fit_cache']] != cfg['train_ids']:
        raise ValueError('Exact ordered FIT records required')
    expected_roles = [role for role in base['roles'] if role != 'fit']
    if [r['role'] for r in cfg['role_inputs']] != expected_roles:
        raise ValueError('Complete ordered evaluation roles required')
    for role in cfg['role_inputs']:
        if role['ids'] != base['roles'][role['role']] or [r['id'] for r in role['records']] != role['ids']:
            raise ValueError('Evaluation profile identities changed')
        lock = doc(checked(role['prediction_lock']))
        audit = doc(checked(role['prior_evaluation_audit']))
        if audit.get('status') != 'PASS' or lock['role'] != role['role']:
            raise ValueError('Prior independent source audit/PRED role mismatch')
        old = {r['id']: r for r in lock['records']}
        lockdir = path(role['prediction_lock']['path']).parent
        for entry in role['records']:
            clock.check()
            previous = old[entry['id']]
            if (path(entry['features']['path']) != (lockdir/previous['path']).resolve() or
                entry['features']['sha256'] != previous['sha256'] or entry['rows'] != previous['rows']):
                raise ValueError('Prior prediction feature identity mismatch')
    # No target or feature numeric arrays are decoded by this metadata stage.
    return cfg, base, worker, core


def loss_values(torch, error, loss):
    if loss == 'mse':
        return error.square()
    if loss == 'asym4':
        return .4*(4.*torch.clamp(error, min=0.).square()+torch.clamp(-error, min=0.).square())
    if loss == 'q90':
        return .9*torch.clamp(error, min=0.)+.1*torch.clamp(-error, min=0.)
    raise ValueError('Unknown loss')


def fixtures(worker):
    torch = worker.torch
    result = worker.arithmetic_fixtures()
    rows = []
    for kind, values, gradients in [
        ('mse', [4.,1.,0.,1.,4.], [-4.,-2.,0.,2.,4.]),
        ('asym4', [1.6,.4,0.,1.6,6.4], [-1.6,-.8,0.,3.2,6.4]),
        ('q90', [.2,.1,0.,.9,1.8], [-.1,-.1,.8,.9,.9])]:
        # clamp at exactly zero chooses its PyTorch subgradient; q90's .8 is
        # valid in [-.1,.9], and is recorded rather than a uniqueness claim.
        e = torch.tensor([-2.,-1.,0.,1.,2.], dtype=torch.float64, device='cuda', requires_grad=True)
        value = loss_values(torch, e, kind)
        grad = torch.autograd.grad(value.sum(), e)[0]
        if not torch.allclose(value, torch.tensor(values, dtype=e.dtype, device=e.device), atol=1e-14, rtol=1e-14):
            raise AssertionError('Loss arithmetic fixture failed: '+kind)
        if not torch.allclose(grad, torch.tensor(gradients, dtype=e.dtype, device=e.device), atol=1e-14, rtol=1e-14):
            raise AssertionError('Loss gradient fixture failed: '+kind)
        rows.append(dict(loss=kind, value=value.detach().cpu().tolist(), gradient=grad.cpu().tolist()))
    return dict(status='PASS', shared_mlp_invariants=result, risk_losses=rows)


def training(args, cfg, base, worker, core, output, clock):
    clock.check(training=True)
    ids = cfg['train_ids'][:2] if args.mode == 'cost-probe' else cfg['train_ids']
    for entry in cfg['fit_cache']:
        if entry['id'] in ids:
            checked(entry)
    records = worker.fit_records(base, ids, clock)
    transform = worker.transforms(core, records, ids)
    x, y, w, profiles, index = worker.training_arrays(core, records, ids, transform, None)
    profiles, index = profiles.astype(np.int64), index.astype(np.int64)
    train_arrays = dict(x=x, y_normalized=y, w=w, profile_ids=profiles, index=index)
    worker.save_arrays(output/'training_arrays.npz', **train_arrays)
    high_temperature = float(np.quantile(np.concatenate([records[p]['y'][core.indices(records[p])] for p in ids]), .95))
    model = worker.new_model(args.seed)
    initial = worker.weights(model)
    worker.save_arrays(output/'weights_initial.npz', **initial)
    torch = worker.torch
    xt = torch.as_tensor(x, device='cuda')
    yt = torch.as_tensor(y, device='cuda')
    wt = torch.as_tensor(w, device='cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=.0001)
    epochs = 2 if args.mode == 'cost-probe' else 200
    trace = []
    for epoch in range(1, epochs+1):
        clock.check(training=True)
        started = time.monotonic()
        model.train()
        order = np.random.Generator(np.random.PCG64(args.seed+epoch-1)).permutation(len(x)).astype(np.int64)
        loss_sum, max_grad = 0., 0.
        for start in range(0, len(x), 4096):
            clock.check(training=True)
            batch = torch.as_tensor(order[start:start+4096], dtype=torch.int64, device='cuda')
            optimizer.zero_grad(set_to_none=True)
            predicted = model(xt[batch]).reshape(-1)
            # Matched MSE follows the original subtraction order exactly.
            pointloss = (predicted-yt[batch]).square() if args.loss == 'mse' else loss_values(torch, yt[batch]-predicted, args.loss)
            objective = (wt[batch]*pointloss).mean()
            if not torch.isfinite(objective):
                raise FloatingPointError('Nonfinite training loss')
            objective.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(objective.detach().cpu())*len(batch)
            max_grad = max(max_grad, float(grad.detach().cpu()))
        torch.cuda.synchronize()
        trace.append(dict(epoch=epoch, online_weighted_loss=loss_sum/len(x),
            max_preclip_gradient_norm=max_grad, seconds=time.monotonic()-started,
            permutation_sha256=worker.array_digest(dict(order=order))))
    clock.check(training=True)
    final = worker.weights(model)
    if any(not np.isfinite(v).all() for v in final.values()):
        raise FloatingPointError('Nonfinite fitted parameters')
    worker.save_arrays(output/'weights_final.npz', **final)
    dump(output/'epoch_trace.json', trace)
    metadata = dict(schema='c16-risk-mlp-model-v001', dataset=cfg['dataset'], loss=args.loss,
        seed=args.seed, fit_ids=ids, epochs=epochs, transform=transform, settings=SETTINGS,
        config_sha256=args.config_sha256, source_sha256=sha(__file__), core_sha256=CORE_SHA,
        shared_worker_sha256=MLP_SHA, high_temperature_threshold_c=high_temperature,
        training_rows=len(x), training_profile_counts={str(p):int(np.sum(profiles == p)) for p in ids},
        training_arrays_sha256=worker.array_digest(train_arrays),
        feature_array_sha256=worker.array_digest(dict(x=x, profile_ids=profiles, index=index)),
        initial_array_sha256=worker.array_digest(initial), final_array_sha256=worker.array_digest(final),
        training_seconds=sum(row['seconds'] for row in trace), evaluation_truth_accessed=False,
        completed_comparator=args.mode == 'train')
    dump(output/'model.json', metadata)
    model_files = {name:sha(output/name) for name in
        ['model.json','weights_initial.npz','weights_final.npz','training_arrays.npz','epoch_trace.json','configuration.json','runtime.json']}
    lock = dict(schema='c16-risk-mlp-model-lock-v001', status='MODEL_LOCKED',
        model=ref(output/'model.json'), files=model_files, source_sha256=sha(__file__),
        config_sha256=args.config_sha256, completed_comparator=args.mode == 'train',
        locked_at_utc=datetime.now(timezone.utc).isoformat())
    dump(output/'model_lock.json', lock)
    return model, metadata, output/'model_lock.json'


def load_model(args, cfg, worker, clock):
    if not args.model_lock or not args.model_lock_sha256:
        raise ValueError('Predict needs exact model-lock path and SHA')
    lockpath = checked(dict(path=args.model_lock, sha256=args.model_lock_sha256))
    lock = doc(lockpath)
    if (lock.get('schema') != 'c16-risk-mlp-model-lock-v001' or lock.get('status') != 'MODEL_LOCKED' or
        lock.get('completed_comparator') is not True or lock['source_sha256'] != sha(__file__) or
        lock['config_sha256'] != args.config_sha256):
        raise ValueError('Qualified same-source full training lock required')
    required = {'model.json','weights_initial.npz','weights_final.npz','training_arrays.npz','epoch_trace.json','configuration.json','runtime.json'}
    if set(lock['files']) != required:
        raise ValueError('Model file coverage changed')
    for name, digest in lock['files'].items():
        clock.check()
        checked(dict(path=rel(lockpath.parent/name), sha256=digest))
    if checked(lock['model']) != lockpath.parent/'model.json':
        raise ValueError('Model location mismatch')
    metadata = doc(lockpath.parent/'model.json')
    if (metadata['fit_ids'] != cfg['train_ids'] or metadata['dataset'] != cfg['dataset'] or
        metadata['loss'] != args.loss or metadata['seed'] != args.seed or metadata['epochs'] != 200 or
        metadata['settings'] != SETTINGS):
        raise ValueError('Model scientific identity changed')
    model = worker.new_model(args.seed)
    worker.load_weights(model, lockpath.parent/'weights_final.npz')
    return model, metadata, lockpath


def predictions(args, cfg, worker, core, model, metadata, model_lock, output, clock):
    started = time.monotonic()
    rows = []
    for role in cfg['role_inputs']:
        destination = output/'predictions'/role['role']
        destination.mkdir(parents=True, exist_ok=False)
        for entry in role['records']:
            clock.check()
            n = entry['rows']
            ix = np.arange(300, n, dtype=np.int64)
            with np.load(checked(entry['features']), allow_pickle=False) as saved:
                if not np.array_equal(saved['index'], ix):
                    raise ValueError('Complete causal endpoint coverage mismatch')
                record = {k:saved[k].astype(np.float64) for k in ['boundary','heat','instant']}
                source_row = saved['source_row']
                if (source_row.shape != (n,) or not np.issubdtype(source_row.dtype, np.integer) or
                    not np.all(np.diff(source_row) > 0)):
                    raise ValueError('Source-row geometry/type/order changed')
                source_row = source_row[ix].astype(np.int64)
            for name, width in [('boundary',10),('heat',15),('instant',8)]:
                if record[name].shape != (n,width) or not np.isfinite(record[name]).all():
                    raise ValueError('Full source feature shape/nonfinite mismatch')
            xx = worker.transformed(core, record, metadata['transform'], ix)
            pp = worker.predict(core, model, record, metadata['transform'], None, ix, clock)
            saved_arrays = dict(index=ix, source_row=source_row, features=xx, prediction=pp)
            filename = destination/f'profile_{entry["id"]}.npz'
            worker.save_arrays(filename, **saved_arrays)
            rows.append(dict(role=role['role'], usage=role['usage'], profile=entry['id'],
                path=rel(filename), sha256=sha(filename), prediction_key='prediction', index_key='index',
                rows=n, evaluated_rows=len(ix), source_features=entry['features'],
                arrays_sha256=worker.array_digest(saved_arrays),
                feature_array_sha256=worker.array_digest(dict(features=xx,index=ix,source_row=source_row)),
                prediction_array_sha256=worker.array_digest(dict(prediction=pp,index=ix,source_row=source_row))))
    lock = dict(schema='c16-risk-mlp-prediction-lock-v001', status='ALL_PREDICTIONS_LOCKED',
        dataset=cfg['dataset'], loss=metadata['loss'], seed=metadata['seed'],
        source_sha256=sha(__file__), config_sha256=args.config_sha256, model_lock=ref(model_lock),
        predictions=rows, truth_accessed=False, all_evaluation_previously_exposed=True,
        locked_at_utc=datetime.now(timezone.utc).isoformat())
    dump(output/'prediction_lock.json', lock)
    return dict(predictions=rows, prediction_lock=ref(output/'prediction_lock.json'),
                prediction_seconds=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['plan','cost-probe','train','predict'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-seconds', type=float, default=600.)
    parser.add_argument('--loss', choices=LOSSES)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument('--model-lock')
    parser.add_argument('--model-lock-sha256')
    args = parser.parse_args()
    if not np.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 3600:
        raise ValueError('Stage cap must be finite in (0,3600]')
    if args.mode != 'plan' and args.loss is None:
        raise ValueError('Explicit loss required')
    output = path(args.output_dir)
    if FOLLOWUP/'runs' not in output.parents:
        raise ValueError('New stage must be below followup-risk-20261110/runs')
    output.mkdir(parents=True, exist_ok=False)
    clock = Clock(args.max_seconds, output)
    conf = dict(vars(args), source=str(Path(__file__).resolve()), source_sha256=sha(__file__),
        output_dir=str(output), pid=os.getpid(), process_group=os.getpgrp() if hasattr(os,'getpgrp') else None,
        argv=list(sys.argv), settings=SETTINGS, started_utc=datetime.now(timezone.utc).isoformat())
    dump(output/'configuration.json', conf)
    try:
        cfg, base, worker, core = configuration(args, clock)
        if args.mode == 'plan':
            result = dict(status='METADATA_PLAN_COMPLETE', dataset=cfg['dataset'],
                jobs=[dict(loss=loss,seed=seed,epochs=200) for seed in SEEDS for loss in LOSSES],
                train_ids=cfg['train_ids'], roles=[dict(role=r['role'],ids=r['ids']) for r in cfg['role_inputs']],
                numerical_arrays_decoded=False, training_executed=False, truth_accessed=False)
        else:
            clock.check(training=args.mode in ('train','cost-probe'))
            runtime = worker.cuda_runtime()
            runtime.update(tf32_matmul_allowed=worker.torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn_allowed=worker.torch.backends.cudnn.allow_tf32,
                deterministic_algorithms=worker.torch.are_deterministic_algorithms_enabled(),
                torch_num_threads=worker.torch.get_num_threads(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
            dump(output/'runtime.json', runtime)
            if args.mode == 'cost-probe':
                dump(output/'fixtures.json', fixtures(worker))
            if args.mode == 'predict':
                model, metadata, lock = load_model(args,cfg,worker,clock)
            else:
                model, metadata, lock = training(args,cfg,base,worker,core,output,clock)
            result = dict(status='TWO_EPOCH_COST_PROBE_COMPLETE' if args.mode == 'cost-probe' else 'MLP_PREDICTIONS_LOCKED',
                dataset=cfg['dataset'], loss=args.loss, seed=args.seed, epochs=metadata['epochs'],
                training_seconds=metadata['training_seconds'] if args.mode != 'predict' else 0.,
                training_rows=metadata['training_rows'], high_temperature_threshold_c=metadata['high_temperature_threshold_c'],
                transform=metadata['transform'], model_lock=ref(lock), truth_accessed=False,
                completed_comparator=args.mode != 'cost-probe')
            if args.mode != 'cost-probe':
                result.update(predictions(args,cfg,worker,core,model,metadata,lock,output,clock))
        clock.check()
        result.update(config_sha256=args.config_sha256,source_sha256=sha(__file__),
                      elapsed_seconds=time.monotonic()-clock.started)
        dump(output/'summary.json', result)
        clock.check()
        files = {p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
        dump(output/'stage_record.json', dict(schema='c16-risk-mlp-stage-v001', status='COMPLETE',
            configuration=conf, result=result, files=files, elapsed_seconds=time.monotonic()-clock.started))
        clock.check()
        print(json.dumps(dict(status='COMPLETE',mode=args.mode,output_dir=str(output),
                              elapsed_seconds=time.monotonic()-clock.started)))
    except Exception as error:
        dump(output/'failure.json', dict(status='FAILED',error=type(error).__name__+': '+str(error),
                                        elapsed_seconds=time.monotonic()-clock.started))
        raise


if __name__ == '__main__':
    main()
