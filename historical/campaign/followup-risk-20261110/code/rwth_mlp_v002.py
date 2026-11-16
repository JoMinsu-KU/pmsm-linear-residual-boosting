"""Fresh RWTH fixed MSE baseline; FIT training and role prediction are separate.

Consumes independently auditable raw X33 caches, never constructs old C15
features or uses its old role/scaler helpers. No target is loaded by prediction.
CONF feature access requires the root's consumed, model-bound admission.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.dont_write_bytecode=True
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FOLLOWUP=ROOT/'followup-risk-20261110'
PROTOCOL=dict(path='followup-risk-20261110/protocols/rwth-replication-v002-frozen.md',sha256='f1d201fe4aa6243f8c77c6a8a85a91268d183d0cd24d48bc35ec86b269082b0f')
PARTITION=dict(path='followup-risk-20261110/configs/rwth-partition-v001.json',sha256='5d5117312eee65bd9811f0ede7326d01d8a0a1aa3d0b0e3fb6ef60ae47959be1')
SHARED=dict(path='code/c15_paper_mlp_v001.py',sha256='4ecd405a408a5ec34a20103390f99893ac3264ab28600de0bbde491b58919dbb')
ARCHIVE_SHA='a6b219a17acbbc2a9899fa7d750e1b0531756eeefcf16efd56009c43a0c4fce6'
ROLES=dict(fit=[38,27,30,23,21,19,29,20,26,37,6,8,35,34,4,15,7,13,18,1],
    calibration=[10,11,22,9,16,31,28,17],confirmation=[5,32,33,36,24,2,25,14,3,12])
SEEDS=[2026111701,2026111702,2026111703]
FEATURE_NAMES=['i','omega','torque','coolant','i_squared','abs_omega','omega_squared','abs_mechanical_power']+[
    drive+'__span'+str(span) for drive in ['coolant','i_squared','abs_omega','omega_squared','abs_mechanical_power']
    for span in [16,64,256,1024,4096]]
SETTINGS=dict(architecture=[33,64,32,1],parameters=4289,activation='ReLU',loss='mse',epochs=200,
    seeds=SEEDS,optimizer='AdamW',learning_rate=.001,betas=[.9,.999],eps=1e-8,weight_decay=.0001,
    batch_size=4096,gradient_norm_cap=None,warmup=0,label_stride=5,feature_std_floor=1e-8,target_iqr_floor=1.,
    permutation='NumPy PCG64(seed+epoch-1); explicit int64',profile_weight='N/(P*n_profile)',
    precision='FP64 cache/scalers, FP32 network/weights, FP64 physical-C output',
    seed_aggregation='arithmetic mean of seed metrics; no seed selection or prediction ensemble')


def sha(value):
    digest=hashlib.sha256()
    with Path(value).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def local(value):
    value=str(value)
    if os.name!='nt' and re.match(r'^[A-Za-z]:[\\/]',value):value='/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
    elif os.name=='nt' and re.match(r'^/mnt/[a-z]/',value):value=value[5].upper()+':/'+value[7:]
    p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
    if p!=ROOT and ROOT not in p.parents:raise ValueError('Path outside campaign')
    return p


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def dump(p,value):
    with Path(p).open('x',encoding='utf-8') as handle:json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)


def ref(p):return dict(path=local(p).relative_to(ROOT).as_posix(),sha256=sha(p))


class Clock:
    def __init__(self,cap,output):self.start=time.monotonic();self.cap=cap;self.output=output;self.inputs={}
    def check(self,training=False):
        if time.monotonic()-self.start>self.cap:raise TimeoutError('Stage wall cap exceeded')
        now=datetime.now(timezone.utc)
        if now>=datetime.fromisoformat('2026-11-11T00:23:59+00:00'):raise TimeoutError('Campaign deadline')
        if training and now>=datetime.fromisoformat('2026-11-10T20:23:59+00:00'):raise TimeoutError('New training cutoff')
        if sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())>1000000000:raise RuntimeError('1GB stage output ceiling')
    def checked(self,entry):
        self.check();p=local(entry['path'])
        if sha(p)!=entry['sha256']:raise ValueError('Exact input SHA mismatch: '+str(p))
        self.inputs[str(p)]=entry['sha256'];return p


def shared_worker(clock):
    spec=importlib.util.spec_from_file_location('rwth_fixed_cuda_primitives',clock.checked(SHARED))
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    return worker


def cache_manifest(entry,clock):
    filename=clock.checked(entry);value=read(filename)
    if (value.get('schema')!='rwth-cache-manifest-v002' or value.get('dataset')!='rwth48v' or
        value['protocol']!=PROTOCOL or value['partition']!=PARTITION or value['roles']!=ROLES or
        value['archive']['sha256']!=ARCHIVE_SHA):raise ValueError('Exact fresh protocol/partition/cache identity')
    clock.checked(PROTOCOL);clock.checked(PARTITION)
    if not isinstance(value['dt_ref'],(float,int)) or not math.isfinite(value['dt_ref']) or value['dt_ref']<=0:raise ValueError('Positive FIT-only dt_ref')
    if value['feature_names']!=FEATURE_NAMES:raise ValueError('Exact33 protocol feature names/order')
    keys=[(x['role'],x['id']) for x in value['records']]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate cache record')
    for item in value['records']:
        if item['role'] not in ROLES or item['id'] not in ROLES[item['role']] or not isinstance(item['rows'],int) or item['rows']<1:raise ValueError('Frozen role/cache record geometry')
        if item['role']=='fit':
            if item['target_key']!='target' or item['truth'] is not None:raise ValueError('FIT target must be in FIT feature cache')
        elif item['target_key'] is not None:raise ValueError('No target key in non-FIT feature cache')
    return value,filename


def confirmation_admission(args,manifest,model_lock,clock):
    if not args.confirmation_admission or not args.confirmation_admission_sha256:raise ValueError('Root confirmation admission required before CONF features')
    entry=dict(path=args.confirmation_admission,sha256=args.confirmation_admission_sha256)
    admitted=clock.checked(entry)
    manifest_ref=manifest.get('confirmation_admission')
    if not isinstance(manifest_ref,dict) or clock.checked(manifest_ref)!=admitted or manifest_ref['sha256']!=entry['sha256']:
        raise ValueError('CONF cache manifest must bind the same root admission')
    value=read(admitted)
    expected=dict(schema='rwth-confirmation-admission-v002',status='ADMITTED_CONSUMED_NO_RESET',
        confirmation_id='C17_RWTH_FIXED_CONFIRMATION_02',candidate_id='C17',alpha=.025,
        allowed_roles=['CONF'],protocol=PROTOCOL,partition=PARTITION,conf_ids=ROLES['confirmation'])
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Exact consumed confirmation admission required')
    key=model_lock.relative_to(ROOT).as_posix()
    if value['bindings'].get(key)!=sha(model_lock):raise ValueError('Current MLP model lock not admitted')
    return ref(admitted)


def rows(entry,clock,fit=False):
    filename=clock.checked(entry['features'])
    with np.load(filename,allow_pickle=False) as saved:
        expected={'profile_id','index','source_row','time','operating','X33'}|({'target'} if fit else set())
        if set(saved.files)!=expected:raise ValueError('Exact target-separated cache fields required')
        data={key:saved[key].copy() for key in saved.files}
    n=entry['rows']
    if data['profile_id'].shape!=() or data['profile_id'].dtype!=np.int64 or int(data['profile_id'])!=entry['id']:raise ValueError('Exact cached profile ID')
    for key in ('index','source_row'):
        if data[key].dtype!=np.int64 or not np.array_equal(data[key],np.arange(n,dtype=np.int64)):raise ValueError('Unfiltered original row identifiers')
    for key,width in [('X33',33),('operating',4)]:
        if data[key].dtype!=np.float64 or data[key].shape!=(n,width) or not np.isfinite(data[key]).all():raise ValueError('Finite FP64 common raw features')
    if data['time'].dtype!=np.float64 or data['time'].shape!=(n,) or not np.isfinite(data['time']).all() or not np.all(np.diff(data['time'])>0):raise ValueError('Strictly increasing original timestamps')
    if fit and (data['target'].dtype!=np.float64 or data['target'].shape!=(n,) or not np.isfinite(data['target']).all()):raise ValueError('Finite FIT target')
    return data


def fit(args,manifest,manifestpath,worker,output,clock):
    if manifest['status'] not in ('READY_FIT_ONLY','READY_FIT_CAL'):raise ValueError('Preconfirmation FIT cache required')
    available=[r for r in manifest['records'] if r['role']=='fit']
    if [r['id'] for r in available]!=ROLES['fit']:raise ValueError('Full ordered FIT20 cache coverage')
    ids=ROLES['fit'][:2] if args.mode=='probe' else ROLES['fit']
    records={r['id']:rows(r,clock,True) for r in available if r['id'] in ids}
    index={pid:np.arange(0,len(records[pid]['target']),5,dtype=np.int64) for pid in ids}
    rawx=np.concatenate([records[p]['X33'][index[p]] for p in ids]);rawy=np.concatenate([records[p]['target'][index[p]] for p in ids])
    mean=rawx.mean(0);std=np.maximum(rawx.std(0),1e-8);mid=float(np.median(rawy));scale=max(float(np.percentile(rawy,75)-np.percentile(rawy,25)),1.)
    transform=dict(input_mean=mean.tolist(),input_std=std.tolist(),y_mid=mid,y_scale=scale,fit_ids=ids,
        warmup=0,label_stride=5,feature_names=manifest['feature_names'],dt_ref=manifest['dt_ref'],
        high_temperature_threshold_c=float(np.quantile(rawy,.95)))
    profiles=np.concatenate([np.full(len(index[p]),p,dtype=np.int64) for p in ids]);local_indices=np.concatenate([index[p] for p in ids])
    source_row=np.concatenate([records[p]['source_row'][index[p]] for p in ids]);times=np.concatenate([records[p]['time'][index[p]] for p in ids])
    x=((rawx-mean)/std).astype(np.float32);y=((rawy-mid)/scale).astype(np.float32)
    weights=np.concatenate([np.full(len(index[p]),len(y)/(len(ids)*len(index[p])),dtype=np.float64) for p in ids]).astype(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(weights).all():raise ValueError('Finite network training arrays')
    for pid in ids:
        if not np.isclose(np.sum(weights[profiles==pid],dtype=np.float64),len(y)/len(ids),rtol=1e-7,atol=1e-6):raise ValueError('Equal profile total loss weights')
    arrays=dict(x=x,y_normalized=y,w=weights,profile_ids=profiles,index=local_indices,source_row=source_row,time=times)
    worker.save_arrays(output/'training_arrays.npz',**arrays)
    model=worker.new_model(args.seed);initial=worker.weights(model);worker.save_arrays(output/'weights_initial.npz',**initial)
    torch=worker.torch
    xt=torch.as_tensor(x,device='cuda');yt=torch.as_tensor(y,device='cuda');wt=torch.as_tensor(weights,device='cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
    epochs=2 if args.mode=='probe' else 200;trace=[]
    for epoch in range(1,epochs+1):
        clock.check(training=True);started=time.monotonic();model.train()
        order=np.random.Generator(np.random.PCG64(args.seed+epoch-1)).permutation(len(y)).astype(np.int64)
        objective_sum,max_grad=0.,0.
        for start in range(0,len(y),4096):
            clock.check(training=True);batch=torch.as_tensor(order[start:start+4096],dtype=torch.int64,device='cuda')
            optimizer.zero_grad(set_to_none=True);error=model(xt[batch]).reshape(-1)-yt[batch]
            loss=(wt[batch]*error.square()).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
            loss.backward()
            gradient=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad,2) for p in model.parameters() if p.grad is not None]),2)
            if not torch.isfinite(gradient):raise FloatingPointError('Nonfinite training gradient norm')
            optimizer.step()
            objective_sum+=float(loss.detach().cpu())*len(batch);max_grad=max(max_grad,float(gradient.detach().cpu()))
        torch.cuda.synchronize()
        trace.append(dict(epoch=epoch,online_weighted_loss=objective_sum/len(y),max_gradient_norm=max_grad,
            seconds=time.monotonic()-started,permutation_sha256=worker.array_digest(dict(order=order))))
    clock.check(training=True);final=worker.weights(model)
    if any(not np.isfinite(value).all() for value in final.values()):raise FloatingPointError('Nonfinite fitted parameters')
    worker.save_arrays(output/'weights_final.npz',**final);dump(output/'epoch_trace.json',trace)
    metadata=dict(schema='rwth-mlp-model-v002',dataset='rwth48v',loss='mse',seed=args.seed,epochs=epochs,fit_ids=ids,
        protocol=PROTOCOL,partition=PARTITION,cache_manifest=ref(manifestpath),shared_cuda_source=SHARED,source_sha256=sha(__file__),
        settings=SETTINGS,transform=transform,training_rows=len(y),training_profile_counts={str(p):len(index[p]) for p in ids},
        training_arrays_sha256=worker.array_digest(arrays),initial_array_sha256=worker.array_digest(initial),
        final_array_sha256=worker.array_digest(final),feature_array_sha256=worker.array_digest(dict(x=x,profile_ids=profiles,index=local_indices)),
        training_seconds=sum(r['seconds'] for r in trace),completed_comparator=args.mode=='train',
        calibration_features_accessed=False,calibration_target_accessed=False,confirmation_accessed=False)
    dump(output/'model.json',metadata)
    files={name:sha(output/name) for name in ['configuration.json','runtime.json','training_arrays.npz','weights_initial.npz','weights_final.npz','epoch_trace.json','model.json']}
    dump(output/'model_lock.json',dict(schema='rwth-mlp-model-lock-v002',status='MODEL_LOCKED',dataset='rwth48v',
        protocol=PROTOCOL,partition=PARTITION,source_sha256=sha(__file__),cache_manifest=ref(manifestpath),
        model=ref(output/'model.json'),seed=args.seed,completed_comparator=args.mode=='train',files=files,
        input_hashes_before_nonfit=dict(clock.inputs),locked_at_utc=datetime.now(timezone.utc).isoformat()))
    return dict(status='RWTH_MLP_FIT_COMPLETE' if args.mode=='train' else 'RWTH_MLP_TWO_EPOCH_PROBE_COMPLETE',
        seed=args.seed,epochs=epochs,training_rows=len(y),training_seconds=metadata['training_seconds'],
        model_lock=ref(output/'model_lock.json'),completed_comparator=args.mode=='train',nonfit_arrays_accessed=False)


def predict(args,manifest,manifestpath,worker,output,clock):
    if not args.model_lock or not args.model_lock_sha256 or args.role not in ('calibration','confirmation'):raise ValueError('Prediction needs exact model and explicit role')
    mpath=clock.checked(dict(path=args.model_lock,sha256=args.model_lock_sha256));lock=read(mpath)
    if (lock['schema']!='rwth-mlp-model-lock-v002' or lock['status']!='MODEL_LOCKED' or lock['completed_comparator'] is not True or
        lock['protocol']!=PROTOCOL or lock['partition']!=PARTITION or lock['source_sha256']!=sha(__file__) or lock['seed']!=args.seed):raise ValueError('Qualified same-source fullFIT model lock')
    admission=None
    if args.role=='confirmation':admission=confirmation_admission(args,manifest,mpath,clock)
    required={'configuration.json','runtime.json','training_arrays.npz','weights_initial.npz','weights_final.npz','epoch_trace.json','model.json'}
    if set(lock['files'])!=required:raise ValueError('Complete immutable training model artifacts')
    for name,digest in lock['files'].items():clock.checked(dict(path=str(mpath.parent/name),sha256=digest))
    modelpath=clock.checked(lock['model'])
    if modelpath!=mpath.parent/'model.json':raise ValueError('Model lock/model directory mismatch')
    metadata=read(modelpath)
    if (metadata['fit_ids']!=ROLES['fit'] or metadata['epochs']!=200 or metadata['seed']!=args.seed or
        metadata['settings']!=SETTINGS or metadata['completed_comparator'] is not True or metadata['source_sha256']!=sha(__file__) or
        metadata['protocol']!=PROTOCOL or metadata['partition']!=PARTITION):raise ValueError('Frozen200epoch fullFIT comparator')
    original,oldpath=cache_manifest(lock['cache_manifest'],clock)
    if metadata['cache_manifest']!=lock['cache_manifest'] or metadata['transform']['feature_names']!=manifest['feature_names'] or original['dt_ref']!=manifest['dt_ref']:
        raise ValueError('Common FIT-only cache feature normalization identity')
    if original['feature_names']!=manifest['feature_names']:raise ValueError('Exact ordered raw33 columns')
    if args.role=='calibration' and manifest['status']!='READY_FIT_CAL':raise ValueError('Explicit prepared CAL cache required')
    if args.role=='confirmation' and manifest['status'] not in ('READY_CONFIRMATION','READY_FIT_CAL_CONF'):raise ValueError('Explicit admitted CONF cache required')
    entries=[entry for entry in manifest['records'] if entry['role']==args.role]
    if [entry['id'] for entry in entries]!=ROLES[args.role]:raise ValueError('Full ordered prediction role')
    model=worker.new_model(args.seed);worker.load_weights(model,mpath.parent/'weights_final.npz');model.eval()
    tr=metadata['transform'];mean=np.asarray(tr['input_mean']);std=np.asarray(tr['input_std']);torch=worker.torch
    records=[];started=time.monotonic()
    for entry in entries:
        value=rows(entry,clock,False);x=((value['X33']-mean)/std).astype(np.float32)
        if not np.isfinite(x).all():raise ValueError('Finite standardized inputs')
        result=[]
        with torch.no_grad():
            for start in range(0,len(x),4096):
                clock.check();raw=model(torch.as_tensor(x[start:start+4096],device='cuda')).reshape(-1)
                result.append(raw.cpu().numpy().astype(np.float64))
        pred=tr['y_mid']+tr['y_scale']*np.concatenate(result)
        if not np.isfinite(pred).all():raise ValueError('Finite physical-C predictions')
        arrays=dict(index=value['index'],source_row=value['source_row'],time=value['time'],features=x,prediction=pred)
        filename=output/f'profile_{entry["id"]}_predictions.npz';worker.save_arrays(filename,**arrays)
        records.append(dict(role=args.role,profile=entry['id'],rows=entry['rows'],**ref(filename),
            source_features=entry['features'],prediction_key='prediction',index_key='index',arrays_sha256=worker.array_digest(arrays)))
    dump(output/'prediction_lock.json',dict(schema='rwth-mlp-prediction-lock-v002',status='ALL_ROLE_PREDICTIONS_LOCKED',
        dataset='rwth48v',role=args.role,seed=args.seed,loss='mse',source_sha256=sha(__file__),protocol=PROTOCOL,
        partition=PARTITION,model_lock=ref(mpath),cache_manifest=ref(manifestpath),confirmation_admission=admission,
        predictions=records,role_targets_accessed=False,locked_at_utc=datetime.now(timezone.utc).isoformat()))
    return dict(status='RWTH_MLP_PREDICTIONS_LOCKED',role=args.role,seed=args.seed,predictions=records,
        prediction_lock=ref(output/'prediction_lock.json'),model_lock=ref(mpath),role_targets_accessed=False,
        confirmation_admission=admission,prediction_seconds=time.monotonic()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['probe','train','predict'],required=True)
    parser.add_argument('--cache-manifest','--manifest',dest='cache_manifest',required=True)
    parser.add_argument('--cache-manifest-sha256','--manifest-sha256',dest='cache_manifest_sha256',required=True)
    parser.add_argument('--output-dir',required=True);parser.add_argument('--max-seconds',type=float,default=600.)
    parser.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0])
    parser.add_argument('--role',choices=['calibration','confirmation'])
    for name in ('model-lock','model-lock-sha256','confirmation-admission','confirmation-admission-sha256'):parser.add_argument('--'+name)
    args=parser.parse_args()
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=1800:raise ValueError('Finite stage cap in(0,1800]')
    output=local(args.output_dir)
    if FOLLOWUP/'runs' not in output.parents:raise ValueError('New followup output required')
    output.mkdir(parents=True,exist_ok=False);clock=Clock(args.max_seconds,output)
    dump(output/'configuration.json',dict(vars(args),source=str(Path(__file__).resolve()),source_sha256=sha(__file__),
        output_dir=str(output),pid=os.getpid(),process_group=os.getpgrp() if hasattr(os,'getpgrp') else None,
        argv=sys.argv,protocol=PROTOCOL,partition=PARTITION,settings=SETTINGS,started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        manifest,manifestpath=cache_manifest(dict(path=args.cache_manifest,sha256=args.cache_manifest_sha256),clock)
        worker=shared_worker(clock);clock.check(training=args.mode in ('probe','train'))
        runtime=worker.cuda_runtime();runtime.update(tf32_matmul_allowed=worker.torch.backends.cuda.matmul.allow_tf32,
            tf32_cudnn_allowed=worker.torch.backends.cudnn.allow_tf32,deterministic_algorithms=worker.torch.are_deterministic_algorithms_enabled(),
            torch_num_threads=worker.torch.get_num_threads(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        dump(output/'runtime.json',runtime)
        if args.mode=='probe':dump(output/'fixtures.json',worker.arithmetic_fixtures())
        result=fit(args,manifest,manifestpath,worker,output,clock) if args.mode in ('probe','train') else predict(args,manifest,manifestpath,worker,output,clock)
        dump(output/'input_pins.json',clock.inputs);clock.check()
        result.update(dataset='rwth48v',protocol=PROTOCOL,partition=PARTITION,source_sha256=sha(__file__),
            cache_manifest=ref(manifestpath),elapsed_seconds=time.monotonic()-clock.start)
        dump(output/'summary.json',result);clock.check()
        files={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
        dump(output/'stage_record.json',dict(schema='rwth-mlp-stage-v002',status='COMPLETE',result=result,files=files,
            source_sha256=sha(__file__),elapsed_seconds=time.monotonic()-clock.start))
        clock.check();print(json.dumps(dict(status='COMPLETE',mode=args.mode,seed=args.seed,elapsed_seconds=time.monotonic()-clock.start)))
    except Exception as error:
        dump(output/'failure.json',dict(status='FAILED',error_type=type(error).__name__,message=str(error),elapsed_seconds=time.monotonic()-clock.start));raise


if __name__=='__main__':main()
