"""Frozen RWTH CAL scalar mixtures and separately admitted CONF evaluation.

Twenty fixed CPU mixtures plus three raw MLP seeds; no model/seed/pair search.
CAL scalar choices are locked; all CONF predictions precede CONF target access.
"""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import zipfile
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.dont_write_bytecode=True
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FOLLOWUP=ROOT/'followup-risk-20261110'
PROTOCOL=dict(path='followup-risk-20261110/protocols/rwth-replication-v002-frozen.md',sha256='f1d201fe4aa6243f8c77c6a8a85a91268d183d0cd24d48bc35ec86b269082b0f')
PARTITION=dict(path='followup-risk-20261110/configs/rwth-partition-v001.json',sha256='5d5117312eee65bd9811f0ede7326d01d8a0a1aa3d0b0e3fb6ef60ae47959be1')
SCALAR=dict(path='followup-risk-20261110/code/gate_worker_v001.py',sha256='d73b6eb70db5c6e57f79a5334c4000d373189d3e49aaaf7371daee9b9a08e25b')
METRIC=dict(path='followup-risk-20261110/code/risk_screen_analysis_v001.py',sha256='7d514ccd7f9471a4e5d8dcc29604d8782c6c0dad300cdc830a253d4e1c759282')
MLP_SOURCE=dict(path='followup-risk-20261110/code/rwth_mlp_v002.py',sha256='4005577d76b689515256e933017c75239335a81c076034e17eedad4ef79d1113')
CPU_SOURCE=dict(path='followup-risk-20261110/code/rwth_cpu_predict_v002.py',sha256='af9222387a346b57828a6a87518b4ef7563dcd119099794b451c3429f3f19cf3')
CPU_FIT=dict(path='followup-risk-20261110/code/rwth_cpu_fit_v002.py',sha256='a22c12eb6962b7ce9a83f956a6cd4d0cf377a940903593a3ed1e0d1ac2abb57a')
CPU_CORE=dict(path='followup-risk-20261110/code/rwth_cpu_core_v002.py',sha256='5c7962dae652958bc032267277dcec429d2f104a94b054c33914b543010bb164')
ROLES=dict(fit=[38,27,30,23,21,19,29,20,26,37,6,8,35,34,4,15,7,13,18,1],
    calibration=[10,11,22,9,16,31,28,17],confirmation=[5,32,33,36,24,2,25,14,3,12])
SEEDS=[2026111701,2026111702,2026111703]
LEFT='direct9_fitcv_selected'
PAIRS=dict(c15='positive_rc_insample_boost',direct200='direct_boost',linear33='linear33_insample_boost',
           free25='unconstrained_rc_insample_boost',positive_prior='positive_rc')
BASE_KEYS=[LEFT,'direct_boost','positive_rc','unconstrained25_prior','linear33_prior',
           'positive_rc_insample_boost','unconstrained_rc_insample_boost','linear33_insample_boost']
RULES=['left','right','half','cal_macro_rmse']
CPU_METHODS=[pair+'__'+rule for pair in PAIRS for rule in RULES]
MLP_METHODS=['mlp__'+str(seed) for seed in SEEDS]
METHODS=CPU_METHODS+MLP_METHODS
PRIMARY='c15__cal_macro_rmse'
ARCHIVE_SHA='a6b219a17acbbc2a9899fa7d750e1b0531756eeefcf16efd56009c43a0c4fce6'


def sha(p):
    value=hashlib.sha256()
    with Path(p).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):value.update(block)
    return value.hexdigest()


def local(value):
    value=str(value)
    if os.name!='nt' and re.match(r'^[A-Za-z]:[\\/]',value):value='/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
    elif os.name=='nt' and re.match(r'^/mnt/[a-z]/',value):value=value[5].upper()+':/'+value[7:]
    p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
    if p!=ROOT and ROOT not in p.parents:raise ValueError('Outside campaign')
    return p


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def dump(p,value):
    with Path(p).open('x',encoding='utf-8') as handle:json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)


def ref(p):return dict(path=local(p).relative_to(ROOT).as_posix(),sha256=sha(p))


class Clock:
    def __init__(self,seconds,output):self.start=time.monotonic();self.seconds=seconds;self.output=output;self.inputs={}
    def check(self,training=False):
        if time.monotonic()-self.start>self.seconds:raise TimeoutError('Stage time cap')
        now=datetime.now(timezone.utc)
        if now>=datetime.fromisoformat('2026-11-11T00:23:59+00:00'):raise TimeoutError('Campaign deadline')
        if training and now>=datetime.fromisoformat('2026-11-10T20:23:59+00:00'):raise TimeoutError('New fitting cutoff')
        if sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())>1000000000:raise RuntimeError('1GB output ceiling')
    def pin(self,entry):
        self.check();p=local(entry['path'])
        if sha(p)!=entry['sha256']:raise ValueError('Exact input SHA mismatch: '+str(p))
        self.inputs[str(p)]=entry['sha256'];return p


def module(entry,name,clock):
    spec=importlib.util.spec_from_file_location(name,clock.pin(entry));result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result);return result


def admitted(args,clock,cache,required_locks):
    if not args.confirmation_admission or not args.confirmation_admission_sha256:raise ValueError('Consumed root CONF admission required')
    entry=dict(path=args.confirmation_admission,sha256=args.confirmation_admission_sha256)
    p=clock.pin(entry);value=read(p)
    expected=dict(schema='rwth-confirmation-admission-v002',status='ADMITTED_CONSUMED_NO_RESET',
        confirmation_id='C17_RWTH_FIXED_CONFIRMATION_02',candidate_id='C17',alpha=.025,
        allowed_roles=['CONF'],protocol=PROTOCOL,partition=PARTITION,conf_ids=ROLES['confirmation'])
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Wrong fresh confirmation admission')
    if clock.pin(cache['confirmation_admission'])!=p or cache['confirmation_admission']['sha256']!=entry['sha256']:
        raise ValueError('Cache/admission identity mismatch')
    for binding in required_locks:
        path=clock.pin(binding);key=path.relative_to(ROOT).as_posix()
        if value['bindings'].get(key)!=binding['sha256']:raise ValueError('A current model is absent from consumed admission')
    return ref(p)


def cache_identity(cache):
    if (cache.get('schema')!='rwth-cache-manifest-v002' or cache.get('dataset')!='rwth48v' or cache['protocol']!=PROTOCOL or
        cache['partition']!=PARTITION or cache['roles']!=ROLES):raise ValueError('Frozen cache/roles identity')


def pin_model_files(lockpath,lock,clock):
    for name,digest in lock['files'].items():
        p=(lockpath.parent/name).resolve()
        if lockpath.parent not in p.parents:raise ValueError('Model artifact path escape')
        clock.pin(dict(path=str(p),sha256=digest))


def mlp_metadata(entry,role,cache_ref,clock):
    p=clock.pin(entry);value=read(p)
    if (value['schema']!='rwth-mlp-prediction-lock-v002' or value['status']!='ALL_ROLE_PREDICTIONS_LOCKED' or
        value['dataset']!='rwth48v' or value['role']!=role or value['seed'] not in SEEDS or value['loss']!='mse' or
        value['source_sha256']!=MLP_SOURCE['sha256'] or value['protocol']!=PROTOCOL or value['partition']!=PARTITION or
        value['role_targets_accessed'] is not False or value['cache_manifest']!=cache_ref or (p.parent/'failure.json').exists()):
        raise ValueError('Complete frozen raw MLP role predictions required')
    clock.pin(MLP_SOURCE);mpath=clock.pin(value['model_lock']);lock=read(mpath)
    if (lock['schema']!='rwth-mlp-model-lock-v002' or lock['status']!='MODEL_LOCKED' or lock['seed']!=value['seed'] or
        lock['completed_comparator'] is not True or lock['source_sha256']!=MLP_SOURCE['sha256'] or
        lock['protocol']!=PROTOCOL or lock['partition']!=PARTITION):raise ValueError('Qualified MLP fullFIT model lock')
    if set(lock['files'])!={'configuration.json','runtime.json','training_arrays.npz','weights_initial.npz','weights_final.npz','epoch_trace.json','model.json'}:
        raise ValueError('All seven MLP model artifacts required')
    pin_model_files(mpath,lock,clock);meta=read(clock.pin(lock['model']))
    if meta['epochs']!=200 or meta['fit_ids']!=ROLES['fit'] or meta['seed']!=value['seed']:raise ValueError('Fixed MLP training definition')
    return value,meta


def component_arrays(bundle,cache,cpu,neural,clock):
    role=bundle['role'];ids=ROLES[role]
    entries=[r for r in cache['records'] if r['role']==role]
    if [r['id'] for r in entries]!=ids:raise ValueError('Complete cache prediction role')
    cpurecords=cpu['records']
    if [r['id'] for r in cpurecords]!=ids:raise ValueError('Complete CPU role IDs')
    for value in neural:
        if [r['profile'] for r in value['predictions']]!=ids:raise ValueError('Complete MLP role IDs')
    result={}
    for ordinal,entry in enumerate(entries):
        row=cpurecords[ordinal]
        if row['profile']!=entry['id'] or row['rows']!=entry['rows'] or row['source_features']!=entry['features']:raise ValueError('CPU exact source role mapping')
        with np.load(clock.pin(row),allow_pickle=False) as data:
            expected={'profile_id','index','source_row','time'}|{'prediction__'+key for key in BASE_KEYS}
            if set(data.files)!=expected:raise ValueError('All eight CPU expert arrays required')
            saved={k:data[k].copy() for k in data.files}
        n=entry['rows'];ix=np.arange(n,dtype=np.int64)
        if saved['profile_id'].dtype!=np.int64 or saved['profile_id'].shape!=() or int(saved['profile_id'])!=entry['id']:raise ValueError('CPU profile scalar identity')
        for key in ('index','source_row'):
            if saved[key].dtype!=np.int64 or not np.array_equal(saved[key],ix):raise ValueError('Complete unfiltered source indices')
        if saved['time'].dtype!=np.float64 or saved['time'].shape!=(n,) or not np.isfinite(saved['time']).all() or not np.all(np.diff(saved['time'])>0):raise ValueError('Original strict timestamp sequence')
        values={key:saved['prediction__'+key] for key in BASE_KEYS}
        for key,value in values.items():
            if value.dtype!=np.float64 or value.shape!=(n,) or not np.isfinite(value).all():raise ValueError('Finite full CPU component')
        for neural_lock in neural:
            item=neural_lock['predictions'][ordinal]
            if item['role']!=role or item['source_features']!=entry['features'] or item['rows']!=n:raise ValueError('MLP exact common feature/row source')
            with np.load(clock.pin(item),allow_pickle=False) as data:
                for key in ('index','source_row','time'):
                    if not np.array_equal(data[key],saved[key]):raise ValueError('CPU and MLP row/time alignment')
                prediction=data['prediction'].copy()
            if prediction.dtype!=np.float64 or prediction.shape!=(n,) or not np.isfinite(prediction).all():raise ValueError('Finite full MLP prediction')
            values['mlp__'+str(neural_lock['seed'])]=prediction
        result[entry['id']]=dict(id=entry['id'],index=ix,source_row=saved['source_row'],time=saved['time'],
            values=values,source_features=entry['features'],truth_ref=entry['truth'])
    return result


def read_truth(rows,clock):
    result={}
    for pid,row in rows.items():
        with np.load(clock.pin(row['truth_ref']),allow_pickle=False) as data:
            if set(data.files)!={'profile_id','index','source_row','time','target'}:raise ValueError('Separated raw role truth schema')
            if int(data['profile_id'])!=pid or any(not np.array_equal(data[key],row[key]) for key in ('index','source_row','time')):raise ValueError('Truth exact original row/time mapping')
            y=data['target'].astype(np.float64)
        if y.shape!=row['index'].shape or not np.isfinite(y).all():raise ValueError('Finite all-row truth')
        result[pid]=y
    return result


def fit_threshold(fitmanifest,clock):
    value=read(clock.pin(fitmanifest));cache_identity(value)
    entries=[r for r in value['records'] if r['role']=='fit']
    if [r['id'] for r in entries]!=ROLES['fit']:raise ValueError('FIT20 high-temperature threshold lineage')
    ys=[]
    for entry in entries:
        with np.load(clock.pin(entry['features']),allow_pickle=False) as saved:
            if saved['target'].shape!=(entry['rows'],):raise ValueError('FIT target geometry')
            ys.append(saved['target'][0::5].astype(np.float64))
    y=np.concatenate(ys)
    if not np.isfinite(y).all():raise ValueError('Finite FIT threshold observations')
    return float(np.quantile(y,.95))


def scalar_models(rows,truth,scalar,clock):
    cfg=dict(scalar_xatol=1e-12,scalar_maxiter=500);models={};ids=list(rows)
    for pair,right in PAIRS.items():
        records=[dict(id=p,d=rows[p]['values'][LEFT],h=rows[p]['values'][right],y=truth[p]) for p in ids]
        final=dict(fit_ids=ids,cal_macro_rmse=scalar.scalar_blend(records,cfg,clock))
        lopo=[]
        for held in ids:
            reduced=[r for r in records if r['id']!=held]
            lopo.append(dict(held_out=held,model=dict(fit_ids=[r['id'] for r in reduced],
                cal_macro_rmse=scalar.scalar_blend(reduced,cfg,clock))))
        models[pair]=dict(right_key=right,final=final,lopo=lopo)
    return dict(schema='rwth-combination-models-v002',dataset='rwth48v',primary=PRIMARY,calibration_ids=ids,
        pair_order=list(PAIRS),rules=RULES,methods=METHODS,left_key=LEFT,pairs=models)


def predictions(rows,models,lopo=False):
    result={}
    for pid,row in rows.items():
        values={}
        for pair,right in PAIRS.items():
            group=models['pairs'][pair];model=next(f['model'] for f in group['lopo'] if f['held_out']==pid) if lopo else group['final']
            d,h=row['values'][LEFT],row['values'][right]
            current=dict(left=d,right=h,half=(d+h)/2,cal_macro_rmse=d+model['cal_macro_rmse']['weight']*(h-d))
            values.update({pair+'__'+key:value for key,value in current.items()})
        values.update({m:row['values'][m] for m in MLP_METHODS})
        if list(values)!=METHODS:raise ValueError('Exact20+3 prediction order')
        result[pid]=values
    return result


def save_predictions(output,role,source_role,rows,values,clock):
    folder=output/'predictions'/role;folder.mkdir(parents=True,exist_ok=False);entries=[]
    for pid,row in rows.items():
        clock.check();arrays=dict(profile_id=np.asarray(pid,dtype=np.int64),index=row['index'],source_row=row['source_row'],time=row['time'])
        arrays.update({'prediction__'+m:values[pid][m] for m in METHODS})
        if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite fixed-panel prediction')
        filename=folder/f'profile_{pid}.npz';np.savez_compressed(filename,**arrays)
        entries.append(dict(role=role,source_role=source_role,profile=pid,rows=len(row['index']),**ref(filename),
            source_features=row['source_features'],truth_ref=row['truth_ref'],model_kind='lopo' if role=='calibration_lopo' else 'final'))
    return entries


def cpu_metadata(entry,role,cache_ref,clock):
    p=clock.pin(entry);value=read(p)
    if (value['schema']!='rwth-cpu-prediction-lock-v002' or value['status']!='CPU_PREDICTIONS_LOCKED' or
        value['dataset']!='rwth48v' or value['role']!=role or value['protocol']!=PROTOCOL or value['partition']!=PARTITION or
        value['cache_manifest']!=cache_ref or value['truth_accessed'] is not False or (p.parent/'failure.json').exists()):raise ValueError('Qualified exact CPU role prediction lock')
    if value['source_sha256']!=CPU_SOURCE['sha256'] or value['core_sha256']!=CPU_CORE['sha256'] or value['fit_source_sha256']!=CPU_FIT['sha256'] or value['columns']!=BASE_KEYS:
        raise ValueError('Frozen CPU predictor/core/fitter source and panel')
    clock.pin(CPU_SOURCE);clock.pin(CPU_CORE)
    mpath=clock.pin(value['model_lock']);lock=read(mpath)
    if (lock['schema']!='rwth-cpu-model-lock-v002' or lock['status']!='CPU_MODELS_LOCKED' or lock['dataset']!='rwth48v' or
        lock['protocol']!=PROTOCOL or lock['partition']!=PARTITION or lock['fit_ids']!=ROLES['fit'] or lock['core_sha256']!=value['core_sha256'] or
        lock['calibration_values_accessed'] is not False or lock['confirmation_values_accessed'] is not False or lock['prediction_columns']!=BASE_KEYS):raise ValueError('Exact fullFIT CPU expert lock')
    if lock['source_sha256']!=CPU_FIT['sha256']:raise ValueError('Frozen fullFIT fitter source')
    clock.pin(CPU_FIT)
    pin_model_files(mpath,lock,clock)
    selection=read(clock.pin(lock['selection']));priors=read(clock.pin(lock['priors']))
    if set(priors)!={'positive25','free25','linear33'}:raise ValueError('All three fixed priors')
    expected_boosters={'direct_boost','positive_rc_insample_boost','unconstrained_rc_insample_boost','linear33_insample_boost'}
    if selection['selected_direct_grid']==0:
        if lock['aliases']!={'direct9_fitcv_selected':'direct_boost'}:raise ValueError('Exact selected-grid0 alias')
    else:
        expected_boosters.add('direct9_fitcv_selected')
        if lock['aliases']!={}:raise ValueError('No unselected aliases')
    if set(lock['boosters'])!=expected_boosters:raise ValueError('All selected base boosters')
    for name,item in lock['boosters'].items():
        for key in ('metadata','pickle','trees'):
            artifact=clock.pin(item[key]);relative=artifact.relative_to(mpath.parent).as_posix()
            if lock['files'].get(relative)!=item[key]['sha256']:raise ValueError('Base artifact absent from final model lock')
    return value,lock


def bundle_metadata(args,clock):
    filename=clock.pin(dict(path=args.components,sha256=args.components_sha256));bundle=read(filename)
    if (bundle['schema']!='rwth-combination-input-v002' or bundle['dataset']!='rwth48v' or
        bundle['role'] not in ('calibration','confirmation') or bundle['protocol']!=PROTOCOL or bundle['partition']!=PARTITION):raise ValueError('Frozen role combination input bundle')
    clock.pin(PROTOCOL);clock.pin(PARTITION)
    cache=read(clock.pin(bundle['cache_manifest']));cache_identity(cache)
    if cache['archive']['sha256']!=ARCHIVE_SHA:raise ValueError('Fixed independent motor archive')
    cpu,cpu_model=cpu_metadata(bundle['cpu_prediction_lock'],bundle['role'],bundle['cache_manifest'],clock)
    if len(bundle['mlp_prediction_locks'])!=3:raise ValueError('Exactly three paired neural seeds')
    neural=[];neural_meta=[]
    for entry in bundle['mlp_prediction_locks']:
        lock,meta=mlp_metadata(entry,bundle['role'],bundle['cache_manifest'],clock);neural.append(lock);neural_meta.append(meta)
    if [x['seed'] for x in neural]!=SEEDS:raise ValueError('All three ordered fixed seeds')
    fit_manifest=cpu_model['fit_manifest'];fit_path=clock.pin(fit_manifest)
    if any(clock.pin(m['cache_manifest'])!=fit_path or m['cache_manifest']['sha256']!=fit_manifest['sha256'] for m in neural_meta):raise ValueError('CPU/MLP common FIT cache')
    base_locks=dict(cpu=cpu['model_lock'],mlp=[m['model_lock'] for m in neural])
    if cpu['base_model_bundle']!=cache['base_model_bundle']:raise ValueError('CPU predictor same locked expert bundle')
    base_bundle=read(clock.pin(cache['base_model_bundle']))
    if (base_bundle['schema']!='rwth-base-model-bundle-v002' or base_bundle['protocol']!=PROTOCOL or base_bundle['partition']!=PARTITION or
        clock.pin(base_bundle['fit_manifest'])!=fit_path or base_bundle['fit_manifest']['sha256']!=fit_manifest['sha256'] or
        base_bundle['cpu']!=base_locks['cpu'] or base_bundle['mlp']!=base_locks['mlp']):raise ValueError('Held cache frozen after all4 base model locks')
    return bundle,cache,cpu,neural,base_locks,fit_manifest,filename


def model_lock_read(entry,clock):
    path=clock.pin(entry);value=read(path)
    if (value['schema']!='rwth-combination-model-lock-v002' or value['status']!='CAL_MODELS_LOCKED' or
        value['protocol']!=PROTOCOL or value['partition']!=PARTITION or value['source_sha256']!=sha(__file__) or
        value['primary']!=PRIMARY or value['methods']!=METHODS or value['confirmation_accessed'] is not False):raise ValueError('Frozen CAL combination model lock')
    if set(value['files'])!={'configuration.json','models.json','fit_threshold.json','component-inputs.json'}:raise ValueError('Exact CAL model artifact set')
    pin_model_files(path,value,clock)
    return path,value,read(path.parent/'models.json')


def calibrate(args,output,clock):
    bundle,cache,cpu,neural,base_locks,fit_manifest,filename=bundle_metadata(args,clock)
    if bundle['role']!='calibration' or cache['status']!='READY_FIT_CAL':raise ValueError('CAL-only scalar fitting')
    rows=component_arrays(bundle,cache,cpu,neural,clock)
    # All base models and complete CAL predictions are validated before labels.
    threshold=fit_threshold(fit_manifest,clock);truth=read_truth(rows,clock)
    scalar=module(SCALAR,'rwth_pinned_scalar_calibration',clock)
    models=scalar_models(rows,truth,scalar,clock)
    dump(output/'models.json',models);dump(output/'fit_threshold.json',dict(high_temperature_threshold_c=threshold,fit_manifest=fit_manifest))
    dump(output/'component-inputs.json',bundle)
    dump(output/'model_lock.json',dict(schema='rwth-combination-model-lock-v002',status='CAL_MODELS_LOCKED',dataset='rwth48v',
        source_sha256=sha(__file__),protocol=PROTOCOL,partition=PARTITION,primary=PRIMARY,methods=METHODS,
        base_model_locks=base_locks,fit_manifest=fit_manifest,calibration_inputs=ref(filename),scalar_dependency=SCALAR,
        calibration_ids=ROLES['calibration'],confirmation_accessed=False,
        files={name:sha(output/name) for name in ['configuration.json','models.json','fit_threshold.json','component-inputs.json']},
        inputs_before_confirmation=dict(clock.inputs),locked_at_utc=datetime.now(timezone.utc).isoformat()))
    entries=save_predictions(output,'calibration_lopo','calibration',rows,predictions(rows,models,True),clock)
    entries+=save_predictions(output,'calibration','calibration',rows,predictions(rows,models,False),clock)
    dump(output/'prediction_lock.json',dict(schema='rwth-combination-prediction-lock-v002',status='ALL_23_PREDICTIONS_LOCKED',
        source_sha256=sha(__file__),dataset='rwth48v',role='calibration',protocol=PROTOCOL,partition=PARTITION,
        primary=PRIMARY,methods=METHODS,model_lock=ref(output/'model_lock.json'),component_inputs=ref(filename),
        cache_manifest=bundle['cache_manifest'],confirmation_admission=None,predictions=entries,
        calibration_truth_accessed=True,confirmation_truth_accessed=False,locked_at_utc=datetime.now(timezone.utc).isoformat()))
    return dict(status='RWTH_CAL_COMBINATIONS_LOCKED',model_lock=ref(output/'model_lock.json'),
        prediction_lock=ref(output/'prediction_lock.json'),primary=PRIMARY,methods=METHODS,
        high_temperature_threshold_c=threshold,confirmation_accessed=False)


def predict_confirmation(args,output,clock):
    if not args.model_lock or not args.model_lock_sha256:raise ValueError('Exact CAL model lock required')
    mpath,lock,models=model_lock_read(dict(path=args.model_lock,sha256=args.model_lock_sha256),clock)
    bundle,cache,cpu,neural,base_locks,_,filename=bundle_metadata(args,clock)
    if bundle['role']!='confirmation' or cache['status']!='READY_CONFIRMATION' or base_locks!=lock['base_model_locks']:
        raise ValueError('Same frozen experts under explicit CONF cache')
    admission=admitted(args,clock,cache,[ref(mpath),base_locks['cpu']]+base_locks['mlp'])
    for value in [cpu]+neural:
        if value.get('confirmation_admission')!=admission:raise ValueError('Every base CONF prediction uses same consumed admission')
    rows=component_arrays(bundle,cache,cpu,neural,clock)
    entries=save_predictions(output,'confirmation','confirmation',rows,predictions(rows,models,False),clock)
    dump(output/'component-inputs.json',bundle)
    dump(output/'prediction_lock.json',dict(schema='rwth-combination-prediction-lock-v002',status='ALL_23_PREDICTIONS_LOCKED',
        source_sha256=sha(__file__),dataset='rwth48v',role='confirmation',protocol=PROTOCOL,partition=PARTITION,
        primary=PRIMARY,methods=METHODS,model_lock=ref(mpath),component_inputs=ref(filename),cache_manifest=bundle['cache_manifest'],
        confirmation_admission=admission,predictions=entries,calibration_truth_accessed=False,confirmation_truth_accessed=False,
        locked_at_utc=datetime.now(timezone.utc).isoformat()))
    return dict(status='RWTH_CONF_23_PREDICTIONS_LOCKED',prediction_lock=ref(output/'prediction_lock.json'),
        model_lock=ref(mpath),primary=PRIMARY,methods=METHODS,confirmation_truth_accessed=False,confirmation_admission=admission)


def conf_truth_from_archive(cache,rows,clock,output):
    if cache['archive']['sha256']!=ARCHIVE_SHA:raise ValueError('Exact confirmed archive bytes')
    archive_path=clock.pin(cache['archive']);entries={x['id']:x for x in cache['records'] if x['role']=='confirmation'}
    truth={};receipts=[]
    destination=output/'truth';destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        for pid,row in rows.items():
            clock.check();entry=entries[pid];name='final/'+str(pid).zfill(3)+'.csv'
            if entry['member']!=name:raise ValueError('Exact frozen CONF member identity')
            info=archive.getinfo(name)
            if info.is_dir() or info.file_size>32000000:raise ValueError('Bounded ordinary CSV member')
            with archive.open(info) as handle:raw=handle.read(info.file_size+1)
            if len(raw)!=info.file_size or hashlib.sha256(raw).hexdigest()!=entry['csv_sha256']:raise ValueError('Exact CSV bytes from admitted feature cache')
            reader=csv.reader(io.StringIO(raw.decode('utf-8-sig')))
            if next(reader)!=['time','i','torque','motor_speed','coolant','rotor','stator','profile_id']:raise ValueError('Fixed rotor/header mapping')
            ys=[];times=[]
            for number,fields in enumerate(reader):
                if number%2048==0:clock.check()
                if len(fields)!=8 or float(fields[7])!=pid:raise ValueError('Exact raw profile/row width')
                times.append(float(fields[0]));ys.append(float(fields[5]))
                # No stator or operating-channel conversion in this target stage.
            y=np.asarray(ys,dtype=np.float64);stamp=np.asarray(times,dtype=np.float64)
            if y.shape!=row['index'].shape or not np.isfinite(y).all() or not np.array_equal(stamp,row['time']):raise ValueError('All raw rotor targets align with locked time/rows')
            filename=destination/f'profile_{pid}_truth.npz'
            np.savez_compressed(filename,profile_id=np.asarray(pid,dtype=np.int64),index=row['index'],source_row=row['source_row'],time=stamp,target=y)
            truth[pid]=y;receipts.append(dict(profile=pid,rows=len(y),member=name,csv_sha256=entry['csv_sha256'],**ref(filename)))
    dump(output/'truth_receipt.json',dict(schema='rwth-confirmation-truth-receipt-v002',archive=cache['archive'],records=receipts,
        rotor_access_after_all23_prediction_lock=True,stator_numeric_values_parsed=False))
    return truth


def metric_mean(metric,role):
    keys=['rmse','under_rmse','over_mae'];seedrows=[metric[m][role] for m in MLP_METHODS]
    rows=[]
    for index,row in enumerate(seedrows[0]['per_profile']):
        if any(s['per_profile'][index]['id']!=row['id'] for s in seedrows):raise ValueError('Aligned MLP seed metric profiles')
        rows.append(dict(id=row['id'],**{key:float(np.mean([s['per_profile'][index][key] for s in seedrows])) for key in keys}))
    return dict(per_profile=rows,summary={'macro_'+key:float(np.mean([s['summary']['macro_'+key] for s in seedrows])) for key in keys},
                seeds=SEEDS,aggregation='arithmetic mean of three seed metrics, not predictions or best seed')


def contrast(candidate,baseline,candidate_name,baseline_name,role):
    if [r['id'] for r in candidate['per_profile']]!=[r['id'] for r in baseline['per_profile']]:raise ValueError('Aligned paired cycles')
    delta=[x['rmse']-y['rmse'] for x,y in zip(candidate['per_profile'],baseline['per_profile'])]
    c,b=candidate['summary'],baseline['summary']
    ratio=lambda key:c[key]/b[key] if b[key]!=0 else None
    return dict(role=role,candidate=candidate_name,comparator=baseline_name,delta_macro_rmse=c['macro_rmse']-b['macro_rmse'],
        rmse_ratio=ratio('macro_rmse'),under_rmse_ratio=ratio('macro_under_rmse'),over_mae_ratio=ratio('macro_over_mae'),
        profile_ids=[r['id'] for r in candidate['per_profile']],per_profile_rmse_differences=delta,
        wins=sum(d<0 for d in delta),ties=sum(d==0 for d in delta),losses=sum(d>0 for d in delta),
        under_rmse_diagnostic_flag=c['macro_under_rmse']>1.05*b['macro_under_rmse']+1e-12,
        over_mae_diagnostic_flag=c['macro_over_mae']>1.25*b['macro_over_mae']+1e-12,
        risk_flags_scope='Descriptive imported diagnostic ratios, not safety requirements or formal decision criteria')


def formal_decision(metrics,mean,models):
    role='confirmation';primary=metrics[PRIMARY][role];direct=metrics['c15__left'][role]
    paired=contrast(primary,direct,PRIMARY,'c15__left',role);delta=paired['per_profile_rmse_differences']
    if paired['profile_ids']!=ROLES['confirmation'] or len(delta)!=10:raise ValueError('Exactly10 predetermined confirmation cycles')
    wins=paired['wins'];ties=paired['ties'];n=10-ties
    numerator=sum(math.comb(n,j) for j in range(wins,n+1)) if n else 1
    denominator=2**n if n else 1;p=numerator/denominator
    average=float(np.mean(delta));formal=average<0 and p<=.025
    comparisons=[]
    for pair in PAIRS:
        if pair!='c15':comparisons.append(contrast(primary,metrics[pair+'__cal_macro_rmse'][role],PRIMARY,pair+'__cal_macro_rmse',role))
    comparisons.append(contrast(primary,metrics['c15__half'][role],PRIMARY,'c15__half',role))
    comparisons.append(contrast(primary,mean,PRIMARY,'mlp_three_seed_metric_mean',role))
    strict=[]
    for item in comparisons:
        baseline=mean if item['comparator']=='mlp_three_seed_metric_mean' else metrics[item['comparator']][role]
        cm,bm=primary['summary']['macro_rmse'],baseline['summary']['macro_rmse']
        strict.append(dict(comparator=item['comparator'],primary_macro_rmse=cm,comparator_macro_rmse=bm,strictly_lower=cm<bm))
    point_condition=all(x['strictly_lower'] for x in strict)
    weight=models['pairs']['c15']['final']['cal_macro_rmse']['weight']
    return dict(schema='rwth-confirmation-decision-v002',candidate='C17',confirmation_id='C17_RWTH_FIXED_CONFIRMATION_02',
        primary=PRIMARY,formal_comparator='c15__left',alpha=.025,profile_ids=ROLES['confirmation'],
        mean_delta=average,per_profile_rmse_differences=delta,wins=wins,ties=ties,losses=paired['losses'],n_eff=n,
        exact_one_sided_sign=dict(numerator=numerator,denominator=denominator,p_value=p,direction='smaller RMSE',zero_ties_excluded=True),
        formal_success=formal,strict_point_mean_controls=strict,contribution_mean_condition=point_condition,
        topic_criterion_pass=formal and point_condition,
        verdict='FORMAL_AND_CONTRIBUTION_CRITERIA_MET' if formal and point_condition else 'FORMAL_PASS_BUT_CONTRIBUTION_NOT_ESTABLISHED' if formal else 'PRIMARY_CONFIRMATION_FAILED',
        primary_cal_weight=weight,primary_collapses_to='direct9' if weight==0 else 'C15' if weight==1 else None,
        leave_one_cycle_out_mean_deltas=[dict(excluded_profile=pid,mean_delta=float(np.mean([d for j,d in enumerate(delta) if j!=i]))) for i,pid in enumerate(ROLES['confirmation'])],
        required_comparisons=comparisons,pending_independent_audit=True,
        scope='One additional motor,10 fixed cycles; cycle independence unverified; no secondary significance or safety guarantee')


def evaluate(args,output,clock):
    if not args.prediction_lock or not args.prediction_lock_sha256:raise ValueError('Exact23-prediction lock required')
    ppath=clock.pin(dict(path=args.prediction_lock,sha256=args.prediction_lock_sha256));lock=read(ppath)
    if (lock['schema']!='rwth-combination-prediction-lock-v002' or lock['status']!='ALL_23_PREDICTIONS_LOCKED' or
        lock['source_sha256']!=sha(__file__) or lock['protocol']!=PROTOCOL or lock['partition']!=PARTITION or
        lock['methods']!=METHODS or lock['primary']!=PRIMARY or lock['confirmation_truth_accessed'] is not False or
        (ppath.parent/'failure.json').exists()):raise ValueError('Qualified fixed-panel prediction lock')
    mpath,model_lock,models=model_lock_read(lock['model_lock'],clock)
    inputs=lock['component_inputs']
    metadata_args=argparse.Namespace(components=inputs['path'],components_sha256=inputs['sha256'])
    bundle,cache,cpu,neural,base_locks,_,_=bundle_metadata(metadata_args,clock)
    if bundle['role']!=lock['role'] or bundle['cache_manifest']!=lock['cache_manifest'] or base_locks!=model_lock['base_model_locks']:
        raise ValueError('Prediction/component/model/cache immutable lineage')
    role=lock['role'];admission=None
    if role=='confirmation':
        admission=admitted(args,clock,cache,[ref(mpath),base_locks['cpu']]+base_locks['mlp'])
        if lock['confirmation_admission']!=admission or any(v.get('confirmation_admission')!=admission for v in [cpu]+neural):raise ValueError('Evaluation same consumed admission')
    elif role!='calibration':raise ValueError('Known evaluation role only')
    expected=[('confirmation',p) for p in ROLES['confirmation']] if role=='confirmation' else [
        (r,p) for r in ('calibration_lopo','calibration') for p in ROLES['calibration']]
    if [(e['role'],e['profile']) for e in lock['predictions']]!=expected:raise ValueError('All predetermined role predictions required')
    cache_records={r['id']:r for r in cache['records'] if r['role']==role}
    rows={};saved={}
    # Decode and validate every saved23 prediction column first, without targets.
    for entry in lock['predictions']:
        pid=entry['profile'];origin=cache_records[pid]
        if entry['source_role']!=role or entry['source_features']!=origin['features'] or entry['truth_ref']!=origin['truth'] or entry['rows']!=origin['rows']:
            raise ValueError('Fixed source row/truth-reference mapping')
        with np.load(clock.pin(entry),allow_pickle=False) as data:
            if set(data.files)!={'profile_id','index','source_row','time'}|{'prediction__'+m for m in METHODS}:raise ValueError('Exactly20+3 locked predictions')
            arrays={k:data[k].copy() for k in data.files}
        n=entry['rows'];ix=np.arange(n,dtype=np.int64)
        if arrays['profile_id'].dtype!=np.int64 or arrays['profile_id'].shape!=() or int(arrays['profile_id'])!=pid:raise ValueError('Saved scalar profile identity')
        for key in ('index','source_row'):
            if arrays[key].dtype!=np.int64 or not np.array_equal(arrays[key],ix):raise ValueError('All original zero-based rows')
        if arrays['time'].dtype!=np.float64 or arrays['time'].shape!=(n,) or not np.all(np.diff(arrays['time'])>0):raise ValueError('Full strict source timestamps')
        for method in METHODS:
            value=arrays['prediction__'+method]
            if value.dtype!=np.float64 or value.shape!=(n,) or not np.isfinite(value).all():raise ValueError('Finite locked complete predictions')
        if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Finite saved prediction geometry')
        with np.load(clock.pin(origin['features']),allow_pickle=False) as cache_arrays:
            for key in ('index','source_row','time'):
                if not np.array_equal(arrays[key],cache_arrays[key]):raise ValueError('Exact original held feature rows/time')
        rows[pid]=dict(id=pid,index=arrays['index'],source_row=arrays['source_row'],time=arrays['time'],truth_ref=origin['truth'])
        saved[(entry['role'],pid)]=arrays
    dump(output/'all_predictions_verified.json',dict(status='ALL23_PREDICTIONS_VERIFIED_BEFORE_TARGETS',prediction_lock=ref(ppath),
        prediction_records=len(saved),methods=METHODS,confirmation_admission=admission,verified_at_utc=datetime.now(timezone.utc).isoformat()))
    if role=='confirmation':truth=conf_truth_from_archive(cache,rows,clock,output)
    else:
        truth=read_truth(rows,clock)
        dump(output/'truth_receipt.json',dict(schema='rwth-calibration-truth-references-v002',records=[
            dict(profile=p,truth_ref=rows[p]['truth_ref']) for p in ROLES['calibration']],stator_numeric_values_parsed=False))
    threshold=read(mpath.parent/'fit_threshold.json')['high_temperature_threshold_c']
    metric=module(METRIC,'rwth_frozen_role_metrics',clock)
    roles=['confirmation'] if role=='confirmation' else ['calibration_lopo','calibration']
    metrics={m:{} for m in METHODS}
    for method in METHODS:
        for r in roles:
            clock.check();values={pid:saved[(r,pid)]['prediction__'+method] for pid in truth}
            metrics[method][r]=metric.role_metrics(truth,values,threshold,None)
    means={r:metric_mean(metrics,r) for r in roles};comparisons=[]
    for r in roles:
        for method in METHODS:comparisons.append(contrast(metrics[method][r],metrics['c15__left'][r],method,'c15__left',r))
        comparisons.append(contrast(means[r],metrics['c15__left'][r],'mlp_three_seed_metric_mean','c15__left',r))
    dump(output/'metrics.json',dict(dataset='rwth48v',high_temperature_threshold_c=threshold,methods=metrics,
        time_unit_independently_verified=False,duration_seconds_reported=False))
    dump(output/'mlp_seed_metric_mean.json',means);dump(output/'comparison.json',comparisons)
    if role=='confirmation':
        decision=formal_decision(metrics,means['confirmation'],models);dump(output/'decision.json',decision)
    else:decision=None
    return dict(status='RWTH_CONFIRMATION_EVALUATION_COMPLETE' if role=='confirmation' else 'RWTH_CALIBRATION_DIAGNOSTICS_COMPLETE',
        role=role,primary=PRIMARY,methods=METHODS,model_lock=ref(mpath),prediction_lock=ref(ppath),
        confirmation_admission=admission,calibration_or_model_selection_performed=False,
        formal_decision=None if decision is None else ref(output/'decision.json'),
        pending_independent_audit=True,stator_numeric_values_parsed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['calibrate','predict','evaluate'],required=True)
    parser.add_argument('--output-dir',required=True);parser.add_argument('--max-seconds',type=float,default=600.)
    for name in ('components','components-sha256','model-lock','model-lock-sha256','prediction-lock','prediction-lock-sha256',
                 'confirmation-admission','confirmation-admission-sha256'):parser.add_argument('--'+name)
    args=parser.parse_args()
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=600:raise ValueError('Finite stage cap in(0,600]')
    if args.mode in ('calibrate','predict') and (not args.components or not args.components_sha256):raise ValueError('Exact component bundle required')
    output=local(args.output_dir)
    if FOLLOWUP/'runs' not in output.parents:raise ValueError('New immutable followup run directory')
    output.mkdir(parents=True,exist_ok=False);clock=Clock(args.max_seconds,output)
    dump(output/'configuration.json',dict(vars(args),source=str(Path(__file__).resolve()),source_sha256=sha(__file__),
        pid=os.getpid(),argv=sys.argv,output_dir=str(output),protocol=PROTOCOL,partition=PARTITION,
        scalar_dependency=SCALAR,metric_dependency=METRIC,started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        clock.pin(PROTOCOL);clock.pin(PARTITION)
        result=calibrate(args,output,clock) if args.mode=='calibrate' else predict_confirmation(args,output,clock) if args.mode=='predict' else evaluate(args,output,clock)
        dump(output/'input_pins.json',clock.inputs);clock.check()
        result.update(source_sha256=sha(__file__),dataset='rwth48v',protocol=PROTOCOL,partition=PARTITION,
            elapsed_seconds=time.monotonic()-clock.start,files={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
        dump(output/'summary.json',result);clock.check()
        print(json.dumps(dict(status=result['status'],mode=args.mode,elapsed_seconds=result['elapsed_seconds'])))
    except Exception as error:
        dump(output/'failure.json',dict(status='FAILED',error_type=type(error).__name__,message=str(error),elapsed_seconds=time.monotonic()-clock.start));raise


if __name__=='__main__':main()
