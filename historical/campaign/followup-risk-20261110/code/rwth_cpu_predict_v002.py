"""Locked RWTH CPU experts on CAL or admitted CONF features; no target decoder."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.dont_write_bytecode=True
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FOLLOWUP=ROOT/'followup-risk-20261110'
CORE_SHA='5c7962dae652958bc032267277dcec429d2f104a94b054c33914b543010bb164'
FIT_SHA='a22c12eb6962b7ce9a83f956a6cd4d0cf377a940903593a3ed1e0d1ac2abb57a'
HELD_SHA='1e1b6eb882c95790c4548dc159b7308fd4080b5440ee50db9dc9714f4dc11525'
COLUMNS=['direct9_fitcv_selected','direct_boost','positive_rc','unconstrained25_prior','linear33_prior',
         'positive_rc_insample_boost','unconstrained_rc_insample_boost','linear33_insample_boost']


def module(filename,name,expected):
    import hashlib
    p=FOLLOWUP/'code'/filename
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('Dependency source changed')
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def predict(args,prep,held,core,out,clock):
    manifestref=dict(path=args.manifest,sha256=args.manifest_sha256)
    manifestpath=clock.pin(manifestref);manifest=prep.read(manifestpath)
    expected_status='READY_FIT_CAL' if args.role=='calibration' else 'READY_CONFIRMATION'
    if (manifest['schema']!='rwth-cache-manifest-v002' or manifest['status']!=expected_status
        or manifest['source_sha256']!=HELD_SHA or manifest['protocol']!=prep.PROTOCOL
        or manifest['partition']!=prep.PARTITION or manifest['roles']!=prep.ROLES
        or manifest['feature_names']!=prep.FEATURE_NAMES or (manifestpath.parent/'failure.json').exists()):
        raise ValueError('Exact current target-separated held cache required')
    bundle,locks=held.base_locks(manifest['base_model_bundle'],manifest['fit_manifest'],prep,clock)
    lockref=dict(path=args.model_lock,sha256=args.model_lock_sha256)
    modelpath=clock.pin(lockref)
    if modelpath!=prep.local(bundle['cpu']['path']) or lockref['sha256']!=bundle['cpu']['sha256']:
        raise ValueError('Predictor CPU lock differs from all-base bundle')
    lock=prep.read(modelpath)
    if (lock['source_sha256']!=FIT_SHA or lock['core_sha256']!=CORE_SHA
        or lock['prediction_columns']!=COLUMNS):raise ValueError('Exact fixed CPU prediction panel')
    admitted=None
    if args.role=='confirmation':
        admitted=held.confirmation_admission(args,prep,clock,locks)
        if manifest['confirmation_admission']!=admitted:raise ValueError('Cache and predictor must bind the same admission')
    priors=prep.read(clock.pin(lock['priors']))
    if set(priors)!=set(core.FAMILIES):raise ValueError('All three fixed prior families')
    for family,value in priors.items():
        if value['family']!=family or value['fit_ids']!=prep.ROLES['fit'] or value['scale']['fit_ids']!=prep.ROLES['fit']:
            raise ValueError('All prior scaling fitted on fullFIT20')
    predictors={}
    for name,entry in lock['boosters'].items():
        metadata=prep.read(clock.pin(entry['metadata']));path=clock.pin(entry['pickle'])
        if metadata['fit_ids']!=prep.ROLES['fit'] or metadata['prior_family']!=entry['prior_family']:
            raise ValueError('Exact fullFIT booster/prior identity')
        # Only byte-pinned local artifacts from the admitted final fitter are loaded.
        with path.open('rb') as handle:model=pickle.load(handle)
        predictors[name]=(model,metadata,entry['prior_family'])
    entries=[e for e in manifest['records'] if e['role']==args.role]
    if [e['id'] for e in entries]!=prep.ROLES[args.role]:raise ValueError('Complete ordered role required')
    records=[]
    for entry in entries:
        clock.check()
        if entry['target_key'] is not None:raise ValueError('Held feature cache cannot include target')
        with np.load(clock.pin(entry['features']),allow_pickle=False) as z:
            if set(z.files)!={'profile_id','index','source_row','time','operating','X33'}:raise ValueError('Target-free exact feature fields')
            data={k:z[k].copy() for k in z.files}
        n=entry['rows'];ix=np.arange(n,dtype=np.int64)
        if (data['profile_id'].shape!=() or data['profile_id'].dtype!=np.int64 or int(data['profile_id'])!=entry['id']
            or data['index'].dtype!=np.int64 or data['source_row'].dtype!=np.int64
            or not np.array_equal(data['index'],ix) or not np.array_equal(data['source_row'],ix)
            or data['X33'].shape!=(n,33) or data['X33'].dtype!=np.float64
            or data['time'].shape!=(n,) or not np.all(np.diff(data['time'])>0)
            or any(not np.isfinite(x).all() for x in data.values())):raise ValueError('Complete finite role geometry')
        values={}
        for name,(model,metadata,family) in predictors.items():
            values[name]=core.predict_boost(model,metadata,data,priors[family] if family else None)
        for alias,original in lock['aliases'].items():values[alias]=values[original].copy()
        for name,family in [('positive_rc','positive25'),('unconstrained25_prior','free25'),('linear33_prior','linear33')]:
            values[name]=core.predict_prior(priors[family],data)
        if set(values)!=set(COLUMNS) or any(v.shape!=(n,) or not np.isfinite(v).all() for v in values.values()):
            raise ValueError('All eight expert predictions must be complete')
        path=out/('profile_'+str(entry['id']).zfill(3)+'_predictions.npz')
        np.savez_compressed(path,profile_id=data['profile_id'],index=ix,source_row=data['source_row'],time=data['time'],
                            **{'prediction__'+name:values[name].astype(np.float64) for name in COLUMNS})
        records.append(dict(id=entry['id'],profile=entry['id'],role=args.role,rows=n,**prep.ref(path),source_features=entry['features']))
    prep.dump(out/'prediction_lock.json',dict(schema='rwth-cpu-prediction-lock-v002',status='CPU_PREDICTIONS_LOCKED',
        dataset='rwth48v',role=args.role,protocol=prep.PROTOCOL,partition=prep.PARTITION,
        source_sha256=prep.sha(__file__),core_sha256=CORE_SHA,fit_source_sha256=FIT_SHA,
        cache_manifest=manifestref,model_lock=prep.ref(modelpath),base_model_bundle=manifest['base_model_bundle'],
        confirmation_admission=admitted,columns=COLUMNS,records=records,truth_accessed=False,
        created_at_utc=datetime.now(timezone.utc).isoformat()))
    prep.dump(out/'input_pins.json',clock.inputs)
    return dict(status='RWTH_CPU_PREDICTIONS_LOCKED',role=args.role,records=len(records),columns=COLUMNS,
        prediction_lock=prep.ref(out/'prediction_lock.json'),model_lock=prep.ref(modelpath),cache_manifest=manifestref,
        truth_accessed=False,confirmation_admission=admitted)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=['calibration','confirmation'],required=True)
    for name in ('manifest','manifest-sha256','model-lock','model-lock-sha256','output-dir'):p.add_argument('--'+name,required=True)
    for name in ('confirmation-admission','confirmation-admission-sha256'):p.add_argument('--'+name)
    p.add_argument('--max-seconds',type=float,default=600.)
    args=p.parse_args()
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=600:raise ValueError('Bounded prediction cap')
    held=module('rwth_prepare_held_v002.py','rwth_bound_held_preparation',HELD_SHA);prep=held.helper()
    core=module('rwth_cpu_core_v002.py','rwth_bound_prediction_core',CORE_SHA)
    out=prep.local(args.output_dir)
    if FOLLOWUP/'runs' not in out.parents:raise ValueError('New followup prediction output')
    out.mkdir(parents=True,exist_ok=False);clock=prep.Clock(args.max_seconds,out)
    prep.dump(out/'configuration.json',dict(vars(args),source_sha256=prep.sha(__file__),core_sha256=CORE_SHA,
        held_source_sha256=HELD_SHA,protocol=prep.PROTOCOL,pid=os.getpid(),argv=sys.argv))
    try:
        clock.pin(dict(path='followup-risk-20261110/code/rwth_prepare_held_v002.py',sha256=HELD_SHA))
        result=predict(args,prep,held,core,out,clock);clock.check()
        result.update(schema='rwth-cpu-prediction-stage-v002',source_sha256=prep.sha(__file__),core_sha256=CORE_SHA,
            elapsed_seconds=time.monotonic()-clock.start,
            files={f.relative_to(out).as_posix():prep.sha(f) for f in sorted(out.rglob('*')) if f.is_file()})
        prep.dump(out/'summary.json',result);clock.check()
        print(json.dumps(dict(status=result['status'],role=args.role,records=result['records'])))
    except Exception as exc:
        prep.dump(out/'failure.json',dict(status='FAILED',error_type=type(exc).__name__,message=str(exc)));raise


if __name__=='__main__':main()
