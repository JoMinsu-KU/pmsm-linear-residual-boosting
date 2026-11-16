"""Prepare CAL after all four base locks, or CONF inputs after consumed admission.

CAL labels are separate files. CONF rotor/stator are never numerically decoded.
"""
import argparse
from datetime import datetime,timezone
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
import zipfile
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.dont_write_bytecode=True
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FOLLOWUP=ROOT/'followup-risk-20261110'
PREP_SHA='ab7d82514f84ad934097fd15ccf83b6f02988c1207b87678ac2314595c95b3a7'
CPU_SHA='a22c12eb6962b7ce9a83f956a6cd4d0cf377a940903593a3ed1e0d1ac2abb57a'
CORE_SHA='5c7962dae652958bc032267277dcec429d2f104a94b054c33914b543010bb164'
MLP_SHA='4005577d76b689515256e933017c75239335a81c076034e17eedad4ef79d1113'
SEEDS=[2026111701,2026111702,2026111703]


def helper():
    import hashlib
    path=FOLLOWUP/'code/rwth_prepare_v002.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=PREP_SHA:raise ValueError('Preparation helper changed')
    spec=importlib.util.spec_from_file_location('rwth_pinned_raw_preparation',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def base_locks(entry,fitref,prep,clock):
    bundle=prep.read(clock.pin(entry))
    if (bundle['schema']!='rwth-base-model-bundle-v002' or bundle['protocol']!=prep.PROTOCOL
        or bundle['partition']!=prep.PARTITION or bundle['fit_manifest']!=fitref or len(bundle['mlp'])!=3):
        raise ValueError('Exact current all-base-model bundle required')
    sources=[('rwth_cpu_fit_v002.py',CPU_SHA),('rwth_cpu_core_v002.py',CORE_SHA),('rwth_mlp_v002.py',MLP_SHA)]
    for filename,digest in sources:clock.pin(dict(path='followup-risk-20261110/code/'+filename,sha256=digest))
    entries=[bundle['cpu']]+bundle['mlp'];seeds=[]
    for pos,ref in enumerate(entries):
        path=clock.pin(ref);lock=prep.read(path)
        if (path.parent/'failure.json').exists():raise ValueError('Failed base-model stage')
        if lock['protocol']!=prep.PROTOCOL or lock['partition']!=prep.PARTITION:raise ValueError('Model protocol/partition')
        for name,digest in lock['files'].items():
            p=(path.parent/name).resolve()
            if path.parent not in p.parents:raise ValueError('Model artifact path escape')
            clock.pin(dict(path=str(p),sha256=digest))
        if pos==0:
            if (lock['schema']!='rwth-cpu-model-lock-v002' or lock['status']!='CPU_MODELS_LOCKED'
                or lock['source_sha256']!=CPU_SHA or lock['core_sha256']!=CORE_SHA
                or lock['fit_manifest']!=fitref or lock['fit_ids']!=prep.ROLES['fit']
                or lock['calibration_values_accessed'] is not False or lock['confirmation_values_accessed'] is not False):
                raise ValueError('Complete FIT-only CPU model lock')
        else:
            if (lock['schema']!='rwth-mlp-model-lock-v002' or lock['status']!='MODEL_LOCKED'
                or lock['source_sha256']!=MLP_SHA or lock['completed_comparator'] is not True
                or lock['cache_manifest']!=fitref):raise ValueError('Completed fullFIT neural comparator')
            metadata=prep.read(clock.pin(lock['model']))
            if (metadata['fit_ids']!=prep.ROLES['fit'] or metadata['epochs']!=200 or metadata['loss']!='mse'
                or metadata['seed']!=lock['seed']):raise ValueError('Fixed fullFIT200epoch neural identity')
            seeds.append(lock['seed'])
    if seeds!=SEEDS:raise ValueError('All three ordered fixed neural seeds required')
    return bundle,entries


def confirmation_admission(args,prep,clock,models):
    if not args.confirmation_admission or not args.confirmation_admission_sha256:
        raise ValueError('Consumed root confirmation admission required')
    ref=dict(path=args.confirmation_admission,sha256=args.confirmation_admission_sha256)
    path=clock.pin(ref);admission=prep.read(path)
    expected=dict(schema='rwth-confirmation-admission-v002',status='ADMITTED_CONSUMED_NO_RESET',
        confirmation_id='C17_RWTH_FIXED_CONFIRMATION_02',candidate_id='C17',alpha=.025,allowed_roles=['CONF'],
        protocol=prep.PROTOCOL,partition=prep.PARTITION,conf_ids=prep.ROLES['confirmation'])
    if any(admission.get(k)!=v for k,v in expected.items()):raise ValueError('Exact one-attempt confirmation admission')
    for model in models:
        key=prep.local(model['path']).relative_to(ROOT).as_posix()
        if admission['bindings'].get(key)!=model['sha256']:raise ValueError('All current base models must be admitted')
    for name,digest in admission['bindings'].items():clock.pin(dict(path=name,sha256=digest))
    return prep.ref(path)


def prepare(args,prep,output,clock):
    fitref=dict(path=args.fit_manifest,sha256=args.fit_manifest_sha256)
    fitpath=clock.pin(fitref);fit=prep.read(fitpath)
    if (fit['schema']!='rwth-cache-manifest-v002' or fit['status']!='READY_FIT_ONLY'
        or fit['source_sha256']!=PREP_SHA or fit['protocol']!=prep.PROTOCOL or fit['partition']!=prep.PARTITION
        or fit['roles']!=prep.ROLES or fit['feature_names']!=prep.FEATURE_NAMES
        or fit['confirmation_member_contents_opened'] is not False or (fitpath.parent/'failure.json').exists()):
        raise ValueError('Exact original qualified FIT-only cache')
    clock.pin(prep.PROTOCOL);clock.pin(prep.PARTITION)
    bundle_ref=dict(path=args.base_models,sha256=args.base_models_sha256)
    _,models=base_locks(bundle_ref,fitref,prep,clock)
    admitted=confirmation_admission(args,prep,clock,models) if args.role=='confirmation' else None
    before=dict(clock.inputs)
    archivepath=clock.pin(fit['archive'])
    if fit['archive']['sha256']!=prep.ARCHIVE_SHA:raise ValueError('Exact archive bytes required')
    dt_ref=fit['dt_ref']
    if not math.isfinite(dt_ref) or dt_ref<=0:raise ValueError('FIT-only positive reference interval')
    features_dir=output/'features'/args.role;features_dir.mkdir(parents=True)
    truths_dir=output/'truths'/args.role
    if args.role=='calibration':truths_dir.mkdir(parents=True)
    records=[];opened=[]
    with zipfile.ZipFile(archivepath) as archive:
        for pid in prep.ROLES[args.role]:
            clock.check();r=prep.parse_member(archive,pid,clock,include_target=args.role=='calibration')
            n=len(r['time']);ix=np.arange(n,dtype=np.int64)
            x=prep.features(r['operating'],r['time'],dt_ref,clock)
            path=features_dir/('profile_'+str(pid).zfill(3)+'.npz')
            np.savez_compressed(path,profile_id=np.asarray(pid,dtype=np.int64),index=ix,source_row=ix,
                                time=r['time'],operating=r['operating'],X33=x)
            truth=None
            if args.role=='calibration':
                targetpath=truths_dir/path.name
                np.savez_compressed(targetpath,profile_id=np.asarray(pid,dtype=np.int64),index=ix,source_row=ix,
                                    time=r['time'],target=r['target'])
                truth=prep.ref(targetpath)
            records.append(dict(id=pid,role=args.role,rows=n,member=r['member'],csv_sha256=r['csv_sha256'],
                                features=prep.ref(path),truth=truth,target_key=None))
            opened.append(r['member'])
    manifest=dict(fit)
    manifest.update(status='READY_FIT_CAL' if args.role=='calibration' else 'READY_CONFIRMATION',
        source_sha256=prep.sha(__file__),preparation_dependency_sha256=PREP_SHA,fit_manifest=fitref,
        records=fit['records']+records if args.role=='calibration' else records,
        base_model_bundle=bundle_ref,base_model_locks=models,confirmation_admission=admitted,
        calibration_member_contents_opened=args.role=='calibration',confirmation_member_contents_opened=args.role=='confirmation',
        confirmation_rotor_values_parsed=False,stator_numeric_values_parsed=False,
        confirmation_attempt_consumed=args.role=='confirmation',fitted_model_count=0,
        opened_csv_members=opened,input_hashes_before_held_members=before,inputs=dict(clock.inputs),
        generated_at_utc=datetime.now(timezone.utc).isoformat())
    # No misleading inherited FIT-only byte count for this role-specific stage.
    manifest.pop('source_bytes',None)
    prep.dump(output/'manifest.json',manifest);prep.dump(output/'input_pins.json',clock.inputs)
    return dict(status='RWTH_HELD_FEATURE_CACHE_READY',role=args.role,manifest=prep.ref(output/'manifest.json'),
        profiles=len(records),rows=sum(e['rows'] for e in records),base_model_bundle=bundle_ref,
        confirmation_admission=admitted,confirmation_rotor_values_parsed=False,stator_numeric_values_parsed=False,
        calibration_labels_separate=args.role=='calibration',models_fitted=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=['calibration','confirmation'],required=True)
    for name in ('fit-manifest','fit-manifest-sha256','base-models','base-models-sha256','output-dir'):p.add_argument('--'+name,required=True)
    for name in ('confirmation-admission','confirmation-admission-sha256'):p.add_argument('--'+name)
    p.add_argument('--max-seconds',type=float,default=300.)
    args=p.parse_args()
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=300:raise ValueError('Finite held preparation cap')
    prep=helper();out=prep.local(args.output_dir)
    if FOLLOWUP/'runs' not in out.parents:raise ValueError('New followup stage output')
    out.mkdir(parents=True,exist_ok=False);clock=prep.Clock(args.max_seconds,out)
    prep.dump(out/'configuration.json',dict(vars(args),source_sha256=prep.sha(__file__),prep_sha256=PREP_SHA,
        protocol=prep.PROTOCOL,pid=os.getpid(),argv=sys.argv,started_at_utc=datetime.now(timezone.utc).isoformat()))
    try:
        clock.pin(dict(path='followup-risk-20261110/code/rwth_prepare_v002.py',sha256=PREP_SHA))
        result=prepare(args,prep,out,clock);clock.check()
        result.update(schema='rwth-held-preparation-summary-v002',source_sha256=prep.sha(__file__),protocol=prep.PROTOCOL,
            elapsed_seconds=time.monotonic()-clock.start,
            files={f.relative_to(out).as_posix():prep.sha(f) for f in sorted(out.rglob('*')) if f.is_file()})
        prep.dump(out/'summary.json',result);clock.check()
        print(__import__('json').dumps(dict(status=result['status'],role=args.role,profiles=result['profiles'])))
    except Exception as exc:
        prep.dump(out/'failure.json',dict(status='FAILED',error_type=type(exc).__name__,message=str(exc)));raise


if __name__=='__main__':main()
