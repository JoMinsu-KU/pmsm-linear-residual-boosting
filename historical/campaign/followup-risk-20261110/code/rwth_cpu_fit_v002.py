"""Bounded FIT-only RWTH CPU probe/CV/final training stages. Root executes.

No CAL or CONF array loader. CV uses every held FIT row and all fixed choices.
"""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.dont_write_bytecode=True
import numpy as np
import sklearn

ROOT=Path(__file__).resolve().parents[2]
FOLLOWUP=ROOT/'followup-risk-20261110'
CORE_SHA='5c7962dae652958bc032267277dcec429d2f104a94b054c33914b543010bb164'
PREP_SHA='ab7d82514f84ad934097fd15ccf83b6f02988c1207b87678ac2314595c95b3a7'


def module(filename,name,expected):
    import hashlib
    path=FOLLOWUP/'code'/filename
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Source dependency changed')
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def save_prediction(prep,directory,pid,record,prediction):
    path=directory/('held_'+str(pid).zfill(3)+'.npz')
    np.savez_compressed(path,profile_id=np.asarray(pid,dtype=np.int64),index=record['index'],
        source_row=record['source_row'],prediction=prediction.astype(np.float64))
    return prep.ref(path)


def probe(core,prep,manifest,records,output,clock):
    ids=manifest['roles']['fit'][:2]
    _,metadata=core.fit_boost(records,ids,5,output/'largest_model',clock,save_pickle=False)
    size=sum(p.stat().st_size for p in (output/'largest_model').rglob('*') if p.is_file())
    result=dict(status='RWTH_HGB_PROBE_COMPLETE',fit_ids=ids,grid_id=5,training_rows=metadata['training_rows'],
        elapsed_training_and_export_seconds=metadata['elapsed_seconds'],model_artifact_bytes=size,
        all_fit_sampled_rows=sum(len(core.indices(records[p])) for p in manifest['roles']['fit']),
        no_held_metrics_computed=True,no_scientific_selection=True)
    prep.dump(output/'probe.json',result)
    return result


def cv_direct(args,core,prep,manifest,records,output,clock):
    if args.grid_id not in range(9):raise ValueError('CV direct requires fixed grid ID0..8')
    ids=manifest['roles']['fit'];rows=[]
    for held in ids:
        clock.check(training=True);train=[p for p in ids if p!=held];folder=output/('fold_'+str(held).zfill(3))
        model,metadata=core.fit_boost(records,train,args.grid_id,folder,clock,save_pickle=False)
        prediction=core.predict_boost(model,metadata,records[held])
        rmse=float(np.sqrt(np.mean((prediction-records[held]['target'])**2)))
        rows.append(dict(held_out=held,fit_ids=train,rmse=rmse,model_metadata=prep.ref(folder/'model.json'),
                         prediction=save_prediction(prep,folder,held,records[held],prediction)))
    table=dict(schema='rwth-direct-cv-v002',grid_id=args.grid_id,parameters=dict(core.BASE,**core.GRID[args.grid_id]),
        ordered_ids=ids,folds=rows,macro_rmse=float(np.mean([r['rmse'] for r in rows])),
        held_scope='all original rows, no warmup; equal profile weight')
    prep.dump(output/'cv.json',table)
    return dict(status='RWTH_DIRECT_CV_COMPLETE',grid_id=args.grid_id,folds=len(rows),cv=prep.ref(output/'cv.json'))


def cv_linear(core,prep,manifest,records,output,clock):
    ids=manifest['roles']['fit'];families={};heldfolder=output/'held_predictions';heldfolder.mkdir()
    for family in core.FAMILIES:
        tables=[];folds=[]
        for li,lam in enumerate(core.LAMBDAS):
            errors=[]
            for held in ids:
                fit_ids=[p for p in ids if p!=held]
                model=core.fit_prior(records,fit_ids,family,lam,clock)
                value=core.predict_prior(model,records[held]);rmse=float(np.sqrt(np.mean((value-records[held]['target'])**2)))
                folder=heldfolder/(family+'_'+str(li));folder.mkdir(exist_ok=True)
                folds.append(dict(held_out=held,lam=lam,model=model,rmse=rmse,
                    prediction=save_prediction(prep,folder,held,records[held],value)))
                errors.append(rmse)
            tables.append(dict(lam=lam,profile_rmse=errors,macro_rmse=float(np.mean(errors))))
        selected=min(tables,key=lambda r:(r['macro_rmse'],r['lam']))['lam']
        families[family]=dict(selected_lambda=selected,cv=tables,folds=folds)
    prep.dump(output/'cv.json',dict(schema='rwth-linear-cv-v002',ordered_ids=ids,families=families,
        held_scope='all original rows; every fold scale fitted only on other19 FIT profiles'))
    return dict(status='RWTH_LINEAR_CV_COMPLETE',folds=180,cv=prep.ref(output/'cv.json'))


def checked_stage(entry,status,args,prep,clock):
    path=clock.pin(entry);stage=prep.read(path)
    if (stage['status']!=status or stage['source_sha256']!=prep.sha(__file__) or stage['core_sha256']!=CORE_SHA
        or stage['manifest']['sha256']!=args.manifest_sha256 or (path.parent/'failure.json').exists()):
        raise ValueError('Exact completed same-source CV stage required')
    for name,digest in stage['files'].items():
        file=(path.parent/name).resolve()
        if path.parent not in file.parents:raise ValueError('CV file path escape')
        clock.pin(dict(path=str(file),sha256=digest))
    return prep.read(clock.pin(stage['cv']))


def final(args,core,prep,manifest,records,output,clock):
    if not args.cv_results or not args.cv_results_sha256:raise ValueError('All frozen CV receipts required')
    bundle=prep.read(clock.pin(dict(path=args.cv_results,sha256=args.cv_results_sha256)))
    if (bundle['schema']!='rwth-cpu-cv-inputs-v002' or bundle['manifest']!=dict(path=args.manifest,sha256=args.manifest_sha256)
        or len(bundle['direct'])!=9):raise ValueError('Exact all-grid CV receipt bundle')
    linear=checked_stage(bundle['linear'],'RWTH_LINEAR_CV_COMPLETE',args,prep,clock)
    directs=[checked_stage(e,'RWTH_DIRECT_CV_COMPLETE',args,prep,clock) for e in bundle['direct']]
    ids=manifest['roles']['fit']
    if linear['ordered_ids']!=ids or set(linear['families'])!=set(core.FAMILIES):raise ValueError('All prior CV families')
    if [r['grid_id'] for r in directs]!=list(range(9)):raise ValueError('All nine ordered direct CV grids')
    for table in directs:
        if table['ordered_ids']!=ids or [f['held_out'] for f in table['folds']]!=ids:raise ValueError('Full direct LOPO coverage')
        if table['parameters']!=dict(core.BASE,**core.GRID[table['grid_id']]):raise ValueError('Fixed direct hyperparameters')
    best=min(directs,key=lambda r:(r['macro_rmse'],r['grid_id']))['grid_id']
    priors={}
    for family in core.FAMILIES:
        item=linear['families'][family]
        if [r['lam'] for r in item['cv']]!=core.LAMBDAS or len(item['folds'])!=60:raise ValueError('Full prior lambda CV')
        selected=min(item['cv'],key=lambda r:(r['macro_rmse'],r['lam']))['lam']
        if item['selected_lambda']!=selected:raise ValueError('Recorded prior selection differs')
        priors[family]=core.fit_prior(records,ids,family,selected,clock)
    prep.dump(output/'priors.json',priors)
    selection=dict(schema='rwth-cpu-selection-v002',selected_direct_grid=best,
        direct_cv=[{k:r[k] for k in ('grid_id','macro_rmse')} for r in directs],
        selected_lambdas={f:priors[f]['lam'] for f in priors},cv_bundle=dict(path=args.cv_results,sha256=args.cv_results_sha256),
        selected_from='FIT20 LOPO only; every held profile uses all rows')
    prep.dump(output/'selection.json',selection)
    boosters={};aliases={}
    jobs=[('direct_boost',0,None),('direct9_fitcv_selected',best,None),
        ('positive_rc_insample_boost',0,'positive25'),('unconstrained_rc_insample_boost',0,'free25'),
        ('linear33_insample_boost',0,'linear33')]
    for name,grid_id,family in jobs:
        if name=='direct9_fitcv_selected' and best==0:
            aliases[name]='direct_boost';continue
        folder=output/name
        _,metadata=core.fit_boost(records,ids,grid_id,folder,clock,prior=priors[family] if family else None,save_pickle=True)
        boosters[name]=dict(metadata=prep.ref(folder/'model.json'),pickle=prep.ref(folder/'model.pkl'),
                            trees=prep.ref(folder/'trees.npz'),prior_family=family)
    lock=dict(schema='rwth-cpu-model-lock-v002',status='CPU_MODELS_LOCKED',dataset='rwth48v',
        source_sha256=prep.sha(__file__),core_sha256=CORE_SHA,protocol=prep.PROTOCOL,partition=prep.PARTITION,
        fit_manifest=dict(path=args.manifest,sha256=args.manifest_sha256),fit_ids=ids,
        selection=prep.ref(output/'selection.json'),priors=prep.ref(output/'priors.json'),boosters=boosters,aliases=aliases,
        prediction_columns=['direct9_fitcv_selected','direct_boost','positive_rc','unconstrained25_prior','linear33_prior',
            'positive_rc_insample_boost','unconstrained_rc_insample_boost','linear33_insample_boost'],
        free25_definition='same normalized25 design/ridge as positive25; no independent intercept; constraints removed',
        calibration_values_accessed=False,confirmation_values_accessed=False,
        files={p.relative_to(output).as_posix():prep.sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        created_at_utc=datetime.now(timezone.utc).isoformat())
    prep.dump(output/'model_lock.json',lock)
    return dict(status='RWTH_CPU_FINAL_MODELS_LOCKED',model_lock=prep.ref(output/'model_lock.json'),
        selected_direct_grid=best,actual_boosters=len(boosters),logical_boosters=5,priors=3,
        calibration_values_accessed=False,confirmation_values_accessed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['probe','cv-direct','cv-linear','final'])
    p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--max-seconds',required=True,type=float)
    p.add_argument('--grid-id',type=int);p.add_argument('--cv-results');p.add_argument('--cv-results-sha256')
    args=p.parse_args();cap={'probe':300,'cv-direct':1200,'cv-linear':600,'final':1200}[args.mode]
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=cap:raise ValueError('Fixed bounded mode cap')
    prep=module('rwth_prepare_v002.py','rwth_preparation_utilities',PREP_SHA)
    core=module('rwth_cpu_core_v002.py','rwth_frozen_numerical_core',CORE_SHA)
    out=prep.local(args.output_dir)
    if FOLLOWUP/'runs' not in out.parents:raise ValueError('New followup run output')
    out.mkdir(parents=True,exist_ok=False);clock=prep.Clock(args.max_seconds,out)
    prep.dump(out/'configuration.json',dict(vars(args),source_sha256=prep.sha(__file__),core_sha256=CORE_SHA,
        prep_sha256=PREP_SHA,protocol=prep.PROTOCOL,pid=os.getpid(),argv=sys.argv,numpy=np.__version__,sklearn=sklearn.__version__))
    try:
        clock.pin(dict(path='followup-risk-20261110/code/rwth_cpu_core_v002.py',sha256=CORE_SHA))
        fixtures=core.synthetic_fixtures();prep.dump(out/'synthetic_fixtures.json',fixtures)
        manifest,records=core.load_fit(dict(path=args.manifest,sha256=args.manifest_sha256),clock,ROOT)
        if args.mode=='probe':result=probe(core,prep,manifest,records,out,clock)
        elif args.mode=='cv-direct':result=cv_direct(args,core,prep,manifest,records,out,clock)
        elif args.mode=='cv-linear':result=cv_linear(core,prep,manifest,records,out,clock)
        else:result=final(args,core,prep,manifest,records,out,clock)
        prep.dump(out/'input_pins.json',clock.inputs);clock.check()
        result.update(schema='rwth-cpu-stage-v002',mode=args.mode,source_sha256=prep.sha(__file__),core_sha256=CORE_SHA,
            protocol=prep.PROTOCOL,manifest=dict(path=args.manifest,sha256=args.manifest_sha256),
            elapsed_seconds=time.monotonic()-clock.start,
            files={f.relative_to(out).as_posix():prep.sha(f) for f in sorted(out.rglob('*')) if f.is_file()})
        prep.dump(out/'summary.json',result);clock.check()
        print(json.dumps(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds'])))
    except Exception as exc:
        prep.dump(out/'failure.json',dict(status='FAILED',error_type=type(exc).__name__,message=str(exc),
                                         elapsed_seconds=time.monotonic()-clock.start));raise


if __name__=='__main__':main()
