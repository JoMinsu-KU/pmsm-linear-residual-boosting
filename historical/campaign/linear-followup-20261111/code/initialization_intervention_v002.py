"""C18 frozen recording-start-state interventions; fit/predict and evaluate separately.

Existing audited numerical primitives are exact source dependencies. No new
lambda, capacity, mixture-weight, warmup, or evaluation-cohort selection occurs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import time
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
import numpy as np
import sklearn

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'linear-followup-20261111'
PROTOCOL_SHA='271c4ec0506f7cea6f0004c305ce9b10f5d3e365a188aed607e3b688e309f914'
CORE_SHA='17c47c41c27b6e92618a44de0f5860f80ffd1357d998a4cfeb805f5e2d8b7b37'
HELPER_SHA='d4aa942ca9db8d8e9319197632cf376730456f3ea5cdc48b9d1c1fd51fc59412'
DIAGNOSTIC_SHA='d799863cf6c7e19db7604a96c409bb70835c06ed9ad53871fc9ead5b510e3434'
METHODS=['d9','linear_prior','linear_residual','linear_half','prior_half']
ARMS=['own_boundary','first_load']
SPANS=[16,64,256,1024,4096]
BASE_KEYS={'d9':'direct9_fitcv_selected','linear_prior':'linear33_ridge','linear_residual':'linear33_insample_boost'}
ROLES={'ipmsm':['calibration','pilot','confirmation'],'paderborn':['development','confirmation']}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def local(value):
    text=str(value)
    if os.name!='nt' and re.match(r'^[A-Za-z]:[\\/]',text):text='/mnt/'+text[0].lower()+'/'+text[3:].replace('\\','/')
    elif os.name=='nt' and re.match(r'^/mnt/[a-z]/',text):text=text[5].upper()+':/'+text[7:]
    p=Path(text);p=(p if p.is_absolute() else ROOT/p).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Campaign-local immutable input/output required')
    return p


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def ref(p):return dict(path=local(p).relative_to(ROOT).as_posix(),sha256=sha(p))
def dump(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)


class Intake:
    def __init__(self,out,cap):
        self.out=out;self.cap=cap;self.start=time.monotonic();self.inputs={};self.audit_cache={};self.checks=0;self.training=False
    def check(self):
        if time.monotonic()-self.start>self.cap:raise TimeoutError('Registered stage cap; preserve incomplete files')
        now=datetime.now(timezone.utc)
        if now>=datetime.fromisoformat('2026-11-12T00:23:59+00:00'):raise TimeoutError('Campaign deadline')
        if self.training and now>=datetime.fromisoformat('2026-11-11T20:23:59+00:00'):raise TimeoutError('New fitting cutoff')
        if sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())>1000000000:raise RuntimeError('1GB stage output cap')
    def pin(self,e):
        self.check();p=local(e['path']);actual=sha(p)
        if actual!=e['sha256']:raise ValueError('Changed pinned artifact: '+str(p))
        self.inputs[str(p)]=actual;return p
    def audit(self,e):
        p=self.pin(e);key=(str(p),e['sha256'])
        if key not in self.audit_cache:
            v=read(p);ids=[v[k] for k in ('auditor_sha256','source_sha256') if k in v]
            if v.get('status')!='PASS' or v.get('failures') or not ids or any(x!=e['auditor_sha256'] for x in ids):raise ValueError('Exact prior independent PASS required')
            self.pin(dict(path=e['auditor_path'],sha256=e['auditor_sha256']))
            ix={}
            for name,digest in v['inputs'].items():ix.setdefault(str(local(name)),set()).add(digest)
            self.audit_cache[key]=(v,ix)
        return self.audit_cache[key]
    def audited(self,e,audit):
        p=self.pin(e);_,ix=self.audit(audit)
        if ix.get(str(p))!={e['sha256']}:raise ValueError('Artifact not bound by prior audit: '+str(p))
        return p
    def arrays(self,e,audit=None,keys=None):
        p=self.audited(e,audit) if audit else self.pin(e)
        with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (keys if keys is not None else z.files)}
    def close(self,x,y,label,atol=1e-9,rtol=1e-11):
        self.checks+=1
        if np.shape(x)!=np.shape(y) or not np.allclose(x,y,atol=atol,rtol=rtol,equal_nan=False):raise ArithmeticError(label)


def module(path,digest,name,a):
    p=a.pin(dict(path=path,sha256=digest));spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def dependencies(a):
    c=module('code/c15_paper_core_v001.py',CORE_SHA,'c18_initialization_old_core',a)
    h=module('code/combination_c12_fit_rc_v001.py',HELPER_SHA,'c18_initialization_old_features',a)
    d=module('linear-followup-20261111/code/linear_diagnostic_v002.py',DIAGNOSTIC_SHA,'c18_initialization_diagnostic_math',a)
    if h.SPANS!=SPANS or h.WARMUP!=300 or h.STRIDE!=5:raise ValueError('Frozen feature geometry')
    return c,h,d


def changed_record(saved,arm,h,a):
    instant=np.asarray(saved['instant']);n=len(instant)
    if instant.shape!=(n,8) or n<=300:raise ValueError('Original full profile and eight instantaneous columns')
    q=instant[:,[1,4,5]];boundary_input=instant[:,[7,6]]
    legacy_b=np.column_stack([h.filtered(boundary_input,s,np.repeat(instant[0,7],2)) for s in SPANS])
    legacy_h=np.column_stack([h.filtered(q,s,np.zeros(3)) for s in SPANS])
    a.close(saved['boundary'],legacy_b,'Original boundary source recreation',atol=0,rtol=0)
    a.close(saved['heat'],legacy_h,'Original heat source recreation',atol=0,rtol=0)
    b=legacy_b.copy();heat=legacy_h.copy();identity_errors=[]
    if arm=='own_boundary':
        b=np.column_stack([h.filtered(boundary_input,s,boundary_input[0]) for s in SPANS])
        delta0=boundary_input[0]-instant[0,7]
        for j,s in enumerate(SPANS):
            expected=(1.-2./(s+1))**np.arange(n,dtype=np.float64)[:,None]*delta0
            actual=b[:,2*j:2*j+2]-legacy_b[:,2*j:2*j+2]
            a.close(actual,expected,'Boundary initialization difference a^t delta0',atol=1e-8,rtol=2e-10)
            identity_errors.append(float(np.max(np.abs(actual-expected))))
        if not np.array_equal(heat,saved['heat']):raise ValueError('Heat is unchanged in boundary-only arm')
    elif arm=='first_load':
        heat=np.column_stack([h.filtered(q,s,q[0]) for s in SPANS])
        for j,s in enumerate(SPANS):
            expected=(1.-2./(s+1))**np.arange(n,dtype=np.float64)[:,None]*q[0]
            actual=heat[:,3*j:3*j+3]-legacy_h[:,3*j:3*j+3]
            # Large squared-speed drives require a bound scaled by the drive,
            # including cancellation when the expected transient approaches0.
            bound=2e-10*(1.+np.maximum(np.max(np.abs(q),axis=0),np.abs(q[0])))
            if np.any(np.abs(actual-expected)>bound[None,:]):raise ArithmeticError('Heat initialization difference a^t delta0')
            a.checks+=1;identity_errors.append(float(np.max(np.abs(actual-expected))))
        if not np.array_equal(b,saved['boundary']):raise ValueError('Boundary is unchanged in heat-only arm')
    else:raise ValueError('Only the two frozen single-factor arms')
    out=dict(boundary=b,heat=heat,instant=instant,ewma=saved.get('ewma'))
    if 'target' in saved:out['y']=saved['target']
    if not all(np.isfinite(out[k]).all() for k in ('boundary','heat','instant')):raise ValueError('Finite changed features')
    return out,dict(rows=n,arm=arm,boundary_observed_initial=boundary_input[0].tolist(),heat_observed_initial=q[0].tolist(),
                    maximum_difference_identity_errors=identity_errors,unchanged_channels_checked=True,legacy_recreated_exactly=True)


def synthetic(h,a):
    # No dataset values: arbitrary three-channel drives and two initial states.
    t=np.arange(97,dtype=np.float64);u=np.column_stack((1.+np.sin(t/7),t/11,2.-t/19))
    z0=np.array([.4,-2.,3.]);z1=np.array([1.7,.2,-.5]);checks=[]
    for s in SPANS:
        left=h.filtered(u,s,z1)-h.filtered(u,s,z0)
        right=(1.-2./(s+1))**t[:,None]*(z1-z0)
        a.close(left,right,'Data-free initial-state exact recurrence identity',atol=1e-12,rtol=1e-12)
        checks.append(dict(span=s,max_error=float(np.max(abs(left-right)))))
    return dict(status='DATA_FREE_INITIALIZATION_IDENTITY_PASS',checks=checks)


def configuration(cfg,a,c):
    ds=cfg['dataset']
    if cfg['schema']!='c18-initialization-config-v002' or ds not in ROLES or cfg['arm'] not in ARMS:raise ValueError('Frozen dataset/arm')
    if cfg['producer']!=ref(Path(__file__)) or cfg['protocol']['sha256']!=PROTOCOL_SHA:raise ValueError('Current source and fixed protocol')
    a.pin(cfg['producer']);a.pin(cfg['protocol']);limits=read(a.pin(cfg['limits']))
    if limits['new_deadline_utc']!='2026-11-12T00:23:59Z' or limits['new_training_cutoff_utc']!='2026-11-11T20:23:59Z':raise ValueError('Original extended limits')
    if cfg['methods']!=METHODS or cfg['weight']!=.5 or cfg['warmup']!=300 or cfg['stride']!=5 or cfg['progress_blocks']!=5 or cfg['formal_confirmation_allowed'] is not False:raise ValueError('No scientific selection changes')
    risk=read(a.pin(cfg['source_config']))
    if risk['dataset']!=ds or risk['train_ids']!=cfg['fit_ids'] or risk['fit_cache']!=cfg['fit_cache']:raise ValueError('Exact old FIT pool')
    if [r['role'] for r in cfg['roles']]!=ROLES[ds]:raise ValueError('All old exposed roles retained')
    oldroles={r['role']:r for r in risk['role_inputs']}
    for r in cfg['roles']:
        old=oldroles[r['role']]
        if r['ids']!=old['ids'] or r['records']!=old['records'] or r['prediction_lock']!=old['prediction_lock']:raise ValueError('Exact old held pools')
    lock=read(a.audited(cfg['old_model_lock'],cfg['selection_audit']))
    expected='direct_boost_7x500' if ds=='ipmsm' else 'direct_boost_15x1000'
    if lock['fit_ids']!=cfg['fit_ids'] or lock['selected_direct9']!=expected or cfg['d9_method']!=expected:raise ValueError('Frozen prior FIT selection')
    for name,job in [('linear','linear_final__linear33__tuned'),('d9','full__'+expected)]:
        rr=cfg['old_'+name+'_receipt'];rp=a.audited(rr,cfg['selection_audit']);receipt=read(rp)
        if local(lock['entries'][job]['path'])!=rp.parent or lock['entries'][job]['receipt_sha256']!=rr['sha256'] or receipt['job']['job_id']!=job or receipt['status']!='FIT_JOB_COMPLETE':raise ValueError('Exact old selected job receipt')
        md=cfg['old_'+name+'_model'];mp=a.audited(md,cfg['selection_audit'])
        if mp.parent!=rp.parent or receipt['files']['model.json']!=md['sha256']:raise ValueError('Exact old selected model receipt binding')
        model=read(mp)
        if model['train_ids']!=cfg['fit_ids']:raise ValueError('Old model exact FIT IDs')
        if name=='linear':
            if model['linear_model']['family']!='linear33' or model['linear_model']['lam']!=cfg['lambda']:raise ValueError('Reused FIT-selected lambda, no reselection')
        else:
            if model['configuration']!=cfg['d9_configuration']:raise ValueError('Reused selected D9 configuration')
    grid=dict(c.GRID[c.DIRECT.index(expected)])
    if dict(c.BASE,**grid)!=cfg['d9_configuration'] or cfg['residual_configuration']!=dict(c.BASE,**c.GRID[0]):raise ValueError('Unchanged booster parameters')
    return grid


def fit_predict(cfg,a,out,c,h,d):
    grid=configuration(cfg,a,c);dump(out/'synthetic_identity.json',synthetic(h,a))
    records={};provenance=[]
    for e in cfg['fit_cache']:
        z=a.arrays(e,cfg['fit_audit'],['boundary','heat','instant','ewma','target'])
        if len(z['target'])!=e['rows']:raise ValueError('Full FIT row count')
        r,info=changed_record(z,cfg['arm'],h,a);records[e['id']]=r;provenance.append(dict(id=e['id'],**info))
    ids=cfg['fit_ids'];a.training=True;a.check()
    linear=c.fit_linear(h,records,ids,'linear33',cfg['lambda'],a)
    x,y,weights,scale,indices=c.training(h,records,ids)
    if linear['scale']['qscale']!=scale['qscale']:raise ValueError('Common FIT-only driving scale')
    prior=np.concatenate([c.predict_linear(h,linear,records[p])[indices[p]] for p in ids]);residual=y-prior
    featuremean=np.asarray(linear['scale']['linear33_mean']);featurestd=np.asarray(linear['scale']['linear33_scale'])
    design=np.column_stack((np.ones(len(y)),(x-featuremean)/featurestd))
    yy=(y-linear['scale']['y_mid'])/linear['scale']['y_scale']
    ww=h.balanced_weights([len(indices[p]) for p in ids]);penalty=np.asarray(linear['penalty'])
    q=design.T@(ww[:,None]*design)+cfg['lambda']*np.diag(penalty);r=design.T@(ww*yy);constant=.5*np.dot(ww,yy*yy)
    np.savez_compressed(out/'training.npz',features=x,target=y,weight=weights,
        profile_id=np.concatenate([np.full(len(indices[p]),p,dtype=np.int64) for p in ids]),
        index=np.concatenate([indices[p].astype(np.int64) for p in ids]),linear_design=design,linear_target=yy,
        linear_weight=ww,prior_prediction=prior,residual_target=residual,Q=q,r=r,constant=np.asarray(constant),penalty=penalty)
    dump(out/'linear_model.json',linear);dump(out/'fit_initialization.json',provenance)
    boosters={};modelmeta={}
    for name,target,g in [('d9',y,grid),('residual',residual,c.GRID[0])]:
        folder=out/name;folder.mkdir();a.check()
        booster,info,binidx=c.fit_booster(h,x,target,weights,g,folder,a)
        np.savez_compressed(folder/'binning_indices.npz',index=binidx.astype(np.int64))
        with (folder/'booster.pkl').open('xb') as f:pickle.dump(booster,f,protocol=5)
        info.update(kind=name,training=ref(out/'training.npz'),binning_indices=ref(folder/'binning_indices.npz'),
                    pickle=ref(folder/'booster.pkl'),numpy=np.__version__,sklearn=sklearn.__version__)
        dump(folder/'model.json',info);boosters[name]=booster;modelmeta[name]=ref(folder/'model.json')
    a.training=False
    dump(out/'fit_inputs.json',a.inputs)
    lock=dict(schema='c18-initialization-model-lock-v001',status='ALL_MODELS_LOCKED',dataset=cfg['dataset'],arm=cfg['arm'],
        source_sha256=sha(__file__),core_sha256=CORE_SHA,helper_sha256=HELPER_SHA,config=ref(local(cfg['_config_path'])),
        fit_ids=ids,methods=METHODS,actual_booster_fits=2,actual_linear_fits=1,linear=ref(out/'linear_model.json'),boosters=modelmeta,
        held_truth_read=False,files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    dump(out/'model_lock.json',lock)
    predrecords=[]
    for role in cfg['roles']:
        a.audited(role['prediction_lock'],role['audit'])
        for e in role['records']:
            z=a.arrays(e['features'],role['audit'],['boundary','heat','instant','index','source_row'])
            record,info=changed_record(z,cfg['arm'],h,a);ix=z['index'];n=len(record['instant'])
            if not np.array_equal(ix,np.arange(300,n)) or e['rows']!=n or len(ix)!=e['rows']-300:raise ValueError('Original full-row metadata and every retained evaluation endpoint')
            features=h.boost_features(record,scale)[ix];pp=c.predict_linear(h,linear,record)[ix]
            dd=boosters['d9'].predict(features);rr=pp+boosters['residual'].predict(features)
            values=dict(d9=dd,linear_prior=pp,linear_residual=rr,linear_half=.5*(dd+rr),prior_half=.5*(dd+pp))
            if not all(v.shape==(len(ix),) and np.isfinite(v).all() for v in values.values()):raise ValueError('Finite full role predictions')
            folder=out/role['role'];folder.mkdir(exist_ok=True);path=folder/('profile_'+str(e['id'])+'.npz')
            np.savez_compressed(path,index=ix.astype(np.int64),source_row=z['source_row'][ix].astype(np.int64),features=features,
                boundary=record['boundary'],heat=record['heat'],instant=record['instant'],**{'prediction__'+k:v for k,v in values.items()})
            predrecords.append(dict(role=role['role'],id=e['id'],rows=len(ix),source_rows=n,prediction=ref(path),source_features=e['features'],
                truth=e['truth'],initialization=info))
            a.check()
    dump(out/'prediction_lock.json',dict(schema='c18-initialization-prediction-lock-v001',status='ALL_ROLE_PREDICTIONS_LOCKED',
        dataset=cfg['dataset'],arm=cfg['arm'],config=ref(local(cfg['_config_path'])),source_sha256=sha(__file__),
        model_lock=ref(out/'model_lock.json'),methods=METHODS,records=predrecords,held_truth_read=False))
    return dict(status='C18_INITIALIZATION_FIT_AND_PREDICT_COMPLETE',dataset=cfg['dataset'],arm=cfg['arm'],
        actual_booster_fits=2,actual_linear_fits=1,fit_rows=len(y),held_profiles=len(predrecords),held_truth_read=False,
        linear_fit_seconds=linear['seconds'],booster_fit_seconds={k:read(local(v['path']))['fit_seconds'] for k,v in modelmeta.items()},
        prediction_lock=ref(out/'prediction_lock.json'),model_lock=ref(out/'model_lock.json'))


def profile_statistics(y,values,ix,a,d):
    methods={k:d.measures(y,values[k]) for k in METHODS}
    whole=dict(methods=methods,decomposition=d.decomposition(y,values['d9'],values['linear_residual'],a),
               residual_effect=d.residual_effect(y,values['linear_prior'],values['linear_residual'],a))
    blocks=[];n=len(y);denominator=whole['decomposition']['rmse_denominator']
    for b in range(5):
        lo=n*b//5;hi=n*(b+1)//5
        if lo==hi:raise ValueError('Frozen cohorts all have at least five evaluation rows')
        vv={k:v[lo:hi] for k,v in values.items()};yy=y[lo:hi]
        dec=d.decomposition(yy,vv['d9'],vv['linear_residual'],a)
        blocks.append(dict(bin=b,n=hi-lo,index_start=int(ix[lo]),index_end_inclusive=int(ix[hi-1]),
            methods={k:d.measures(yy,v) for k,v in vv.items()},decomposition=dec,
            residual_effect=d.residual_effect(yy,vv['linear_prior'],vv['linear_residual'],a),
            primary_delta_rmse_contribution=(hi-lo)/n*dec['delta_mse']/denominator if denominator else 0.))
    a.close(sum(v['primary_delta_rmse_contribution'] for v in blocks),whole['decomposition']['delta_rmse'],'Block contributions add to whole profile')
    return dict(whole=whole,progress_blocks=blocks)


def summary_rows(rows,key):
    output={}
    for method in METHODS:
        entries=[r[key]['whole']['methods'][method] for r in rows];n=sum(v['n'] for v in entries)
        output[method]=dict(profiles=len(rows),n=n,**{'macro_'+k:float(np.mean([v[k] for v in entries])) for k in ['rmse','mae','bias','under_rmse','over_mae']},
            pooled_rmse=math.sqrt(sum(v['n']*v['mse'] for v in entries)/n))
    return output


def evaluate(cfg,a,out,c,h,d,prediction_ref):
    configuration(cfg,a,c)
    lock=read(a.pin(prediction_ref))
    if lock['schema']!='c18-initialization-prediction-lock-v001' or lock['source_sha256']!=sha(__file__) or lock['config']!=ref(local(cfg['_config_path'])) or lock['methods']!=METHODS or lock['held_truth_read'] is not False:raise ValueError('Exact completed prediction lock')
    ml=read(a.pin(lock['model_lock']));root=local(lock['model_lock']['path']).parent
    if ml['config']!=lock['config'] or ml['dataset']!=cfg['dataset'] or ml['arm']!=cfg['arm'] or ml['source_sha256']!=sha(__file__):raise ValueError('Exact model lock')
    for name,digest in ml['files'].items():a.pin(dict(path=str(root/name),sha256=digest))
    expected=[(r['role'],e['id']) for r in cfg['roles'] for e in r['records']]
    if [(r['role'],r['id']) for r in lock['records']]!=expected:raise ValueError('All frozen role records before any truth read')
    rolemap={r['role']:r for r in cfg['roles']}
    for e in lock['records']:
        original=next(v for v in rolemap[e['role']]['records'] if v['id']==e['id'])
        if e['source_features']!=original['features'] or e['truth']!=original['truth'] or e['source_rows']!=original['rows'] or e['rows']!=e['source_rows']-300:raise ValueError('Original labels, full source rows and retained evaluation-row lineage')
        a.pin(e['prediction'])
    # All current models and all role predictions are pinned above; only now
    # are existing exposed evaluation targets decoded.
    rows=[]
    for e in lock['records']:
        role=rolemap[e['role']];pred=a.arrays(e['prediction']);truth=a.arrays(e['truth'],role['audit'],['index','target'])
        if not np.array_equal(truth['index'],pred['index']):raise ValueError('Prediction/truth endpoint alignment')
        y=truth['target'];baseline=a.arrays(e['source_features'],role['audit'],['index','source_row']+['prediction__'+v for v in BASE_KEYS.values()])
        if not np.array_equal(baseline['index'],pred['index']) or not np.array_equal(baseline['source_row'][pred['index']],pred['source_row']):raise ValueError('Baseline original source row alignment')
        old={k:baseline['prediction__'+v] for k,v in BASE_KEYS.items()};old['linear_half']=(old['d9']+old['linear_residual'])/2.;old['prior_half']=(old['d9']+old['linear_prior'])/2.
        values={k:pred['prediction__'+k] for k in METHODS};new=profile_statistics(y,values,pred['index'],a,d);original=profile_statistics(y,old,pred['index'],a,d)
        change={k:new['whole']['methods'][k]['rmse']-original['whole']['methods'][k]['rmse'] for k in METHODS}
        relative_change=new['whole']['decomposition']['delta_rmse']-original['whole']['decomposition']['delta_rmse']
        row=dict(dataset=cfg['dataset'],arm=cfg['arm'],role=e['role'],profile=e['id'],intervention=new,legacy=original,
            rmse_change_vs_legacy=change,half_minus_d9_change_vs_legacy=relative_change,
            block_contribution_changes=[n['primary_delta_rmse_contribution']-o['primary_delta_rmse_contribution'] for n,o in zip(new['progress_blocks'],original['progress_blocks'])])
        folder=out/e['role'];folder.mkdir(exist_ok=True);path=folder/('profile_'+str(e['id'])+'.json');dump(path,row);rows.append(row)
    results=[]
    for role in cfg['roles']:
        rr=[r for r in rows if r['role']==role['role']];new=summary_rows(rr,'intervention');old=summary_rows(rr,'legacy')
        result=dict(dataset=cfg['dataset'],arm=cfg['arm'],role=role['role'],ids=role['ids'],intervention=new,legacy=old,
            half_minus_d9=new['linear_half']['macro_rmse']-new['d9']['macro_rmse'],
            legacy_half_minus_d9=old['linear_half']['macro_rmse']-old['d9']['macro_rmse'],
            half_minus_d9_change_vs_legacy=float(np.mean([r['half_minus_d9_change_vs_legacy'] for r in rr])),
            per_profile=[dict(profile=r['profile'],rmse_change_vs_legacy=r['rmse_change_vs_legacy'],half_minus_d9_change_vs_legacy=r['half_minus_d9_change_vs_legacy']) for r in rr])
        dump(out/role['role']/'summary.json',result);results.append(result)
    dump(out/'results.json',results)
    return dict(status='C18_INITIALIZATION_EVALUATION_COMPLETE',dataset=cfg['dataset'],arm=cfg['arm'],profiles=len(rows),
        roles=len(results),methods=METHODS,prediction_lock=prediction_ref,new_training_fits=0,
        all_data_previously_exposed=True,formal_test_performed=False,independent_confirmation=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['fit-predict','evaluate'],required=True)
    for key in ('config','config-sha256','output-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--prediction-lock');p.add_argument('--prediction-lock-sha256');p.add_argument('--max-seconds',type=float,default=1200.)
    args=p.parse_args()
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=1200:raise ValueError('Finite registered stage cap <=1200s')
    out=local(args.output_dir)
    if WORK/'runs' not in out.parents:raise ValueError('New C18 run directory required')
    out.mkdir(parents=True,exist_ok=False);a=Intake(out,args.max_seconds)
    dump(out/'configuration.json',dict(vars(args),source=str(Path(__file__).resolve()),source_sha256=sha(__file__),pid=os.getpid(),argv=sys.argv))
    try:
        cfg=read(a.pin(dict(path=args.config,sha256=args.config_sha256)));cfg['_config_path']=args.config
        c,h,d=dependencies(a)
        if args.mode=='fit-predict':result=fit_predict(cfg,a,out,c,h,d)
        else:
            if not args.prediction_lock or not args.prediction_lock_sha256:raise ValueError('Explicit saved prediction lock required')
            result=evaluate(cfg,a,out,c,h,d,dict(path=args.prediction_lock,sha256=args.prediction_lock_sha256))
        dump(out/'input_pins.json',a.inputs);a.check()
        result.update(source_sha256=sha(__file__),config=dict(path=args.config,sha256=args.config_sha256),identity_checks=a.checks,
            elapsed_seconds=time.monotonic()-a.start,files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
        dump(out/'summary.json',result);a.check();print(json.dumps(dict(status=result['status'],dataset=cfg['dataset'],arm=cfg['arm'],elapsed_seconds=result['elapsed_seconds'])))
    except Exception as exc:
        dump(out/'failure.json',dict(status='FAILED',error_type=type(exc).__name__,message=str(exc)));raise


if __name__=='__main__':main()
