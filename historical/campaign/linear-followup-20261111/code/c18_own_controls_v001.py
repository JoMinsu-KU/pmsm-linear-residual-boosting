"""Fixed own_boundary strong controls. Root alone registers/runs every stage.

Reuses independently audited own_boundary FIT/held features and baseline scores.
CPU and CUDA stages are separate processes. No new model selection, new data,
confirmation reset, calibration, scheduler, or early stopping occurs.
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
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.dont_write_bytecode=True
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'linear-followup-20261111'
PROTOCOL_SHA='00aee40005399af1df692671e757d998f937eeb939817a5251b6ff478a92de0d'
RUNTIME_SHA='a5397df92507fb59736448d44dc1906e2ee0cbae90580ad83e90e150067e4d4b'
SOURCES={
 'code/c15_paper_core_v001.py':'17c47c41c27b6e92618a44de0f5860f80ffd1357d998a4cfeb805f5e2d8b7b37',
 'code/combination_c12_fit_rc_v001.py':'d4aa942ca9db8d8e9319197632cf376730456f3ea5cdc48b9d1c1fd51fc59412',
 'code/c15_paper_mlp_v001.py':'4ecd405a408a5ec34a20103390f99893ac3264ab28600de0bbde491b58919dbb',
 'code/c15_neural_rc_core_v001.py':'52d897657163391de6e1e42a51dad04550d5ca6f1dfc2a60f2e0041ec3f68a86',
 'followup-risk-20261110/code/risk_mlp_v001.py':'54ec228ba97251b1b427845a57c8bd0f906729cab4d950e798eb8ffd9bc5cd7f',
 'linear-followup-20261111/code/initialization_intervention_v002.py':'63108ce0700e7646dce34663b88542b4273eab052fd5bbb9d30d7d6aea39b046',
 'linear-followup-20261111/code/linear_diagnostic_v002.py':'d799863cf6c7e19db7604a96c409bb70835c06ed9ad53871fc9ead5b510e3434'}
BASE_METHODS=['d9','linear_prior','linear_residual','linear_half','prior_half']
NEW_CPU=['direct200','direct200_half','capacity_direct','free_prior','free_residual','free_half']
CPU_METHODS=BASE_METHODS+NEW_CPU
SEEDS=[2026111701,2026111702,2026111703]
MLP_METHODS=['mse_mlp_'+str(s) for s in SEEDS]
METHODS=CPU_METHODS+MLP_METHODS
BOOST_BASE=dict(learning_rate=.05,min_samples_leaf=20,l2_regularization=.1,loss='squared_error',
                early_stopping=False,random_state=2026111201)
DIRECT200=dict(BOOST_BASE,max_leaf_nodes=15,max_iter=200)
MLP_SETTINGS=dict(architecture=[33,64,32,1],parameters=4289,activation='ReLU',loss='mse',epochs=200,
    seeds=SEEDS,optimizer='AdamW',learning_rate=.001,betas=[.9,.999],eps=1e-8,weight_decay=.0001,
    batch_size=4096,gradient_norm_cap=1.,warmup=300,label_stride=5,feature_std_floor=1e-8,target_iqr_floor=1.,
    permutation='NumPy PCG64(seed+epoch-1); full permutation; explicit int64',
    profile_weight='N/(P*n_profile); weighted minibatch mean including final partial batch',
    precision='FP64 transforms, FP32 CUDA network and weights, FP64 physical outputs',
    scheduler=None,early_stopping=False,checkpoint='last epoch200',
    seed_aggregation='mean of seed metrics; no seed selection or prediction ensemble')


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
    if not p.is_relative_to(ROOT):raise ValueError('Campaign-local path required')
    return p


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def ref(p):return dict(path=local(p).relative_to(ROOT).as_posix(),sha256=sha(p))
def dump(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


class Intake:
    def __init__(self,out,cap):
        self.out=out;self.cap=cap;self.start=time.monotonic();self.inputs={};self.audit_cache={};self.training=False;self.checks=0
    def check(self):
        if time.monotonic()-self.start>self.cap:raise TimeoutError('Registered stage cap; preserve partial output')
        now=datetime.now(timezone.utc)
        if now>=datetime.fromisoformat('2026-11-12T00:23:59+00:00'):raise TimeoutError('Campaign deadline')
        if self.training and now>=datetime.fromisoformat('2026-11-11T20:23:59+00:00'):raise TimeoutError('New fitting cutoff')
        if sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())>1000000000:raise RuntimeError('1GB stage output cap')
    def pin(self,e):
        self.check();p=local(e['path']);digest=sha(p)
        if digest!=e['sha256'] or (str(p) in self.inputs and self.inputs[str(p)]!=digest):raise ValueError('Pinned artifact changed: '+str(p))
        self.inputs[str(p)]=digest;return p
    def audit(self,e):
        p=self.pin(e);key=(str(p),e['sha256'])
        if key not in self.audit_cache:
            doc=read(p);identities=[doc[k] for k in ('auditor_sha256','source_sha256') if k in doc]
            if doc.get('status')!='PASS' or doc.get('failures') or not identities or any(v!=e['auditor_sha256'] for v in identities):raise ValueError('Exact independent PASS required')
            self.pin(dict(path=e['auditor_path'],sha256=e['auditor_sha256']))
            index={}
            for name,digest in doc['inputs'].items():index.setdefault(str(local(name)),set()).add(digest)
            self.audit_cache[key]=(doc,index)
        return self.audit_cache[key]
    def audited(self,e,audit):
        p=self.pin(e);_,index=self.audit(audit)
        if index.get(str(p))!={e['sha256']}:raise ValueError('Artifact absent/conflicting in prior independent audit: '+str(p))
        return p
    def arrays(self,e,audit=None,keys=None):
        p=self.audited(e,audit) if audit else self.pin(e)
        with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (keys if keys is not None else z.files)}
    def close(self,x,y,label,atol=1e-10,rtol=1e-12):
        self.checks+=1
        if np.shape(x)!=np.shape(y) or not np.allclose(x,y,atol=atol,rtol=rtol,equal_nan=False):raise ArithmeticError(label)


def module(rel,name,a):
    p=a.pin(dict(path=rel,sha256=SOURCES[rel]));spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def configuration(cfg,a):
    if (cfg['schema']!='c18-own-controls-config-v001' or cfg['dataset'] not in ['ipmsm','paderborn'] or cfg['arm']!='own_boundary'
        or cfg['scientific_execution_admitted'] is not True or cfg['formal_confirmation_allowed'] is not False
        or cfg['all_data_previously_exposed'] is not True or cfg['producer']!=ref(__file__)
        or cfg['protocol']['sha256']!=PROTOCOL_SHA or cfg['runtime_addendum']['sha256']!=RUNTIME_SHA
        or cfg['methods']!=METHODS or cfg['mlp_settings']!=MLP_SETTINGS or cfg['source_dependencies']!=SOURCES):
        raise ValueError('Frozen own-controls scientific configuration')
    a.pin(cfg['producer']);a.pin(cfg['protocol']);a.pin(cfg['runtime_addendum']);limits=read(a.pin(cfg['limits']))
    if limits['new_deadline_utc']!='2026-11-12T00:23:59Z' or limits['new_training_cutoff_utc']!='2026-11-11T20:23:59Z':raise ValueError('Original extended limits')
    for name,digest in SOURCES.items():a.pin(dict(path=name,sha256=digest))
    old=read(a.audited(cfg['own_config'],cfg['own_audit']))
    if old['dataset']!=cfg['dataset'] or old['arm']!='own_boundary' or old['fit_ids']!=cfg['fit_ids'] or old['roles']!=cfg['roles']:raise ValueError('Exact own baseline role/feature sources')
    lock=read(a.audited(cfg['own_model_lock'],cfg['own_audit']))
    pred=read(a.audited(cfg['own_prediction_lock'],cfg['own_audit']))
    if (lock['schema']!='c18-initialization-model-lock-v001' or lock['status']!='ALL_MODELS_LOCKED'
        or lock['source_sha256']!=SOURCES['linear-followup-20261111/code/initialization_intervention_v002.py']
        or lock['config']!=cfg['own_config'] or lock['dataset']!=cfg['dataset'] or lock['arm']!='own_boundary'
        or lock['fit_ids']!=cfg['fit_ids'] or lock['held_truth_read'] is not False
        or pred['schema']!='c18-initialization-prediction-lock-v001' or pred['status']!='ALL_ROLE_PREDICTIONS_LOCKED'
        or pred['source_sha256']!=lock['source_sha256'] or pred['model_lock']!=cfg['own_model_lock']
        or pred['config']!=cfg['own_config'] or pred['methods']!=BASE_METHODS or pred['held_truth_read'] is not False):
        raise ValueError('Audited immutable own model/prediction lock contract')
    folder=local(cfg['own_model_lock']['path']).parent
    for name,digest in lock['files'].items():
        p=(folder/name).resolve()
        if not p.is_relative_to(folder):raise ValueError('Own model artifact path escape')
        a.pin(dict(path=str(p),sha256=digest))
    if cfg['training']!=ref(folder/'training.npz') or cfg['own_linear_model']!=lock['linear']:raise ValueError('Own training/linear artifact identity')
    a.audited(cfg['training'],cfg['own_audit']);linear=read(a.audited(cfg['own_linear_model'],cfg['own_audit']))
    if linear['family']!='linear33' or linear['scale']['ids']!=cfg['fit_ids']:raise ValueError('Old own linear33 FIT normalization')
    expected=[(r['role'],e['id']) for r in cfg['roles'] for e in r['records']]
    if [(e['role'],e['id']) for e in pred['records']]!=expected:raise ValueError('All held own records before any new fitting')
    original={(r['role'],e['id']):e for r in cfg['roles'] for e in r['records']}
    for e in pred['records']:
        prior=original[(e['role'],e['id'])]
        if e['source_rows']!=prior['rows'] or e['rows']!=prior['rows']-300 or e['truth']!=prior['truth'] or e['source_features']!=prior['features']:raise ValueError('Source rows/retained rows/truth lineage')
        a.audited(e['prediction'],cfg['own_audit'])
    oldlock=read(a.audited(cfg['old_model_lock'],cfg['selection_audit']))
    freereceipt=read(a.audited(cfg['old_free_receipt'],cfg['selection_audit']))
    freemodel=read(a.audited(cfg['old_free_model'],cfg['selection_audit']))
    job='linear_final__free__tuned';entry=oldlock['entries'][job]
    if (oldlock['fit_ids']!=cfg['fit_ids'] or entry['receipt_sha256']!=cfg['old_free_receipt']['sha256']
        or local(entry['path'])!=local(cfg['old_free_receipt']['path']).parent
        or local(cfg['old_free_model']['path']).parent!=local(entry['path'])
        or freereceipt['status']!='FIT_JOB_COMPLETE' or freereceipt['job']['job_id']!=job
        or freereceipt['files']['model.json']!=cfg['old_free_model']['sha256'] or freemodel['train_ids']!=cfg['fit_ids']
        or freemodel['linear_model']['family']!='free' or len(freemodel['linear_model']['coef'])!=26
        or freemodel['linear_model']['lam']!=cfg['free_lambda']):raise ValueError('Original FIT-selected free25+intercept26 lineage')
    if cfg['free_lambda'] not in [.0001,.01,1.] or cfg['direct200_configuration']!=DIRECT200:raise ValueError('Fixed free lambda/direct200 parameters')
    d9=old['d9_configuration'];capacity=dict(d9,max_iter=d9['max_iter']+math.ceil(200*14/(d9['max_leaf_nodes']-1)))
    if cfg['d9_configuration']!=d9 or cfg['capacity_configuration']!=capacity:raise ValueError('Original fixed capacity rule')
    for key in ['qscale','y_mid','y_scale']:
        a.close(linear['scale'][key],freemodel['linear_model']['scale'][key],'Own and original FIT unchanged drive/target statistics')
    if [e['seed'] for e in cfg['original_mlp']]!=SEEDS:raise ValueError('All original fixed MSE seed references')
    for entry in cfg['original_mlp']:
        original=read(a.audited(entry['model'],entry['audit']))
        a.audited(entry['initial_weights'],entry['audit']);a.audited(entry['epoch_trace'],entry['audit'])
        if (original['source_sha256']!=SOURCES['followup-risk-20261110/code/risk_mlp_v001.py']
            or original['dataset']!=cfg['dataset'] or original['seed']!=entry['seed'] or original['loss']!='mse'
            or original['epochs']!=200 or original['fit_ids']!=cfg['fit_ids']):raise ValueError('Original completed matched MSE training identity')
        for name in ['architecture','parameters','activation','optimizer','learning_rate','betas','eps','weight_decay','batch_size','gradient_norm_cap']:
            if original['settings'][name]!=MLP_SETTINGS[name]:raise ValueError('Original MSE optimization setting changed: '+name)
    return old,lock,pred,linear


def fit_arrays(cfg,a,ownlinear):
    z=a.arrays(cfg['training'],cfg['own_audit'],['features','target','weight','profile_id','index'])
    x,y,w,p,ix=(z[k] for k in ['features','target','weight','profile_id','index'])
    n=len(y)
    if x.dtype!=np.float64 or y.dtype!=np.float64 or w.dtype!=np.float64 or x.shape!=(n,33) or w.shape!=(n,) or p.shape!=(n,) or ix.shape!=(n,):raise ValueError('Exact full FIT cache geometry/dtype')
    if not all(np.isfinite(v).all() for v in z.values()) or not np.issubdtype(p.dtype,np.integer) or not np.issubdtype(ix.dtype,np.integer):raise ValueError('Finite FIT arrays and integer identity')
    old=read(a.pin(cfg['own_config']));counts=[];parts=[];indices=[]
    for entry in old['fit_cache']:
        ii=np.arange(300,entry['rows'],5,dtype=np.int64);counts.append(len(ii));parts.append(np.full(len(ii),entry['id'],dtype=np.int64));indices.append(ii)
    if [e['id'] for e in old['fit_cache']]!=cfg['fit_ids'] or not np.array_equal(p,np.concatenate(parts)) or not np.array_equal(ix,np.concatenate(indices)):raise ValueError('Exact sampled FIT row identity/order')
    expected=np.concatenate([np.full(k,n/(len(counts)*k),dtype=np.float64) for k in counts])
    a.close(w,expected,'Mean-one exact profile weights',atol=1e-12,rtol=1e-13)
    a.close(np.median(y),ownlinear['scale']['y_mid'],'FIT target median')
    a.close(max(float(np.percentile(y,75)-np.percentile(y,25)),1.),ownlinear['scale']['y_scale'],'FIT target IQR')
    return x,y,w,p.astype(np.int64),ix.astype(np.int64)


def held_arrays(cfg,entry,a):
    z=a.arrays(entry['prediction'],cfg['own_audit'],['index','source_row','features']+['prediction__'+m for m in BASE_METHODS])
    n=entry['rows'];x=z['features']
    if x.dtype!=np.float64 or x.shape!=(n,33) or not np.array_equal(z['index'],np.arange(300,entry['source_rows'],dtype=np.int64)):
        raise ValueError('All retained own held feature rows')
    if z['source_row'].shape!=(n,) or any(z['prediction__'+m].shape!=(n,) for m in BASE_METHODS) or not all(np.isfinite(v).all() for v in z.values()):raise ValueError('Finite own baseline output/row geometry')
    a.close(z['prediction__linear_half'],.5*(z['prediction__d9']+z['prediction__linear_residual']),'Preserved H identity',atol=0,rtol=0)
    return z


def free_design(x,scale):
    return np.column_stack((np.ones(len(x)),(x[:,:10]-scale['y_mid'])/scale['y_scale'],x[:,10:25]))


def free_predict(model,x):
    return model['scale']['y_mid']+model['scale']['y_scale']*(free_design(x,model['scale'])@np.asarray(model['coef']))


def save_lock(out,cfg,kind,extra):
    value=dict(schema='c18-own-controls-model-lock-v001',status='ALL_MODELS_LOCKED',dataset=cfg['dataset'],arm='own_boundary',
        kind=kind,source_sha256=sha(__file__),protocol=cfg['protocol'],config=cfg['_ref'],fit_ids=cfg['fit_ids'],
        held_truth_read=False,files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()},**extra)
    dump(out/'model_lock.json',value);return ref(out/'model_lock.json')


def prediction_lock(out,cfg,kind,model,methods,records,seed=None):
    value=dict(schema='c18-own-controls-prediction-lock-v001',status='ALL_ROLE_PREDICTIONS_LOCKED',dataset=cfg['dataset'],arm='own_boundary',
        kind=kind,seed=seed,source_sha256=sha(__file__),protocol=cfg['protocol'],config=cfg['_ref'],model_lock=model,
        methods=methods,records=records,held_truth_read=False)
    dump(out/'prediction_lock.json',value);return ref(out/'prediction_lock.json')


def cpu_fit(cfg,a,out,priorlock,ownlinear):
    # These pinned CPU imports intentionally hide CUDA; never execute on MLP path.
    c=module('code/c15_paper_core_v001.py','c18_own_cpu_core',a)
    h=module('code/combination_c12_fit_rc_v001.py','c18_own_cpu_helper',a)
    import sklearn
    x,y,w,profiles,index=fit_arrays(cfg,a,ownlinear)
    scale={k:ownlinear['scale'][k] for k in ['qscale','y_mid','y_scale']};scale['ids']=cfg['fit_ids']
    design=free_design(x,scale);target=(y-scale['y_mid'])/scale['y_scale']
    # Preserve the original C12 arithmetic: sum-one linear weights are built
    # independently, rather than dividing already rounded mean-one HGB weights.
    linear_w=h.balanced_weights([int(np.sum(profiles==pid)) for pid in cfg['fit_ids']])
    penalty=np.r_[0.,np.ones(25)];lam=cfg['free_lambda']
    q=design.T@(linear_w[:,None]*design)+lam*np.diag(penalty);r=design.T@(linear_w*target);constant=.5*np.dot(linear_w,target*target)
    a.training=True;a.check();started=time.monotonic();coef,cert=c.quadratic(q,r,constant,False,False)
    free=dict(schema='c18-own-free26-v001',family='free',definition='25 normalized response features plus independent unpenalized intercept;26 coefficients',
        fit_ids=cfg['fit_ids'],lam=lam,scale=scale,coef=coef.tolist(),penalty=penalty.tolist(),certificate=cert,seconds=time.monotonic()-started,
        original_lambda_source=cfg['old_free_model'],nonnegative=False,equality=False,intercept_penalized=False)
    pp=free_predict(free,x);residual=y-pp
    dump(out/'free_prior.json',free)
    np.savez_compressed(out/'training.npz',features=x,target=y,weight=w,profile_id=profiles,index=index,
        free_design=design,free_target=target,free_weight=linear_w,free_prior_prediction=pp,free_residual_target=residual,
        Q=q,r=r,constant=np.asarray(constant),penalty=penalty)
    boosters={};metadata={}
    for name,yy,config in [('direct200',y,cfg['direct200_configuration']),('capacity_direct',y,cfg['capacity_configuration']),('free_residual',residual,cfg['direct200_configuration'])]:
        folder=out/name;folder.mkdir();grid={k:config[k] for k in ['max_iter','max_leaf_nodes','l2_regularization']}
        if dict(c.BASE,**grid)!=config:raise ValueError('Exact frozen booster configuration')
        model,info,binidx=c.fit_booster(h,x,yy,w,grid,folder,a)
        np.savez_compressed(folder/'binning_indices.npz',index=binidx.astype(np.int64))
        with (folder/'booster.pkl').open('xb') as f:pickle.dump(model,f,protocol=5)
        info.update(kind=name,feature_scale=scale,training=ref(out/'training.npz'),target_key='free_residual_target' if name=='free_residual' else 'target',
            prior=ref(out/'free_prior.json') if name=='free_residual' else None,binning_indices=ref(folder/'binning_indices.npz'),
            pickle=ref(folder/'booster.pkl'),numpy=np.__version__,sklearn=sklearn.__version__)
        dump(folder/'model.json',info);boosters[name]=model
        metadata[name]=dict(metadata=ref(folder/'model.json'),trees=ref(folder/'booster.npz'),pickle=ref(folder/'booster.pkl'))
    a.training=False
    model_lock=save_lock(out,cfg,'cpu',dict(actual_booster_fits=3,actual_linear_fits=1,free_prior=ref(out/'free_prior.json'),
        training=ref(out/'training.npz'),boosters=metadata,source_own_training=cfg['training'],reused_baseline_lock=cfg['own_model_lock']))
    entries=[]
    for entry in priorlock['records']:
        z=held_arrays(cfg,entry,a);xx=z['features'];values={m:z['prediction__'+m] for m in BASE_METHODS}
        fp=free_predict(free,xx);direct=boosters['direct200'].predict(xx);fr=fp+boosters['free_residual'].predict(xx)
        values.update(direct200=direct,direct200_half=.5*(values['d9']+direct),capacity_direct=boosters['capacity_direct'].predict(xx),
            free_prior=fp,free_residual=fr,free_half=.5*(values['d9']+fr))
        if not all(v.shape==(entry['rows'],) and np.isfinite(v).all() for v in values.values()):raise ValueError('Finite CPU predictions')
        folder=out/entry['role'];folder.mkdir(exist_ok=True);p=folder/('profile_'+str(entry['id'])+'.npz')
        np.savez_compressed(p,index=z['index'].astype(np.int64),source_row=z['source_row'].astype(np.int64),**{'prediction__'+m:values[m] for m in CPU_METHODS})
        entries.append(dict(role=entry['role'],id=entry['id'],rows=entry['rows'],source_rows=entry['source_rows'],prediction=ref(p),
            source_own_prediction=entry['prediction'],truth=entry['truth']))
    pred=prediction_lock(out,cfg,'cpu',model_lock,CPU_METHODS,entries)
    return dict(status='C18_OWN_CPU_FIT_PREDICT_COMPLETE',kind='cpu',actual_booster_fits=3,actual_linear_fits=1,actual_mlp_fits=0,
        fit_rows=len(y),held_profiles=len(entries),model_lock=model_lock,prediction_lock=pred,held_truth_read=False,
        booster_fit_seconds={k:read(local(v['metadata']['path']))['fit_seconds'] for k,v in metadata.items()},linear_fit_seconds=free['seconds'])


def mlp_fit(cfg,a,out,priorlock,ownlinear,seed):
    if seed not in SEEDS:raise ValueError('Fixed three seeds')
    worker=module('code/c15_paper_mlp_v001.py','c18_own_shared_cuda_primitives',a)
    runtime=worker.cuda_runtime();torch=worker.torch
    runtime.update(tf32_matmul_allowed=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn_allowed=torch.backends.cudnn.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),torch_num_threads=torch.get_num_threads())
    dump(out/'runtime.json',runtime)
    rawx,rawy,raww,profiles,index=fit_arrays(cfg,a,ownlinear)
    mean=rawx.mean(0);std=np.maximum(rawx.std(0),1e-8);mid=float(np.median(rawy));scale=max(float(np.percentile(rawy,75)-np.percentile(rawy,25)),1.)
    transform=dict(input_mean=mean.tolist(),input_std=std.tolist(),y_mid=mid,y_scale=scale,fit_ids=cfg['fit_ids'],warmup=300,label_stride=5,
        rc_scale={k:ownlinear['scale'][k] for k in ['qscale','y_mid','y_scale']},source_own_training=cfg['training'])
    x=((rawx-mean)/std).astype(np.float32);y=((rawy-mid)/scale).astype(np.float32)
    counts=[int(np.sum(profiles==pid)) for pid in cfg['fit_ids']]
    w=np.concatenate([np.full(k,len(y)/(len(counts)*k),dtype=np.float64) for k in counts]).astype(np.float32)
    arrays=dict(x=x,y_normalized=y,w=w,profile_ids=profiles,index=index)
    worker.save_arrays(out/'training_arrays.npz',**arrays)
    if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Finite transformed FIT values')
    model=worker.new_model(seed);initial=worker.weights(model)
    original_entry=next(e for e in cfg['original_mlp'] if e['seed']==seed)
    original_model=read(a.audited(original_entry['model'],original_entry['audit']))
    if worker.array_digest(initial)!=original_model['initial_array_sha256']:raise ValueError('Original/own exact paired seeded initialization changed')
    worker.save_arrays(out/'weights_initial.npz',**initial)
    xt=torch.as_tensor(x,device='cuda');yt=torch.as_tensor(y,device='cuda');wt=torch.as_tensor(w,device='cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
    trace=[];a.training=True;a.check()
    for epoch in range(1,201):
        a.check();started=time.monotonic();model.train()
        order=np.random.Generator(np.random.PCG64(seed+epoch-1)).permutation(len(y)).astype(np.int64)
        total,maxgrad=0.,0.
        for lo in range(0,len(y),4096):
            a.check();batch=torch.as_tensor(order[lo:lo+4096],dtype=torch.int64,device='cuda');optimizer.zero_grad(set_to_none=True)
            error=model(xt[batch]).reshape(-1)-yt[batch];loss=(wt[batch]*error.square()).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Finite weighted MSE')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach().cpu())*len(batch);maxgrad=max(maxgrad,float(grad.detach().cpu()))
        torch.cuda.synchronize();trace.append(dict(epoch=epoch,online_weighted_loss=total/len(y),max_preclip_gradient_norm=maxgrad,
            seconds=time.monotonic()-started,permutation_sha256=worker.array_digest(dict(order=order))))
    a.check();a.training=False;final=worker.weights(model)
    if any(not np.isfinite(v).all() for v in final.values()):raise FloatingPointError('Finite final MLP parameters')
    worker.save_arrays(out/'weights_final.npz',**final);dump(out/'epoch_trace.json',trace)
    metadata=dict(schema='c18-own-mlp-model-v001',dataset=cfg['dataset'],arm='own_boundary',loss='mse',seed=seed,epochs=200,
        settings=MLP_SETTINGS,fit_ids=cfg['fit_ids'],training_rows=len(y),training_profile_counts={str(p):int(np.sum(profiles==p)) for p in cfg['fit_ids']},
        transform=transform,source_sha256=sha(__file__),shared_worker_sha256=SOURCES['code/c15_paper_mlp_v001.py'],
        training_arrays_sha256=worker.array_digest(arrays),initial_array_sha256=worker.array_digest(initial),final_array_sha256=worker.array_digest(final),
        training_seconds=sum(v['seconds'] for v in trace),original_model=original_entry['model'],initialization_exactly_matches_original=True,held_truth_read=False)
    dump(out/'model.json',metadata)
    model_lock=save_lock(out,cfg,'mlp',dict(seed=seed,actual_mlp_fits=1,model=ref(out/'model.json'),weights=ref(out/'weights_final.npz'),
        initial_weights=ref(out/'weights_initial.npz'),training=ref(out/'training_arrays.npz'),source_own_training=cfg['training']))
    entries=[];model.eval();method='mse_mlp_'+str(seed)
    for entry in priorlock['records']:
        z=held_arrays(cfg,entry,a);xx=((z['features']-mean)/std).astype(np.float32);values=[]
        if not np.isfinite(xx).all():raise ValueError('Finite standardized held features')
        with torch.no_grad():
            for lo in range(0,len(xx),4096):
                a.check();v=model(torch.as_tensor(xx[lo:lo+4096],device='cuda')).reshape(-1);values.append(v.cpu().numpy().astype(np.float64))
        pred=mid+scale*np.concatenate(values)
        if pred.shape!=(entry['rows'],) or not np.isfinite(pred).all():raise ValueError('Finite all-row MLP predictions')
        folder=out/entry['role'];folder.mkdir(exist_ok=True);p=folder/('profile_'+str(entry['id'])+'.npz')
        worker.save_arrays(p,index=z['index'].astype(np.int64),source_row=z['source_row'].astype(np.int64),features=xx,**{'prediction__'+method:pred})
        entries.append(dict(role=entry['role'],id=entry['id'],rows=entry['rows'],source_rows=entry['source_rows'],prediction=ref(p),
            source_own_prediction=entry['prediction'],truth=entry['truth']))
    lock=prediction_lock(out,cfg,'mlp',model_lock,[method],entries,seed)
    return dict(status='C18_OWN_MLP_FIT_PREDICT_COMPLETE',kind='mlp',seed=seed,actual_mlp_fits=1,actual_booster_fits=0,actual_linear_fits=0,
        fit_rows=len(y),training_seconds=metadata['training_seconds'],held_profiles=len(entries),model_lock=model_lock,prediction_lock=lock,held_truth_read=False)


def stage(entry,cfg,a,kind,seed=None):
    p=a.pin(entry);summary=read(p)
    expected='C18_OWN_CPU_FIT_PREDICT_COMPLETE' if kind=='cpu' else 'C18_OWN_MLP_FIT_PREDICT_COMPLETE'
    if (summary['status']!=expected or summary['kind']!=kind or summary['source_sha256']!=sha(__file__) or summary['config']!=cfg['_ref']
        or summary['dataset']!=cfg['dataset'] or summary['held_truth_read'] is not False or (p.parent/'failure.json').exists()
        or (kind=='mlp' and summary['seed']!=seed)):raise ValueError('Exact completed source/config/seed stage required')
    for name,digest in summary['files'].items():
        f=(p.parent/name).resolve()
        if not f.is_relative_to(p.parent):raise ValueError('Stage path escape')
        a.pin(dict(path=str(f),sha256=digest))
    ml=read(a.pin(summary['model_lock']));pl=read(a.pin(summary['prediction_lock']))
    if (ml['status']!='ALL_MODELS_LOCKED' or ml['config']!=cfg['_ref'] or ml['source_sha256']!=sha(__file__) or ml['kind']!=kind
        or pl['status']!='ALL_ROLE_PREDICTIONS_LOCKED' or pl['config']!=cfg['_ref'] or pl['source_sha256']!=sha(__file__)
        or pl['model_lock']!=summary['model_lock'] or pl['kind']!=kind or pl['held_truth_read'] is not False
        or ml['held_truth_read'] is not False or pl['methods']!=(CPU_METHODS if kind=='cpu' else ['mse_mlp_'+str(seed)])
        or local(summary['model_lock']['path']).parent!=p.parent or local(summary['prediction_lock']['path']).parent!=p.parent
        or (kind=='mlp' and (ml['seed']!=seed or pl['seed']!=seed))):
        raise ValueError('Complete current model and prediction locks')
    return summary,pl


def metrics(y,p):
    e=p-y;u=np.maximum(-e,0);o=np.maximum(e,0)
    return dict(n=len(y),mse=float(np.mean(e*e)),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(np.abs(e))),bias=float(e.mean()),
        under_rmse=float(np.sqrt(np.mean(u*u))),over_mae=float(o.mean()),max_under=float(u.max()),max_over=float(o.max()),max_absolute_error=float(np.max(np.abs(e))))


def evaluate(cfg,a,out,priorlock,components_ref):
    bundle=read(a.pin(components_ref))
    if bundle['schema']!='c18-own-controls-components-v001' or bundle['config']!=cfg['_ref'] or bundle['dataset']!=cfg['dataset'] or len(bundle['mlp'])!=3:raise ValueError('Full three-seed component bundle')
    loaded=[stage(bundle['cpu'],cfg,a,'cpu')]+[stage(e,cfg,a,'mlp',seed) for e,seed in zip(bundle['mlp'],SEEDS)]
    expected=[(e['role'],e['id']) for e in priorlock['records']]
    bycomponent=[]
    for summary,pl in loaded:
        if [(e['role'],e['id']) for e in pl['records']]!=expected:raise ValueError('Every model/all held role predictions locked before labels')
        indexed={}
        for e,old in zip(pl['records'],priorlock['records']):
            if e['source_own_prediction']!=old['prediction'] or e['truth']!=old['truth'] or e['rows']!=old['rows'] or e['source_rows']!=old['source_rows']:raise ValueError('Exact old exposed row/truth lineage')
            a.pin(e['prediction']);indexed[(e['role'],e['id'])]=e
        bycomponent.append(indexed)
    dump(out/'evaluation_input_lock.json',dict(schema='c18-own-controls-evaluation-input-lock-v001',config=cfg['_ref'],components=components_ref,
        prediction_locks=[s['prediction_lock'] for s,p in loaded],all_predictions_verified_before_truth=True,locked_at_utc=datetime.now(timezone.utc).isoformat()))
    # Only this separate stage opens existing exposed held labels, after all four
    # component stages/model locks/all dataset-specific cohorts are pinned above.
    roles={v['role']:v for v in cfg['roles']};rows=[]
    for old in priorlock['records']:
        key=(old['role'],old['id']);truth=a.arrays(old['truth'],roles[old['role']]['audit'],['index','target']);values={};ix=None;source_row=None
        for (summary,pl),indexed in zip(loaded,bycomponent):
            z=a.arrays(indexed[key]['prediction'])
            if ix is None:ix=z['index'];source_row=z['source_row']
            if not np.array_equal(ix,z['index']) or not np.array_equal(source_row,z['source_row']) or not np.array_equal(ix,truth['index']):raise ValueError('All models/truth exact endpoint pairing')
            values.update({m:z['prediction__'+m] for m in pl['methods']})
        if set(values)!=set(METHODS):raise ValueError('Complete frozen14 method panel')
        if not all(v.shape==truth['target'].shape and np.isfinite(v).all() for v in values.values()):raise ValueError('Finite paired evaluation arrays')
        mm={m:metrics(truth['target'],values[m]) for m in METHODS}
        row=dict(dataset=cfg['dataset'],role=old['role'],profile=old['id'],n=len(ix),methods=mm,
            mean_seed_metrics={k:float(np.mean([mm[m][k] for m in MLP_METHODS])) for k in mm[MLP_METHODS[0]] if k!='n'},
            delta_vs_d9={m:mm[m]['rmse']-mm['d9']['rmse'] for m in METHODS},delta_vs_linear_residual={m:mm[m]['rmse']-mm['linear_residual']['rmse'] for m in METHODS})
        folder=out/old['role'];folder.mkdir(exist_ok=True);dump(folder/('profile_'+str(old['id'])+'.json'),row);rows.append(row)
    summaries=[]
    for role in cfg['roles']:
        rr=[r for r in rows if r['role']==role['role']];mm={}
        for method in METHODS:
            vals=[r['methods'][method] for r in rr];n=sum(v['n'] for v in vals)
            mm[method]=dict(profiles=len(vals),n=n,**{'macro_'+k:float(np.mean([v[k] for v in vals])) for k in ['rmse','mae','bias','under_rmse','over_mae','max_under','max_over','max_absolute_error']},
                pooled_rmse=math.sqrt(sum(v['n']*v['mse'] for v in vals)/n))
        seedsummary={k:dict(mean=float(np.mean([mm[m][k] for m in MLP_METHODS])),minimum=float(min(mm[m][k] for m in MLP_METHODS)),maximum=float(max(mm[m][k] for m in MLP_METHODS))) for k in mm[MLP_METHODS[0]] if k not in ['n','profiles']}
        comparisons={}
        for left,right in [('linear_half',m) for m in METHODS if m!='linear_half']+[('linear_residual',m) for m in METHODS if m!='linear_residual']:
            dif=np.asarray([r['methods'][left]['rmse']-r['methods'][right]['rmse'] for r in rr])
            comparisons[left+'__minus__'+right]=dict(macro_delta=float(np.mean(dif)),wins=int(np.sum(dif<0)),ties=int(np.sum(dif==0)),losses=int(np.sum(dif>0)),profile_deltas=dif.tolist())
        result=dict(dataset=cfg['dataset'],role=role['role'],ids=role['ids'],methods=mm,mlp_seed_metric_summary=seedsummary,
            comparisons=comparisons,all_data_previously_exposed=True,formal_test_performed=False)
        dump(out/role['role']/'summary.json',result);summaries.append(result)
    dump(out/'results.json',dict(schema='c18-own-controls-results-v001',dataset=cfg['dataset'],arm='own_boundary',roles=summaries,per_profile=rows,
        seed_scope='All three fixed seeds; metric means/ranges, not prediction ensemble',new_fits=0))
    return dict(status='C18_OWN_CONTROLS_EVALUATION_COMPLETE',kind='evaluation',methods=METHODS,profiles=len(rows),roles=len(summaries),
        components=components_ref,results=ref(out/'results.json'),formal_test_performed=False,independent_confirmation=False,new_training_fits=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['plan','cpu-fit-predict','mlp-fit-predict','evaluate'],required=True)
    for name in ['config','config-sha256','output-dir']:p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--components');p.add_argument('--components-sha256')
    p.add_argument('--max-seconds',type=float,default=1200.)
    args=p.parse_args()
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=3600:raise ValueError('Registered cap in(0,3600]')
    if args.mode=='mlp-fit-predict' and args.max_seconds>900:raise ValueError('Frozen CUDA stage cap900seconds')
    out=local(args.output_dir)
    if WORK/'runs' not in out.parents:raise ValueError('New C18 output directory only')
    out.mkdir(parents=True,exist_ok=False);a=Intake(out,args.max_seconds)
    dump(out/'configuration.json',dict(vars(args),source=str(Path(__file__).resolve()),source_sha256=sha(__file__),output_dir=str(out),
        pid=os.getpid(),process_group=os.getpgrp() if hasattr(os,'getpgrp') else None,argv=list(sys.argv),numpy=np.__version__,started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        entry=dict(path=args.config,sha256=args.config_sha256);cfg=read(a.pin(entry));cfg['_ref']=ref(local(args.config))
        old,ml,pl,linear=configuration(cfg,a)
        if args.mode=='plan':result=dict(status='C18_OWN_CONTROLS_PLAN_COMPLETE',kind='plan',actual_booster_fits=0,actual_linear_fits=0,actual_mlp_fits=0,
            planned_per_dataset=dict(cpu_boosters=3,free26_prior=1,mlp=3),methods=METHODS,numeric_arrays_decoded=False)
        elif args.mode=='cpu-fit-predict':result=cpu_fit(cfg,a,out,pl,linear)
        elif args.mode=='mlp-fit-predict':result=mlp_fit(cfg,a,out,pl,linear,args.seed)
        else:
            if not args.components or not args.components_sha256:raise ValueError('Full completed components required')
            result=evaluate(cfg,a,out,pl,dict(path=args.components,sha256=args.components_sha256))
        dump(out/'input_pins.json',a.inputs);a.check()
        result.update(dataset=cfg['dataset'],arm='own_boundary',source_sha256=sha(__file__),config=cfg['_ref'],protocol=cfg['protocol'],
            identity_checks=a.checks,elapsed_seconds=time.monotonic()-a.start,
            files={f.relative_to(out).as_posix():sha(f) for f in sorted(out.rglob('*')) if f.is_file()})
        dump(out/'summary.json',result);a.check()
        dump(out/'stage_record.json',dict(schema='c18-own-controls-stage-v001',status='COMPLETE',source_sha256=sha(__file__),result=result,
            files={f.relative_to(out).as_posix():sha(f) for f in sorted(out.rglob('*')) if f.is_file()},elapsed_seconds=time.monotonic()-a.start))
        a.check();print(json.dumps(dict(status=result['status'],mode=args.mode,dataset=cfg['dataset'],seed=args.seed,elapsed_seconds=result['elapsed_seconds'])))
    except Exception as exc:
        dump(out/'failure.json',dict(status='FAILED',error_type=type(exc).__name__,message=str(exc),elapsed_seconds=time.monotonic()-a.start));raise


if __name__=='__main__':main()
