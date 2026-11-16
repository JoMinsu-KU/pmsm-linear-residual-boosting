"""Independent artifact audit of C18 one-factor filter initialization arms.

Scientific producer imports, optimizer replay and pickle execution are absent.
The caller/root must register actual artifact audit invocations separately.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'linear-followup-20261111'
CAPACITY_AUDIT_SHA='bc498182739bd0bb09ba071cf9e4607fd098f6c4f55bf783129f4d371b261d21'
DIAGNOSTIC_AUDIT_SHA='f13268f4c5ad6ccabd5dc5e11b5e4ec13782bdc966505f9589b413fba85a57bd'
SPANS=(16,64,256,1024,4096)
METHODS=['d9','linear_prior','linear_residual','linear_half','prior_half']
PRODUCER_SHA='63108ce0700e7646dce34663b88542b4273eab052fd5bbb9d30d7d6aea39b046'
PROTOCOL_SHA='271c4ec0506f7cea6f0004c305ce9b10f5d3e365a188aed607e3b688e309f914'
CORE_SHA='17c47c41c27b6e92618a44de0f5860f80ffd1357d998a4cfeb805f5e2d8b7b37'
HELPER_SHA='d4aa942ca9db8d8e9319197632cf376730456f3ea5cdc48b9d1c1fd51fc59412'
ROLES={'ipmsm':['calibration','pilot','confirmation'],'paderborn':['development','confirmation']}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()


def dependency(filename,expected,name):
    path=Path(__file__).with_name(filename)
    if sha(path)!=expected:raise ValueError('Independent helper changed: '+filename)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


K=dependency('c18_capacity_audit_v002.py',CAPACITY_AUDIT_SHA,'c18_init_capacity_oracle')
D=dependency('c18_diagnostic_audit_v001.py',DIAGNOSTIC_AUDIT_SHA,'c18_init_metric_oracle')
I=K.I; C=K.C


def local(value):return K.local(value)
def pin(a,entry):return K.pin(a,entry)
def obj(a,entry):return I.read(pin(a,entry))
def exact(a,x,y,label):return K.exact(a,x,y,label)
def close(a,x,y,label,atol=2e-9,rtol=2e-12):
    return K.P.close_array(a,x,y,label,atol=atol,rtol=rtol)


def filtered(a,u,span,initial):
    """Explicit independent recurrence, not producer/scipy filter execution."""
    u=np.asarray(u,dtype=np.float64)
    decay=1.-2./(span+1.)
    output=np.empty_like(u);output[0]=initial
    for t in range(1,len(u)):
        if t%10000==0:a.budget()
        output[t]=decay*output[t-1]+(1.-decay)*u[t]
    return output


def features_from_operating(a,operating,arm):
    a.require(arm in ('own_boundary','first_load'),'Exactly one prospective initialization change')
    op=np.asarray(operating,dtype=np.float64)
    a.require(op.ndim==2 and op.shape[1]==7 and len(op)>300 and np.isfinite(op).all(),
              'Finite original seven observed channels; no rotor input')
    current=np.hypot(op[:,0],op[:,1]);speed=np.abs(op[:,4]);voltage=np.hypot(op[:,2],op[:,3])
    instant=np.stack((current,current**2,voltage,speed,speed**2,current*speed,op[:,5],op[:,6]),axis=1)
    boundary_input=op[:,[6,5]];heat_input=instant[:,[1,4,5]]
    init_boundary=boundary_input[0] if arm=='own_boundary' else np.full(2,op[0,6])
    init_heat=heat_input[0] if arm=='first_load' else np.zeros(3)
    boundary=np.concatenate([filtered(a,boundary_input,s,init_boundary) for s in SPANS],axis=1)
    heat=np.concatenate([filtered(a,heat_input,s,init_heat) for s in SPANS],axis=1)
    # EWMA is an unchanged legacy scaling metadata component, not an extra input.
    ewma=np.concatenate([instant]+[filtered(a,heat_input,s,heat_input[0]) for s in SPANS],axis=1)
    return dict(boundary=boundary,heat=heat,instant=instant,ewma=ewma)


def check_closed_form_delta(a,old,new,arm):
    n=len(new['instant']);time_index=np.arange(n,dtype=np.float64)
    a.require(old['boundary'].shape==(n,10) and old['heat'].shape==(n,15),'Original audited filter geometry')
    for k,span in enumerate(SPANS):
        power=(1.-2./(span+1.))**time_index
        bdelta=np.array([0.,new['instant'][0,6]-new['instant'][0,7]]) if arm=='own_boundary' else np.zeros(2)
        hdelta=new['instant'][0,[1,4,5]] if arm=='first_load' else np.zeros(3)
        close(a,new['boundary'][:,2*k:2*k+2],old['boundary'][:,2*k:2*k+2]+power[:,None]*bdelta,
              'Analytic boundary initialization difference/span'+str(span))
        close(a,new['heat'][:,3*k:3*k+3],old['heat'][:,3*k:3*k+3]+power[:,None]*hdelta,
              'Analytic heating initialization difference/span'+str(span))
    close(a,new['instant'],old['instant'],'Unchanged instantaneous observed features',atol=0.,rtol=0.)


def training_reference(a,records,ids):
    indexes={p:np.arange(300,len(records[p]['target']),5,dtype=np.int64) for p in ids}
    instant=np.concatenate([records[p]['instant'][indexes[p]] for p in ids])
    target=np.concatenate([records[p]['target'][indexes[p]] for p in ids])
    ri=max(float(np.sqrt(np.mean(instant[:,0]**2))),1e-8)
    rs=max(float(np.sqrt(np.mean(instant[:,3]**2))),1e-8)
    qscale=np.tile([ri**2,rs**2,ri*rs],5)
    x=np.concatenate([np.column_stack((records[p]['boundary'],records[p]['heat']/qscale,records[p]['instant']))[indexes[p]] for p in ids])
    lengths=[len(indexes[p]) for p in ids]
    w=np.concatenate([np.full(n,1./(len(ids)*n)) for n in lengths])
    ewma=np.concatenate([records[p]['ewma'][indexes[p]] for p in ids])
    scale=dict(ids=ids,qscale=qscale.tolist(),y_mid=float(np.median(target)),
        y_scale=max(float(np.percentile(target,75)-np.percentile(target,25)),1.),
        ewma_mean=ewma.mean(0).tolist(),ewma_scale=np.maximum(ewma.std(0),1e-8).tolist(),
        linear33_mean=x.mean(0).tolist(),linear33_scale=np.maximum(x.std(0),1e-8).tolist())
    return dict(features=x,target=target,weight=len(target)*w,
        profile_id=np.concatenate([np.full(len(indexes[p]),p,dtype=np.int64) for p in ids]),
        index=np.concatenate([indexes[p] for p in ids]),linear_weight=w),scale,indexes


def linear_certificate(a,model,training):
    """Normal equations and objective-gap qualification, no refitting."""
    x=training['linear_design'];y=training['linear_target'];w=training['linear_weight']
    coefficient=np.asarray(model['coef'],dtype=np.float64)
    penalty=np.r_[0.,np.ones(33)];lam=model['lam']
    a.require(x.shape==(len(y),34) and coefficient.shape==(34,) and lam>0,'Linear33 fixed ridge geometry')
    q=x.T@(w[:,None]*x)+lam*np.diag(penalty)
    r=x.T@(w*y);constant=.5*float(np.dot(w,y*y))
    close(a,training['Q'],q,'Independent ridge normal-equation matrix',atol=1e-9,rtol=1e-10)
    close(a,training['r'],r,'Independent ridge right-hand side',atol=1e-9,rtol=1e-10)
    close(a,training['constant'],constant,'Independent ridge constant')
    exact(a,training['penalty'],penalty,'Unpenalized intercept and33 penalized coefficients')
    np.linalg.cholesky(q)
    residual=q@coefficient-r
    gap=.5*float(residual@np.linalg.solve(q,residual))
    primal=float(.5*coefficient@q@coefficient-r@coefficient+constant)
    certificate=model['certificate']
    a.require(math.isfinite(gap) and gap>=-1e-12 and gap<=certificate['gap_tolerance'],
              'Independent positive-definite ridge objective gap qualified')
    close(a,certificate['primal'],primal,'Saved ridge primal',atol=1e-9,rtol=1e-10)
    close(a,certificate['stationarity'],float(np.max(np.abs(residual))),'Saved normal-equation residual',atol=2e-9,rtol=1e-8)
    return dict(objective=primal,objective_gap=gap,maximum_normal_equation_residual=float(np.max(np.abs(residual))))


def validate_config(a,args):
    cfg=obj(a,dict(path=args.config,sha256=args.config_sha256))
    a.require(cfg['schema']=='c18-initialization-config-v002' and cfg['dataset'] in ROLES
              and cfg['arm'] in ('own_boundary','first_load'),'Prospective dataset/one-factor arm')
    a.require(cfg['producer']['sha256']==PRODUCER_SHA==args.producer_sha256,'Frozen producer identity')
    pin(a,cfg['producer']);pin(a,cfg['protocol']);pin(a,cfg['limits'])
    a.require(cfg['protocol']['sha256']==PROTOCOL_SHA,'Frozen initialization protocol')
    a.same(cfg['methods'],METHODS,'All five fixed methods')
    a.require(cfg['weight']==.5 and cfg['warmup']==300 and cfg['stride']==5 and cfg['progress_blocks']==5
              and cfg['formal_confirmation_allowed'] is False and cfg['all_data_previously_exposed'] is True,
              'No tuning, endpoint or formal-test changes')
    for filename,digest in [('c15_paper_core_v001.py',CORE_SHA),('combination_c12_fit_rc_v001.py',HELPER_SHA)]:
        pin(a,dict(path='code/'+filename,sha256=digest))
    source=obj(a,cfg['source_config'])
    a.same(cfg['fit_ids'],source['train_ids'],'Original FIT IDs')
    a.same(cfg['fit_cache'],source['fit_cache'],'Exact original FIT source refs')
    a.same([v['role'] for v in cfg['roles']],ROLES[cfg['dataset']],'All old nonFIT roles')
    for now,old in zip(cfg['roles'],source['role_inputs']):
        for key in ('role','ids','records','prediction_lock'):a.same(now[key],old[key],'Original role/'+key)
        K.audit_index(a,now['audit'])
    lock=I.read(K.audited(a,cfg['old_model_lock'],cfg['selection_audit']))
    expected='direct_boost_7x500' if cfg['dataset']=='ipmsm' else 'direct_boost_15x1000'
    a.require(cfg['d9_method']==lock['selected_direct9']==expected,'Original FIT-CV D9 capacity')
    a.same(lock['fit_ids'],cfg['fit_ids'],'Original selected model FIT pool')
    for name,job in [('linear','linear_final__linear33__tuned'),('d9','full__'+expected)]:
        receipt_path=K.audited(a,cfg['old_'+name+'_receipt'],cfg['selection_audit'])
        receipt=I.read(receipt_path)
        a.require(local(lock['entries'][job]['path'])==receipt_path.parent
                  and lock['entries'][job]['receipt_sha256']==cfg['old_'+name+'_receipt']['sha256'],
                  'Exact selected job receipt '+name)
        a.require(receipt['status']=='FIT_JOB_COMPLETE' and receipt['job']['job_id']==job,'Qualified selected job '+name)
        model_path=K.audited(a,cfg['old_'+name+'_model'],cfg['selection_audit']);old_model=I.read(model_path)
        a.require(model_path.parent==receipt_path.parent and receipt['files']['model.json']==cfg['old_'+name+'_model']['sha256'],
                  'Old selected model exact receipt binding')
        a.same(old_model['train_ids'],cfg['fit_ids'],'Original model train IDs')
        if name=='linear':
            a.require(old_model['linear_model']['family']=='linear33' and old_model['linear_model']['lam']==cfg['lambda'],
                      'Frozen prior FIT-selected lambda')
        else:a.same(cfg['d9_configuration'],old_model['configuration'],'Frozen D9 configuration')
    expected_base=dict(learning_rate=.05,min_samples_leaf=20,l2_regularization=.1,loss='squared_error',
                       early_stopping=False,random_state=2026111201)
    a.same(cfg['residual_configuration'],dict(expected_base,max_leaf_nodes=15,max_iter=200),'Fixed residual model capacity')
    a.same(cfg['d9_configuration'],dict(expected_base,max_leaf_nodes=7 if cfg['dataset']=='ipmsm' else 15,
                                        max_iter=500 if cfg['dataset']=='ipmsm' else 1000),'Exactly original D9 parameters')
    return cfg


def stage(a,path,digest,mode,cfg,args):
    folder=local(path)
    a.require(not (folder/'failure.json').exists(),'No incomplete/failed producer stage')
    summary=obj(a,dict(path=str(folder/'summary.json'),sha256=digest))
    status='C18_INITIALIZATION_FIT_AND_PREDICT_COMPLETE' if mode=='fit-predict' else 'C18_INITIALIZATION_EVALUATION_COMPLETE'
    a.require(summary['status']==status and summary['source_sha256']==PRODUCER_SHA
              and summary['dataset']==cfg['dataset'] and summary['arm']==cfg['arm'],'Completed exact initialization stage')
    a.require(local(summary['config']['path'])==local(args.config) and summary['config']['sha256']==args.config_sha256,
              'Stage exact configuration')
    for name,digest in summary['files'].items():
        path=(folder/name).resolve()
        a.require(path.is_relative_to(folder),'Safe stage-relative artifact')
        pin(a,dict(path=str(path),sha256=digest))
    invocation=I.read(folder/'configuration.json')
    a.require(invocation['mode']==mode and invocation['source_sha256']==PRODUCER_SHA
              and invocation['config_sha256']==args.config_sha256,'Actual bounded invocation')
    return folder,summary


def checked_features(a,entry,report,arm,fit):
    old=a.arrays(K.audited(a,entry,report))
    a.require('operating' in old,'Previously source-audited raw operating cache available')
    new=features_from_operating(a,old['operating'],arm)
    check_closed_form_delta(a,old,new,arm)
    if fit:
        a.require(old['target'].shape==(len(new['instant']),),'Full FIT target coverage')
        new['target']=old['target']
        close(a,new['ewma'],old['ewma'],'Unchanged legacy EWMA scaling features')
    return old,new


def training_artifacts(a,cfg,folder,model_lock):
    records={}
    for e in cfg['fit_cache']:
        old,new=checked_features(a,e,cfg['fit_audit'],cfg['arm'],True)
        a.require(len(new['target'])==e['rows'],'Original complete FIT profile retained')
        records[e['id']]=new
    expected,scale,indexes=training_reference(a,records,cfg['fit_ids'])
    model=obj(a,model_lock['linear'])
    a.require(model['family']=='linear33' and model['lam']==cfg['lambda'],'Same selected unconstrained ridge family/lambda')
    a.same(model['constraint_flags'],dict(nonnegative=False,equality=False,intercept=True,intercept_penalized=False),
           'Unpenalized explicit intercept; no thermal coefficient constraints')
    a.same(model['scale'],scale,'Independent FIT-only scales',atol=2e-8,rtol=2e-11)
    a.same(model['penalty'],[0.]+[1.]*33,'Linear penalty declaration')
    training=a.arrays(folder/'training.npz')
    keys=set(expected)|{'linear_design','linear_target','prior_prediction','residual_target','Q','r','constant','penalty'}
    a.require(set(training)==keys,'Exact original FIT training export schema')
    for key,value in expected.items():
        a.require(training[key].dtype==value.dtype,'Training dtype/'+key)
        if key=='features':close(a,training[key],value,'All independently generated changed FIT features')
        else:exact(a,training[key],value,'Original complete training field/'+key)
    # The source-generated training features have just been independently
    # checked at every element; exact tree branching uses these saved numbers.
    x=training['features'];s=model['scale']
    close(a,np.mean(x,axis=0),s['linear33_mean'],'Serialized design mean')
    close(a,np.maximum(np.std(x,axis=0),1e-8),s['linear33_scale'],'Serialized design standard deviation')
    design=np.column_stack((np.ones(len(x)),(x-np.asarray(s['linear33_mean']))/np.asarray(s['linear33_scale'])))
    yy=(training['target']-s['y_mid'])/s['y_scale']
    exact(a,training['linear_design'],design,'All independently assembled standardized design values')
    exact(a,training['linear_target'],yy,'Correct normalized linear target')
    coef=np.asarray(model['coef'],dtype=np.float64)
    prior=s['y_mid']+s['y_scale']*(design@coef)
    close(a,training['prior_prediction'],prior,'Full FIT in-sample prior prediction',atol=1e-8,rtol=2e-11)
    close(a,training['residual_target'],training['target']-prior,'Full FIT residual construction',atol=1e-8,rtol=2e-11)
    certificate=linear_certificate(a,model,training)
    a.require(model['certificate']['nonnegative'] is False and model['certificate']['equality'] is False,
              'Ridge certificate matches unconstrained model')
    close(a,model['certificate']['gap_tolerance'],1e-8*(1.+abs(certificate['objective'])),'Original certificate tolerance',atol=1e-14,rtol=1e-10)
    return model,training,certificate


def booster_artifacts(a,cfg,entry,training,kind):
    metadata_path=pin(a,entry);folder=metadata_path.parent;info=I.read(metadata_path)
    expected=cfg['d9_configuration'] if kind=='d9' else cfg['residual_configuration']
    a.same(info['configuration'],expected,'Unchanged '+kind+' booster settings')
    a.require(info['kind']==kind and info['fit_rows']==len(training['target']),'Booster exact target role/row count')
    pin(a,info['training']);pin(a,info['pickle'])
    target=training['target'] if kind=='d9' else training['residual_target']
    for key,value in [('features_sha256',training['features']),('target_sha256',target),('sample_weight_sha256',training['weight'])]:
        a.require(info[key]==K.original_training_digest(value,cfg['dataset']),'Exact legacy header+array training digest/'+kind+'/'+key)
    binsaved=a.arrays(pin(a,info['binning_indices']))['index']
    limit=info['binning_subsample'];seed=info['binning_random_state'];n=len(target)
    selected=np.random.RandomState(seed).choice(n,limit,replace=False) if limit is not None and n>limit else np.arange(n,dtype=np.int64)
    exact(a,binsaved,selected,'Frozen binning sample/index generation')
    a.require(info['binning_subsample_active']==(len(selected)<n),'Correct binning subsample declaration')
    tree_ref=dict(path=str(folder/info['model']['path']),sha256=info['model']['sha256'])
    trees=C.numeric_nodes(a,pin(a,tree_ref));nt=expected['max_iter']
    required={'baseline','n_iter'}|{f'tree_{j:03d}' for j in range(nt)}|{f'bin_thresholds_{j:02d}' for j in range(33)}
    a.require(set(trees)==required,'Every frozen booster tree and bin table exported')
    exact(a,trees['n_iter'],np.asarray([nt],dtype=np.int64),'Exact fixed number of boosting iterations')
    close(a,trees['baseline'],np.asarray([[np.average(target,weights=training['weight'])]]),'Independent weighted booster intercept')
    for j in range(33):
        ordered=np.sort(training['features'][selected,j]);unique=np.unique(ordered)
        if len(unique)<=255:thresholds=(unique[:-1]+unique[1:])*.5
        else:
            positions=(np.linspace(0,100,256)[1:-1]/100)*(len(ordered)-1)
            thresholds=.5*(ordered[np.floor(positions).astype(np.int64)]+ordered[np.ceil(positions).astype(np.int64)])
        close(a,trees[f'bin_thresholds_{j:02d}'],thresholds,'Independent FIT midpoint thresholds/'+kind+'/'+str(j))
        a.require(np.all(np.diff(trees[f'bin_thresholds_{j:02d}'])>=0),'Valid nondecreasing thresholds including legitimate duplicates')
    for j in range(nt):
        a.require(np.count_nonzero(trees[f'tree_{j:03d}']['is_leaf'])<=expected['max_leaf_nodes'],'Frozen leaf capacity')
    for j in sorted({0,nt//2,nt-1}):C.traverse(a,trees[f'tree_{j:03d}'],training['features'],kind+'/FIT/'+str(j),True)
    return trees


def raw_predict(a,trees,x,tag):
    result=np.full(len(x),float(trees['baseline'].item()),dtype=np.float64)
    for j in range(int(trees['n_iter'][0])):
        result+=C.traverse(a,trees[f'tree_{j:03d}'],x,tag+'/'+str(j))
    a.replayed[tag]=dict(rows=len(x),trees=int(trees['n_iter'][0]),all_rows_all_trees=True)
    return result


def prediction_artifacts(a,cfg,lock,model,boosters):
    expected=[(r['role'],v['id']) for r in cfg['roles'] for v in r['records']]
    a.require([(v['role'],v['id']) for v in lock['records']]==expected,'Every original nonFIT profile in order')
    output={}
    for role in cfg['roles']:
        K.audited(a,role['prediction_lock'],role['audit'])
        for source in role['records']:
            entry=next(v for v in lock['records'] if v['role']==role['role'] and v['id']==source['id'])
            a.same(entry['source_features'],source['features'],'Original feature lineage')
            a.same(entry['truth'],source['truth'],'Original target reference, no decoding yet')
            old,independent=checked_features(a,source['features'],role['audit'],cfg['arm'],False)
            saved=a.arrays(pin(a,entry['prediction']))
            keys={'index','source_row','features','boundary','heat','instant'}|{'prediction__'+k for k in METHODS}
            a.require(set(saved)==keys,'All five method predictions and complete feature artifacts')
            ix=np.arange(300,len(independent['instant']),dtype=np.int64)
            exact(a,old['index'],ix,'Original complete retained indices')
            exact(a,saved['index'],ix,'Intervention preserves every evaluated index')
            exact(a,saved['source_row'],old['source_row'][ix],'Original raw source rows')
            a.require(entry['source_rows']==source['rows']==len(independent['instant'])
                      and entry['rows']==len(ix)==entry['source_rows']-300,
                      'Explicit full raw rows and retained evaluation rows')
            for key in ('boundary','heat','instant'):
                close(a,saved[key],independent[key],'All independent changed held '+key)
            s=model['scale']
            x=np.column_stack((saved['boundary'],saved['heat']/np.asarray(s['qscale']),saved['instant']))[ix]
            exact(a,saved['features'],x,'Exact tree inputs from validated saved primitive features')
            design=np.column_stack((np.ones(len(ix)),(x-np.asarray(s['linear33_mean']))/np.asarray(s['linear33_scale'])))
            prior=s['y_mid']+s['y_scale']*(design@np.asarray(model['coef']))
            dd=raw_predict(a,boosters['d9'],x,role['role']+'/'+str(source['id'])+'/d9')
            rr=prior+raw_predict(a,boosters['residual'],x,role['role']+'/'+str(source['id'])+'/residual')
            values=dict(d9=dd,linear_prior=prior,linear_residual=rr,linear_half=(dd+rr)*.5,prior_half=(dd+prior)*.5)
            for key,value in values.items():
                close(a,saved['prediction__'+key],value,'All raw-node/affine held predictions/'+key,atol=1e-8,rtol=2e-11)
            # Full predicted arrays have been validated. Use the exact saved
            # values for later arithmetic, preserving producer FP64 evaluation.
            output[(role['role'],source['id'])]=dict(index=ix,source_row=saved['source_row'],
                values={key:saved['prediction__'+key] for key in METHODS},source=source,role=role)
    return output


def profile_statistics(y,values,index):
    whole=dict(methods={key:D.metrics(y,values[key]) for key in METHODS},
        decomposition=D.half_accounting(y,values['d9'],values['linear_residual']),
        residual_effect=D.residual_accounting(y,values['linear_prior'],values['linear_residual']))
    blocks=[];n=len(y);denom=whole['decomposition']['rmse_denominator']
    for b in range(5):
        lo,hi=n*b//5,n*(b+1)//5;yy=y[lo:hi];pp={k:v[lo:hi] for k,v in values.items()}
        a0=pp['d9']-yy;a1=pp['linear_half']-yy
        blocks.append(dict(bin=b,n=hi-lo,index_start=int(index[lo]),index_end_inclusive=int(index[hi-1]),
            methods={k:D.metrics(yy,v) for k,v in pp.items()},
            decomposition=D.half_accounting(yy,pp['d9'],pp['linear_residual']),
            residual_effect=D.residual_accounting(yy,pp['linear_prior'],pp['linear_residual']),
            primary_delta_rmse_contribution=float(np.sum(a1*a1-a0*a0)/n/denom) if denom else 0.))
    return dict(whole=whole,progress_blocks=blocks)


def summarize(rows,key):
    output={}
    for method in METHODS:
        entries=[row[key]['whole']['methods'][method] for row in rows];n=sum(v['n'] for v in entries)
        output[method]=dict(profiles=len(rows),n=n,
            **{'macro_'+k:math.fsum(v[k] for v in entries)/len(entries) for k in ['rmse','mae','bias','under_rmse','over_mae']},
            pooled_rmse=math.sqrt(math.fsum(v['n']*v['mse'] for v in entries)/n))
    return output


def evaluation_artifacts(a,args,cfg,predictions,prediction_ref):
    folder,summary=stage(a,args.evaluation_dir,args.evaluation_sha256,'evaluate',cfg,args)
    a.same(summary['prediction_lock'],prediction_ref,'Evaluation consumes exact model-validated prediction lock')
    a.require(summary['new_training_fits']==0 and summary['formal_test_performed'] is False
              and summary['independent_confirmation'] is False and summary['all_data_previously_exposed'] is True,
              'Exploratory descriptive evaluation only')
    rows=[]
    old_keys=dict(d9='direct9_fitcv_selected',linear_prior='linear33_ridge',linear_residual='linear33_insample_boost')
    # All current model and every prediction have been audited before truth decoding.
    for (role,pid),record in predictions.items():
        source,report=record['source'],record['role']['audit']
        truth=a.arrays(K.audited(a,source['truth'],report))
        exact(a,truth['index'],record['index'],'Original target row alignment')
        baseline=a.arrays(K.audited(a,source['features'],report))
        old={k:baseline['prediction__'+v] for k,v in old_keys.items()}
        old['linear_half']=(old['d9']+old['linear_residual'])*.5
        old['prior_half']=(old['d9']+old['linear_prior'])*.5
        new=profile_statistics(truth['target'],record['values'],record['index'])
        legacy=profile_statistics(truth['target'],old,record['index'])
        changes={k:new['whole']['methods'][k]['rmse']-legacy['whole']['methods'][k]['rmse'] for k in METHODS}
        row=dict(dataset=cfg['dataset'],arm=cfg['arm'],role=role,profile=pid,intervention=new,legacy=legacy,
            rmse_change_vs_legacy=changes,
            half_minus_d9_change_vs_legacy=new['whole']['decomposition']['delta_rmse']-legacy['whole']['decomposition']['delta_rmse'],
            block_contribution_changes=[n['primary_delta_rmse_contribution']-o['primary_delta_rmse_contribution']
                                        for n,o in zip(new['progress_blocks'],legacy['progress_blocks'])])
        a.same(I.read(folder/role/('profile_'+str(pid)+'.json')),row,'All five-method/five-block profile metrics and contrasts',atol=1e-8,rtol=1e-10)
        rows.append(row)
    results=[]
    for role in cfg['roles']:
        selected=[v for v in rows if v['role']==role['role']]
        new,old=summarize(selected,'intervention'),summarize(selected,'legacy')
        result=dict(dataset=cfg['dataset'],arm=cfg['arm'],role=role['role'],ids=role['ids'],intervention=new,legacy=old,
            half_minus_d9=new['linear_half']['macro_rmse']-new['d9']['macro_rmse'],
            legacy_half_minus_d9=old['linear_half']['macro_rmse']-old['d9']['macro_rmse'],
            half_minus_d9_change_vs_legacy=math.fsum(v['half_minus_d9_change_vs_legacy'] for v in selected)/len(selected),
            per_profile=[dict(profile=v['profile'],rmse_change_vs_legacy=v['rmse_change_vs_legacy'],
                             half_minus_d9_change_vs_legacy=v['half_minus_d9_change_vs_legacy']) for v in selected])
        a.same(I.read(folder/role['role']/'summary.json'),result,'Full role aggregate and relative-change contrast',atol=1e-8,rtol=1e-10)
        results.append(result)
    a.same(I.read(folder/'results.json'),results,'Every original role result',atol=1e-8,rtol=1e-10)
    a.require(summary['profiles']==len(rows) and summary['roles']==len(results),'All evaluation coverage')
    for name,digest in I.read(folder/'input_pins.json').items():pin(a,dict(path=name,sha256=digest))
    a.require(not (folder/'failure.json').exists(),'No late evaluation failure')
    return results


def run(a,args):
    cfg=validate_config(a,args)
    folder,summary=stage(a,args.fit_dir,args.fit_sha256,'fit-predict',cfg,args)
    model_lock=obj(a,summary['model_lock']);prediction_ref=summary['prediction_lock'];lock=obj(a,prediction_ref)
    for value,kind,status in [(model_lock,'model','ALL_MODELS_LOCKED'),(lock,'prediction','ALL_ROLE_PREDICTIONS_LOCKED')]:
        a.require(value['schema']=='c18-initialization-'+kind+'-lock-v001' and value['status']==status
                  and value['dataset']==cfg['dataset'] and value['arm']==cfg['arm'] and value['source_sha256']==PRODUCER_SHA,
                  'Completed immutable '+kind+' lock')
        a.same(value['methods'],METHODS,'Lock exact five methods')
        a.require(value['held_truth_read'] is False,'Producer target-free fit/predict declaration')
        a.require(local(value['config']['path'])==local(args.config) and value['config']['sha256']==args.config_sha256,
                  'Lock exact config identity')
    a.same(lock['model_lock'],summary['model_lock'],'Prediction-to-model lock linkage')
    a.require(model_lock['actual_booster_fits']==summary['actual_booster_fits']==2
              and model_lock['actual_linear_fits']==summary['actual_linear_fits']==1,'Exactly two boosters and one linear fit per arm')
    a.require(model_lock['core_sha256']==CORE_SHA and model_lock['helper_sha256']==HELPER_SHA,'Frozen numerical dependencies')
    for name,digest in model_lock['files'].items():
        a.require(summary['files'].get(name)==digest,'Stage and model artifact hash agreement')
        pin(a,dict(path=str(folder/name),sha256=digest))
    forbidden={local(v['truth']['path']) for r in cfg['roles'] for v in r['records']}
    for receipt_name in ('fit_inputs.json','input_pins.json'):
        inputs=I.read(folder/receipt_name)
        a.require(not forbidden.intersection(local(p) for p in inputs),'Held truth absent from fit/predict decoded input receipt')
        for name,digest in inputs.items():pin(a,dict(path=name,sha256=digest))
    model,training,certificate=training_artifacts(a,cfg,folder,model_lock)
    a.require(summary['fit_rows']==len(training['target']),'Summary full training-row count')
    boosters={k:booster_artifacts(a,cfg,model_lock['boosters'][k],training,k) for k in ('d9','residual')}
    predictions=prediction_artifacts(a,cfg,lock,model,boosters)
    a.require(summary['held_profiles']==len(predictions),'Complete original evaluation profile coverage')
    results=evaluation_artifacts(a,args,cfg,predictions,prediction_ref) if args.evaluation_dir else None
    a.require(not (folder/'failure.json').exists(),'No late fit/predict failure')
    a.finish_pins()
    return dict(dataset=cfg['dataset'],arm=cfg['arm'],fit_profiles=len(cfg['fit_ids']),held_profiles=len(predictions),
        linear_certificate=certificate,evaluation_audited=results is not None,cohorts=results,
        fit_node_counts_scope='First, middle and final tree of each booster; optimizer and leaf training not replayed',
        prediction_scope='All trees and all retained rows of every original nonFIT profile',
        feature_scope='Independent explicit recurrence from previously raw-source-audited operating caches plus analytic initial-state difference')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('config','config-sha256','fit-dir','fit-sha256','producer-sha256','output-dir'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--evaluation-dir');parser.add_argument('--evaluation-sha256')
    parser.add_argument('--max-seconds',type=float,default=1200.)
    args=parser.parse_args()
    if bool(args.evaluation_dir)!=bool(args.evaluation_sha256):raise ValueError('Evaluation directory and summary SHA are paired')
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=1800:raise ValueError('Finite bounded audit cap <=1800s')
    out=local(args.output_dir)
    if WORK/'runs' not in out.parents:raise ValueError('New C18 audit output only')
    out.mkdir(parents=True,exist_ok=False);a=I.Audit(args.max_seconds);results={}
    try:
        for filename,digest in [('c18_capacity_audit_v002.py',CAPACITY_AUDIT_SHA),('c18_diagnostic_audit_v001.py',DIAGNOSTIC_AUDIT_SHA)]:
            pin(a,dict(path=str(Path(__file__).with_name(filename)),sha256=digest))
        results=run(a,args)
    except Exception as exc:a.failures.append(dict(check='All mandatory audit operations completed',detail=type(exc).__name__+': '+str(exc)))
    report=dict(schema='c18-initialization-independent-audit-v001',status='HOLD' if a.failures else 'PASS',
        auditor_sha256=sha(__file__),producer_sha256=args.producer_sha256,config_sha256=args.config_sha256,
        fit_summary_sha256=args.fit_sha256,evaluation_summary_sha256=args.evaluation_sha256,
        checks=a.checks,failures=a.failures,inputs=a.inputs,replayed=a.replayed,results=results,
        elapsed_seconds=time.monotonic()-a.start,no_scientific_producer_import=True,no_pickle_execution=True,
        scope='Independent initialization features, FIT ridge equations/scales/weights, saved raw-tree forward and complete descriptive metrics; no optimizer replay or new formal test')
    raw=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)
    if len(raw.encode('utf-8'))>100000000:raise RuntimeError('Audit report100MB ceiling')
    with (out/'report.json').open('x',encoding='utf-8') as stream:stream.write(raw)
    print(json.dumps(dict(status=report['status'],checks=a.checks,failures=len(a.failures))))
    return 2 if a.failures else 0


if __name__=='__main__':raise SystemExit(main())
