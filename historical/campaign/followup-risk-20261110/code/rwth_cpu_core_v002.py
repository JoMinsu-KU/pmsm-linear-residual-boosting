"""Frozen-protocol RWTH numerical core. No execution at import.

All learning callers supply only fixed FIT records. No CSV or held-target loader.
"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

PROTOCOL_SHA='f1d201fe4aa6243f8c77c6a8a85a91268d183d0cd24d48bc35ec86b269082b0f'
PREP_SHA='ab7d82514f84ad934097fd15ccf83b6f02988c1207b87678ac2314595c95b3a7'
LAMBDAS=[.0001,.01,1.]
FAMILIES=['positive25','free25','linear33']
GRID=[dict(max_leaf_nodes=l,max_iter=n,l2_regularization=p) for l,n,p in
      [(15,200,.1),(15,500,.1),(15,1000,.1),(31,200,.1),(31,500,.1),(31,1000,.1),(7,200,.1),(7,500,.1),(15,100,1.)]]
BASE=dict(learning_rate=.05,min_samples_leaf=20,max_bins=255,loss='squared_error',
          early_stopping=False,random_state=2026111201)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def array_sha(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def load_fit(manifest_entry,clock,root):
    path=clock.pin(manifest_entry)
    manifest=json.loads(path.read_text(encoding='utf-8-sig'))
    if (manifest['schema']!='rwth-cache-manifest-v002' or manifest['status']!='READY_FIT_ONLY'
        or manifest['dataset']!='rwth48v' or manifest['source_sha256']!=PREP_SHA
        or manifest['protocol']['sha256']!=PROTOCOL_SHA or manifest['warmup']!=0
        or manifest['train_stride']!=5 or manifest['evaluation_stride']!=1
        or manifest['calibration_member_contents_opened'] is not False
        or manifest['confirmation_member_contents_opened'] is not False):
        raise ValueError('Exact unexposed FIT-only preparation required')
    clock.pin(manifest['protocol']);clock.pin(manifest['partition'])
    clock.pin(dict(path='followup-risk-20261110/code/rwth_prepare_v002.py',sha256=PREP_SHA))
    if (path.parent/'failure.json').exists():raise ValueError('Preparation failure marker')
    records={};ids=manifest['roles']['fit']
    if len(ids)!=20 or len(set(ids))!=20 or [e['id'] for e in manifest['records']]!=ids:
        raise ValueError('Exact twenty FIT records required')
    for e in manifest['records']:
        if e['role']!='fit' or e['truth'] is not None or e['target_key']!='target':raise ValueError('FIT label role contract')
        with np.load(clock.pin(e['features']),allow_pickle=False) as z:
            data={k:z[k].copy() for k in z.files}
        if set(data)!={'profile_id','index','source_row','time','operating','X33','target'}:raise ValueError('FIT exact cache fields')
        n=e['rows'];index=np.arange(n,dtype=np.int64)
        if (data['profile_id'].shape!=() or int(data['profile_id'])!=e['id']
            or data['index'].dtype!=np.int64 or data['source_row'].dtype!=np.int64
            or not np.array_equal(data['index'],index) or not np.array_equal(data['source_row'],index)
            or data['X33'].shape!=(n,33) or data['operating'].shape!=(n,4)
            or data['target'].shape!=(n,) or data['time'].shape!=(n,)
            or any(not np.isfinite(v).all() for v in data.values()) or not np.all(np.diff(data['time'])>0)):
            raise ValueError('FIT finite feature/target/index geometry')
        data.update(id=e['id'],source=e['features']);records[e['id']]=data
    return manifest,records


def indices(record):
    return np.arange(0,len(record['target']),5,dtype=np.int64)


def scaling(records,ids):
    x=np.concatenate([records[p]['X33'][indices(records[p])] for p in ids])
    y=np.concatenate([records[p]['target'][indices(records[p])] for p in ids])
    return dict(fit_ids=list(ids),x_mean=x.mean(0).tolist(),x_std=np.maximum(x.std(0),1e-8).tolist(),
        y_mid=float(np.median(y)),y_scale=max(float(np.percentile(y,75)-np.percentile(y,25)),1.),
        drive_rms=np.maximum(np.sqrt(np.mean(x[:,4:8]**2,axis=0)),1e-8).tolist())


def x_scaled(record,scale):
    return (record['X33']-np.asarray(scale['x_mean']))/np.asarray(scale['x_std'])


def design(record,scale,family):
    if family=='linear33':return np.column_stack((np.ones(len(record['X33'])),x_scaled(record,scale)))
    x=record['X33']
    return np.column_stack(((x[:,8:13]-scale['y_mid'])/scale['y_scale'],
                            x[:,13:33]/np.repeat(np.asarray(scale['drive_rms']),5)))


def row_weights(records,ids):
    counts=[len(indices(records[p])) for p in ids];n=sum(counts)
    if not all(counts):raise ValueError('Every FIT profile contributes')
    return np.concatenate([np.full(k,n/(len(ids)*k),dtype=np.float64) for k in counts])


def solve_qp(q,r,constant,positive):
    np.linalg.cholesky(q)
    n=len(r);e=np.r_[np.ones(5),np.zeros(n-5)] if positive else np.zeros(n)
    nu=0.;iteration=1
    if positive:
        theta=np.zeros(n);theta[0]=1.;free=[0]
        tol=2e-10*(1+float(np.max(np.abs(r))))
        for iteration in range(1,1001):
            f=np.asarray(sorted(free),dtype=np.int64);qf=q[np.ix_(f,f)];ef=e[f]
            solved=np.linalg.solve(qf,np.column_stack((r[f],ef)))
            denominator=float(ef@solved[:,1])
            if denominator<=0:raise ArithmeticError('Active face lost feasible coolant coordinate')
            nu=float((ef@solved[:,0]-1)/denominator)
            candidate=np.zeros(n);candidate[f]=solved[:,0]-nu*solved[:,1]
            negative=[j for j in f if candidate[j]<-1e-12]
            if negative:
                alpha,hit=min((theta[j]/(theta[j]-candidate[j]),int(j)) for j in negative)
                theta+=alpha*(candidate-theta);theta[hit]=0.;free.remove(hit);continue
            theta=candidate
            gradient=q@theta-r+nu*e
            inactive=[j for j in range(n) if j not in free]
            if inactive:
                add=min(inactive,key=lambda j:(gradient[j],j))
                if gradient[add]<-tol:free.append(add);continue
            break
        else:raise ArithmeticError('Positive QP active-set iteration cap')
    else:theta=np.linalg.solve(q,r)
    gradient=q@theta-r+nu*e;mu=np.maximum(gradient,0.) if positive else np.zeros(n)
    argument=r-nu*e+mu;z=np.linalg.solve(q,argument)
    primal=float(.5*theta@q@theta-r@theta+constant)
    dual=float(constant-nu-.5*argument@z)
    gap=float(.5*(theta-z)@q@(theta-z)+mu@theta+nu*(1-e@theta))
    tolerance=1e-8*(1+abs(primal));equality=abs(float(e@theta)-1) if positive else 0.
    certificate=dict(cholesky_pass=True,iterations=iteration,primal=primal,dual=dual,direct_gap=primal-dual,
        stable_gap=gap,gap_tolerance=tolerance,equality_residual=equality,min_coefficient=float(theta.min()),
        nu=nu,mu=mu.tolist(),stationarity=float(np.max(np.abs(gradient-mu))),
        complementarity=float(np.max(np.abs(mu*theta))),positive_constraints=positive)
    if (not np.isfinite(theta).all() or equality>1e-10 or (positive and theta.min()<-1e-10)
        or gap<-1e-10 or gap>tolerance or abs(primal-dual-gap)>tolerance):
        raise ArithmeticError('QP fixed numerical qualification failed: '+json.dumps(certificate))
    return theta,certificate


def fit_prior(records,ids,family,lam,clock):
    if family not in FAMILIES or lam not in LAMBDAS:raise ValueError('Frozen prior grid')
    clock.check(training=True);scale=scaling(records,ids)
    a=np.concatenate([design(records[p],scale,family)[indices(records[p])] for p in ids])
    u=np.concatenate([(records[p]['target'][indices(records[p])]-scale['y_mid'])/scale['y_scale'] for p in ids])
    w=row_weights(records,ids)/len(u);penalty=np.ones(a.shape[1])
    if family=='linear33':penalty[0]=0.
    q=2*(a.T@(w[:,None]*a)+lam*np.diag(penalty));r=2*a.T@(w*u);constant=float(w@(u*u))
    coef,cert=solve_qp(q,r,constant,family=='positive25')
    clock.check(training=True)
    return dict(family=family,lam=lam,fit_ids=list(ids),scale=scale,coef=coef.tolist(),
        Q=q.tolist(),r=r.tolist(),constant=constant,penalty=penalty.tolist(),certificate=cert,
        training_design_sha256=array_sha(a),training_target_sha256=array_sha(u),training_weights_sha256=array_sha(w))


def predict_prior(model,record):
    value=model['scale']['y_mid']+model['scale']['y_scale']*(design(record,model['scale'],model['family'])@np.asarray(model['coef']))
    if not np.isfinite(value).all():raise FloatingPointError('Finite prior prediction required')
    return value


def fit_boost(records,ids,grid_id,directory,clock,prior=None,save_pickle=False):
    if grid_id not in range(9):raise ValueError('Fixed nine-grid ID required')
    directory.mkdir(parents=True,exist_ok=False);clock.check(training=True);start=time.monotonic()
    scale=scaling(records,ids);xs=[];ys=[];profiles=[];rows=[]
    for pid in ids:
        rec=records[pid];ix=indices(rec)
        xs.append(x_scaled(rec,scale)[ix]);ys.append(rec['target'][ix]-(predict_prior(prior,rec)[ix] if prior is not None else 0.))
        profiles.append(np.full(len(ix),pid,dtype=np.int64));rows.append(ix)
    x,y=np.concatenate(xs),np.concatenate(ys);w=row_weights(records,ids)
    model=HistGradientBoostingRegressor(**dict(BASE,**GRID[grid_id]))
    model.fit(x,y,sample_weight=w);clock.check(training=True)
    if model.n_iter_!=GRID[grid_id]['max_iter']:raise ArithmeticError('Every fixed HGB iteration required')
    nodes=dict(baseline=model._baseline_prediction.copy(),n_iter=np.asarray([model.n_iter_],dtype=np.int64),
               n_bins_non_missing=model._bin_mapper.n_bins_non_missing_.copy())
    for j,stage in enumerate(model._predictors):
        if len(stage)!=1:raise ValueError('Single-target booster required')
        nodes['tree_'+str(j).zfill(4)]=stage[0].nodes.copy()
    for j,threshold in enumerate(model._bin_mapper.bin_thresholds_):nodes['bin_thresholds_'+str(j).zfill(2)]=threshold.copy()
    np.savez_compressed(directory/'trees.npz',**nodes)
    np.savez_compressed(directory/'training.npz',profile_id=np.concatenate(profiles),index=np.concatenate(rows),
                        source_row=np.concatenate(rows),target=y,weight=w)
    if save_pickle:
        with (directory/'model.pkl').open('xb') as f:pickle.dump(model,f,protocol=5)
    metadata=dict(schema='rwth-hgb-model-v002',fit_ids=list(ids),grid_id=grid_id,parameters=dict(BASE,**GRID[grid_id]),
        scale=scale,training_rows=len(y),training_X_sha256=array_sha(x),training_target_sha256=array_sha(y),
        training_weight_sha256=array_sha(w),target_kind='prior_residual' if prior is not None else 'rotor_celsius',
        prior_family=None if prior is None else prior['family'],histogram_subsample=None,
        histogram_subsample_reason='All FIT sampled rows below200000; no binning subsample',
        numpy=np.__version__,elapsed_seconds=time.monotonic()-start,
        files={p.name:sha(p) for p in sorted(directory.iterdir()) if p.is_file()})
    if len(y)>200000:raise ValueError('New unsupported histogram subsampling scope')
    dump(directory/'model.json',metadata);clock.check(training=True)
    return model,metadata


def predict_boost(model,metadata,record,prior=None):
    value=model.predict(x_scaled(record,metadata['scale']))
    if prior is not None:value=value+predict_prior(prior,record)
    if not np.isfinite(value).all():raise FloatingPointError('Finite full booster prediction')
    return value.astype(np.float64)


def synthetic_fixtures():
    q=2*np.eye(25);r=np.zeros(25)
    theta,cert=solve_qp(q,r,0.,True)
    if not np.allclose(theta[:5],.2,atol=1e-12,rtol=0) or not np.allclose(theta[5:],0,atol=1e-12,rtol=0):
        raise ArithmeticError('Five-coolant simplex analytic fixture')
    if cert['stable_gap']>1e-12:raise ArithmeticError('Synthetic convex qualification')
    free,_=solve_qp(q,np.arange(25,dtype=float),0.,False)
    if not np.array_equal(free,np.arange(25,dtype=float)/2):raise ArithmeticError('Free QP analytic fixture')
    return dict(status='PASS',fixtures=['five_boundary_simplex','free_linear_analytic'],no_dataset_values=True)
