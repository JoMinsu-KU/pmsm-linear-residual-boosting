"""Independent staged C15 paper artifact audit; no worker import or retraining.

Root registers every actual invocation. Default/help and --stage synthetic read
no data. A separate linked admission is required before any CONF value access.
All evaluated points are replayed from saved numeric tree nodes, never pickle.
CV training-node counts use five predeclared iterations; final training counts
use every saved tree. Split optimization / leaf fitting is NOT re-executed.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np
from threadpoolctl import threadpool_limits

WORKER = '40b49b6d564084bdc92bcbbb944b95f1b53999233352e4b0dc80e55c35e728d5'
CORE = '17c47c41c27b6e92618a44de0f5860f80ffd1357d998a4cfeb805f5e2d8b7b37'
HELPER = '31289f420877fbcbf64343fd1515ea8f56cdbf769685c8d5778a932b98f50b3e'
PRIMARY = 'positive_rc_insample_boost'
COMPARATOR = 'direct9_fitcv_selected'
LAMBDAS = [.0001, .01, 1.]
FAMILIES = ['positive', 'free', 'ewma', 'linear33', 'nn_only', 'eq_only', 'unconstrained25']
FACTORIAL = ['positive', 'nn_only', 'eq_only', 'unconstrained25']
GRID = [(15,200,.1),(15,500,.1),(15,1000,.1),(31,200,.1),(31,500,.1),
        (31,1000,.1),(7,200,.1),(7,500,.1),(15,100,1.)]
DIRECT = ['direct_boost','direct_boost_15x500','direct_boost_15x1000',
          'direct_boost_31x200','direct_boost_31x500','direct_boost_31x1000',
          'direct_boost_7x200','direct_boost_7x500','direct_boost_15x100_l2_1']
OLD15 = ['positive_rc','unconstrained_rc_ridge','direct_boost','positive_rc_oof_boost',
         'unconstrained_rc_oof_boost','mean_A_B','ewma_ridge',PRIMARY,
         'unconstrained_rc_insample_boost'] + DIRECT[1:6] + ['direct_boost_fitcv_selected']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()


def array_sha(x):
    x=np.ascontiguousarray(x); h=hashlib.sha256()
    h.update(str(x.dtype).encode('ascii')); h.update(json.dumps(list(x.shape)).encode('ascii'))
    h.update(x.tobytes()); return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def arrays(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}


def resolve(root,path):
    p=Path(path); return p.resolve() if p.is_absolute() else (root/p).resolve()


class Audit:
    def __init__(self,seconds):
        self.start=time.monotonic(); self.seconds=seconds; self.checks=0
        self.failures=[]; self.bounds=[]; self.bin_diagnostics=[]; self.inputs={}
        self.coverage=[]; self.results={}; self.replayed={}

    def budget(self):
        if time.monotonic()-self.start>self.seconds: raise TimeoutError('Registered audit stage cap')

    def check(self,condition,message,detail=None):
        self.budget(); self.checks+=1
        if not bool(condition): self.failures.append(dict(check=message,detail=detail))

    def require(self,condition,message):
        self.check(condition,message)
        if not condition: raise ValueError(message)

    def equal(self,a,b,message):
        self.check(np.array_equal(np.asarray(a),np.asarray(b)),message)

    def close(self,a,b,message,atol=2e-9,rtol=2e-10):
        aa,bb=np.asarray(a),np.asarray(b)
        okay=aa.shape==bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all()
        if okay: okay=bool(np.allclose(aa,bb,atol=atol,rtol=rtol))
        detail=None if okay else dict(shape_a=list(aa.shape),shape_b=list(bb.shape),atol=atol,rtol=rtol,
            maximum_error=float(np.max(np.abs(aa-bb))) if aa.shape==bb.shape and aa.size else None)
        self.check(okay,message,detail)

    def entry(self,root,entry):
        p=resolve(root,entry['path']); actual=sha(p)
        self.require(actual==entry['sha256'],'Pinned input '+str(p))
        self.inputs[str(p)]=actual; return p


def helper(a):
    p=Path(__file__).with_name('audit_combination_c12_fit_rc_v002.py')
    a.require(sha(p)==HELPER,'Pinned independently authored C12 audit helper')
    spec=importlib.util.spec_from_file_location('independent_c12_audit_helper',p)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    a.inputs[str(p)]=HELPER; return module


def specs():
    d={'positive_rc':dict(kind='linear',prior='positive__tuned'),
       'unconstrained_rc_ridge':dict(kind='linear',prior='free__tuned'),
       'ewma_ridge':dict(kind='linear',prior='ewma__tuned'),
       'linear33_ridge':dict(kind='linear',prior='linear33__tuned'),
       'positive_rc_oof_boost':dict(kind='residual',prior='positive__tuned',target='oof'),
       'unconstrained_rc_oof_boost':dict(kind='residual',prior='free__tuned',target='oof'),
       PRIMARY:dict(kind='residual',prior='positive__tuned',target='insample'),
       'unconstrained_rc_insample_boost':dict(kind='residual',prior='free__tuned',target='insample'),
       'linear33_insample_boost':dict(kind='residual',prior='linear33__tuned',target='insample'),
       'mean_A_B':dict(kind='mean',components=['positive_rc','direct_boost']),
       'direct_boost_fitcv_selected':dict(kind='selected',grid_count=6),
       COMPARATOR:dict(kind='selected',grid_count=9)}
    for j,n in enumerate(DIRECT): d[n]=dict(kind='direct',grid_index=j)
    for family in FACTORIAL:
        for mode in ('common','tuned'):
            if family=='positive' and mode=='tuned': continue
            name='ablation_'+family+'_'+mode; prior=family+'__'+mode
            d[name]=dict(kind='linear',prior=prior)
            d[name+'_residual']=dict(kind='residual',prior=prior,target='insample')
    return {n:d[n] for n in OLD15+[n for n in d if n not in OLD15]}


def expected_jobs(ids):
    jobs=[]
    for f in FAMILIES:
        for li,l in enumerate(LAMBDAS):
            for p in ids: jobs.append(dict(job_id=f'linear_cv__{f}__l{li}__p{p}',kind='linear_cv',family=f,lambda_index=li,lam=l,held=p))
    for f in FAMILIES: jobs.append(dict(job_id='linear_final__'+f+'__tuned',kind='linear_final',family=f,mode='tuned'))
    for f in FACTORIAL: jobs.append(dict(job_id='linear_final__'+f+'__common',kind='linear_final',family=f,mode='common'))
    for gi in range(9):
        for p in ids: jobs.append(dict(job_id=f'direct_cv__g{gi}__p{p}',kind='direct_cv',grid_index=gi,held=p))
    d=specs()
    for n in sorted(d,key=lambda n:bool(d[n].get('prior','').endswith('__common'))):
        if d[n]['kind'] in ('direct','residual'): jobs.append(dict(job_id='full__'+n,kind='full',method=n,spec=d[n]))
    return {j['job_id']:j for j in jobs}


def load_manifest(a,args):
    root=Path(args.project_root).resolve(); p=resolve(root,args.manifest); d=read(p)
    a.inputs[str(p)]=sha(p)
    a.require(d['schema']=='c15-paper-dataset-v001' and d['scientific_execution_admitted'] is True,'Admitted dataset manifest')
    a.require(d['worker_sha256']==WORKER and d['core_sha256']==CORE,'Exact reviewed producer source pins')
    for filename,digest in [('c15_paper_suite_v001.py',WORKER),('c15_paper_core_v001.py',CORE)]:
        a.require(sha(Path(__file__).with_name(filename))==digest,'Producer source retained: '+filename)
    ids=[p for role in d['roles'].values() for p in role]
    a.require(len(ids)==len(set(ids)) and all(type(p) is int for p in ids),'Disjoint integer profile roles')
    a.require([v['id'] for v in d['fit_cache']]==d['roles']['fit'],'Exact ordered FIT cache IDs')
    for entry in d['source_files'].values(): a.entry(root,entry)
    if d['dataset']=='ipmsm':
        a.require(d['roles']==dict(fit=[0,2,3,5,7,8,11,14],calibration=[1,6,12],pilot=[4,9,13],confirmation=[10,15]),'Unchanged original profile roles')
    else:
        split=read(a.entry(root,d['source_files']['split']))
        a.require(d['roles']==split['roles'],'Unchanged external premeasurement split')
        inventory=read(a.entry(root,d['source_files']['inventory']))
        ordered=sorted(inventory['profile_ids'],key=lambda p:(hashlib.sha256(('c15-emt-v001:'+str(p)).encode()).hexdigest(),p))
        a.require(ordered==d['roles']['fit']+d['roles']['development']+d['roles']['confirmation'],'Salted metadata-only 41/14/14 order')
        a.require([len(d['roles'][r]) for r in ('fit','development','confirmation')]==[41,14,14],'Exact external profile counts')
    return root,p,d


def raw_records(a,root,d,ids,source,targets=True):
    """Decode only explicitly requested profiles; other measurement tokens stay bytes."""
    columns=source.get('operating_columns',d['evaluation_source']['operating_columns'])
    target=source.get('target_column',d['evaluation_source']['target_column'])
    raw={}
    if source['format']=='profile_csv':
        entries={v['id']:v for v in source['files']}
        for p in ids:
            path=a.entry(root,entries[p]); op=[]; yy=[]
            with path.open(encoding='utf-8-sig',newline='') as f:
                for row in csv.DictReader(f):
                    op.append([float(row[k]) for k in columns])
                    if targets: yy.append(float(row[target]))
            a.require(len(op)==entries[p]['rows'],'Exact raw row count '+str(p))
            raw[p]=dict(operating=np.asarray(op,np.float64),source_row=np.arange(len(op),dtype=np.int64))
            if targets: raw[p]['y']=np.asarray(yy,np.float64)
    else:
        path=a.entry(root,d['source_files'][source['csv_source_key']]); inv=read(a.entry(root,d['source_files'][source['inventory_source_key']]))
        op={p:[] for p in ids}; yy={p:[] for p in ids}; rows={p:[] for p in ids}; counts={p:0 for p in inv['profile_ids']}
        with path.open('rb') as f:
            header=next(csv.reader([f.readline().decode('utf-8-sig')]))
            a.require(header==inv['header'],'Exact external raw header')
            ip=header.index('profile_id'); ix=[header.index(k) for k in columns]; iy=header.index(target)
            n=0
            for n,line in enumerate(f,1):
                if n%4096==0: a.budget()
                tokens=line.rstrip(b'\r\n').split(b',')
                if len(tokens)!=len(header): raise ValueError('Raw CSV field count')
                token=tokens[ip].strip(); pid=int(token.split(b'.',1)[0]); counts[pid]+=1
                if pid in op:
                    op[pid].append([float(tokens[j]) for j in ix]); rows[pid].append(n-1)
                    if targets: yy[pid].append(float(tokens[iy]))
        a.require(n==inv['row_count'] and all(counts[p]==inv['profile_row_counts'][str(p)] for p in counts),'All metadata profile counts')
        for p in ids:
            raw[p]=dict(operating=np.asarray(op[p],np.float64),source_row=np.asarray(rows[p],np.int64))
            if targets: raw[p]['y']=np.asarray(yy[p],np.float64)
    for p,r in raw.items():
        a.require(r['operating'].ndim==2 and r['operating'].shape[1]==7 and len(r['operating'])>332,'Valid profile geometry '+str(p))
        a.require(all(np.isfinite(v).all() for v in r.values()),'No nonfinite raw values '+str(p))
    return raw


def fit_records(a,h,root,d):
    raw=raw_records(a,root,d,d['roles']['fit'],d['audit_fit_sources']); records={}
    for item in d['fit_cache']:
        p=item['id']; r=raw[p]; f=h.features(r['operating'],a)
        f['y']=r['y']; records[p]=f
        cache=arrays(a.entry(root,item))
        a.require(len(r['y'])==item['rows'],'FIT cache raw length '+str(p))
        a.equal(cache['operating'],r['operating'],'FIT raw operating '+str(p))
        a.equal(cache[item.get('target_key','target')],r['y'],'FIT raw target '+str(p))
        if 'source_row' in cache: a.equal(cache['source_row'],r['source_row'],'FIT raw global rows '+str(p))
        for key in ('instant','boundary','heat','ewma'): a.close(cache[key],f[key],'Independent causal recurrence '+str(p)+'/'+key)
        # Preserve validated producer arithmetic for exact feature hashes and
        # split-boundary replay; the independent recurrence comparison above
        # is still performed on every value, without widening its tolerance.
        f['_stored_features']={k:cache[k] for k in ('instant','boundary','heat','ewma')}
    return records


def load_jobs(a,root,mp,d,args):
    lock=None
    if args.model_lock:
        lp=resolve(root,args.model_lock); lock=read(lp); a.inputs[str(lp)]=sha(lp)
        a.require(lock['schema']=='c15-paper-model-lock-v001' and lock['manifest_sha256']==sha(mp),'Model lock identity')
        a.require(lock['worker_sha256']==WORKER and lock['core_sha256']==CORE,'Model lock source identity')
        a.require(lock['fit_ids']==d['roles']['fit'] and lock['methods']==specs(),'Model lock FIT/methods')
        items=list(lock['entries'].values())
    else:
        dep=resolve(root,args.dependencies); source=read(dep); a.inputs[str(dep)]=sha(dep)
        a.require(source['schema']=='c15-paper-dependencies-v001' and source['manifest_sha256']==sha(mp),'Dependency manifest identity')
        items=source['jobs']
    valid=expected_jobs(d['roles']['fit']); jobs={}
    for item in items:
        folder=resolve(root,item['path']); receipt=read(folder/'receipt.json'); key=receipt['job']['job_id']
        a.require(sha(folder/'receipt.json')==item['receipt_sha256'],'Receipt pinned '+key)
        a.inputs[str(folder/'receipt.json')]=item['receipt_sha256']
        a.require(key in valid and key not in jobs and receipt['job']==valid[key],'Fixed unique job '+key)
        a.require(receipt['status']=='FIT_JOB_COMPLETE' and not (folder/'failure.json').exists(),'Completed job '+key)
        a.require(receipt['manifest_sha256']==sha(mp) and receipt['worker_sha256']==WORKER and receipt['core_sha256']==CORE,'Receipt lineage '+key)
        for name,digest in receipt['files'].items():
            p=(folder/name).resolve(); a.require(folder in p.parents,'Artifact stays in job directory')
            a.entry(folder,dict(path=name,sha256=digest))
        model=read(folder/'model.json')
        a.require(model['job']==valid[key] and model['fit_role_ids']==d['roles']['fit'] and model['manifest_sha256']==sha(mp),'Model job lineage '+key)
        jobs[key]=dict(root=folder,model=model,receipt=receipt)
    if lock: a.require(set(jobs)==set(valid),'All planned jobs present at model lock')
    return jobs,lock


def independent_scale(h,records,ids,family):
    scale=h.scaler(records,ids)
    if family=='linear33':
        f=np.concatenate([h.boost_x(records[p],scale)[h.indices(records[p])] for p in ids])
        scale['linear33_mean']=f.mean(0); scale['linear33_scale']=np.maximum(f.std(0),1e-8)
    return scale


def design(h,record,scale,family):
    if family=='linear33':
        f=h.boost_x(record,scale)
        return np.column_stack([np.ones(len(f)),(f-scale['linear33_mean'])/scale['linear33_scale']])
    return h.design(record,scale,'positive' if family in FACTORIAL else family)


def linear_pred(h,model,record):
    s=model['scale']; return s['y_mid']+s['y_scale']*(design(h,record,s,model['family'])@model['coef'])


def verify_linear(a,h,model,records,ids,family,lam,tag):
    if family in ('positive','free','ewma'):
        h.verify_linear(a,model,records,ids,family,lam,tag)
    else:
        a.require(model['family']==family and model['lam']==lam,tag+': family/lambda')
        scale=independent_scale(h,records,ids,family)
        for k,v in scale.items():
            if k=='ids': a.equal(model['scale'][k],v,tag+': fold-only IDs')
            else: a.close(model['scale'][k],v,tag+': fold-only '+k)
        xs=[design(h,records[p],scale,family)[h.indices(records[p])] for p in ids]
        ys=[(records[p]['y'][h.indices(records[p])]-scale['y_mid'])/scale['y_scale'] for p in ids]
        x,y=np.concatenate(xs),np.concatenate(ys); w=np.concatenate([np.full(len(v),1/(len(ids)*len(v))) for v in ys])
        b=np.asarray(model['coef']); dim=34 if family=='linear33' else 25
        a.require(b.shape==(dim,) and np.isfinite(b).all(),tag+': coefficient shape')
        penalty=np.ones(dim)
        if family=='linear33': penalty[0]=0
        a.equal(model['penalty'],penalty,tag+': ridge penalty/intercept')
        q=x.T@(w[:,None]*x)+lam*np.diag(penalty); r=x.T@(w*y); constant=.5*np.dot(w,y*y)
        np.linalg.cholesky(q); a.check(True,tag+': SPD Hessian')
        nn,eq=family=='nn_only',family=='eq_only'; e=np.r_[np.ones(10),np.zeros(dim-10)] if eq else np.zeros(dim)
        cert=model['certificate']; nu=float(cert['nu']); mu=np.asarray(cert['mu'])
        a.require(mu.shape==b.shape and np.isfinite(mu).all() and np.all(mu>=0),tag+': feasible dual')
        if not nn: a.equal(mu,np.zeros(dim),tag+': absent inequality dual')
        if not eq: a.close(nu,0.,tag+': absent equality dual',atol=0,rtol=0)
        z=np.linalg.solve(q,r-nu*e+mu)
        primal=.5*np.dot(w,(y-x@b)**2)+.5*lam*np.dot(penalty*b,b)
        dual=constant-nu-.5*np.dot(r-nu*e+mu,z)
        gap=.5*(b-z)@q@(b-z)+mu@b+nu*(1-e@b)
        tolerance=1e-8*(1+abs(cert['primal']))
        expected=dict(primal=primal,dual=dual,direct_gap=cert['primal']-cert['dual'],stable_gap=gap,
            gap_tolerance=tolerance,equality_residual=abs(e@b-1) if eq else 0.,min_coefficient=b.min(),
            stationarity=np.max(np.abs(q@b-r+nu*e-mu)),complementarity=np.max(np.abs(mu*b)))
        for k,v in expected.items(): a.close(cert[k],v,tag+': independent '+k,atol=2e-8 if k in ('primal','dual') else 2e-9)
        a.check(cert['nonnegative']==nn and cert['equality']==eq,tag+': certificate constraint identity')
        a.check((not nn or b.min()>=-1e-10) and (not eq or abs(e@b-1)<=1e-10),tag+': primal feasibility')
        a.check(-1e-10<=cert['stable_gap']<=tolerance and abs(cert['direct_gap']-cert['stable_gap'])<=tolerance,tag+': original unchanged gap qualification')
        a.check(-3e-9<=gap<=tolerance+2e-9 and abs(primal-dual-gap)<=tolerance+4e-8,tag+': independent floating certificate')
        a.bounds.append(dict(fit=tag,objective_normalized=float(primal),dual_normalized=float(dual),stable_gap_normalized=float(gap),approximate_floating_point_certificate=True))
    intercept=family in ('free','ewma','linear33')
    a.check(model['constraint_flags']==dict(nonnegative=family in ('positive','nn_only'),equality=family in ('positive','eq_only'),intercept=intercept,intercept_penalized=False),tag+': declared constraint geometry')


def metrics(pred,truth):
    err=np.asarray(pred)-np.asarray(truth)
    return dict(n=len(err),squared_error_sum=float(np.dot(err,err)),rmse=float(np.sqrt(np.dot(err,err)/len(err))),
        mae=float(np.abs(err).mean()),max_absolute_error=float(np.abs(err).max()),max_underprediction=float(max(0.,-err.min())))


def verify_metrics(a,saved,pred,y,tag):
    value=metrics(pred,y)
    for k,v in value.items(): a.close(saved[k],v,tag+': '+k,atol=2e-8)
    return value


def selection(a,jobs,ids,family=None,count=None):
    rows=[]
    for j in range(3 if family else count):
        prefix=f'linear_cv__{family}__l{j}__p' if family else f'direct_cv__g{j}__p'
        values=[]
        for p in ids:
            key=prefix+str(p); a.require(key in jobs,'Complete selection fold '+key)
            values.append(jobs[key]['model']['held_metrics']['rmse'])
        row=dict(profile_rmse=values,macro_rmse=float(np.mean(values)))
        if family: row['lam']=LAMBDAS[j]
        else: row.update(method=DIRECT[j],grid_index=j)
        rows.append(row)
    best=min(range(len(rows)),key=lambda j:(rows[j]['macro_rmse'],j))
    return best,rows


def compare_rows(a,saved,expected,tag):
    a.require(len(saved)==len(expected),tag+': complete row count')
    for j,(s,e) in enumerate(zip(saved,expected)):
        a.require(set(s)==set(e),tag+': schema '+str(j))
        for k,v in e.items():
            if isinstance(v,str): a.check(s[k]==v,tag+': '+k)
            else: a.close(s[k],v,tag+': '+k)


def resolve_alias(a,jobs,key):
    seen=set()
    while 'alias_job_id' in jobs[key]['model']:
        a.require(key not in seen,'Acyclic alias'); seen.add(key); key=jobs[key]['model']['alias_job_id']
        a.require(key in jobs,'Alias dependency present')
    return key,jobs[key]


def training(h,records,ids):
    s=h.scaler(records,ids); indices={p:h.indices(records[p]) for p in ids}
    ys=[records[p]['y'][indices[p]] for p in ids]; y=np.concatenate(ys)
    x=np.concatenate([h.boost_x(records[p],s)[indices[p]] for p in ids])
    w=len(y)*np.concatenate([np.full(len(v),1/(len(ids)*len(v))) for v in ys])
    return x,y,w,s,indices


def tree(a,node,x,tag,counts=False):
    needed={'value','count','feature_idx','num_threshold','left','right','is_leaf','is_categorical'}
    a.require(needed<=set(node.dtype.names or []),'Numeric tree node schema '+tag)
    a.require(np.all(node['is_categorical']==0) and np.isfinite(node['value']).all() and np.isfinite(node['num_threshold']).all(),'Finite numerical tree '+tag)
    leaf=node['is_leaf'].astype(bool); state=np.zeros(len(x),dtype=np.int64)
    visits=np.zeros(len(node),dtype=np.int64); visits[0]=len(x)
    for depth in range(len(node)+1):
        a.budget(); moving=np.flatnonzero(~leaf[state])
        if not len(moving): break
        at=state[moving]; j=node['feature_idx'][at].astype(np.int64)
        a.require(np.all(j<x.shape[1]),'Tree feature indices '+tag)
        to=np.where(x[moving,j]<=node['num_threshold'][at],node['left'][at],node['right'][at]).astype(np.int64)
        a.require(np.all(to>at) and np.all(to<len(node)),'Forward acyclic child indices '+tag)
        state[moving]=to
        if counts: visits+=np.bincount(to,minlength=len(node))
    else: raise ValueError('Nonterminating tree')
    if counts: a.equal(node['count'],visits,'All training node counts '+tag)
    return node['value'][state]


def verify_booster_schema(a,data,info,tag):
    n=info['configuration']['max_iter']; leaves=info['configuration']['max_leaf_nodes']
    required={'baseline','n_iter'}|{'tree_%03d'%j for j in range(n)}|{'bin_thresholds_%02d'%j for j in range(33)}
    a.require(set(data)==required,'Exact exported tree/bin keys '+tag)
    a.equal(data['n_iter'],[n],'Fixed iteration count '+tag)
    a.require(data['baseline'].shape==(1,1) and np.isfinite(data['baseline']).all(),'Finite scalar initial response '+tag)
    a.close(info['model']['baseline'],data['baseline'],'Export baseline metadata '+tag)
    for j in range(n):
        node=data['tree_%03d'%j]
        a.check(np.count_nonzero(node['is_leaf'])<=leaves,'Leaf capacity '+tag+'/'+str(j))
    return n


def replay(a,data,x,tag):
    n=int(data['n_iter'][0]); value=np.full(len(x),data['baseline'].item(),np.float64)
    for j in range(n): value+=tree(a,data['tree_%03d'%j],x,tag+'/'+str(j))
    a.replayed[tag]=dict(rows=len(x),trees=n,all_points_all_trees=True)
    return value


def binning(a,data,x,info,saved_rows,tag):
    seed=int(np.random.RandomState(2026111201).randint(np.iinfo(np.uint32).max,dtype='u8'))
    a.check(info['binning_random_state']==seed,'sklearn1.5.1 exact derived bin seed '+tag)
    limit=info['binning_subsample']; a.check(limit==200000,'Registered sklearn binning sample cap '+tag)
    rows=np.random.RandomState(seed).choice(len(x),limit,replace=False) if len(x)>limit else np.arange(len(x),dtype=np.int64)
    a.equal(saved_rows,rows,'Exact binning row indices '+tag)
    a.check(info['binning_subsample_active']==(len(rows)<len(x)),'Binning subsample status '+tag)
    for j in range(33):
        values=np.sort(x[rows,j]); distinct=np.unique(values)
        a.require(np.isfinite(values).all() and np.max(np.abs(values))<1e290,'Finite safe binning scale '+tag)
        if len(distinct)<=255: expected=.5*(distinct[:-1]+distinct[1:])
        else:
            positions=(np.linspace(0,100,256)[1:-1]/100)*(len(values)-1)
            expected=.5*(values[np.floor(positions).astype(int)]+values[np.ceil(positions).astype(int)])
        bins=data['bin_thresholds_%02d'%j]
        a.check(np.all(np.diff(bins)>=0),'Legitimate nondecreasing percentile bins '+tag+'/'+str(j))
        a.close(bins,expected,'Independent midpoint bin reconstruction '+tag+'/'+str(j))


def audit_job(a,h,jobs,key,records,ids):
    entry=jobs[key]; info=entry['model']; job=info['job']; folder=entry['root']; kind=job['kind']
    train=[p for p in ids if p!=job['held']] if kind in ('linear_cv','direct_cv') else ids
    a.equal(info['train_ids'],train,'Exclusion and ordered training profiles '+key)
    if kind in ('linear_cv','linear_final'):
        if kind=='linear_cv': lam=job['lam']
        else:
            at,rows=selection(a,jobs,ids,family=job['family']); li=at if job['mode']=='tuned' else 0; lam=LAMBDAS[li]
            a.check(info['selected_lambda_index']==li and info['global_tuned_lambda_index']==at,'FIT-only lambda choice '+key)
            compare_rows(a,info['cv'],rows,'All lambda CV scores '+key)
            if job['mode']=='common' and at==0:
                alias='linear_final__'+job['family']+'__tuned'
                a.check(info.get('alias_job_id')==alias and info['actual_linear_fits']==0,'Exact linear alias '+key)
                a.check(info['linear_model']==jobs[alias]['model']['linear_model'],'Aliased model identity '+key)
            else: a.check(info['actual_linear_fits']==1 and 'alias_job_id' not in info,'One actual linear fit '+key)
        verify_linear(a,h,info['linear_model'],records,train,job['family'],lam,key)
        if kind=='linear_cv':
            p=job['held']; ix=h.indices(records[p]); held=arrays(folder/'held.npz')
            a.equal(held['index'],ix,'Held endpoint order '+key)
            pred=linear_pred(h,info['linear_model'],records[p])[ix]
            a.close(held['prediction'],pred,'All held linear predictions '+key,atol=2e-8)
            verify_metrics(a,info['held_metrics'],pred,records[p]['y'][ix],key)
        a.coverage.append(dict(job=key,linear_certificate=True,held_all_points=kind=='linear_cv'))
        return
    if 'alias_job_id' in info:
        spec=job['spec']; family,mode=spec['prior'].split('__')
        prior=jobs['linear_final__'+spec['prior']]['model']['linear_model']
        tuned=jobs['linear_final__'+family+'__tuned']['model']['linear_model']
        a.check(mode=='common' and prior==tuned and info['actual_booster_fits']==0,'Exact common-lambda residual alias '+key)
        alias,alias_entry=resolve_alias(a,jobs,key)
        a.check(alias_entry['model']['job']['spec']==dict(kind='residual',prior=family+'__tuned',target=spec['target']),'Alias target meaning '+key)
        a.coverage.append(dict(job=key,alias=alias,actual_fit=False)); return
    x,y,w,s,ix=training(h,records,train)
    for k,v in s.items():
        if k=='ids': a.equal(info['scale'][k],v,'Booster training-only IDs '+key)
        else: a.close(info['scale'][k],v,'Booster training-only '+k+' '+key)
    ti=arrays(folder/'training_index.npz'); target=y; prior_pred=None; spec=job.get('spec',{})
    if kind=='full' and spec['kind']=='residual':
        prior=jobs['linear_final__'+spec['prior']]['model']['linear_model']
        a.check(info['prior']==prior,'Exact saved prior dependency '+key)
        family=spec['prior'].split('__')[0]
        if spec['target']=='oof':
            li,_=selection(a,jobs,ids,family=family)
            prior_pred=np.concatenate([linear_pred(h,jobs[f'linear_cv__{family}__l{li}__p{p}']['model']['linear_model'],records[p])[ix[p]] for p in train])
            rule='selected_lambda_nonnested_OOF_residual'
        else:
            prior_pred=np.concatenate([linear_pred(h,prior,records[p])[ix[p]] for p in train]); rule='fullFIT_insample_residual'
        target=y-prior_pred; a.close(ti['prior_prediction'],prior_pred,'Exact residual-prior targets '+key,atol=2e-8)
    else: rule='raw_target'
    a.check(info['target_construction']==rule,'Residual/direct construction label '+key)
    a.equal(ti['profile_ids'],np.concatenate([np.full(len(ix[p]),p,dtype=np.int64) for p in train]),'Training profile rows '+key)
    a.equal(ti['local_indices'],np.concatenate([ix[p] for p in train]),'Training local endpoints '+key)
    a.close(ti['sample_weight'],w,'Equal total profile weights '+key)
    a.close(ti['fit_target'],target,'Degree-C training response '+key,atol=2e-8)
    # Hash the exact feature arithmetic built from independently validated
    # caches and saved fold scales. Use this same validated representation for
    # numeric split boundaries: independent scale reduction may differ by an
    # ulp even when the prescribed scale has been verified to full tolerance.
    exact_x=np.concatenate([h.boost_x(records[p]['_stored_features'],info['scale'])[ix[p]] for p in train])
    a.close(exact_x,x,'Independent versus exact producer feature arithmetic '+key)
    a.check(info['features_sha256']==array_sha(exact_x),'Exact saved feature matrix hash '+key)
    for field,value in [('target',ti['fit_target']),('sample_weight',ti['sample_weight'])]:
        a.check(info[field+'_sha256']==array_sha(value),'Saved training array hash '+key+'/'+field)
    a.check(info['fit_rows']==len(y) and info['actual_booster_fits']==1,'Training row/fit count '+key)
    gi=job['grid_index'] if kind=='direct_cv' else spec.get('grid_index',0); leaves,n,penalty=GRID[gi]
    config=dict(learning_rate=.05,min_samples_leaf=20,l2_regularization=penalty,loss='squared_error',early_stopping=False,random_state=2026111201,max_leaf_nodes=leaves,max_iter=n)
    a.check(info['configuration']==config,'Frozen HGB configuration '+key)
    data=arrays(a.entry(folder,info['model'])); verify_booster_schema(a,data,info,key)
    a.close(data['baseline'].item(),np.dot(w,target)/w.sum(),'Initial weighted residual/raw mean '+key,atol=2e-8)
    binning(a,data,exact_x,info,ti['binning_row_indices'],key)
    chosen=sorted(set([0,1,n//2,n-2,n-1])) if kind=='direct_cv' else list(range(n))
    for j in chosen: tree(a,data['tree_%03d'%j],exact_x,key+'/FIT/'+str(j),counts=True)
    if kind=='direct_cv':
        p=job['held']; held=arrays(folder/'held.npz'); index=h.indices(records[p]); a.equal(held['index'],index,'Direct held endpoints '+key)
        pred=replay(a,data,h.boost_x(records[p]['_stored_features'],info['scale'])[index],key+'/held')
        a.close(held['prediction'],pred,'Every direct held tree prediction '+key,atol=2e-8)
        verify_metrics(a,info['held_metrics'],pred,records[p]['y'][index],key)
    a.coverage.append(dict(job=key,training_nodecount_iterations=chosen,training_nodecount_all_rows=True,
        all_training_trees=kind=='full',held_prediction_all_rows_all_trees=kind=='direct_cv',split_search_and_leaf_fitting_retrained=False))


def audit_selection(a,jobs,ids,lock):
    a.require(lock is not None,'Selection audit requires final model lock')
    for count in (6,9):
        at,rows=selection(a,jobs,ids,count=count)
        a.check(lock['selected_direct'+str(count)]==DIRECT[at],'FIT-only direct'+str(count)+' choice')
        compare_rows(a,lock['direct_cv'+str(count)],rows,'All direct'+str(count)+' CV selection rows')
    a.check(lock['primary']==PRIMARY and lock['main_comparator']==COMPARATOR,'Unchanged primary contrast')
    a.check(len(specs())==35 and all(n in specs() for n in OLD15),'35 logical entries preserve old15')
    a.check(lock['selection_uses_FIT_only'] is True and lock['evaluation_operating_decoded'] is False and lock['evaluation_targets_decoded'] is False,'Model lock declared before evaluation access')
    a.check(lock['actual_booster_fits']==sum(j['model'].get('actual_booster_fits',0) for j in jobs.values()),'Actual booster fits exclude aliases')
    a.check(lock['actual_linear_fits']==sum(1 if j['model']['job']['kind']=='linear_cv' else j['model'].get('actual_linear_fits',0) for j in jobs.values()),'Actual linear fits exclude aliases')


def require_job_audits(a,args,root,jobs):
    a.require(bool(args.job_audits),'Selection consolidation requires completed independent job audits')
    covered=set(); reports=[]
    for name in args.job_audits:
        p=resolve(root,name); report=read(p); digest=sha(p); a.inputs[str(p)]=digest
        a.require(report['status']=='PASS' and report['stage']=='jobs' and report['auditor_sha256']==sha(__file__),
                  'Current independent job-audit PASS '+str(p))
        a.require(report['producer_sha256']==WORKER and report['core_sha256']==CORE,'Prior audit source linkage')
        for item in report['coverage']:
            key=item['job']; a.require(key in jobs,'Audited job belongs to current model lock')
            receipt=jobs[key]['root']/'receipt.json'
            a.require(report['inputs'].get(str(receipt))==sha(receipt),'Audited receipt is current exact model '+key)
            covered.add(key)
        reports.append(dict(path=str(p),sha256=digest))
    a.require(covered==set(jobs),'Every logical job independently audited, including aliases')
    a.results.update(all_job_audits=reports,complete_job_coverage=len(covered))


def require_selection_audit(a,args,root):
    a.require(bool(args.selection_audit),'Evaluation audit requires complete selection/job audit consolidation')
    p=resolve(root,args.selection_audit); report=read(p); a.inputs[str(p)]=sha(p)
    a.require(report['status']=='PASS' and report['stage']=='selection' and report['auditor_sha256']==sha(__file__),
              'Current independent selection audit PASS')
    model=resolve(root,args.model_lock)
    a.require(report['inputs'].get(str(model))==sha(model),'Selection audit binds current exact model lock')
    a.results['selection_audit']=dict(path=str(p),sha256=sha(p),complete_job_coverage=report['results']['complete_job_coverage'])


def inference(primary,comparison):
    """Independent profile-unit paired inference, C15 minus comparator."""
    p,b=np.asarray(primary,np.float64),np.asarray(comparison,np.float64); d=p-b; n=len(d)
    rng=np.random.Generator(np.random.PCG64(2026111601)); indices=rng.integers(0,n,size=(10000,n))
    draws=d[indices].mean(1); bounds=np.quantile(draws,[.025,.975],method='linear')
    non_ties=d[np.abs(d)>1e-10]; improved=int(np.count_nonzero(non_ties<0)); m=len(non_ties)
    pvalue=sum(math.comb(m,k) for k in range(improved,m+1))/2**m
    return dict(profile_count=n,paired_difference_C15_minus_comparator=d.tolist(),mean_difference_C=float(d.mean()),
        relative_macro_reduction=float(1-p.mean()/b.mean()),bootstrap_draws=10000,bootstrap_rng='NumPy PCG64',
        bootstrap_seed=2026111601,bootstrap_unit='paired operating profiles',bootstrap_percentile95_C=bounds.tolist(),
        bootstrap_quantile='linear',numerical_tie_tolerance_C=1e-10,ties=n-m,improved_profiles=improved,non_tied_profiles=m,
        exact_one_sided_sign_p=float(pvalue),one_sided_alpha=.025,
        directional_rule_met=bool(d.mean()<0 and pvalue<=.025),
        sign_test_estimand='direction probability, not the mean difference',
        scope='descriptive except the frozen Paderborn CONF14 contrast; no cross-motor inference')


def gate(a,args,root,mp,lock):
    if args.role!='confirmation': return
    a.require(bool(args.confirmation_admission),'Explicit root confirmation admission required BEFORE measurements')
    path=resolve(root,args.confirmation_admission); g=read(path); a.inputs[str(path)]=sha(path)
    expected=dict(schema='c15-paper-confirmation-admission-v001',manifest_sha256=sha(mp),
        model_lock_sha256=sha(resolve(root,args.model_lock)),worker_sha256=WORKER,core_sha256=CORE,
        primary=PRIMARY,main_comparator=COMPARATOR,numeric_operating_and_target_access_admitted=True)
    a.require(all(g.get(k)==v for k,v in expected.items()),'Confirmation gate exact model/source/contrast linkage')
    a.entry(root,g['confirmation_protocol'])


def audit_evaluation(a,h,args,root,mp,d,jobs,lock):
    a.require(args.role in d['roles'] and args.role!='fit','Explicit evaluation role')
    gate(a,args,root,mp,lock) # must precede even operating input access
    pp=resolve(root,args.prediction_lock); pl=read(pp); a.inputs[str(pp)]=sha(pp)
    a.require(pl['schema']=='c15-paper-prediction-lock-v001' and pl['manifest_sha256']==sha(mp),'Prediction lock dataset lineage')
    a.require(pl['model_lock_sha256']==sha(resolve(root,args.model_lock)) and pl['role']==args.role,'Prediction lock model/role lineage')
    a.require(pl['methods']==list(specs()) and pl['all_role_predictions_before_target_decode'] is True and pl['target_values_decoded'] is False,'Complete prediction lock before targets')
    ids=d['roles'][args.role]; a.require([v['id'] for v in pl['records']]==ids,'Complete ordered role profiles')
    raw=raw_records(a,root,d,ids,d['evaluation_source']); evalroot=resolve(root,args.evaluation_dir)
    # Bind the audited truth/metric files to the completed producer stage and
    # retain their hashes in the audit report for subsequent window diagnostics.
    stage_path=a.entry(evalroot,dict(path='summary.json',sha256=sha(evalroot/'summary.json')))
    stage_summary=read(stage_path)
    a.require(stage_summary['status']=='STAGE_COMPLETE_REQUIRES_AUDIT' and stage_summary['stage']=='evaluate',
              'Completed producer evaluation stage')
    a.require(stage_summary['worker_sha256']==WORKER and stage_summary['core_sha256']==CORE and stage_summary['manifest_sha256']==sha(mp),
              'Evaluation stage source/dataset lineage')
    for name,digest in stage_summary['artifact_sha256'].items():
        artifact=(evalroot/name).resolve(); a.require(evalroot in artifact.parents,'Evaluation artifact remains in stage directory')
        a.entry(evalroot,dict(path=name,sha256=digest))
    def evaluated_file(name):
        return a.entry(evalroot,dict(path=name,sha256=stage_summary['artifact_sha256'][name]))
    saved_rows=read(evaluated_file('metrics.json')); saved_summary=read(evaluated_file('metrics_summary.json')); rows=[]
    a.require(saved_summary['prediction_lock_sha256']==sha(pp) and saved_summary['role']==args.role,'Evaluation binds exact predictions')
    a.require(len(saved_rows)==35*len(ids),'All profile-method metric rows')
    models={}
    for name,s in specs().items():
        if s['kind'] in ('direct','residual'):
            key,entry=resolve_alias(a,jobs,'full__'+name)
            if key not in models:
                data=arrays(a.entry(entry['root'],entry['model']['model'])); verify_booster_schema(a,data,entry['model'],key)
                models[key]=(data,entry['model'])
    legacy=None
    if args.legacy_lock:
        lp=resolve(root,args.legacy_lock); legacy=read(lp); a.inputs[str(lp)]=sha(lp)
        a.require(d['dataset']=='ipmsm' and args.role in ('calibration','pilot'),'Legacy comparison only exposed original roles')
        for item in legacy['audit_reports']:
            report=read(a.entry(root,item)); a.require(report['status']=='PASS','Prior original audit PASS')
    for item in pl['records']:
        p=item['id']; saved=arrays(a.entry(pp.parent,item)); r=raw[p]; record=h.features(r['operating'],a); ix=np.arange(300,len(r['y']),dtype=np.int64)
        a.require('target' not in saved and 'y' not in saved,'Prediction artifact contains no truth '+str(p))
        a.equal(saved['operating'],r['operating'],'Raw evaluation operating '+str(p)); a.equal(saved['source_row'],r['source_row'],'Raw evaluation row lineage '+str(p)); a.equal(saved['index'],ix,'Every evaluation endpoint '+str(p))
        for k,v in record.items(): a.close(saved[k],v,'Independent evaluation recurrence '+str(p)+'/'+k)
        values={}; correction={}
        for name,s in specs().items():
            if s['kind']=='linear': values[name]=linear_pred(h,jobs['linear_final__'+s['prior']]['model']['linear_model'],record)[ix]
            elif s['kind'] in ('direct','residual'):
                key,_=resolve_alias(a,jobs,'full__'+name)
                if key not in correction:
                    # Every cached feature was independently reconstructed and
                    # checked above. Preserve its exact arithmetic at numeric
                    # split boundaries, consistently with held/FIT replay.
                    data,info=models[key]; correction[key]=replay(a,data,h.boost_x(saved,info['scale'])[ix],str(p)+'/'+key)
                value=correction[key].copy()
                if s['kind']=='residual': value+=linear_pred(h,jobs['linear_final__'+s['prior']]['model']['linear_model'],record)[ix]
                values[name]=value
        values['mean_A_B']=.5*(values['positive_rc']+values['direct_boost'])
        values['direct_boost_fitcv_selected']=values[lock['selected_direct6']]
        values[COMPARATOR]=values[lock['selected_direct9']]
        truth=arrays(evaluated_file('profile_%d_truth.npz'%p)); a.equal(truth['index'],ix,'Truth endpoint identity '+str(p)); a.equal(truth['target'],r['y'][ix],'Raw target identity '+str(p))
        for name in specs():
            a.close(saved['prediction__'+name],values[name],'Full raw-node prediction '+str(p)+'/'+name,atol=2e-8)
            matched=[v for v in saved_rows if v['id']==p and v['method']==name]
            a.require(len(matched)==1 and matched[0]['role']==args.role,'Exactly one profile-method metric')
            row=verify_metrics(a,matched[0],values[name],r['y'][ix],str(p)+'/'+name)
            rows.append(dict(id=p,role=args.role,method=name,**row))
        if legacy:
            matches=[v for v in legacy['predictions'] if v['id']==p and v['role']==args.role]
            a.require(len(matches)==1,'One pinned old15 prediction artifact '+str(p))
            old=arrays(a.entry(root,matches[0])); a.equal(old['index'],ix,'Legacy endpoint preservation '+str(p))
            for name in OLD15: a.close(old['prediction__'+name],values[name],'Old15 independent reproduction '+str(p)+'/'+name,atol=2e-8)
    summary={}
    for name in specs():
        block=[r for r in rows if r['method']==name]; n=sum(r['n'] for r in block)
        value=dict(profile_count=len(block),n=n,macro_rmse=float(np.mean([r['rmse'] for r in block])),
            pooled_rmse=float(np.sqrt(sum(r['squared_error_sum'] for r in block)/n)),macro_mae=float(np.mean([r['mae'] for r in block])),
            max_absolute_error=max(r['max_absolute_error'] for r in block),max_underprediction=max(r['max_underprediction'] for r in block))
        for k,v in value.items(): a.close(saved_summary['summary'][name][k],v,'All summary metrics '+name+'/'+k,atol=2e-8)
        summary[name]=value
    p=[next(r['rmse'] for r in rows if r['id']==v and r['method']==PRIMARY) for v in ids]
    b=[next(r['rmse'] for r in rows if r['id']==v and r['method']==COMPARATOR) for v in ids]
    inf=inference(p,b); a.close(saved_summary['relative_macro_reduction'],inf['relative_macro_reduction'],'Primary relative macro reduction')
    for pid,x,y in zip(ids,p,b): a.close(saved_summary['paired_rmse_gain'][str(pid)],y-x,'Saved paired gain sign '+str(pid))
    inf['primary_confirmatory_role']=d['dataset']=='paderborn' and args.role=='confirmation'
    a.results.update(metrics=rows,summary=summary,inference=inf,legacy_old15_reproduced=legacy is not None,
        legacy_reproduction_scope='Only when pinned legacy artifacts and prior PASS audits supplied')


def synthetic(a):
    """No scientific source imports, data access or model fitting."""
    a.check(len(specs())==35 and len(expected_jobs(list(range(8))))==272,'Fixed35 panel and 30N+32 job count')
    positive=inference([1.]*14,[2.]*14)
    a.close(positive['exact_one_sided_sign_p'],2**-14,'Analytic all-win exact sign p',atol=0,rtol=0)
    a.equal(positive['bootstrap_percentile95_C'],[-1.,-1.],'Constant paired bootstrap fixture')
    tied=inference([2.]*14,[2.]*14); a.check(tied['ties']==14 and tied['exact_one_sided_sign_p']==1 and not tied['directional_rule_met'],'All-tie direction fixture')
    under=metrics(np.array([2.,4.,6.]),np.array([1.,3.,5.])); a.close(under['max_underprediction'],0.,'Nonnegative-clipped underprediction')
    a.results['no_dataset_values_accessed']=True


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root'); ap.add_argument('--manifest')
    ap.add_argument('--stage',choices=['synthetic','jobs','selection','evaluation'],required=True)
    ap.add_argument('--dependencies'); ap.add_argument('--model-lock'); ap.add_argument('--job-ids',nargs='+')
    ap.add_argument('--job-audits',nargs='+'); ap.add_argument('--selection-audit')
    ap.add_argument('--prediction-lock'); ap.add_argument('--evaluation-dir'); ap.add_argument('--legacy-lock')
    ap.add_argument('--role',choices=['calibration','pilot','development','confirmation'])
    ap.add_argument('--confirmation-admission'); ap.add_argument('--report',required=True)
    ap.add_argument('--max-seconds',type=float,default=3600.)
    args=ap.parse_args()
    if not np.isfinite(args.max_seconds) or not 0<args.max_seconds<=3600: raise ValueError('Audit stage cap <=3600s')
    output=Path(args.report).resolve()
    if output.exists(): raise FileExistsError('Immutable audit report already exists')
    a=Audit(args.max_seconds); error=None
    try:
        with threadpool_limits(limits=1):
            if args.stage=='synthetic': synthetic(a)
            else:
                if not args.project_root or not args.manifest: raise ValueError('Dataset manifest/project root required')
                root,mp,d=load_manifest(a,args); h=helper(a); jobs,lock=load_jobs(a,root,mp,d,args)
                if args.stage=='jobs':
                    if not args.job_ids or len(set(args.job_ids))!=len(args.job_ids): raise ValueError('Explicit unique job IDs required')
                    records=fit_records(a,h,root,d)
                    for key in args.job_ids:
                        a.require(key in jobs,'Requested completed job present'); audit_job(a,h,jobs,key,records,d['roles']['fit'])
                elif args.stage=='selection':
                    require_job_audits(a,args,root,jobs)
                    audit_selection(a,jobs,d['roles']['fit'],lock)
                else:
                    require_selection_audit(a,args,root)
                    audit_selection(a,jobs,d['roles']['fit'],lock)
                    audit_evaluation(a,h,args,root,mp,d,jobs,lock)
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc)); a.failures.append(dict(check='Audit execution exception',detail=error))
    result=dict(schema='c15-paper-independent-audit-v001',status='PASS' if not a.failures else 'HOLD',stage=args.stage,
        auditor_sha256=sha(__file__),producer_sha256=WORKER,core_sha256=CORE,independent_helper_sha256=HELPER,
        checks=a.checks,failures=a.failures,elapsed_seconds=time.monotonic()-a.start,inputs=a.inputs,
        coverage=a.coverage,linear_precision_bounds=a.bounds,replayed=a.replayed,results=a.results,
        limitations=['No scientific worker imports, pickle loads, retraining, split search or leaf optimization replay.',
            'CV training node counts cover fixed five iterations; full models cover all training trees when their jobs are audited.',
            'Every reported evaluation prediction is replayed for all points and trees; this stage does not imply unrun job audits passed.',
            'Source order and saved locks support exposure provenance; a saved artifact cannot prove unrecorded human behavior.',
            'Floating QP certificates are numerical bounds, not exact rational proofs.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],checks=a.checks,failures=len(a.failures),report=str(output)),ensure_ascii=False))
    if a.failures: raise SystemExit(2)


if __name__=='__main__': main()
