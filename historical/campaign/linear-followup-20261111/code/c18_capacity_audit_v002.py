"""Independent saved-tree audit of the exploratory C18 capacity control.

No sklearn/scientific-worker import, pickle load, fitting, timing experiment,
or new hypothesis test. Prior raw-source audits establish input-cache lineage;
this audit independently rebuilds the complete training design and weights.
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
CPU_AUDIT_SHA='0fba461b3bc8a5a9a034732061abff9eba1e13635dc325fe72f1bee3992d0794'
PRODUCER_SHA='6b647c0c02656dc5f13133655b084f44acdaf7a2a4e2b2eb8bc7475679370ba1'
PROTOCOL_SHA='8784420257e84b1fc44026c7ad4280001b7177c748e3f5833067fa6cb9f4ab51'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def dependency():
    p=ROOT/'followup-risk-20261110/code/rwth_cpu_audit_v002.py'
    if sha(p)!=CPU_AUDIT_SHA:raise ValueError('Frozen independent raw-tree helper changed')
    s=importlib.util.spec_from_file_location('c18_independent_tree_helper',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

C=dependency();I=C.I;P=C.P

def local(value):
    # Legacy locks retain their original Windows absolute root, also under WSL.
    value=str(value).replace('\\','/')
    if '/topic-selection-20261105/' in value:value=value.split('/topic-selection-20261105/',1)[1]
    return I.path_in(ROOT,value)

def pin(a,e):
    p=local(e['path'])
    if p in a._pins:
        a.require(a._pins[p]==e['sha256'],'One immutable identity per input')
        return p
    return a.pin(dict(path=str(p),sha256=e['sha256']))

def ref(p):return dict(path=local(p).relative_to(ROOT).as_posix(),sha256=sha(local(p)))
def obj(a,e):return I.read(pin(a,e))
def arr(a,e):return a.arrays(pin(a,e))
def exact(a,x,y,label):P.close_array(a,x,y,label,exact=True)
def close(a,x,y,label):P.close_array(a,x,y,label,atol=1e-9,rtol=2e-12)

def audit_index(a,e):
    cache=getattr(a,'_report_indexes',{})
    key=(e['path'],e['sha256'])
    if key in cache:return cache[key]
    r=obj(a,e)
    a.require(r.get('status')=='PASS' and not r.get('failures'),'Prior complete independent PASS')
    identities=[r[k] for k in ('auditor_sha256','source_sha256') if k in r]
    a.require(bool(identities) and all(x==e['auditor_sha256'] for x in identities),'Unambiguous prior auditor identity')
    pin(a,dict(path=e['auditor_path'],sha256=e['auditor_sha256']))
    index={}
    for p,h in r['inputs'].items():index.setdefault(local(p),set()).add(h)
    cache[key]=index;a._report_indexes=cache;return index

def audited(a,e,report):
    p=pin(a,e);index=audit_index(a,report)
    a.require(index.get(p)=={e['sha256']},'Prior raw-source/forward audit binds '+p.name)
    return p

def features(z,scale,ds):
    if ds=='rwth':return (z['X33']-np.asarray(scale['x_mean']))/np.asarray(scale['x_std'])
    return np.column_stack((z['boundary'],z['heat']/np.asarray(scale['qscale']),z['instant']))

def training(a,cfg,lineage):
    ds=cfg['dataset'];name='rwth48v' if ds=='rwth' else ds
    item=next(x for x in lineage['datasets'] if x['dataset']==name)
    expected=[dict(id=e['id'],features=dict(path=e['path'],sha256=e['sha256'])) for e in item['fit_cache']]
    a.same(cfg['fit'],expected,'Original ordered complete FIT cache references')
    a.require(cfg['warmup']==(0 if ds=='rwth' else 300) and cfg['stride']==5,'Frozen sampling rule')
    records=[];indices=[]
    for e in cfg['fit']:
        p=audited(a,e['features'],item['fit_audit']);z=a.arrays(p)
        n=len(z['target']);ix=np.arange(cfg['warmup'],n,5,dtype=np.int64)
        a.require(len(ix)>0 and z['target'].shape==(n,),'Nonempty finite FIT targets')
        if ds=='rwth':
            a.require(z['X33'].shape==(n,33),'RWTH common raw33 geometry')
            exact(a,z['index'],np.arange(n,dtype=np.int64),'RWTH all original FIT rows')
        else:
            a.require(z['boundary'].shape==(n,10) and z['heat'].shape==(n,15) and z['instant'].shape==(n,8),'Legacy 10+15+8 geometry')
        records.append((e['id'],z));indices.append(ix)
    if ds=='rwth':
        raw=np.concatenate([z['X33'][ix] for (_,z),ix in zip(records,indices)])
        scale=dict(x_mean=raw.mean(0).tolist(),x_std=np.maximum(raw.std(0),1e-8).tolist())
    else:
        instant=np.concatenate([z['instant'][ix] for (_,z),ix in zip(records,indices)])
        ri=max(float(np.sqrt(np.mean(instant[:,0]**2))),1e-8)
        rw=max(float(np.sqrt(np.mean(instant[:,3]**2))),1e-8)
        scale=dict(qscale=np.tile([ri**2,rw**2,ri*rw],5).tolist())
    x=np.concatenate([features(z,scale,ds)[ix] for (_,z),ix in zip(records,indices)])
    y=np.concatenate([z['target'][ix] for (_,z),ix in zip(records,indices)])
    n=len(y);m=len(records)
    weights=(np.concatenate([np.full(len(ix),n/(m*len(ix))) for ix in indices]) if ds=='rwth'
             else n*np.concatenate([np.full(len(ix),1./(m*len(ix))) for ix in indices]))
    ids=np.concatenate([np.full(len(ix),p,dtype=np.int64) for (p,_),ix in zip(records,indices)])
    return dict(X=x,target=y,weight=weights,profile_id=ids,index=np.concatenate(indices)),scale

def old_model(a,cfg):
    lock=obj(a,cfg['old_model_lock']);ds=cfg['dataset']
    if ds=='rwth':
        e=lock['boosters']['direct9_fitcv_selected'];metadata=obj(a,e['metadata'])
        a.same(e['metadata'],cfg['old_d9_metadata'],'Selected original RWTH D9 metadata')
        nodes=C.numeric_nodes(a,pin(a,e['trees']))
        return metadata,nodes,'tree_%04d','bin_thresholds_%02d'
    name='direct_boost_7x500' if ds=='ipmsm' else 'direct_boost_15x1000'
    a.require(lock['selected_direct9']==name,'Original FIT-CV selected D9 identity')
    e=lock['entries']['full__'+name];folder=local(e['path'])
    receipt=obj(a,dict(path=str(folder/'receipt.json'),sha256=e['receipt_sha256']))
    a.require(receipt['status']=='FIT_JOB_COMPLETE','Completed original D9')
    for name,h in receipt['files'].items():pin(a,dict(path=str(folder/name),sha256=h))
    metadata=I.read(folder/'model.json');nodes=C.numeric_nodes(a,folder/'booster.npz')
    return metadata,nodes,'tree_%03d','bin_thresholds_%02d'

def original_training_digest(value,dataset):
    """Match each frozen producer's exact array-header serialization."""
    if dataset=='rwth':return C.array_sha(value)
    if dataset not in ('ipmsm','paderborn'):raise ValueError('Known frozen dataset required')
    value=np.ascontiguousarray(value)
    h=hashlib.sha256()
    h.update(str(value.dtype).encode('ascii'))
    h.update(json.dumps(list(value.shape)).encode('ascii'))
    h.update(value.tobytes())
    return h.hexdigest()


def trees_and_prefix(a,cfg,model,expected,scale):
    saved=arr(a,model['training'])
    a.require(set(saved)==set(expected),'Exact complete training export fields')
    for key,x in expected.items():
        a.require(saved[key].dtype==x.dtype,'Training dtype '+key)
        exact(a,saved[key],x,'Independent FIT training '+key)
    a.same(model['scale'],scale,'Independent FIT-only scales',atol=0.,rtol=0.)
    a.same(model['fit_ids'],[e['id'] for e in cfg['fit']],'Exact original FIT IDs')
    old,nodes0,fmt0,binfmt0=old_model(a,cfg)
    prior=dict(old['parameters'] if cfg['dataset']=='rwth' else old['configuration'])
    prior.setdefault('max_bins',255)
    a.require(prior['max_leaf_nodes']==cfg['prior_d9_leaves'] and prior['max_iter']==cfg['prior_d9_iterations'],'Original D9 capacity identity')
    n0=prior['max_iter'];leaves=prior['max_leaf_nodes']
    n=n0+math.ceil(2800/(leaves-1));params=dict(prior,max_iter=n)
    a.same(cfg['parameters'],params,'Only iteration count changed from D9')
    a.same(model['parameters'],params,'Registered capacity model parameters')
    a.require(model['rows']==len(expected['target']),'All original training rows')
    if cfg['dataset']=='rwth':
        digestkeys=['training_X_sha256','training_target_sha256','training_weight_sha256']
    else:digestkeys=['features_sha256','target_sha256','sample_weight_sha256']
    for key,value in zip(digestkeys,[expected['X'],expected['target'],expected['weight']]):
        a.require(old[key]==original_training_digest(value,cfg['dataset']),'Original D9 identical training digest '+key)
    trees=C.numeric_nodes(a,pin(a,model['trees']));pin(a,model['pickle'])
    required={'baseline','n_iter'}|{f'tree_{j:04d}' for j in range(n)}|{f'bin_threshold_{j:02d}' for j in range(33)}
    a.require(set(trees)==required and trees['n_iter'].shape==() and int(trees['n_iter'])==n,'Every registered raw tree exported')
    close(a,trees['baseline'],np.asarray([[np.average(expected['target'],weights=expected['weight'])]]),'Independent weighted intercept')
    exact(a,trees['baseline'],nodes0['baseline'],'Original D9 intercept preserved')
    for j in range(33):exact(a,trees[f'bin_threshold_{j:02d}'],nodes0[binfmt0%j],'Original D9 bin thresholds '+str(j))
    for j in range(n0):
        aa,bb=trees[f'tree_{j:04d}'],nodes0[fmt0%j]
        a.require(aa.shape==bb.shape and aa.dtype.names==bb.dtype.names,'D9 prefix raw-node schema '+str(j))
        for field in bb.dtype.names:exact(a,aa[field],bb[field],'Original D9 prefix tree '+str(j)+'/'+field)
    actual=0
    for j in range(n):
        nodes=trees[f'tree_{j:04d}'];actual+=int(np.count_nonzero(nodes['is_leaf']==0))
        a.require(np.count_nonzero(nodes['is_leaf'])<=leaves,'Frozen maximum leaves '+str(j))
    a.require(model['actual_internal_nodes']==actual and model['upper_internal_nodes']==n*(leaves-1),'Actual versus maximum internal split counts')
    # Audit structure and full training row counts on three representative trees.
    for j in sorted({0,n//2,n-1}):C.traverse(a,trees[f'tree_{j:04d}'],expected['X'],'FIT/'+str(j),True)
    return trees,n0,n

def measure(y,p):
    e=p-y
    return dict(rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(np.abs(e))),bias=float(np.mean(e)),
                under_mae=float(np.mean(np.maximum(-e,0))),over_mae=float(np.mean(np.maximum(e,0))),mse=float(np.mean(e*e)))

def replay(a,trees,x,n0,n,tag):
    p=np.full(len(x),float(trees['baseline'].item()),dtype=np.float64);prefix=None
    for j in range(n):
        p+=C.traverse(a,trees[f'tree_{j:04d}'],x,tag+'/'+str(j))
        if j+1==n0:prefix=p.copy()
    a.replayed[tag]=dict(rows=len(x),trees=n,all_rows_all_trees=True)
    return p,prefix

def held(a,cfg,lineage,lock,trees,n0,n,scale):
    name='rwth48v' if cfg['dataset']=='rwth' else cfg['dataset']
    cohorts=[c for c in lineage['cohorts'] if c['dataset']==name]
    expected_ids=[[c['role'],p] for c in cohorts for p in c['ids']]
    a.same([[e['role'],e['id']] for e in cfg['held']],expected_ids,'All exposed cohorts/profiles without selection')
    a.same([[e['role'],e['id']] for e in lock['records']],expected_ids,'Complete new prediction coverage')
    result={}
    for cohort in cohorts:
        role=cohort['role'];report=cohort['audit'];audit_index(a,report)
        pl=I.read(audited(a,cohort['prediction_lock'],report))
        original={r['profile']:r for r in pl['predictions'] if r['role']==role}
        rows=[]
        for pid in cohort['ids']:
            cfgentry=next(e for e in cfg['held'] if e['role']==role and e['id']==pid)
            entry=next(e for e in lock['records'] if e['role']==role and e['id']==pid)
            a.same({k:entry[k] for k in ('role','id','features','truth')},cfgentry,'Exact locked input references')
            z=a.arrays(audited(a,entry['features'],report));truth=a.arrays(audited(a,entry['truth'],report))
            originalentry=original[pid];old=a.arrays(audited(a,originalentry,report));new=arr(a,entry['prediction'])
            a.require(set(new)=={'index','source_row','X','prediction','d9_prefix'},'Exact saved capacity prediction schema')
            ix=z['index'];xx=features(z,scale,cfg['dataset'])
            source=z['source_row']
            if cfg['dataset']!='rwth':xx=xx[ix];source=source[ix]
            a.require(ix.dtype==np.int64 and np.all(np.diff(ix)==1),'All consecutive original evaluated rows')
            exact(a,new['index'],ix,'Retained endpoint indices');exact(a,truth['index'],ix,'Aligned truth rows')
            exact(a,new['source_row'],source,'Original source rows');exact(a,old['index'],ix,'Existing comparison row alignment')
            exact(a,old['source_row'],source,'Existing comparison source rows');exact(a,new['X'],xx,'Exact independently rebuilt held features')
            a.require(new['X'].shape==(len(ix),33) and new['prediction'].shape==truth['target'].shape==(len(ix),),'Complete finite prediction geometry')
            pred,prefix=replay(a,trees,new['X'],n0,n,role+'/'+str(pid))
            close(a,new['prediction'],pred,'All raw-tree final predictions');close(a,new['d9_prefix'],prefix,'All raw-tree D9 prefix predictions')
            close(a,prefix,old['prediction__c15__left'],'Mandatory original D9 prefix predictions')
            values=dict(capacity_direct=pred,d9=old['prediction__c15__left'],linear_residual=old['prediction__linear33__right'],
                        linear_half=old['prediction__linear33__half'],direct200=old['prediction__direct200__right'],direct200_half=old['prediction__direct200__half'])
            close(a,values['linear_half'],.5*(values['d9']+values['linear_residual']),'Original fixed-half construction')
            a.require(all(v.shape==(len(ix),) and np.isfinite(v).all() for v in values.values()),'Complete finite fixed comparison panel')
            rows.append(dict(id=pid,n=len(ix),metrics={k:measure(truth['target'],v) for k,v in values.items()}))
        methods=list(rows[0]['metrics']);total=sum(r['n'] for r in rows)
        summary={k:dict(macro_rmse=float(np.mean([r['metrics'][k]['rmse'] for r in rows])),
                        pooled_rmse=float(math.sqrt(math.fsum(r['n']*r['metrics'][k]['mse'] for r in rows)/total)),
                        macro_mae=float(np.mean([r['metrics'][k]['mae'] for r in rows]))) for k in methods}
        result[role]=dict(profile_count=len(rows),n=total,per_profile=rows,summary=summary,
                         capacity_minus_linear_half_macro_rmse=summary['capacity_direct']['macro_rmse']-summary['linear_half']['macro_rmse'])
    return result

def run(a,args):
    cfgref=dict(path=args.config,sha256=args.config_sha256);cfg=obj(a,cfgref)
    a.require(cfg['schema']=='c18-capacity-config-v001' and cfg['dataset'] in ('ipmsm','paderborn','rwth') and cfg['all_evaluation_exposed'] is True,'Exploratory fixed capacity design')
    a.require(cfg['protocol']['sha256']==PROTOCOL_SHA,'Frozen prospective capacity protocol');pin(a,cfg['protocol']);pin(a,cfg['limits'])
    lineage=obj(a,dict(path=args.lineage,sha256=args.lineage_sha256))
    a.require(lineage['all_data_exposed'] is True and lineage['formal_confirmation_allowed'] is False,'No renewed confirmation claim')
    folder=local(args.input_dir);a.require(not(folder/'failure.json').exists(),'No failed capacity producer marker')
    summary=obj(a,dict(path=str(folder/'summary.json'),sha256=args.input_sha256))
    a.require(summary['status']=='CAPACITY_CONTROL_FIT_AND_PREDICTED' and summary['actual_fits']==1 and summary['dataset']==cfg['dataset'],'One completed fixed capacity fit')
    a.require(summary['source']['sha256']==args.producer_sha256==PRODUCER_SHA,'Frozen capacity producer identity');pin(a,summary['source'])
    lock=obj(a,summary['prediction_lock'])
    a.same(lock['config'],dict(path=local(args.config).relative_to(ROOT).as_posix(),sha256=args.config_sha256),'Exact pre-fit config identity')
    a.require(summary['held_truth_read'] is False and lock['held_truth_read'] is False,'Producer declares target-free evaluation prediction')
    model=obj(a,lock['model']);a.require(model['schema']=='c18-capacity-model-v001' and model['dataset']==cfg['dataset'],'Saved model identity')
    expected,scale=training(a,cfg,lineage);trees,n0,n=trees_and_prefix(a,cfg,model,expected,scale)
    a.require(summary['training_rows']==len(expected['target']) and summary['iterations']==n and summary['held_profiles']==len(cfg['held']),'Completed summary coverage')
    results=held(a,cfg,lineage,lock,trees,n0,n,scale)
    a.require(not(folder/'failure.json').exists(),'Producer remains completed without failure marker')
    a.results=dict(dataset=cfg['dataset'],cohorts=results,actual_internal_nodes=model['actual_internal_nodes'],maximum_internal_nodes=model['upper_internal_nodes'],
                   iterations=n,original_d9_iterations=n0,model_artifact_bytes={k:pin(a,model[k]).stat().st_size for k in ('trees','pickle')},
                   producer_reported_fit_seconds=model['fit_seconds'],all_results_exploratory=True,
                   timing_scope='Producer-reported fit duration; no independent timing or inference-cost measurement; NPZ and pickle are alternative serializations, not summed deployment size.')
    a.finish_pins()

def main():
    ap=argparse.ArgumentParser()
    for name in ('config','config-sha256','lineage','lineage-sha256','input-dir','input-sha256','producer-sha256','output-dir'):ap.add_argument('--'+name,required=True)
    ap.add_argument('--max-seconds',type=float,default=900);args=ap.parse_args()
    out=local(args.output_dir);out.mkdir(parents=True,exist_ok=False);a=I.Audit(args.max_seconds)
    try:run(a,args)
    except Exception as e:a.failures.append(dict(check='Audit terminated',detail=type(e).__name__+': '+str(e)))
    report=dict(status='PASS' if not a.failures else 'HOLD',stage='c18_capacity',auditor_sha256=sha(Path(__file__)),producer_sha256=args.producer_sha256,
                checks=a.checks,failures=a.failures,inputs=a.inputs,replayed=a.replayed,results=a.results,elapsed_seconds=time.monotonic()-a.start,
                scope='Independent complete FIT design/weights/scales, exact original D9 raw-node prefix, all saved held raw-tree predictions and descriptive metrics. Cache raw-source lineage inherited from hash-bound prior independent PASS. No optimizer replay, pickle execution, timing experiment, new hypothesis test or fresh confirmation.')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],checks=a.checks,failures=len(a.failures),report=str(out/'report.json'))))
    if a.failures:raise SystemExit(2)

if __name__=='__main__':main()
