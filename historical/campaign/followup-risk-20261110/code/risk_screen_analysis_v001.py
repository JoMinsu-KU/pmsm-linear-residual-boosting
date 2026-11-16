"""CAL-only policy selection and exhaustive exploratory risk tradeoff reporting."""
import argparse,hashlib,json,os,time
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
KEYS=['rmse','mae','under_rmse','under_mae','over_mae','under_q95','under_q99','max_under','max_over',
      'under_gt5_longest_samples','under_gt5_longest_seconds']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(e):
    p=ROOT/e['path'];assert sha(p)==e['sha256'],str(p);return p
def dump(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def longest(mask):
    changes=np.diff(np.r_[False,mask,False].astype(np.int8));starts=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1)
    return int(np.max(ends-starts)) if len(starts) else 0
def measures(y,p,dt,mask=None):
    mask=np.ones(len(y),dtype=bool) if mask is None else mask
    if not np.any(mask):return None
    e=y[mask]-p[mask];u=np.maximum(e,0);o=np.maximum(-e,0)
    nrun=longest(mask & ((y-p)>5))
    return dict(n=int(mask.sum()),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),
        under_rmse=float(np.sqrt(np.mean(u*u))),under_mae=float(np.mean(u)),over_mae=float(np.mean(o)),
        under_q95=float(np.quantile(u,.95)),under_q99=float(np.quantile(u,.99)),max_under=float(np.max(u)),
        max_over=float(np.max(o)),under_gt5_longest_samples=nrun,under_gt5_longest_seconds=None if dt is None else nrun*dt)
def summarize(rows):
    n=sum(r['n'] for r in rows)
    v=dict(profile_count=len(rows),n=n)
    for k in KEYS:
        vv=[r[k] for r in rows]
        v['macro_'+k]=float(np.mean(vv)) if vv and all(x is not None for x in vv) else None
    v['pooled_rmse']=float(np.sqrt(sum(r['rmse']**2*r['n'] for r in rows)/n)) if n else None
    return v
def role_metrics(truth,preds,threshold,dt):
    rows=[];high=[];excluded=[]
    for pid in truth:
        y=truth[pid];pp=preds[pid];assert y.shape==pp.shape and np.isfinite(pp).all()
        row=dict(id=pid,**measures(y,pp,dt));hi=measures(y,pp,dt,y>=threshold)
        row['high_temperature']=None if hi is None else dict(id=pid,**hi)
        if hi is None:excluded.append(pid)
        else:high.append(dict(id=pid,**hi))
        rows.append(row)
    summary=summarize(rows);summary['high_temperature']=dict(eligible_profile_ids=[r['id'] for r in high],
        excluded_profile_ids=excluded,**summarize(high))
    return dict(per_profile=rows,summary=summary)
def main():
    p=argparse.ArgumentParser();p.add_argument('--components',required=True);p.add_argument('--components-sha256',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--max-seconds',type=int,default=600);a=p.parse_args()
    start=time.monotonic();out=ROOT/a.output_dir;out.mkdir(parents=True,exist_ok=False)
    ip=checked(dict(path=a.components,sha256=a.components_sha256));inputs=json.loads(ip.read_text(encoding='utf-8'))
    cfg=json.loads(checked(inputs['config']).read_text(encoding='utf-8'));checked(cfg['protocol'])
    assert inputs['dataset']==cfg['dataset'];assert cfg['all_evaluation_previously_exposed']
    dump(out/'configuration.json',dict(source=str(Path(__file__).resolve()),pid=os.getpid(),components=a.components,
        components_sha256=a.components_sha256,output_dir=str(out.resolve())))
    dump(out/'component-inputs.json',inputs)
    train=[]
    for entry in cfg['fit_cache']:
        with np.load(checked(entry),allow_pickle=False) as z:train.append(z['target'][300::5].astype(np.float64))
    threshold=float(np.quantile(np.concatenate(train),.95));dt=cfg['metrics']['sample_interval_seconds']
    truth={};indices={}
    for rr in cfg['role_inputs']:
        truth[rr['role']]={};indices[rr['role']]={}
        for e in rr['records']:
            with np.load(checked(e['truth']),allow_pickle=False) as z:
                ix=z['index'];y=z['target'].astype(np.float64)
                assert np.array_equal(ix,np.arange(300,e['rows'])) and np.isfinite(y).all()
                truth[rr['role']][e['id']]=y;indices[rr['role']][e['id']]=ix
    comp={};metadata={}
    for c in inputs['components']:
        assert c['id'] not in comp
        metadata[c['id']]=c;comp[c['id']]={r:{} for r in truth}
        for e in c['predictions']:
            with np.load(checked(e),allow_pickle=False) as z:
                ix=z[e.get('index_key','index')];pp=z[e.get('prediction_key','prediction')].astype(np.float64)
                assert np.array_equal(ix,indices[e['role']][e['profile']]) and len(pp)==len(ix) and np.isfinite(pp).all()
                comp[c['id']][e['role']][e['profile']]=pp
        assert all(list(comp[c['id']][r])==list(truth[r]) for r in truth)
    settings=[];metric={};cal=cfg['calibration_role']
    def add(s):
        assert s['id'] not in metric
        settings.append(s);metric[s['id']]={}
        for r in truth:
            pred={}
            for pid in truth[r]:
                base=comp[s['base']][r][pid]
                if s['other'] is not None:pp=(1-s['weight'])*base+s['weight']*comp[s['other']][r][pid]
                else:pp=base+(s['shift'] or 0)
                pred[pid]=pp
            metric[s['id']][r]=role_metrics(truth[r],pred,threshold,dt)
        assert time.monotonic()-start<a.max_seconds,'Analysis cap'
    for c in inputs['components']:
        add(dict(id=c['id']+'__raw',family='context' if c['backbone']=='context' else 'unadjusted',base=c['id'],
            other=None,weight=None,shift=0.,backbone=c['backbone'],seed=c['seed']))
    groups=[]
    for cid,c in metadata.items():
        if c['loss']!='mse' or c['backbone']=='context':continue
        baseline=cid+'__raw';group=dict(backbone=c['backbone'],seed=c['seed'],base_setting_id=baseline,families={})
        errors=np.concatenate([truth[cal][pid]-comp[cid][cal][pid] for pid in truth[cal]])
        weights=np.concatenate([np.full(len(truth[cal][pid]),1/(len(truth[cal])*len(truth[cal][pid]))) for pid in truth[cal]])
        order=np.argsort(errors,kind='stable');ev=errors[order];cdf=np.cumsum(weights[order]);cdf[-1]=1.
        shifts=list(cfg['calibration']['shift_grid_c'])
        for q in cfg['calibration']['shift_residual_quantiles']:
            shifts.append(max(0.,float(ev[min(np.searchsorted(cdf,q,side='left'),len(ev)-1)])))
        shifts=list(dict.fromkeys(shifts));family=[]
        for i,value in enumerate(shifts):
            sid=cid+'__shift_'+str(i);family.append(sid)
            add(dict(id=sid,family='shift',base=cid,other=None,weight=None,shift=float(value),backbone=c['backbone'],seed=c['seed']))
        group['families']['shift']=family
        for loss,family_name in [('asym4','asym_mix'),('q90','quantile_mix')]:
            other=[k for k,v in metadata.items() if v['backbone']==c['backbone'] and v['seed']==c['seed'] and v['loss']==loss]
            if not other:continue
            assert len(other)==1;family=[]
            for i,w in enumerate(cfg['calibration']['mix_weights']):
                sid=cid+'__'+family_name+'_'+str(i);family.append(sid)
                add(dict(id=sid,family=family_name,base=cid,other=other[0],weight=float(w),shift=None,backbone=c['backbone'],seed=c['seed']))
            group['families'][family_name]=family
        groups.append(group)
    # Every choice uses CAL only, with settings order as final tie-break.
    policies=[]
    for g in groups:
        b=metric[g['base_setting_id']][cal]['summary']
        for rr in cfg['calibration']['rmse_budget_ratios']:
            for oo in cfg['calibration']['over_mae_budget_ratios']:
                for family,ids in g['families'].items():
                    feasible=[(i,sid) for i,sid in enumerate(ids) if metric[sid][cal]['summary']['macro_rmse']<=rr*b['macro_rmse']+1e-12
                        and metric[sid][cal]['summary']['macro_over_mae']<=oo*b['macro_over_mae']+1e-12]
                    assert feasible
                    _,selected=min(feasible,key=lambda pair:tuple(metric[pair[1]][cal]['summary'][k] for k in
                        ['macro_under_rmse','macro_rmse','macro_over_mae'])+(pair[0],))
                    violations={}
                    for role in truth:
                        bm=metric[g['base_setting_id']][role]['summary'];sm=metric[selected][role]['summary']
                        violations[role]=dict(rmse=sm['macro_rmse']>rr*bm['macro_rmse']+1e-12,
                            over_mae=sm['macro_over_mae']>oo*bm['macro_over_mae']+1e-12)
                    policies.append(dict(backbone=g['backbone'],seed=g['seed'],family=family,rmse_ratio=rr,over_mae_ratio=oo,
                        selected_setting_id=selected,base_setting_id=g['base_setting_id'],calibration_role=cal,evaluation_budget_violations=violations))
    dump(out/'settings.json',settings);dump(out/'policies.json',policies)
    dump(out/'metrics.json',dict(dataset=cfg['dataset'],high_temperature_threshold_c=threshold,settings=metric))
    dump(out/'summary.json',dict(status='EXPLORATORY_RISK_ANALYSIS_COMPLETE',dataset=cfg['dataset'],settings=len(settings),
        policies=len(policies),high_temperature_threshold_c=threshold,elapsed_seconds=time.monotonic()-start,
        independent_confirmation=False,files={f.name:sha(f) for f in out.iterdir() if f.is_file()}))
    print(json.dumps(dict(status='EXPLORATORY_RISK_ANALYSIS_COMPLETE',settings=len(settings),policies=len(policies))))
if __name__=='__main__':main()
