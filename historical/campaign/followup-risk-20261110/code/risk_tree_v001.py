"""Frozen C16 direct HGB controls. FIT only; locks all predictions before truth access."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import sklearn,joblib
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def checked(e):
    p=ROOT/e['path'];assert sha(p)==e['sha256'],str(p);return p
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True)
    p.add_argument('--loss',choices=['mse','q90'],required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--max-seconds',type=int,default=600);a=p.parse_args();start=time.monotonic()
    out=ROOT/a.output_dir;out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(checked({'path':a.config,'sha256':a.config_sha256}).read_text(encoding='utf-8'))
    assert cfg['scientific_execution_admitted'] and cfg['schema']=='c16-risk-dataset-v001'
    checked(cfg['protocol']);checked(cfg['original_manifest'])
    dump(out/'configuration.json',dict(source=str(Path(__file__).resolve()),pid=os.getpid(),argv=sys.argv,
        output_dir=str(out.resolve()),config=a.config,config_sha256=a.config_sha256,loss=a.loss,
        started_utc=datetime.now(timezone.utc).isoformat()))
    def check():
        assert time.monotonic()-start<a.max_seconds,'Stage cap'
        assert sum(q.stat().st_size for q in out.rglob('*') if q.is_file())<2_000_000_000
    records=[]
    for e in cfg['fit_cache']:
        check()
        with np.load(checked(e),allow_pickle=False) as z:
            ix=np.arange(300,e['rows'],5,dtype=np.int64)
            records.append((e['id'],{k:z[k][ix].astype(np.float64) for k in ('instant','boundary','heat','target')},ix))
    assert [r[0] for r in records]==cfg['train_ids']
    instant=np.concatenate([r[1]['instant'] for r in records]);y=np.concatenate([r[1]['target'] for r in records])
    ir=max(float(np.sqrt(np.mean(instant[:,0]**2))),1e-8);wr=max(float(np.sqrt(np.mean(instant[:,3]**2))),1e-8)
    qs=np.tile([ir**2,wr**2,ir*wr],5)
    def features(z):return np.column_stack([z['boundary'],z['heat']/qs,z['instant']])
    x=np.concatenate([features(r[1]) for r in records]);mu=x.mean(0);sd=np.maximum(x.std(0),1e-8);x=(x-mu)/sd
    weight=np.concatenate([np.full(len(r[2]),len(y)/(len(records)*len(r[2]))) for r in records])
    assert np.isfinite(x).all() and np.isfinite(y).all()
    transform=dict(qscale=qs.tolist(),input_mean=mu.tolist(),input_std=sd.tolist(),train_ids=cfg['train_ids'],
        high_temperature_threshold_c=float(np.quantile(y,.95)),training_rows=len(y))
    dump(out/'transform.json',transform)
    np.savez_compressed(out/'training_index.npz',profile=np.concatenate([np.full(len(r[2]),r[0],dtype=np.int64) for r in records]),
        index=np.concatenate([r[2] for r in records]),weight=weight)
    model=HistGradientBoostingRegressor(loss='squared_error' if a.loss=='mse' else 'quantile',
        quantile=None if a.loss=='mse' else .9,max_leaf_nodes=15,max_iter=200,learning_rate=.05,min_samples_leaf=20,
        l2_regularization=.1,early_stopping=False,random_state=2026111701)
    fit_start=time.monotonic();model.fit(x,y,sample_weight=weight);fit_seconds=time.monotonic()-fit_start;check()
    joblib.dump(model,out/'model.joblib',compress=3)
    pred=[]
    for role in cfg['role_inputs']:
        for e in role['records']:
            check()
            with np.load(checked(e['features']),allow_pickle=False) as z:
                ix=z['index'].astype(np.int64);assert np.array_equal(ix,np.arange(300,e['rows']))
                xx=(features({k:z[k][ix].astype(np.float64) for k in ('instant','boundary','heat')})-mu)/sd
                pp=model.predict(xx);assert pp.shape==ix.shape and np.isfinite(pp).all()
                source_row=z['source_row'][ix]
            name=role['role']+'__profile_'+str(e['id'])+'.npz'
            np.savez_compressed(out/name,index=ix,source_row=source_row,prediction=pp)
            pred.append(dict(role=role['role'],profile=e['id'],path=(out/name).relative_to(ROOT).as_posix(),
                sha256=sha(out/name),prediction_key='prediction',index_key='index'))
    dump(out/'prediction_lock.json',dict(config_sha256=a.config_sha256,truth_accessed=False,predictions=pred))
    dump(out/'summary.json',dict(status='HGB_FIT_PREDICTIONS_LOCKED',dataset=cfg['dataset'],loss=a.loss,seed=2026111701,
        config_sha256=a.config_sha256,source_sha256=sha(__file__),fit_seconds=fit_seconds,elapsed_seconds=time.monotonic()-start,
        transform=transform,sklearn=sklearn.__version__,numpy=np.__version__,predictions=pred,
        files={f.name:sha(f) for f in out.iterdir() if f.is_file()}))
    print(json.dumps(dict(status='HGB_FIT_PREDICTIONS_LOCKED',dataset=cfg['dataset'],loss=a.loss,fit_seconds=fit_seconds)))
if __name__=='__main__':main()
