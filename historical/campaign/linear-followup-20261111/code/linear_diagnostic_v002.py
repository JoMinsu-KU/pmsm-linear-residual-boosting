"""C18 fixed-half diagnostic: existing exposed predictions, no fitting or tests."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'linear-followup-20261111'
METHODS=['d9','linear_prior','linear_residual','linear_half','linear_cal_mix',
         'direct200','direct200_half','free25_residual','free25_half','prior_half']
PRIMARY='linear_half'
PAIRS=['linear_residual','direct200','free25_residual','linear_prior']
COHORTS=[('ipmsm','calibration'),('ipmsm','pilot'),('ipmsm','confirmation'),
         ('paderborn','development'),('paderborn','confirmation'),
         ('rwth48v','calibration'),('rwth48v','confirmation')]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def local(value):
    value=str(value)
    if os.name!='nt' and re.match(r'^[A-Za-z]:[\\/]',value):value='/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
    elif os.name=='nt' and re.match(r'^/mnt/[a-z]/',value):value=value[5].upper()+':/'+value[7:]
    path=Path(value);path=(path if path.is_absolute() else ROOT/path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Outside campaign')
    return path


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def ref(path):return dict(path=local(path).relative_to(ROOT).as_posix(),sha256=sha(path))


class Intake:
    def __init__(self,cap,out):self.start=time.monotonic();self.cap=cap;self.out=out;self.inputs={};self.audit_cache={};self.checks=0
    def check(self):
        if time.monotonic()-self.start>self.cap:raise TimeoutError('Registered stage time cap')
        if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-11-12T00:23:59+00:00'):raise TimeoutError('Extended campaign deadline')
        if sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())>100000000:raise RuntimeError('100MB diagnostic output ceiling')
    def pin(self,e):
        self.check();path=local(e['path']);actual=sha(path)
        if actual!=e['sha256']:raise ValueError('Changed pinned artifact: '+str(path))
        self.inputs[str(path)]=actual;return path
    def audit(self,e):
        path=self.pin(e);key=(str(path),e['sha256'])
        if key not in self.audit_cache:
            value=read(path)
            if value.get('status')!='PASS' or value.get('failures'):raise ValueError('Prior independent complete PASS required')
            identities=[value[k] for k in ('auditor_sha256','source_sha256') if k in value]
            if not identities or any(v!=e['auditor_sha256'] for v in identities):raise ValueError('Exact independent auditor identity; conflicting legacy/current keys rejected')
            self.pin(dict(path=e['auditor_path'],sha256=e['auditor_sha256']))
            index={}
            for name,digest in value['inputs'].items():index.setdefault(str(local(name)),set()).add(digest)
            self.audit_cache[key]=(value,index)
        return self.audit_cache[key]
    def audited(self,e,report):
        path=self.pin(e);_,index=self.audit(report)
        if index.get(str(path))!={e['sha256']}:raise ValueError('Exact artifact not bound by prior independent PASS: '+str(path))
        return path
    def arrays(self,e,report=None,keys=None):
        path=self.audited(e,report) if report else self.pin(e)
        with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
    def close(self,x,y,label,atol=1e-9,rtol=1e-11):
        self.checks+=1
        if np.shape(x)!=np.shape(y) or not np.allclose(x,y,atol=atol,rtol=rtol,equal_nan=False):raise ArithmeticError(label)


def feature_view(saved,kind):
    if kind=='legacy':
        x=np.column_stack((saved['boundary'],saved['heat'],saved['instant']))
        observed=saved['instant'][:,[0,3,6]].astype(np.float64)
    elif kind=='rwth':
        x=saved['X33'];observed=np.column_stack((np.abs(x[:,0]),np.abs(x[:,1]),x[:,3]))
    else:raise ValueError('Known fixed feature interpretation')
    if x.ndim!=2 or x.shape[1]!=33 or not np.isfinite(x).all():raise ValueError('Finite original33 features')
    return x.astype(np.float64),observed


def fit_reference(item,a):
    a.pin(item['reference_config']);blocks=[];obs=[];ys=[];ids=[]
    for e in item['fit_cache']:
        keys=['boundary','heat','instant','target'] if item['kind']=='legacy' else ['X33','target']
        data=a.arrays(e,item['fit_audit'],keys);x,o=feature_view(data,item['kind'])
        if len(x)!=e['rows'] or data['target'].shape!=(len(x),):raise ValueError('Full FIT reference geometry')
        ix=np.arange(item['warmup'],len(x),5,dtype=np.int64)
        if not len(ix):raise ValueError('No FIT endpoint may disappear')
        blocks.append(x[ix]);obs.append(o[ix]);ys.append(data['target'][ix]);ids.append(e['id'])
    x=np.concatenate(blocks);o=np.concatenate(obs);y=np.concatenate(ys)
    if ids!=item['fit_ids'] or not np.isfinite(y).all():raise ValueError('Frozen FIT IDs and finite labels')
    conditions={}
    for name,v in zip(['current','abs_speed','coolant','target_temperature'],[o[:,0],o[:,1],o[:,2],y]):
        q=np.quantile(v,[.05,.5,.95]);conditions[name]=dict(min=float(v.min()),q05=float(q[0]),median=float(q[1]),q95=float(q[2]),max=float(v.max()))
    return dict(dataset=item['dataset'],kind=item['kind'],fit_ids=ids,training_endpoints=len(y),warmup=item['warmup'],stride=5,
        feature_min=x.min(0).tolist(),feature_max=x.max(0).tolist(),conditions=conditions,
        scope='Pooled actual FIT training endpoints only; coordinate ranges are descriptive, not a deployment OOD guarantee')


def measures(y,p):
    e=p-y;under=np.maximum(-e,0);over=np.maximum(e,0)
    return dict(n=len(y),mse=float(np.mean(e*e)),rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),
        bias=float(e.mean()),error_variance=float(np.mean((e-e.mean())**2)),under_rmse=float(np.sqrt(np.mean(under**2))),
        over_mae=float(over.mean()),max_under=float(under.max()),max_over=float(over.max()))


def decomposition(y,d,r,a):
    ed=d-y;er=r-y;h=(d+r)/2.;eh=h-y
    md=float(np.mean(ed**2));mr=float(np.mean(er**2));mh=float(np.mean(eh**2))
    disagreement=float(np.mean((d-r)**2));expert=.5*(mr-md);diversity=-.25*disagreement;delta=mh-md
    mu_d=float(ed.mean());mu_r=float(er.mean());var_d=float(np.mean((ed-mu_d)**2));var_r=float(np.mean((er-mu_r)**2))
    covariance=float(np.mean((ed-mu_d)*(er-mu_r)));bias_delta=((mu_d+mu_r)/2.)**2-mu_d**2
    variance_delta=.25*(var_d+var_r+2*covariance)-var_d
    a.close(delta,expert+diversity,'Exact half MSE decomposition');a.close(delta,bias_delta+variance_delta,'Exact bias/variance decomposition')
    denominator=math.sqrt(mh)+math.sqrt(md)
    terms=[expert,diversity,bias_delta,variance_delta]
    terms=[v/denominator for v in terms] if denominator else [0.]*4
    rmse_delta=math.sqrt(mh)-math.sqrt(md)
    a.close(rmse_delta,sum(terms[:2]),'Exact cycle RMSE attribution')
    return dict(mse_d9=md,mse_expert=mr,mse_half=mh,delta_mse=delta,expert_quality_term=expert,
        disagreement=disagreement,diversity_term=diversity,mu_d9=mu_d,mu_expert=mu_r,
        variance_d9=var_d,variance_expert=var_r,error_covariance=covariance,
        centered_error_correlation=covariance/math.sqrt(var_d*var_r) if var_d>0 and var_r>0 else None,
        bias_squared_delta=bias_delta,variance_delta=variance_delta,rmse_denominator=denominator,delta_rmse=rmse_delta,
        rmse_expert_quality_term=terms[0],rmse_diversity_term=terms[1],rmse_bias_squared_term=terms[2],rmse_variance_term=terms[3])


def residual_effect(y,prior,r,a):
    error=prior-y;correction=r-prior;cross=float(2*np.mean(error*correction));energy=float(np.mean(correction**2))
    change=float(np.mean((r-y)**2)-np.mean(error**2));a.close(change,cross+energy,'Exact residual correction contribution')
    return dict(prior_mse=float(np.mean(error**2)),hybrid_mse=float(np.mean((r-y)**2)),delta_mse=change,
        error_correction_cross_term=cross,correction_energy=energy,correction_mean=float(correction.mean()),
        correction_rms=math.sqrt(energy))


def slice_result(y,values,mask,a):
    n=int(mask.sum())
    if not n:return dict(n=0,methods=None,decomposition=None,residual_effect=None)
    yy=y[mask];vv={k:v[mask] for k,v in values.items()}
    return dict(n=n,methods={k:measures(yy,v) for k,v in vv.items()},
        decomposition={k:decomposition(yy,vv['d9'],vv[k],a) for k in PAIRS},
        residual_effect=residual_effect(yy,vv['linear_prior'],vv['linear_residual'],a))


def profile(item,entry,panel,reference,a,rwth_prior=None,cpu_record=None):
    fields=['index','source_row']+['prediction__'+k for k in ['c15__left','linear33__right','linear33__half',
        'linear33__cal_macro_rmse','direct200__right','direct200__half','free25__right','free25__half']]
    saved=a.arrays(entry,item['audit'],fields);ix=saved['index'].astype(np.int64);n=len(ix)
    truth=a.arrays(entry['truth_ref'],item['audit'],['index','target'])
    if not np.array_equal(truth['index'],ix) or not np.all(np.diff(ix)==1):raise ValueError('All original contiguous evaluation endpoints')
    y=truth['target'].astype(np.float64)
    if y.shape!=(n,) or not np.isfinite(y).all():raise ValueError('Finite aligned original labels')
    if item['kind']=='legacy':
        data=a.arrays(entry['source_features'],item['audit'],['boundary','heat','instant','source_row','prediction__linear33_ridge'])
        features,observed=feature_view(data,'legacy');features=features[ix];observed=observed[ix]
        prior=data['prediction__linear33_ridge'].astype(np.float64)
        if not np.array_equal(saved['source_row'],data['source_row'][ix]):raise ValueError('Original legacy source rows')
    else:
        data=a.arrays(entry['source_features'],item['audit'],['X33','index','source_row'])
        features,observed=feature_view(data,'rwth')
        if not np.array_equal(data['index'],ix) or not np.array_equal(saved['source_row'],data['source_row']):raise ValueError('Original RWTH source rows')
        scale=rwth_prior['scale'];coef=np.asarray(rwth_prior['coef'],dtype=np.float64)
        if rwth_prior['family']!='linear33' or coef.shape!=(34,):raise ValueError('Frozen34-coefficient linear33 prior')
        standardized=(features-np.asarray(scale['x_mean']))/np.asarray(scale['x_std'])
        prior=scale['y_mid']+scale['y_scale']*(np.column_stack((np.ones(n),standardized))@coef)
        cpu=a.arrays(cpu_record,item['audit'],['prediction__linear33_prior'])
        a.close(prior,cpu['prediction__linear33_prior'],'Frozen RWTH prior replay matches audited saved base column',atol=1e-8,rtol=2e-11)
    mapping={'d9':'c15__left','linear_residual':'linear33__right','linear_half':'linear33__half',
        'linear_cal_mix':'linear33__cal_macro_rmse','direct200':'direct200__right','direct200_half':'direct200__half',
        'free25_residual':'free25__right','free25_half':'free25__half'}
    values={k:saved['prediction__'+v].astype(np.float64) for k,v in mapping.items()}
    values['linear_prior']=prior;values['prior_half']=(values['d9']+prior)/2.;values={k:values[k] for k in METHODS}
    for k,v in values.items():
        if v.shape!=(n,) or not np.isfinite(v).all():raise ValueError('All fixed finite methods: '+k)
    for label,right in [('linear_half','linear_residual'),('direct200_half','direct200'),('free25_half','free25_residual')]:
        a.close(values[label],(values['d9']+values[right])/2.,'Exact existing fixed half '+label,atol=0.,rtol=0.)
    whole=slice_result(y,values,np.ones(n,dtype=bool),a);blocks=[]
    for b in range(5):
        lo=n*b//5;hi=n*(b+1)//5;mask=np.zeros(n,dtype=bool);mask[lo:hi]=True
        current=slice_result(y,values,mask,a);current.update(bin=b,index_start=None if lo==hi else int(ix[lo]),
            index_end_inclusive=None if lo==hi else int(ix[hi-1]),evaluated_fraction_start=b/5,evaluated_fraction_end=(b+1)/5)
        # Contributions use the whole-profile RMSE denominator so their sum is
        # the original profile ΔRMSE, not the average of block RMSE differences.
        denominator=whole['decomposition']['linear_residual']['rmse_denominator']
        dd=current['decomposition']['linear_residual'] if current['n'] else None
        current['primary_delta_rmse_contribution']=(current['n']/n*dd['delta_mse']/denominator) if dd and denominator else 0.
        blocks.append(current)
    a.close(sum(v['primary_delta_rmse_contribution'] for v in blocks),whole['decomposition']['linear_residual']['delta_rmse'],'Five blocks sum to whole-profile ΔRMSE')
    conditions={};outside={}
    for name,v in zip(['current','abs_speed','coolant','target_temperature'],[observed[:,0],observed[:,1],observed[:,2],y]):
        q=reference['conditions'][name];masks={'below_fit_q05':v<q['q05'],'central_fit_q05_q95':(v>=q['q05'])&(v<=q['q95']),'above_fit_q95':v>q['q95']}
        conditions[name]={key:slice_result(y,values,mask,a) for key,mask in masks.items()}
        outside[name]=dict(below_fit_min_fraction=float(np.mean(v<q['min'])),above_fit_max_fraction=float(np.mean(v>q['max'])))
    oob=(features<np.asarray(reference['feature_min']))|(features>np.asarray(reference['feature_max']))
    stats=dict(any_coordinate_outside_fraction=float(np.mean(np.any(oob,axis=1))),
        mean_coordinate_outside_fraction=float(oob.mean()),per_coordinate_outside_fraction=np.mean(oob,axis=0).tolist())
    row=dict(dataset=item['dataset'],role=item['role'],profile=entry['profile'],original_index_start=int(ix[0]),
        original_index_end=int(ix[-1]),whole=whole,progress_blocks=blocks,condition_strata=conditions,
        outside_fit_ranges=outside,feature_range_diagnostic=stats,all_original_rows_retained=True)
    arrays=dict(index=ix,source_row=saved['source_row'].astype(np.int64),target=y,
        observed_current=observed[:,0],observed_abs_speed=observed[:,1],observed_coolant=observed[:,2],feature_outside_fit=oob)
    arrays.update({'prediction__'+k:v for k,v in values.items()})
    return row,arrays


def aggregate(rows,a):
    summary={}
    for method in METHODS:
        entries=[r['whole']['methods'][method] for r in rows];n=sum(v['n'] for v in entries)
        summary[method]=dict(profiles=len(entries),n=n,**{'macro_'+k:float(np.mean([v[k] for v in entries])) for k in ['rmse','mae','bias','under_rmse','over_mae']},
            pooled_rmse=math.sqrt(sum(v['n']*v['mse'] for v in entries)/n))
    terms=['delta_rmse','rmse_expert_quality_term','rmse_diversity_term','rmse_bias_squared_term','rmse_variance_term']
    attribution={p:{k:float(np.mean([r['whole']['decomposition'][p][k] for r in rows])) for k in terms} for p in PAIRS}
    a.close(attribution['linear_residual']['delta_rmse'],summary[PRIMARY]['macro_rmse']-summary['d9']['macro_rmse'],'MacroRMSE unchanged by attribution')
    differences=[r['whole']['decomposition']['linear_residual']['delta_rmse'] for r in rows]
    return dict(methods=summary,macro_rmse_attribution=attribution,
        primary=dict(delta_macro_rmse=summary[PRIMARY]['macro_rmse']-summary['d9']['macro_rmse'],
            relative_reduction=1-summary[PRIMARY]['macro_rmse']/summary['d9']['macro_rmse'] if summary['d9']['macro_rmse'] else None,
            per_profile_deltas=[dict(profile=r['profile'],delta_rmse=d) for r,d in zip(rows,differences)],
            wins=sum(d<0 for d in differences),ties=sum(d==0 for d in differences),losses=sum(d>0 for d in differences)),
        formal_test_performed=False,independent_confirmation=False)


def run(config,a,out):
    if config['schema']!='c18-linear-diagnostic-v001' or config['primary']!=PRIMARY or config['weight']!=.5 or config['all_data_exposed'] is not True:
        raise ValueError('Exact new exploratory fixed-half configuration')
    if [(x['dataset'],x['role']) for x in config['cohorts']]!=COHORTS:raise ValueError('All seven exposed roles required')
    if config['producer']['sha256']!=sha(__file__) or local(config['producer']['path'])!=Path(__file__).resolve():raise ValueError('Exact diagnostic source frozen in config')
    if config['progress_blocks']!=5 or config['condition_quantiles']!=[.05,.95] or config['formal_confirmation_allowed'] is not False:raise ValueError('Frozen diagnostic, not a new confirmation')
    a.pin(config['protocol']);a.pin(config['limits'])
    references={x['dataset']:fit_reference(x,a) for x in config['datasets']};dump(out/'fit_reference.json',references)
    results=[];all_records=[];checks=[]
    for item in config['cohorts']:
        lock=read(a.audited(item['prediction_lock'],item['audit']))
        entries=[v for v in lock['predictions'] if v['role']==item['role']]
        if [v['profile'] for v in entries]!=item['ids']:raise ValueError('Complete fixed role coverage')
        metrics=read(a.audited(item['metrics'],item['audit']))['methods']
        cpu_rows={};prior=None
        if item['kind']=='rwth':
            cpu_lock=read(a.audited(item['cpu_prediction_lock'],item['audit']))
            cpu_rows={x['id']:x for x in cpu_lock['records']}
            prior=read(a.audited(item['priors'],item['model_audit']))['linear33']
            if prior['fit_ids']!=config['datasets'][2]['fit_ids']:raise ValueError('Original RWTH FIT-only prior')
        rows=[];folder=out/(item['dataset']+'__'+item['role']);folder.mkdir()
        for entry in entries:
            a.check();e=dict(entry)
            if item['kind']=='rwth' and item['role']=='confirmation':e['truth_ref']=item['truths'][str(entry['profile'])]
            row,arrays=profile(item,e,lock,references[item['dataset']],a,prior,cpu_rows.get(e['profile']))
            for method,old in [('d9','c15__left'),('linear_residual','linear33__right'),('linear_half','linear33__half'),
                ('linear_cal_mix','linear33__cal_macro_rmse'),('direct200_half','direct200__half'),('free25_half','free25__half')]:
                previous=next(x for x in metrics[old][item['role']]['per_profile'] if x['id']==e['profile'])
                for k in ('rmse','under_rmse','over_mae'):a.close(row['whole']['methods'][method][k],previous[k],'Reproduced original exposed metric '+method+'/'+k)
            npz=folder/('profile_'+str(e['profile'])+'.npz');np.savez_compressed(npz,**arrays)
            row['arrays']=ref(npz);dump(folder/('profile_'+str(e['profile'])+'.json'),row)
            rows.append(row);all_records.append(dict(dataset=item['dataset'],role=item['role'],profile=e['profile'],arrays=ref(npz)))
        result=dict(dataset=item['dataset'],role=item['role'],ids=item['ids'],summary=aggregate(rows,a))
        dump(folder/'summary.json',result);results.append(result)
    if len(all_records)!=54:raise ValueError('All54 exposed profile-role records required')
    dump(out/'results.json',results);dump(out/'diagnostic_arrays_lock.json',dict(primary=PRIMARY,methods=METHODS,records=all_records))
    return dict(status='C18_FIXED_HALF_DIAGNOSTIC_COMPLETE',primary=PRIMARY,methods=METHODS,cohorts=7,profiles=54,
        new_training_fits=0,new_weight_optimization=False,new_formal_tests=False,all_data_previously_exposed=True,
        pending_independent_audit=True,identity_checks=a.checks,interpretation='Exact error accounting and descriptive strata, not causal proof or new confirmation')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','config-sha256','output-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--max-seconds',type=float,default=600.);args=p.parse_args()
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=600:raise ValueError('Finite bounded600second stage')
    out=local(args.output_dir)
    if WORK/'runs' not in out.parents:raise ValueError('New C18 output only')
    out.mkdir(parents=True,exist_ok=False);a=Intake(args.max_seconds,out)
    dump(out/'configuration.json',dict(vars(args),source=str(Path(__file__).resolve()),source_sha256=sha(__file__),pid=os.getpid(),argv=sys.argv))
    try:
        config=read(a.pin(dict(path=args.config,sha256=args.config_sha256)));result=run(config,a,out)
        dump(out/'input_pins.json',a.inputs);a.check()
        result.update(source_sha256=sha(__file__),config=dict(path=args.config,sha256=args.config_sha256),elapsed_seconds=time.monotonic()-a.start,
            files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
        dump(out/'summary.json',result);a.check();print(json.dumps(dict(status=result['status'],profiles=result['profiles'],elapsed_seconds=result['elapsed_seconds'])))
    except Exception as exc:dump(out/'failure.json',dict(status='FAILED',error_type=type(exc).__name__,message=str(exc)));raise


if __name__=='__main__':main()
