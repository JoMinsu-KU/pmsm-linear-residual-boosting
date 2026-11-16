"""Independent audit of five fixed expert pairs and five scalar rules.

No scientific producer import, model training, new source data or hypothesis test.
Pinned independent C17/C16 audit functions supply scalar and metric primitives.
"""
import argparse
import hashlib
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

ROOT=Path(__file__).resolve().parents[2]
FOLLOWUP=ROOT/'followup-risk-20261110'
GATE_AUDIT_SHA='d8ffcb5c43a9f59c7c8b57179e0ccabf5007aa9450ff7c94693a8c5ea831ca74'
RISK_AUDIT_SHA='2cfb48953741515e8d9815073075bcc3e46c717c5110837d280acdb93a2bd035'
PROTOCOL_SHA='eb06ca878a603ba954bc5f53613f5d16749da87ba041798ab22241aa794fffab'
LEFT='prediction__direct9_fitcv_selected'
RIGHT_KEYS=['prediction__positive_rc_insample_boost','prediction__direct_boost',
    'prediction__linear33_insample_boost','prediction__unconstrained_rc_insample_boost','prediction__positive_rc']
PAIRS=dict(zip(['c15','direct200','linear33','free25','positive_prior'],RIGHT_KEYS))
RULES=['left','right','half','cal_mse','cal_macro_rmse']
METHODS=[pair+'__'+rule for pair in PAIRS for rule in RULES]
PRIMARY='c15__cal_macro_rmse'
PROTOCOL=dict(path='followup-risk-20261110/protocols/combination-controls-v002.md',sha256=PROTOCOL_SHA)
SCALAR_SOURCE=dict(path='followup-risk-20261110/code/gate_worker_v001.py',sha256='d73b6eb70db5c6e57f79a5334c4000d373189d3e49aaaf7371daee9b9a08e25b')
METRIC_SOURCE=dict(path='followup-risk-20261110/code/risk_screen_analysis_v001.py',sha256='7d514ccd7f9471a4e5d8dcc29604d8782c6c0dad300cdc830a253d4e1c759282')


def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def local(value):
    p=(ROOT/value).resolve()
    if p!=ROOT and ROOT not in p.parents:raise ValueError('Outside campaign')
    return p


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def independent_dependencies():
    filename=FOLLOWUP/'code/gate_audit_v001.py'
    if sha(filename)!=GATE_AUDIT_SHA:raise ValueError('Independent gate audit source changed')
    spec=importlib.util.spec_from_file_location('independent_c17_scalar_primitives',filename)
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    risk=gate.shared()
    return gate,risk


def metadata_binding(a,report,entry):
    suffix=entry['path'].replace('\\','/')
    hits=[digest for name,digest in report['inputs'].items() if name.replace('\\','/').endswith('/'+suffix)]
    a.require(bool(hits) and all(x==entry['sha256'] for x in hits),'Full prior audit binds input/'+suffix)


def threshold_reference(a,risk):
    rows=[]
    for entry in risk['fit_cache']:
        value=a.arrays(a.pin(entry))
        a.require(value['target'].shape==(entry['rows'],),'FIT target row geometry')
        rows.append(value['target'][300::5].astype(np.float64))
    return float(np.quantile(np.concatenate(rows),.95))


def component_sources(a,risk):
    records={}
    for role in risk['role_inputs']:
        lockpath=a.pin(role['prediction_lock']);lock=read(lockpath)
        report=read(a.pin(role['prior_evaluation_audit']))
        a.require(report.get('status')=='PASS' and report.get('stage')=='evaluation','Prior full evaluation audit PASS')
        a.same(lock['role'],role['role'],'Source lock role')
        a.same([r['id'] for r in lock['records']],role['ids'],'Source lock complete ordered IDs')
        for entry,previous in zip(role['records'],lock['records']):
            a.same(entry['rows'],previous['rows'],'Source original rows')
            a.require(local(entry['features']['path'])==(lockpath.parent/previous['path']).resolve() and entry['features']['sha256']==previous['sha256'],'Source feature exact lock mapping')
            for pin in (entry['features'],entry['truth'],role['prediction_lock']):metadata_binding(a,report,pin)
            data=a.arrays(a.pin(entry['features']));ix=np.arange(300,entry['rows'],dtype=np.int64)
            a.require(np.array_equal(data['index'],ix),'All300: endpoints')
            a.require(data['source_row'].shape==(entry['rows'],) and np.issubdtype(data['source_row'].dtype,np.integer),'Original source rows')
            values={}
            for key in [LEFT]+RIGHT_KEYS:
                a.require(data[key].dtype==np.float64 and data[key].shape==ix.shape,'Pinned full expert prediction/'+key)
                values[key]=data[key]
            records[(role['role'],entry['id'])]=dict(id=entry['id'],role=role['role'],index=ix,
                source_row=data['source_row'][ix],components=values,source_features=entry['features'],truth_ref=entry['truth'])
    return records


def attach_target(a,record):
    if 'y' not in record:
        data=a.arrays(a.pin(record['truth_ref']))
        a.require(np.array_equal(data['index'],record['index']) and data['target'].shape==record['index'].shape,'Exact truth alignment')
        record['y']=data['target'].astype(np.float64)
    return record['y']


def pair_records(records,right):
    return [dict(id=r['id'],d=r['components'][LEFT],h=r['components'][right],y=r['y']) for r in records]


def scalar_check(a,gate,model,records,tag):
    # The original scalar protocol uses equal profile weight, not pooled rows.
    cross=np.asarray([np.mean((r['h']-r['d'])*(r['y']-r['d'])) for r in records])
    delta2=np.asarray([np.mean((r['h']-r['d'])**2) for r in records])
    numerator,denominator=float(cross.mean()),float(delta2.mean())
    weight=float(np.clip(numerator/denominator,0.,1.)) if denominator!=0 else 0.
    a.same(model['fit_ids'],[r['id'] for r in records],tag+'/fit IDs')
    a.same(model['cal_mse'],dict(weight=weight,numerator=numerator,denominator=denominator),tag+'/closed form')
    scalar=model['cal_macro_rmse'];w=scalar['weight']
    a.require(isinstance(w,(float,int)) and math.isfinite(w) and 0<=w<=1,tag+'/scalar feasibility')
    value=gate.scalar_objective(records,w);lo,hi=gate.subgradients(records,w)
    a.same(scalar['objective'],value,tag+'/scalar full objective')
    a.same(scalar['endpoint_objectives'],[dict(weight=x,objective=gate.scalar_objective(records,x)) for x in (0.,1.)],tag+'/endpoints')
    a.same(scalar['subgradient_interval'],[lo,hi],tag+'/subgradient interval')
    slope=0. if lo<=0<=hi else lo if lo>0 else hi
    gap=max(0.,float(slope*w-min(0.,slope)))
    a.same(scalar['convex_gap'],gap,tag+'/convex objective gap')
    best,best_w=gate.scalar_oracle(records)
    a.check(value-best<=1e-8*(1+abs(value)),tag+'/independent derivative-bisection optimum',dict(actual=value,oracle=best,oracle_weight=best_w))
    optimizer=scalar['optimizer']
    a.require(optimizer['success'] is True and optimizer['xatol']==1e-12 and optimizer['maxiter']==500 and 0<=optimizer['x']<=1,tag+'/fixed bounded solver')
    a.same(optimizer['objective'],gate.scalar_objective(records,optimizer['x']),tag+'/bounded optimizer objective')
    chosen=min((gate.scalar_objective(records,x),x) for x in [0.,1.,optimizer['x']])
    a.check(abs(chosen[0]-value)<=1e-9*(1+abs(value)),tag+'/endpoint-aware bounded choice')
    return dict(objective=value,independent_oracle_objective=best,weight=w,convex_gap=gap)


def correlation_reference(left,right,truth):
    e1=left-truth;e2=right-truth;c1=e1-e1.mean();c2=e2-e2.mean()
    lv,rv=float(np.mean(c1*c1)),float(np.mean(c2*c2));cov=float(np.mean(c1*c2))
    denominator=math.sqrt(lv*rv)
    return dict(evaluated_rows=len(truth),error_sign='prediction_minus_truth',left_mean_error=float(e1.mean()),
        right_mean_error=float(e2.mean()),left_centered_variance=lv,right_centered_variance=rv,
        centered_covariance=cov,centered_error_correlation=None if denominator==0 else cov/denominator,
        correlation_defined=denominator!=0,uncentered_mean_error_product=float(np.mean(e1*e2)),
        scope='descriptive complementarity; not a causal explanation')


def statistics_reference(calrecords):
    result=dict(profile_ids=np.asarray([r['id'] for r in calrecords],dtype=np.int64),
                counts=np.asarray([len(r['index']) for r in calrecords],dtype=np.int64))
    for pair,key in PAIRS.items():
        rows=pair_records(calrecords,key)
        result[pair+'__mse_cross']=np.asarray([np.mean((r['h']-r['d'])*(r['y']-r['d'])) for r in rows],dtype=np.float64)
        result[pair+'__mse_delta2']=np.asarray([np.mean((r['h']-r['d'])**2) for r in rows],dtype=np.float64)
    return result


def replay_predictions(a,gate,lock,models,risk,source):
    cal=risk['calibration_role'];calids=models['calibration_ids']
    expected=[('calibration_lopo',cal,p) for p in calids]+[(r['role'],r['role'],p) for r in risk['role_inputs'] for p in r['ids']]
    a.same([[e['role'],e['source_role'],e['profile']] for e in lock['predictions']],[list(x) for x in expected],'Complete ordered25-method prediction cohorts')
    result={}
    for entry in lock['predictions']:
        role,srole,pid=entry['role'],entry['source_role'],entry['profile'];rec=source[(srole,pid)]
        a.same(entry['source_features'],rec['source_features'],'Prediction exact source features')
        a.same(entry['truth_ref'],rec['truth_ref'],'Prediction exact source truth')
        is_lopo=role=='calibration_lopo'
        a.same(entry['model_kind'],'lopo' if is_lopo else 'final','Prediction model kind')
        a.same(entry['held_out'],pid if is_lopo else None,'Prediction held ID')
        data=a.arrays(a.pin(entry))
        a.require(set(data)=={'index','source_row'}|{'prediction__'+method for method in METHODS},'Exactly25 saved methods')
        a.require(data['index'].dtype==np.int64 and data['source_row'].dtype==np.int64,'Identifier int64')
        gate.exact(a,data['index'],rec['index'],'Exact local indices')
        gate.exact(a,data['source_row'],rec['source_row'],'Exact source rows')
        a.same(entry['evaluated_rows'],len(rec['index']),'Complete prediction row count')
        for pair,right in PAIRS.items():
            item=models['pairs'][pair]
            model=next(x['model'] for x in item['lopo'] if x['held_out']==pid) if is_lopo else item['final']
            d,h=rec['components'][LEFT],rec['components'][right];delta=h-d
            expected_values=dict(left=d,right=h,half=(d+h)/2,cal_mse=d+model['cal_mse']['weight']*delta,
                cal_macro_rmse=d+model['cal_macro_rmse']['weight']*delta)
            for rule,value in expected_values.items():
                key='prediction__'+pair+'__'+rule
                a.require(data[key].dtype==np.float64,'FP64 prediction/'+key)
                gate.close(a,data[key],value,'Full prediction replay/'+key)
        result[(role,pid)]=data
    return result


def compare(metrics,role,candidate,baseline):
    c,b=metrics[candidate][role],metrics[baseline][role];cs,bs=c['summary'],b['summary']
    difference=[x['rmse']-y['rmse'] for x,y in zip(c['per_profile'],b['per_profile'])]
    ratio=lambda key:cs[key]/bs[key] if bs[key]!=0 else None
    return dict(role=role,candidate=candidate,comparator=baseline,delta_macro_rmse=cs['macro_rmse']-bs['macro_rmse'],
        rmse_ratio=ratio('macro_rmse'),under_rmse_ratio=ratio('macro_under_rmse'),over_mae_ratio=ratio('macro_over_mae'),
        under_rmse_diagnostic_flag=cs['macro_under_rmse']>1.05*bs['macro_under_rmse']+1e-12,
        over_mae_diagnostic_flag=cs['macro_over_mae']>1.25*bs['macro_over_mae']+1e-12,
        per_profile_rmse_differences=difference,profile_ids=[x['id'] for x in c['per_profile']],
        wins=sum(x<0 for x in difference),ties=sum(x==0 for x in difference),losses=sum(x>0 for x in difference),
        zero_baseline_ratio_policy='null ratio; exact inequality flag without denominator floor')


def metric_replay(a,risk_audit,risk,source,predictions,lock,threshold):
    roles=['calibration_lopo']+[r['role'] for r in risk['role_inputs']]
    ids={r['role']:r['ids'] for r in risk['role_inputs']};ids['calibration_lopo']=ids[risk['calibration_role']]
    result={m:{} for m in METHODS};complementarity=[]
    for role in roles:
        original=risk['calibration_role'] if role=='calibration_lopo' else role
        for method in METHODS:
            profiles=[]
            for pid in ids[role]:
                row=source[(original,pid)];y=attach_target(a,row);saved=predictions[(role,pid)]
                profiles.append((pid,y,saved['prediction__'+method],saved['index']))
            result[method][role]=risk_audit.group_metrics(profiles,risk['metrics']['sample_interval_seconds'],threshold)
        for pid in ids[role]:
            data=predictions[(role,pid)];y=source[(original,pid)]['y']
            complementarity.append(dict(role=role,profile=pid,**correlation_reference(data['prediction__c15__left'],data['prediction__c15__right'],y)))
    comparisons=[]
    for role in roles:
        comparisons.extend(compare(result,role,m,'c15__left') for m in METHODS)
        comparisons.extend(compare(result,role,PRIMARY,p+'__cal_macro_rmse') for p in PAIRS if p!='c15')
    return (dict(dataset=risk['dataset'],high_temperature_threshold_c=threshold,methods=result),comparisons,
        dict(primary_pair='c15',records=complementarity,scope='Previously exposed data, descriptive only; no pair selection or causal attribution'))


def reproduce_primary(a,gate,args,models,predictions,lock):
    report=read(a.pin(dict(path=args.reference_gate_audit,sha256=args.reference_gate_audit_sha256)))
    a.require(report.get('status')=='PASS' and report['source_sha256']==GATE_AUDIT_SHA and
        report['producer_sha256']==SCALAR_SOURCE['sha256'] and report['config_sha256']==args.config_sha256,'Original C17 same-config independent audit PASS')
    refdir,summary=gate.pin_summary(a,args.reference_gate_fit_dir,args.reference_gate_fit_sha256)
    a.require(summary['status']=='GATE_MODELS_PREDICTIONS_LOCKED' and summary['source_sha256']==SCALAR_SOURCE['sha256'] and
        summary['config_sha256']==args.config_sha256,'Exact qualified original C17 fit')
    def binding(p,digest):
        metadata_binding(a,report,dict(path=p.relative_to(ROOT).as_posix(),sha256=digest))
    binding(refdir/'summary.json',args.reference_gate_fit_sha256)
    for name,digest in summary['files'].items():binding(refdir/name,digest)
    original=read(refdir/'models.json');predpath=a.pin(summary['prediction_lock']);oldlock=read(predpath)
    a.require(a.pin(oldlock['model_lock'])==refdir/'model_lock.json','Original C17 lock chain')
    current=models['pairs']['c15']
    def compare_models(now,old,tag):
        a.same(now['fit_ids'],old['fit_ids'],tag+'/IDs')
        a.same(now['cal_mse'],old['fixed_cal_mse'],tag+'/closed-form exact reproduction',atol=1e-12,rtol=1e-12)
        a.same(now['cal_macro_rmse'],old['fixed_cal_macro_rmse'],tag+'/scalar exact reproduction',atol=1e-12,rtol=1e-12)
    compare_models(current['final'],original['final'],'Primary final')
    a.same([x['held_out'] for x in current['lopo']],[x['held_out'] for x in original['lopo']],'Primary all old LOPO IDs')
    for now,old in zip(current['lopo'],original['lopo']):compare_models(now['model'],old['model'],'Primary LOPO/'+str(now['held_out']))
    a.same([[e['role'],e['profile']] for e in oldlock['predictions']],[[e['role'],e['profile']] for e in lock['predictions']],'Primary old/new role ordering')
    mapping=dict(left='direct9',right='C15',half='fixed_half',cal_mse='fixed_cal_mse',cal_macro_rmse='fixed_cal_macro_rmse')
    for entry in oldlock['predictions']:
        metadata_binding(a,report,entry)
        old=a.arrays(a.pin(entry));now=predictions[(entry['role'],entry['profile'])]
        gate.exact(a,now['index'],old['index'],'Old/new primary local rows')
        gate.exact(a,now['source_row'],old['source_row'],'Old/new primary source rows')
        for newrule,oldrule in mapping.items():
            gate.close(a,now['prediction__c15__'+newrule],old['prediction__'+oldrule],
                       'Full primary original C17 prediction reproduction/'+newrule,atol=1e-11,rtol=1e-12)
    return dict(reference_gate_audit_sha256=args.reference_gate_audit_sha256,
        original_prediction_lock_sha256=sha(predpath),all_final_and_lopo_weights=True,all_five_primary_rules_all_endpoints=True)


def audit_stage(a,args,gate,risk_audit):
    cfg,risk=gate.load_configuration(a,args)
    a.pin(PROTOCOL);a.pin(SCALAR_SOURCE);a.pin(METRIC_SOURCE)
    a.pin(dict(path='followup-risk-20261110/code/gate_audit_v001.py',sha256=GATE_AUDIT_SHA))
    a.pin(dict(path='followup-risk-20261110/code/risk_audit_v001.py',sha256=RISK_AUDIT_SHA))
    a.pin(dict(path='followup-risk-20261110/code/combination_controls_v002.py',sha256=args.producer_sha256))
    folder,summary=gate.pin_summary(a,args.fit_dir,args.fit_sha256)
    a.require(summary['status']=='COMBINATION_MODELS_PREDICTIONS_LOCKED' and summary['source_sha256']==args.producer_sha256 and
        summary['config_sha256']==args.config_sha256 and summary['dataset']==cfg['dataset'],'Qualified same-source fit stage')
    a.same(summary['protocol'],PROTOCOL,'Frozen v002 protocol')
    a.same(summary['methods'],METHODS,'Exactly25 fit methods')
    a.same(summary['primary'],PRIMARY,'Unchanged primary pair/rule')
    conf=read(folder/'configuration.json')
    a.require(conf['mode']=='fit-predict' and conf['source_sha256']==args.producer_sha256 and conf['config_sha256']==args.config_sha256,'Fit configuration source/mode')
    a.same(conf['scalar_dependency'],SCALAR_SOURCE,'Scientific scalar dependency pin')
    a.same(conf['metric_dependency'],METRIC_SOURCE,'Scientific metric dependency pin')
    mpath=a.pin(summary['model_lock']);ppath=a.pin(summary['prediction_lock'])
    a.require(mpath==folder/'model_lock.json' and ppath==folder/'prediction_lock.json','Own model/prediction locks')
    model_lock,lock=read(mpath),read(ppath)
    for value,kind,status in ((model_lock,'model','MODELS_LOCKED'),(lock,'prediction','ALL_PREDICTIONS_LOCKED')):
        a.require(value['schema']=='c17-combination-'+kind+'-lock-v002' and value['status']==status,'Qualified '+kind+' lock')
        a.require(value['source_sha256']==args.producer_sha256 and value['config_sha256']==args.config_sha256 and value['dataset']==cfg['dataset'],'Lock source/config/dataset')
        a.same(value['protocol'],PROTOCOL,'Lock prospective v002 protocol')
        a.same(value['methods'],METHODS,'Lock method panel');a.same(value['primary'],PRIMARY,'Lock primary')
        a.require(value['calibration_truth_accessed'] is True and value['evaluation_truth_accessed'] is False,'CAL-only fitting boundary')
    a.require(lock['independent_confirmation'] is False and lock['all_evaluation_previously_exposed'] is True,'Exposed scope')
    a.require(a.pin(lock['model_lock'])==mpath,'Predictions bind exact model lock')
    files={'configuration.json','fit_threshold.json','calibration_statistics.npz','models.json'}
    a.require(set(model_lock['files'])==files,'Exact locked scalar-model artifacts')
    for name,digest in model_lock['files'].items():
        a.require(summary['files'][name]==digest,'Model/stage dual binding')
        a.pin(dict(path=str(folder/name),sha256=digest))
    input_pins=read(folder/'input_pins.json')
    for name,digest in input_pins.items():a.pin(dict(path=name,sha256=digest))
    for name,digest in model_lock['input_hashes_before_evaluation_features'].items():
        a.require(input_pins.get(name)==digest,'Before-evaluation-feature input unchanged')
    forbidden={str(local(e['truth']['path'])) for role in risk['role_inputs'] if role['role']!=risk['calibration_role'] for e in role['records']}
    a.check(not forbidden.intersection(str(local(name)) for name in input_pins),'Held truth absent from fit input pins')
    threshold=threshold_reference(a,risk)
    expected_threshold=dict(high_temperature_threshold_c=threshold,fit_ids=risk['train_ids'],
        fit_endpoint_counts={str(e['id']):len(range(300,e['rows'],5)) for e in risk['fit_cache']},sample_scope='300::5')
    a.same(read(folder/'fit_threshold.json'),expected_threshold,'Independent FIT95 reporting threshold')
    source=component_sources(a,risk)
    calrole=next(r for r in risk['role_inputs'] if r['role']==risk['calibration_role'])
    calrecords=[source[(calrole['role'],p)] for p in calrole['ids']]
    for record in calrecords:attach_target(a,record)
    expected_stats=statistics_reference(calrecords);saved_stats=a.arrays(folder/'calibration_statistics.npz')
    a.require(set(saved_stats)==set(expected_stats),'Exact five-pair statistic fields')
    for key,value in expected_stats.items():
        a.require(saved_stats[key].dtype==value.dtype,'Statistic exact dtype/'+key)
        if value.dtype.kind=='i':gate.exact(a,saved_stats[key],value,'Statistic/'+key)
        else:gate.close(a,saved_stats[key],value,'Statistic/'+key)
    models=read(folder/'models.json')
    a.require(models['schema']=='c17-combination-models-v002' and models['dataset']==cfg['dataset'],'Models schema/dataset')
    a.same(models['left_key'],LEFT,'Fixed left expert');a.same(models['pair_order'],list(PAIRS),'Five fixed ordered right components')
    a.same(models['rules'],RULES,'Five equal-budget rules');a.same(models['methods'],METHODS,'Complete25-model panel')
    a.same(models['primary'],PRIMARY,'No favorable-primary substitution')
    a.same(models['calibration_role'],calrole['role'],'Correct CAL role');a.same(models['calibration_ids'],calrole['ids'],'Correct CAL IDs')
    a.require(list(models['pairs'])==list(PAIRS),'Exact pair dictionary order')
    qualified=[]
    for pair,key in PAIRS.items():
        item=models['pairs'][pair];a.same(item['right_key'],key,'Exact right prediction column/'+pair)
        records=pair_records(calrecords,key)
        qualified.append(dict(pair=pair,kind='final',**scalar_check(a,gate,item['final'],records,pair+'/final')))
        a.same([f['held_out'] for f in item['lopo']],calrole['ids'],'Complete CAL-LOPO/'+pair)
        for fold in item['lopo']:
            rows=[r for r in records if r['id']!=fold['held_out']]
            qualified.append(dict(pair=pair,kind='calibration_lopo',held_out=fold['held_out'],
                **scalar_check(a,gate,fold['model'],rows,pair+'/LOPO/'+str(fold['held_out']))))
    predictions=replay_predictions(a,gate,lock,models,risk,source)
    reproduction=reproduce_primary(a,gate,args,models,predictions,lock)
    a.same(summary['pair_count'],5,'Five complete pairs');a.same(summary['models'],5*(len(calrecords)+1),'Every scalar fit counted')
    a.same(summary['predictions'],len(predictions),'Every output prediction record')
    a.results=dict(dataset=cfg['dataset'],methods=METHODS,primary=PRIMARY,scalar_qualifications=qualified,
        original_primary_reproduction=reproduction,prediction_records=len(predictions),independent_confirmation=False)
    if args.evaluation_dir:
        evaluated,evsum=gate.pin_summary(a,args.evaluation_dir,args.evaluation_sha256)
        a.require(evsum['status']=='EXPLORATORY_COMBINATION_EVALUATION_COMPLETE' and evsum['source_sha256']==args.producer_sha256 and
            evsum['config_sha256']==args.config_sha256 and evsum['dataset']==cfg['dataset'],'Qualified evaluation same-source identity')
        a.require(a.pin(evsum['prediction_lock'])==ppath,'Evaluation binds audited25 predictions')
        a.require(evsum['model_selection_performed'] is False and evsum['pair_selection_performed'] is False and
            evsum['new_base_fit_performed'] is False and evsum['independent_confirmation'] is False and evsum['all_evaluation_previously_exposed'] is True,'Evaluation interpretation flags')
        a.same(evsum['metric_dependency'],METRIC_SOURCE,'Evaluation metric dependency')
        a.same(evsum['methods'],METHODS,'Evaluation all25 methods');a.same(evsum['primary'],PRIMARY,'Evaluation fixed primary')
        a.same(evsum['roles'],['calibration_lopo']+[r['role'] for r in risk['role_inputs']],'Evaluation full ordered roles')
        a.same(evsum['high_temperature_threshold_c'],threshold,'Evaluation FIT-only threshold')
        evconf=read(evaluated/'configuration.json')
        a.require(evconf['mode']=='evaluate' and evconf['source_sha256']==args.producer_sha256 and evconf['config_sha256']==args.config_sha256,'Evaluation configuration')
        metrics,comparisons,correlations=metric_replay(a,risk_audit,risk,source,predictions,lock,threshold)
        a.same(read(evaluated/'metrics.json'),metrics,'All25 method risk metrics',atol=1e-8,rtol=1e-10)
        a.same(read(evaluated/'comparison.json'),comparisons,'All29 comparisons per cohort',atol=1e-8,rtol=1e-10)
        a.same(read(evaluated/'complementarity.json'),correlations,'Every primary error-correlation diagnostic',atol=1e-9,rtol=1e-10)
        a.results.update(metrics=metrics,comparisons=comparisons,complementarity=correlations,evaluation_audited=True)
    else:a.results['evaluation_audited']=False
    a.replayed=dict(scalar_controls='Every final and CAL-LOPO fit for all5 pairs; independent closed form and derivative bisection',
        predictions='Every endpoint for every one of25 fixed methods',
        metrics='All available evaluation metrics/29 contrasts per cohort/primary correlations if evaluation supplied',
        prior_primary='All final/LOPO coefficients and5 primary predictions reproduce independently audited C17 v001',
        base_models='No model refit; exact original audited prediction columns reused',
        independent_shared_dependencies='gate_audit_v001 scalar/lineage primitives; risk_audit_v001 generic integrity and risk metrics')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','config-sha256','fit-dir','fit-sha256','producer-sha256','output-dir',
                 'reference-gate-fit-dir','reference-gate-fit-sha256','reference-gate-audit','reference-gate-audit-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--evaluation-dir');parser.add_argument('--evaluation-sha256')
    parser.add_argument('--max-seconds',type=float,default=600.)
    args=parser.parse_args()
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=600:raise ValueError('Finite audit cap in(0,600]')
    if bool(args.evaluation_dir)!=bool(args.evaluation_sha256):raise ValueError('Both optional evaluation bindings required together')
    out=local(args.output_dir)
    if FOLLOWUP not in out.parents:raise ValueError('New audit output inside followup required')
    out.mkdir(parents=True,exist_ok=False)
    gate,risk=independent_dependencies();a=risk.Audit(args.max_seconds);error=None
    try:
        audit_stage(a,args,gate,risk);a.finish_pins()
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        a.failures.append(dict(check='All required audit operations complete',detail=error))
    report=dict(schema='c17-combination-independent-audit-v002',status='PASS' if not a.failures else 'FAIL',
        source_sha256=sha(__file__),producer_sha256=args.producer_sha256,config_sha256=args.config_sha256,
        gate_audit_dependency_sha256=GATE_AUDIT_SHA,risk_audit_dependency_sha256=RISK_AUDIT_SHA,
        checks=a.checks,failures=a.failures,inputs=a.inputs,replayed=a.replayed,results=a.results,
        elapsed_seconds=time.monotonic()-a.start,error=error,no_scientific_producer_import=True,no_base_model_fit=True)
    raw=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)
    if len(raw.encode('utf-8'))>200000000:raise RuntimeError('Audit report200MB cap')
    with (out/'report.json').open('x',encoding='utf-8') as handle:handle.write(raw)
    print(json.dumps(dict(status=report['status'],checks=a.checks,failures=len(a.failures),elapsed_seconds=report['elapsed_seconds'])))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
