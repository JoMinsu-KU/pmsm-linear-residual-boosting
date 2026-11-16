"""Identical CAL-only nonnegative shift selection for fixed C16 loss backbones.

Adaptive exploratory analysis of already exposed roles. No refitting, mixing,
new data, independent-confirmation claim, or favorable-seed selection.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
FOLLOWUP=ROOT/'followup-risk-20261110'
PROTOCOL=dict(path='followup-risk-20261110/protocols/risk-calibration-symmetry-v002.md',
              sha256='202554d6723531e76ab8210a7e7f63048fb0ba14ec38187685ae1bdb969f9260')
METRIC=dict(path='followup-risk-20261110/code/risk_screen_analysis_v001.py',
            sha256='7d514ccd7f9471a4e5d8dcc29604d8782c6c0dad300cdc830a253d4e1c759282')
AUDIT_SHA='2cfb48953741515e8d9815073075bcc3e46c717c5110837d280acdb93a2bd035'
SEEDS=[2026111701,2026111702,2026111703]
LOSSES=['mse','asym4','q90']
SHIFT_GRID=[0,.1,.2,.3,.4,.5,.75,1,1.25,1.5,2,2.5,3,4,5,7.5,10]
QUANTILES=[.5,.6,.7,.8,.9,.95]
RMSE_RATIOS=[1,1.02,1.05,1.10]
OVER_RATIOS=[1,1.10,1.25,1.50]


def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def local(value):
    p=(ROOT/value).resolve()
    if ROOT!=p and ROOT not in p.parents:raise ValueError('Path outside campaign')
    return p


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def dump(p,value):
    with Path(p).open('x',encoding='utf-8') as handle:json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)


def ref(p):
    return dict(path=local(p).relative_to(ROOT).as_posix(),sha256=sha(p))


class Clock:
    def __init__(self,seconds,output):
        self.started=time.monotonic();self.seconds=seconds;self.output=output;self.inputs={}
    def check(self):
        if time.monotonic()-self.started>self.seconds:raise TimeoutError('Analysis stage cap')
        if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-11-11T00:23:59+00:00'):raise TimeoutError('Campaign deadline')
        if sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())>100000000:raise RuntimeError('100MB output cap')
    def checked(self,entry):
        self.check();p=local(entry['path'])
        if sha(p)!=entry['sha256']:raise ValueError('Pinned artifact mismatch: '+str(p))
        self.inputs[str(p)]=entry['sha256'];return p


def audit_binding(report,entry):
    suffix=entry['path'].replace('\\','/')
    values=[digest for name,digest in report['inputs'].items() if name.replace('\\','/').endswith('/'+suffix)]
    if not values or any(x!=entry['sha256'] for x in values):raise ValueError('Independent report does not bind reused input')


def initialize(args,clock):
    inputs=read(clock.checked(dict(path=args.components,sha256=args.components_sha256)))
    cfg=read(clock.checked(inputs['config']));clock.checked(PROTOCOL);clock.checked(cfg['protocol'])
    old=read(clock.checked(cfg['original_manifest']))
    if (cfg['schema']!='c16-risk-dataset-v001' or cfg['scientific_execution_admitted'] is not True or
        cfg['dataset']!=inputs['dataset'] or old['dataset']!=cfg['dataset'] or cfg['all_evaluation_previously_exposed'] is not True or
        old['fit_cache']!=cfg['fit_cache'] or old['roles']['fit']!=cfg['train_ids']):raise ValueError('Original exposed roles/FIT binding')
    cal=cfg['calibration']
    if (cal['shift_grid_c']!=SHIFT_GRID or cal['shift_residual_quantiles']!=QUANTILES or
        cal['rmse_budget_ratios']!=RMSE_RATIOS or cal['over_mae_budget_ratios']!=OVER_RATIOS or
        cal['primary_rmse_ratio']!=1.05 or cal['primary_over_mae_ratio']!=1.25):raise ValueError('Frozen calibration settings changed')
    expected_roles=[r for r in old['roles'] if r!='fit']
    if [r['role'] for r in cfg['role_inputs']]!=expected_roles:raise ValueError('Complete original roles')
    source={};ids_used=set(cfg['train_ids']);cal_roles=[]
    for role in cfg['role_inputs']:
        if (role['ids']!=old['roles'][role['role']] or [e['id'] for e in role['records']]!=role['ids'] or
            len(set(role['ids']))!=len(role['ids']) or ids_used.intersection(role['ids'])):raise ValueError('Exact disjoint role IDs')
        ids_used.update(role['ids'])
        if role['usage']=='calibration':cal_roles.append(role['role'])
        elif role['usage']!='exploratory_evaluation':raise ValueError('Unexpected role usage')
        prior=read(clock.checked(role['prior_evaluation_audit']));lockpath=clock.checked(role['prediction_lock']);lock=read(lockpath)
        if prior['status']!='PASS' or prior['stage']!='evaluation' or lock['role']!=role['role'] or [e['id'] for e in lock['records']]!=role['ids']:
            raise ValueError('Prior source independent audit/lock identity')
        for entry,previous in zip(role['records'],lock['records']):
            if (entry['rows']!=previous['rows'] or entry['features']['sha256']!=previous['sha256'] or
                local(entry['features']['path'])!=(lockpath.parent/previous['path']).resolve()):raise ValueError('Source feature identity')
            for item in (entry['features'],entry['truth'],role['prediction_lock']):audit_binding(prior,item)
            source[(role['role'],entry['id'])]=entry
    if cal_roles!=[cfg['calibration_role']]:raise ValueError('Exactly one original CAL role')
    metricpath=clock.checked(METRIC)
    spec=importlib.util.spec_from_file_location('c16_v002_frozen_scientific_metrics',metricpath)
    metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
    return inputs,cfg,source,metric


def components(inputs,cfg,source,clock):
    metadata={};predictions={};reports={}
    expected=[(r['role'],p) for r in cfg['role_inputs'] for p in r['ids']]
    for item in inputs['components']:
        cid=item['id'];backbone=item['backbone']
        if cid in metadata or backbone not in ('hgb','mlp','context'):raise ValueError('Unique known component identities')
        if [(x['role'],x['profile']) for x in item['predictions']]!=expected:raise ValueError('Complete ordered prediction cohort')
        if backbone in ('mlp','hgb'):
            receiptpath=clock.checked(item['source_receipt']);receipt=read(receiptpath)
            if (receipt['dataset']!=cfg['dataset'] or receipt['config_sha256']!=inputs['config']['sha256'] or
                receipt['loss']!=item['loss'] or receipt['seed']!=item['seed'] or (receiptpath.parent/'failure.json').exists()):raise ValueError('Complete same-identity component receipt')
            sourcename='risk_mlp_v001.py' if backbone=='mlp' else 'risk_tree_v001.py'
            clock.checked(dict(path='followup-risk-20261110/code/'+sourcename,sha256=receipt['source_sha256']))
            if item['predictions']!=receipt['predictions']:raise ValueError('Exact producer prediction collection')
            lockpath=clock.checked(receipt['prediction_lock']) if backbone=='mlp' else receiptpath.parent/'prediction_lock.json'
            if backbone=='hgb':clock.checked(dict(path=str(lockpath),sha256=receipt['files']['prediction_lock.json']))
            lock=read(lockpath)
            if lock['predictions']!=item['predictions'] or lock['config_sha256']!=inputs['config']['sha256'] or lock['truth_accessed'] is not False:
                raise ValueError('Prediction lock/config/no-target identity')
            if backbone=='mlp':
                if receipt['status']!='MLP_PREDICTIONS_LOCKED' or receipt['completed_comparator'] is not True:raise ValueError('Completed MLP required')
                report=read(clock.checked(item['audit']))
                clock.checked(dict(path='followup-risk-20261110/code/risk_audit_v001.py',sha256=AUDIT_SHA))
                if (report['status']!='PASS' or report['mode']!='mlp' or report['auditor_sha256']!=AUDIT_SHA or
                    report['config_sha256']!=inputs['config']['sha256'] or report['results']['loss']!=item['loss'] or
                    report['results']['seed']!=item['seed'] or report['producer_sha256']!=receipt['source_sha256']):raise ValueError('Independent MLP forward audit identity')
                audit_binding(report,item['source_receipt'])
                for entry in item['predictions']:audit_binding(report,entry)
                reports[cid]=report
            elif receipt['status']!='HGB_FIT_PREDICTIONS_LOCKED':raise ValueError('Completed HGB required')
        values={role['role']:{} for role in cfg['role_inputs']}
        for entry in item['predictions']:
            key=entry['role'],entry['profile'];origin=source[key];ix=np.arange(300,origin['rows'],dtype=np.int64)
            with np.load(clock.checked(entry),allow_pickle=False) as data:
                if not np.array_equal(data[entry.get('index_key','index')],ix):raise ValueError('Full component endpoint alignment')
                value=data[entry.get('prediction_key','prediction')].astype(np.float64)
                if value.shape!=ix.shape or not np.isfinite(value).all():raise ValueError('Finite full scalar prediction')
            if backbone=='context':
                allowed={'prediction__direct9_fitcv_selected','prediction__positive_rc_insample_boost','prediction__linear33_insample_boost'}
                if entry.get('prediction_key') not in allowed or entry['path']!=origin['features']['path'] or entry['sha256']!=origin['features']['sha256']:
                    raise ValueError('Exact prior contextual prediction identity')
            values[key[0]][key[1]]=value
        metadata[cid]=item;predictions[cid]=values
    seeds=sorted({m['seed'] for m in metadata.values() if m['backbone']=='mlp'})
    if seeds not in ([SEEDS[0]],SEEDS):raise ValueError('Complete initial neural seed or fixed three-seed panel required')
    for seed in seeds:
        group=[c for c in metadata.values() if c['backbone']=='mlp' and c['seed']==seed]
        if len(group)!=3 or {c['loss'] for c in group}!=set(LOSSES):raise ValueError('Three matched losses per seed required')
        for field in ('initial_array_sha256','training_array_sha256','permutation_digests'):
            observed=[reports[c['id']]['results'][field] for c in group]
            if not all(x==observed[0] for x in observed):raise ValueError('Unpaired neural '+field)
    trees=[c for c in metadata.values() if c['backbone']=='hgb']
    if len(trees)!=2 or {c['loss'] for c in trees}!={'mse','q90'} or any(c['seed']!=SEEDS[0] for c in trees):raise ValueError('Both fixed tree controls required')
    context=[c for c in metadata.values() if c['backbone']=='context']
    keys={c['predictions'][0].get('prediction_key') for c in context}
    if len(context)!=3 or keys!={'prediction__direct9_fitcv_selected','prediction__positive_rc_insample_boost','prediction__linear33_insample_boost'} or any(c['seed'] is not None or c['loss']!='context' for c in context):raise ValueError('Exact three contextual controls required')
    if any(len({e.get('prediction_key') for e in c['predictions']})!=1 for c in context):raise ValueError('Context identity fixed across roles')
    return metadata,predictions,seeds


def truths(role,source,clock):
    result={}
    for pid in role['ids']:
        entry=source[(role['role'],pid)];ix=np.arange(300,entry['rows'],dtype=np.int64)
        with np.load(clock.checked(entry['truth']),allow_pickle=False) as data:
            y=data['target'].astype(np.float64)
            if not np.array_equal(data['index'],ix) or y.shape!=ix.shape or not np.isfinite(y).all():raise ValueError('Complete aligned target cohort')
        result[pid]=y
    return result


def shifts(truth,predictions):
    error=np.concatenate([truth[p]-predictions[p] for p in truth])
    weight=np.concatenate([np.full(len(truth[p]),1/(len(truth)*len(truth[p]))) for p in truth])
    order=np.argsort(error,kind='stable');values=error[order];cdf=np.cumsum(weight[order]);cdf[-1]=1.
    candidate=list(SHIFT_GRID)
    for quantile in QUANTILES:
        position=min(int(np.searchsorted(cdf,quantile,side='left')),len(values)-1)
        candidate.append(max(0.,float(values[position])))
    return list(dict.fromkeys(candidate))


def selection(settings,metrics,metadata,cal):
    policies=[]
    for cid,item in metadata.items():
        if item['backbone']=='context':continue
        baseline=[k for k,m in metadata.items() if m['backbone']==item['backbone'] and m['seed']==item['seed'] and m['loss']=='mse']
        if len(baseline)!=1:raise ValueError('Exactly one paired raw MSE budget reference')
        base_id=baseline[0]+'__raw';base=metrics[base_id][cal]['summary']
        candidates=[s['id'] for s in settings if s['base']==cid and s['family']=='shift']
        for rr in RMSE_RATIOS:
            for oo in OVER_RATIOS:
                feasible=[(i,sid) for i,sid in enumerate(candidates) if
                    metrics[sid][cal]['summary']['macro_rmse']<=rr*base['macro_rmse']+1e-12 and
                    metrics[sid][cal]['summary']['macro_over_mae']<=oo*base['macro_over_mae']+1e-12]
                chosen=None
                if feasible:
                    _,chosen=min(feasible,key=lambda pair:tuple(metrics[pair[1]][cal]['summary'][k] for k in
                        ('macro_under_rmse','macro_rmse','macro_over_mae'))+(pair[0],))
                policies.append(dict(backbone=item['backbone'],seed=item['seed'],loss=item['loss'],family='shift',
                    rmse_ratio=rr,over_mae_ratio=oo,selected_setting_id=chosen,base_setting_id=base_id,
                    calibration_role=cal,status='SELECTED' if chosen is not None else 'NO_FEASIBLE_CAL_SETTING',
                    candidate_setting_ids=candidates,feasible_setting_ids=[sid for _,sid in feasible]))
    return policies


def aggregate_seeds(policies,metrics,seeds,roles):
    groups=[]
    for rr in RMSE_RATIOS:
        for oo in OVER_RATIOS:
            for loss in LOSSES:
                chosen={p['seed']:p for p in policies if p['backbone']=='mlp' and p['loss']==loss and p['rmse_ratio']==rr and p['over_mae_ratio']==oo}
                for role in roles:
                    rows=[]
                    for seed in seeds:
                        policy=chosen[seed];sid=policy['selected_setting_id']
                        rows.append(dict(seed=seed,status=policy['status'],selected_setting_id=sid,
                            metrics=None if sid is None else metrics[sid][role]['summary'],
                            evaluation_budget_violations=None if sid is None else policy['evaluation_budget_violations'][role]))
                    complete=all(r['metrics'] is not None for r in rows)
                    mean=None
                    if complete:
                        keys=[key for key in rows[0]['metrics'] if key.startswith('macro_')]+['pooled_rmse']
                        mean={key:None if any(r['metrics'][key] is None for r in rows) else float(np.mean([r['metrics'][key] for r in rows])) for key in keys}
                    groups.append(dict(backbone='mlp',loss=loss,rmse_ratio=rr,over_mae_ratio=oo,role=role,
                        expected_seeds=seeds,per_seed=rows,status='ALL_INCLUDED_SEEDS_FEASIBLE' if complete else 'INCOMPLETE_FEASIBILITY',
                        mean_of_seed_metrics=mean,three_seed_panel=len(seeds)==3))
    return dict(seed_scope='THREE_PAIRED_SEEDS' if len(seeds)==3 else 'INITIAL_SEED_IMPLEMENTATION_CHECK',
        seeds=seeds,aggregation='arithmetic mean of per-seed metrics; no ensemble, no feasible-seed-only mean',groups=groups)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--components',required=True);parser.add_argument('--components-sha256',required=True)
    parser.add_argument('--output-dir',required=True);parser.add_argument('--max-seconds',type=float,default=600.)
    args=parser.parse_args()
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=600:raise ValueError('Finite cap in(0,600]')
    output=local(args.output_dir)
    if FOLLOWUP/'runs' not in output.parents:raise ValueError('New followup run output required')
    output.mkdir(parents=True,exist_ok=False);clock=Clock(args.max_seconds,output)
    dump(output/'configuration.json',dict(vars(args),source=str(Path(__file__).resolve()),source_sha256=sha(__file__),
        output_dir=str(output),pid=os.getpid(),argv=sys.argv,protocol=PROTOCOL,metric_dependency=METRIC,
        started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        inputs,cfg,source,metric=initialize(args,clock)
        dump(output/'component-inputs.json',inputs)
        metadata,comp,seeds=components(inputs,cfg,source,clock)
        train=[]
        for entry in cfg['fit_cache']:
            with np.load(clock.checked(entry),allow_pickle=False) as data:train.append(data['target'][300::5].astype(np.float64))
        target=np.concatenate(train)
        if not np.isfinite(target).all():raise ValueError('Finite FIT targets')
        threshold=float(np.quantile(target,.95));dt=cfg['metrics']['sample_interval_seconds'];cal=cfg['calibration_role']
        calrole=next(r for r in cfg['role_inputs'] if r['role']==cal)
        truth={cal:truths(calrole,source,clock)}
        settings=[];metrics={}
        for item in inputs['components']:
            settings.append(dict(id=item['id']+'__raw',family='context' if item['backbone']=='context' else 'unadjusted',
                base=item['id'],other=None,weight=None,shift=0.,backbone=item['backbone'],seed=item['seed'],loss=item['loss']))
        for cid,item in metadata.items():
            if item['backbone']=='context':continue
            for i,value in enumerate(shifts(truth[cal],comp[cid][cal])):
                settings.append(dict(id=cid+'__shift_'+str(i),family='shift',base=cid,other=None,weight=None,
                    shift=float(value),backbone=item['backbone'],seed=item['seed'],loss=item['loss']))
        def measure(setting,role):
            clock.check();pp={pid:comp[setting['base']][role][pid]+setting['shift'] for pid in truth[role]}
            return metric.role_metrics(truth[role],pp,threshold,dt)
        for setting in settings:metrics[setting['id']]={cal:measure(setting,cal)}
        selections=selection(settings,metrics,metadata,cal)
        dump(output/'settings.json',settings);dump(output/'policy_selections.json',selections)
        dump(output/'calibration_metrics.json',dict(dataset=cfg['dataset'],high_temperature_threshold_c=threshold,
            settings={sid:values[cal] for sid,values in metrics.items()}))
        dump(output/'calibration_lock.json',dict(schema='c16-risk-calibration-selection-lock-v002',status='CAL_ONLY_SELECTIONS_LOCKED',
            dataset=cfg['dataset'],protocol=PROTOCOL,components=dict(path=args.components,sha256=args.components_sha256),
            config=inputs['config'],source_sha256=sha(__file__),seeds=seeds,calibration_role=cal,
            files={name:sha(output/name) for name in ['settings.json','policy_selections.json','calibration_metrics.json']},
            evaluation_truth_accessed=False,input_hashes_before_evaluation_truth=dict(clock.inputs),
            locked_at_utc=datetime.now(timezone.utc).isoformat()))
        # Fitting/selection is now immutable; only reporting follows.
        for role in cfg['role_inputs']:
            name=role['role']
            if name==cal:continue
            truth[name]=truths(role,source,clock)
            for setting in settings:metrics[setting['id']][name]=measure(setting,name)
        policies=[]
        for selection_row in selections:
            row=dict(selection_row);sid=row['selected_setting_id'];violations=None
            if sid is not None:
                violations={}
                for role in truth:
                    baseline=metrics[row['base_setting_id']][role]['summary'];selected=metrics[sid][role]['summary']
                    violations[role]=dict(rmse=selected['macro_rmse']>row['rmse_ratio']*baseline['macro_rmse']+1e-12,
                        over_mae=selected['macro_over_mae']>row['over_mae_ratio']*baseline['macro_over_mae']+1e-12)
            row['evaluation_budget_violations']=violations;policies.append(row)
        dump(output/'policies.json',policies)
        dump(output/'metrics.json',dict(dataset=cfg['dataset'],high_temperature_threshold_c=threshold,settings=metrics))
        dump(output/'paired_seed_summary.json',aggregate_seeds(policies,metrics,seeds,list(truth)))
        dump(output/'input_pins.json',clock.inputs)
        clock.check()
        result=dict(status='EXPLORATORY_RISK_CALIBRATION_V002_COMPLETE',dataset=cfg['dataset'],settings=len(settings),policies=len(policies),
            no_feasible_policy_count=sum(r['selected_setting_id'] is None for r in policies),
            high_temperature_threshold_c=threshold,seeds=seeds,complete_three_seed_panel=len(seeds)==3,
            protocol=PROTOCOL,source_sha256=sha(__file__),config_sha256=inputs['config']['sha256'],
            metric_source_sha256=METRIC['sha256'],independent_confirmation=False,new_fit_performed=False,
            evaluation_used_for_selection=False,elapsed_seconds=time.monotonic()-clock.started,
            files={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
        dump(output/'summary.json',result);clock.check()
        print(json.dumps(dict(status=result['status'],settings=len(settings),policies=len(policies),elapsed_seconds=result['elapsed_seconds'])))
    except Exception as error:
        dump(output/'failure.json',dict(status='FAILED',error_type=type(error).__name__,message=str(error),elapsed_seconds=time.monotonic()-clock.started))
        raise


if __name__=='__main__':main()
