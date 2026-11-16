"""Independent C17 saved-input/convex-fit/prediction/metric audit.

No scientific worker import, base-regressor fit, optimizer training or network.
The pinned independent C16 audit supplies only generic integrity and risk metrics.
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
    os.environ[name] = '1'
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP = ROOT/'followup-risk-20261110'
SHARED_SHA = '2cfb48953741515e8d9815073075bcc3e46c717c5110837d280acdb93a2bd035'
PROTOCOL_SHA = '2a1aa6f2c815840a3419b6ae6bb05b0a68a8fa97f8780966cce4bd8614b3f227'
METHODS = ['direct9','C15','fixed_half','fixed_cal_mse','fixed_cal_macro_rmse','load_threshold','observable_simplex_gate']
DIRECT = 'prediction__direct9_fitcv_selected'
HYBRID = 'prediction__positive_rc_insample_boost'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def location(value):
    p = (ROOT/value).resolve()
    if ROOT != p and ROOT not in p.parents:
        raise ValueError('Path outside campaign')
    return p


def shared():
    path = FOLLOWUP/'code/risk_audit_v001.py'
    if sha(path) != SHARED_SHA:
        raise ValueError('Independent audit dependency changed')
    spec = importlib.util.spec_from_file_location('c17_independent_metric_dependency',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close(a, actual, expected, label, atol=1e-9, rtol=1e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    a.require(actual.shape == expected.shape, label+' shape')
    a.check(np.isfinite(actual).all() and np.allclose(actual,expected,atol=atol,rtol=rtol),label,
            {'max_abs':float(np.max(np.abs(actual-expected))) if actual.size else 0.})


def exact(a, actual, expected, label):
    actual, expected = np.asarray(actual),np.asarray(expected)
    a.require(actual.shape == expected.shape,label+' shape')
    a.check(np.array_equal(actual,expected),label)


def load_configuration(a, args):
    cfg = read(a.pin(dict(path=args.config,sha256=args.config_sha256)))
    a.require(cfg.get('schema') == 'c17-gate-dataset-v001' and cfg.get('scientific_execution_admitted') is True,'C17 admitted config')
    a.same(cfg['protocol'],dict(path='followup-risk-20261110/protocols/c17-gate-v001-frozen.md',sha256=PROTOCOL_SHA),'Frozen protocol identity')
    a.pin(cfg['protocol'])
    expected = dict(direct_key=DIRECT,hybrid_key=HYBRID,ridge_lambda=.01,scalar_xatol=1e-12,
        scalar_maxiter=500,simplex_feasibility_tolerance=1e-10,negative_face_tolerance=1e-12,
        fw_gap_relative_tolerance=1e-8,methods=METHODS,max_output_bytes=100000000)
    for key,value in expected.items():
        a.same(cfg[key],value,'Config/'+key,atol=0,rtol=0)
    risk = read(a.pin(cfg['risk_config']))
    a.require(risk['dataset'] == cfg['dataset'] and risk['all_evaluation_previously_exposed'] is True,'Same exposed dataset')
    a.pin(risk['protocol'])
    original = read(a.pin(risk['original_manifest']))
    a.require(risk['schema']=='c16-risk-dataset-v001' and original['dataset']==cfg['dataset'],'Original dataset/schema identity')
    a.same(risk['train_ids'],original['roles']['fit'],'Original FIT identity')
    a.same(risk['fit_cache'],original['fit_cache'],'Original FIT cache lineage')
    a.same([r['role'] for r in risk['role_inputs']],[r for r in original['roles'] if r!='fit'],'Original complete ordered roles')
    used=set(risk['train_ids']);calibration=[]
    a.require(len(used)==len(risk['train_ids']),'Unique FIT profiles')
    for role in risk['role_inputs']:
        a.same(role['ids'],original['roles'][role['role']],'Original role IDs')
        a.same([e['id'] for e in role['records']],role['ids'],'Ordered role records')
        a.require(len(set(role['ids']))==len(role['ids']) and not used.intersection(role['ids']),'Disjoint unique role profiles')
        used.update(role['ids'])
        if role['usage']=='calibration':calibration.append(role['role'])
        else:a.require(role['usage']=='exploratory_evaluation','Only exposed evaluation usage')
    a.same(calibration,[risk['calibration_role']],'Exactly one declared CAL role')
    return cfg,risk


def features(record, scale):
    instant,heat = record['instant'],record['heat']
    current,speed = instant[:,0],instant[:,3]
    if not (np.isfinite(current).all() and np.isfinite(speed).all() and np.min(current)>=0 and np.min(speed)>=0):
        raise ValueError('Current/speed must be finite nonnegative audited quantities')
    return np.column_stack([np.log1p(current/scale['I_rms']),np.log1p(speed/scale['omega_rms']),
                            (heat[:,0]-heat[:,9])/scale['I_rms']**2])


def transform_reference(a, risk):
    records,ys,counts = [],[],{}
    for item in risk['fit_cache']:
        saved = a.arrays(a.pin(item))
        n = item['rows'];ix=np.arange(300,n,5,dtype=np.int64)
        a.require(saved['instant'].shape == (n,8) and saved['heat'].shape == (n,15) and saved['target'].shape == (n,), 'FIT geometry')
        records.append(dict(instant=saved['instant'][ix].astype(np.float64),heat=saved['heat'][ix].astype(np.float64)))
        ys.append(saved['target'][ix].astype(np.float64));counts[str(item['id'])]=len(ix)
    joined = {key:np.concatenate([x[key] for x in records]) for key in ('instant','heat')}
    scale=dict(I_rms=max(float(np.sqrt(np.mean(joined['instant'][:,0]**2))),1e-8),
               omega_rms=max(float(np.sqrt(np.mean(joined['instant'][:,3]**2))),1e-8))
    value=features(joined,scale);target=np.concatenate(ys)
    scale.update(mean=value.mean(0).tolist(),std=np.maximum(value.std(0),1e-8).tolist(),
        y_scale=max(float(np.quantile(target,.75)-np.quantile(target,.25)),1.),
        high_temperature_threshold_c=float(np.quantile(target,.95)),fit_ids=risk['train_ids'],fit_endpoint_counts=counts)
    return scale


def sources(a,risk,scale):
    result={}
    for role in risk['role_inputs']:
        lockpath=a.pin(role['prediction_lock']);lock=read(lockpath)
        prior=read(a.pin(role['prior_evaluation_audit']))
        a.require(prior.get('status')=='PASS' and prior.get('stage')=='evaluation','Source independent evaluation audit PASS')
        a.same(lock['role'],role['role'],'Prior prediction lock role')
        a.same([r['id'] for r in lock['records']],role['ids'],'Prior prediction lock ordered IDs')
        a.same([e['id'] for e in role['records']],role['ids'],'All source IDs')
        old={x['id']:x for x in lock['records']}
        for entry in role['records']:
            n=entry['rows'];ix=np.arange(300,n,dtype=np.int64)
            a.same(old[entry['id']]['rows'],n,'Prior prediction lock full rows')
            # Prior reports may carry native/WSL absolute prefixes. Bind every
            # matching campaign-relative suffix, preserving duplicate aliases.
            # This is metadata only: held truth is not decoded by fit-only audit.
            for pin in (entry['features'],entry['truth'],role['prediction_lock']):
                suffix=pin['path'].replace('\\','/')
                hits=[digest for name,digest in prior['inputs'].items() if name.replace('\\','/').endswith('/'+suffix)]
                a.require(bool(hits) and all(digest==pin['sha256'] for digest in hits),'Prior full audit binds exact reused input/'+suffix)
            p=a.pin(entry['features'])
            a.require(p==(lockpath.parent/old[entry['id']]['path']).resolve() and entry['features']['sha256']==old[entry['id']]['sha256'],'Original prediction lock identity')
            saved=a.arrays(p)
            exact(a,saved['index'],ix,'Full causal endpoints')
            a.require(saved['instant'].shape==(n,8) and saved['heat'].shape==(n,15),'Role source feature shape')
            a.require(saved['source_row'].shape==(n,) and np.issubdtype(saved['source_row'].dtype,np.integer),'Source row dtype/shape')
            raw=features(dict(instant=saved['instant'][ix].astype(np.float64),heat=saved['heat'][ix].astype(np.float64)),scale)
            z=(raw-np.asarray(scale['mean']))/np.asarray(scale['std'])
            ramps=np.clip(.5+.25*z,0.,1.)
            phi=np.column_stack([np.zeros(len(ix)),np.ones(len(ix)),ramps[:,0],1-ramps[:,0],ramps[:,1],1-ramps[:,1],ramps[:,2],1-ramps[:,2]])
            d,h=saved[DIRECT],saved[HYBRID]
            a.require(d.dtype==np.float64 and h.dtype==np.float64 and d.shape==ix.shape and h.shape==ix.shape,'Frozen two expert predictions')
            result[(role['role'],entry['id'])]=dict(id=entry['id'],role=role['role'],index=ix,
                source_row=saved['source_row'][ix],v=raw,z=z,phi=phi,d=d,h=h,truth_ref=entry['truth'],source_features=entry['features'])
    return result


def target(a, record):
    value=a.arrays(a.pin(record['truth_ref']))
    exact(a,value['index'],record['index'],'Truth endpoint alignment')
    a.require(value['target'].shape==record['index'].shape and value['target'].dtype==np.float64,'Truth shape/dtype')
    return value['target']


def statistic(record,scale):
    delta=record['h']-record['d'];residual=record['y']-record['d'];n=len(delta)
    design=delta[:,None]*record['phi']/scale['y_scale'];response=residual/scale['y_scale']
    return dict(gram=design.T@design/n,linear=design.T@response/n,constant=float(response@response/n),
                mse_cross=float(delta@residual/n),mse_delta2=float(delta@delta/n),count=n)


def scalar_objective(records,w):
    return float(np.mean([np.sqrt(np.mean((r['y']-r['d']-w*(r['h']-r['d']))**2)) for r in records]))


def subgradients(records,w):
    low,high=[],[]
    for r in records:
        delta=r['h']-r['d'];res=r['y']-r['d']-w*delta
        rmse=float(np.sqrt(np.mean(res*res)))
        if rmse==0.:
            bound=float(np.sqrt(np.mean(delta*delta)));lo,hi=-bound,bound
        else:
            lo=hi=-float(np.mean(delta*res))/rmse
        low.append(lo);high.append(hi)
    return float(np.mean(low)),float(np.mean(high))


def scalar_oracle(records):
    # Convex derivative bisection is independent of the producer bounded solver.
    left,right=0.,1.
    for _ in range(100):
        middle=(left+right)/2;lo,hi=subgradients(records,middle)
        if lo<=0<=hi:
            left=right=middle;break
        if hi<0:left=middle
        else:right=middle
    candidates=[0.,1.,left,right,(left+right)/2]
    return min((scalar_objective(records,w),w) for w in candidates)


def qp_reference(stats,weight):
    theta_ref=np.r_[1-weight,weight,np.zeros(6)]
    q=2*(np.mean([s['gram'] for s in stats],axis=0)+.01*np.eye(8))
    r=2*(np.mean([s['linear'] for s in stats],axis=0)+.01*theta_ref)
    constant=float(np.mean([s['constant'] for s in stats])+.01*(theta_ref@theta_ref))
    return q,r,constant,theta_ref


def independent_faces(q,r,constant):
    np.linalg.cholesky(q)
    faces=[];qualified=[]
    for mask in range(1,256):
        active=np.array([j for j in range(8) if (mask>>j)&1],dtype=np.int64)
        # Direct bordered KKT system, unlike a two-SPD-solve equality formula.
        kkt=np.zeros((len(active)+1,len(active)+1))
        kkt[:-1,:-1]=q[np.ix_(active,active)];kkt[-1,:-1]=1;kkt[:-1,-1]=1
        solution=np.linalg.solve(kkt,np.r_[r[active],1.])[:-1]
        feasible=bool(np.min(solution)>=-1e-12)
        value=dict(mask=mask,feasible=feasible,minimum_unclipped_coordinate=float(solution.min()))
        if feasible:
            theta=np.zeros(8);theta[active]=np.maximum(solution,0);theta/=theta.sum()
            objective=float(.5*theta@q@theta-r@theta+constant)
            value['objective']=objective;qualified.append((objective,mask,theta))
        faces.append(value)
    return faces,min(qualified,key=lambda x:(x[0],x[1]))


def expected_predictions(record,model):
    delta=record['h']-record['d'];theta=np.asarray(model['gate']['theta'])
    gate=np.clip(record['phi']@theta,0.,1.);rule=model['load_threshold']['rule']
    # Final schema resolves rule to an integer or documented name below.
    weights=[np.zeros(len(delta)),np.ones(len(delta)),(record['v'][:,2]>=0).astype(float),(record['v'][:,2]<0).astype(float)]
    rule_index=rule if isinstance(rule,int) else ['direct9','C15','v3_ge_zero','v3_lt_zero'].index(rule)
    return dict(direct9=record['d'],C15=record['h'],fixed_half=(record['d']+record['h'])/2,
        fixed_cal_mse=record['d']+model['fixed_cal_mse']['weight']*delta,
        fixed_cal_macro_rmse=record['d']+model['fixed_cal_macro_rmse']['weight']*delta,
        load_threshold=record['d']+weights[rule_index]*delta,
        observable_simplex_gate=record['d']+gate*delta),gate


def verify_model(a, model, records, statistics, tag):
    a.same(model['fit_ids'],[r['id'] for r in records],tag+'/gate CAL fit IDs')
    stats=[statistics[r['id']] for r in records]
    numerator=float(np.mean([s['mse_cross'] for s in stats]));denominator=float(np.mean([s['mse_delta2'] for s in stats]))
    weight=float(np.clip(numerator/denominator,0,1)) if denominator else 0.
    a.same(model['fixed_cal_mse']['weight'],weight,tag+'/MSE closed form')
    a.same(model['fixed_cal_mse']['numerator'],numerator,tag+'/MSE numerator')
    a.same(model['fixed_cal_mse']['denominator'],denominator,tag+'/MSE denominator')
    scalar=model['fixed_cal_macro_rmse'];w=scalar['weight']
    a.require(isinstance(w,(int,float)) and math.isfinite(w) and 0<=w<=1,tag+'/scalar feasible')
    value=scalar_objective(records,w);lower,upper=subgradients(records,w)
    oracle,oracle_w=scalar_oracle(records)
    a.same(scalar['objective'],value,tag+'/scalar objective')
    a.same(scalar['endpoint_objectives'],[dict(weight=w,objective=scalar_objective(records,w)) for w in (0.,1.)],tag+'/scalar endpoints')
    close(a,scalar['subgradient_interval'],[lower,upper],tag+'/scalar subgradient')
    subgradient=0. if lower<=0<=upper else lower if lower>0 else upper
    convex_gap=float(subgradient*w-min(0.,subgradient))
    a.same(scalar['convex_gap'],max(0.,convex_gap),tag+'/scalar convex certificate')
    a.check(value-oracle<=1e-8*(1+abs(value)),tag+'/scalar independent bisection objective optimality',
        dict(chosen_objective=value,oracle_objective=oracle,oracle_weight=oracle_w))
    a.check(convex_gap>=-1e-10,tag+'/scalar nonnegative convex bound',dict(gap=convex_gap))
    optimizer=scalar['optimizer']
    a.require(optimizer['success'] is True and 0<=optimizer['x']<=1 and optimizer['xatol']==1e-12 and optimizer['maxiter']==500,
              tag+'/fixed bounded solver contract')
    a.same(optimizer['objective'],scalar_objective(records,optimizer['x']),tag+'/optimizer objective replay')
    selected=min([(scalar_objective(records,ww),ww) for ww in (0.,1.,optimizer['x'])],key=lambda item:(item[0],item[1]))
    a.check(abs(value-selected[0])<=1e-9*(1+abs(value)),tag+'/endpoint plus bounded candidate selection')
    threshold_scores=[]
    for rule in range(4):
        scores=[]
        for rec in records:
            ww=[0.,1.,(rec['v'][:,2]>=0).astype(float),(rec['v'][:,2]<0).astype(float)][rule]
            pred=rec['d']+ww*(rec['h']-rec['d'])
            scores.append(np.sqrt(np.mean((rec['y']-pred)**2)))
        threshold_scores.append(float(np.mean(scores)))
    rule=min(range(4),key=lambda j:(threshold_scores[j],j))
    a.same(model['load_threshold']['rule'],rule,tag+'/threshold chosen rule')
    close(a,model['load_threshold']['cal_macro_rmse'],threshold_scores,tag+'/all four threshold candidates')
    q,r,constant,reference=qp_reference(stats,weight)
    gate=model['gate'];theta=np.asarray(gate['theta'],dtype=np.float64)
    a.require(theta.shape==(8,) and np.isfinite(theta).all(),tag+'/finite eight coefficient')
    close(a,gate['theta_ref'],reference,tag+'/reference constant shrinkage')
    close(a,gate['Q'],q,tag+'/QP Gram')
    close(a,gate['r'],r,tag+'/QP linear')
    a.same(gate['constant'],constant,tag+'/QP constant')
    a.check(theta.min()>=-1e-10 and abs(theta.sum()-1)<=1e-10,tag+'/simplex feasibility')
    objective=float(.5*theta@q@theta-r@theta+constant)
    gradient=q@theta-r;gap=float(gradient@theta-gradient.min())
    tolerance=1e-8*(1+abs(objective))
    a.same(gate['objective'],objective,tag+'/QP selected objective')
    a.same(gate['ridge_lambda'],.01,tag+'/fixed lambda',atol=0,rtol=0)
    a.check(gap>=-1e-10 and gap<=tolerance,tag+'/independent FW global suboptimality bound',dict(gap=gap,tolerance=tolerance))
    certificate=dict(spd_cholesky_pass=True,equality_residual=abs(float(theta.sum())-1),minimum_coefficient=float(theta.min()),
        fw_gap=max(0.,gap),raw_fw_gap=gap,gap_tolerance=tolerance,feasibility_tolerance=1e-10,qualified=True)
    a.same(gate['certificate'],certificate,tag+'/all certificate fields')
    faces,best=independent_faces(q,r,constant)
    a.require(len(gate['faces'])==255,tag+'/all255 support receipts')
    for saved,expected in zip(gate['faces'],faces):
        a.same(saved['mask'],expected['mask'],tag+'/face mask')
        a.same(saved['feasible'],expected['feasible'],tag+'/face feasibility')
        a.same(saved['minimum_unclipped_coordinate'],expected['minimum_unclipped_coordinate'],tag+'/face minimum coordinate')
        if expected['feasible']:
            a.same(saved['objective'],expected['objective'],tag+'/feasible face objective')
    mask=gate['selected_support_mask']
    a.require(isinstance(mask,int) and 1<=mask<=255,tag+'/selected support index')
    a.check(faces[mask-1]['feasible'],tag+'/selected support is feasible')
    a.check(all(theta[j]==0 for j in range(8) if not ((mask>>j)&1)),tag+'/selected support contains coefficients')
    a.check(abs(objective-best[0])<=tolerance,tag+'/all-face independent optimum',dict(objective=objective,oracle=best[0]))
    # Both global certificate and all-face enumeration are independent of the
    # producer optimizer. Roundoff-tied masks need not identify distinct models.
    return dict(fit_ids=model['fit_ids'],qp_gap=gap,qp_tolerance=tolerance,
                selected_objective=objective,enumerated_best_objective=best[0],
                scalar_objective=value,scalar_oracle_objective=oracle,scalar_convex_gap=convex_gap)


def pin_summary(a, directory, expected):
    directory=location(directory)
    summary_path=a.pin(dict(path=str(directory/'summary.json'),sha256=expected))
    summary=read(summary_path)
    # The producer's exact map/status keys are fixed with its source handoff.
    a.require(not (directory/'failure.json').exists(),'No producer failure artifact')
    for name,digest in summary['files'].items():
        p=(directory/name).resolve()
        a.require(directory in p.parents,'Producer file map stays inside stage')
        a.pin(dict(path=str(p),sha256=digest))
    return directory,summary


def verify_statistics(a, saved, cal_records, scale):
    stats={r['id']:statistic(r,scale) for r in cal_records}
    expected=dict(profile_ids=np.asarray([r['id'] for r in cal_records],dtype=np.int64),
        counts=np.asarray([len(r['index']) for r in cal_records],dtype=np.int64),
        gram=np.asarray([stats[r['id']]['gram'] for r in cal_records]),
        linear=np.asarray([stats[r['id']]['linear'] for r in cal_records]),
        constant=np.asarray([stats[r['id']]['constant'] for r in cal_records]),
        mse_cross=np.asarray([stats[r['id']]['mse_cross'] for r in cal_records]),
        mse_delta2=np.asarray([stats[r['id']]['mse_delta2'] for r in cal_records]))
    a.require(set(saved)==set(expected),'Statistics exact fields')
    for key,value in expected.items():
        a.check(saved[key].dtype==value.dtype,'Statistics dtype/'+key)
        if key in ('profile_ids','counts'):exact(a,saved[key],value,'Statistics/'+key)
        else:close(a,saved[key],value,'Statistics/'+key)
    return stats


def verify_predictions(a, lock, risk, source, models):
    calibration=risk['calibration_role']
    expected_keys=[(r['role'],p) for r in risk['role_inputs'] for p in r['ids']]
    expected_keys += [('calibration_lopo',p) for p in models['calibration_ids']]
    items=lock['predictions'];keys=[(e['role'],e['profile']) for e in items]
    a.require(len(keys)==len(set(keys)) and set(keys)==set(expected_keys),'Complete unique all-role plus CAL-LOPO predictions')
    folds={r['held_out']:r['model'] for r in models['lopo']}
    arrays={};diagnostics={}
    for entry in items:
        role,pid=entry['role'],entry['profile'];lopo=role=='calibration_lopo'
        source_role=calibration if lopo else role
        rec=source[(source_role,pid)];model=folds[pid] if lopo else models['final']
        a.same(entry['source_role'],source_role,'Saved source role')
        a.same(entry['model_kind'],'lopo' if lopo else 'final','Prediction model role')
        a.same(entry['source_features'],rec['source_features'],'Prediction original feature identity')
        a.same(entry['truth_ref'],rec['truth_ref'],'Prediction separate truth identity')
        if lopo:a.same(entry['held_out'],pid,'LOPO held profile')
        saved=a.arrays(a.pin(entry))
        a.require(set(saved)=={'index','source_row','gate_weight'}|{'prediction__'+m for m in METHODS},'Exact seven-method prediction schema')
        a.check(saved['index'].dtype==np.int64 and saved['source_row'].dtype==np.int64,'Explicit int64 prediction identifiers')
        exact(a,saved['index'],rec['index'],'All endpoints retained')
        exact(a,saved['source_row'],rec['source_row'],'Exact source rows retained')
        values,weight=expected_predictions(rec,model)
        close(a,saved['gate_weight'],weight,'Gate weight all rows',atol=2e-11,rtol=1e-11)
        a.check(weight.min()>=-1e-10 and weight.max()<=1+1e-10,'All gate weights in convex expert hull')
        for method in METHODS:
            a.check(saved['prediction__'+method].dtype==np.float64,'Prediction FP64/'+method)
            close(a,saved['prediction__'+method],values[method],'Full saved prediction replay/'+method)
        difference=values['observable_simplex_gate']-values['fixed_cal_mse']
        diagnostics[(role,pid)]=dict(role=role,profile=pid,weight_min=float(weight.min()),weight_mean=float(weight.mean()),
            weight_max=float(weight.max()),weight_std=float(weight.std()),
            versus_fixed_mse_rms_difference=float(np.sqrt(np.mean(difference**2))),
            versus_fixed_mse_max_absolute_difference=float(np.max(np.abs(difference))),label_free=True)
        arrays[(role,pid)]=saved
    return arrays,diagnostics


def verify_metrics(a, shared_audit, saved, risk, scale, source, arrays):
    role_ids={r['role']:r['ids'] for r in risk['role_inputs']}
    role_ids['calibration_lopo']=next(r['ids'] for r in risk['role_inputs'] if r['role']==risk['calibration_role'])
    truth={key:rec.get('y') if 'y' in rec else target(a,rec) for key,rec in source.items()}
    methods={}
    for method in METHODS:
        methods[method]={}
        for role,ids in role_ids.items():
            source_role=risk['calibration_role'] if role=='calibration_lopo' else role
            profiles=[(pid,truth[(source_role,pid)],arrays[(role,pid)]['prediction__'+method],arrays[(role,pid)]['index']) for pid in ids]
            methods[method][role]=shared_audit.group_metrics(profiles,risk['metrics']['sample_interval_seconds'],scale['high_temperature_threshold_c'])
    expected=dict(dataset=risk['dataset'],high_temperature_threshold_c=scale['high_temperature_threshold_c'],methods=methods)
    a.same(saved,expected,'All-role independent metrics',atol=1e-8,rtol=1e-10)
    return expected


def comparison_reference(metrics,roles):
    result=[]
    for role in roles:
        gate=metrics['methods']['observable_simplex_gate'][role]
        for comparator in ('direct9','fixed_cal_macro_rmse'):
            baseline=metrics['methods'][comparator][role]
            g,b=gate['summary'],baseline['summary']
            delta=[x['rmse']-y['rmse'] for x,y in zip(gate['per_profile'],baseline['per_profile'])]
            ratio=lambda key:g[key]/b[key] if b[key]!=0 else None
            result.append(dict(role=role,comparator=comparator,delta_macro_rmse=g['macro_rmse']-b['macro_rmse'],
                rmse_ratio=ratio('macro_rmse'),under_rmse_ratio=ratio('macro_under_rmse'),over_mae_ratio=ratio('macro_over_mae'),
                under_rmse_diagnostic_flag=g['macro_under_rmse']>1.05*b['macro_under_rmse']+1e-12,
                over_mae_diagnostic_flag=g['macro_over_mae']>1.25*b['macro_over_mae']+1e-12,
                per_profile_rmse_differences=delta,profile_ids=[x['id'] for x in gate['per_profile']],
                wins=sum(x<0 for x in delta),ties=sum(x==0 for x in delta),losses=sum(x>0 for x in delta),
                zero_baseline_ratio_policy='null ratio; exact inequality flag without denominator floor'))
    return result


def execute_audit(a,args,shared_audit):
    cfg,risk=load_configuration(a,args)
    a.pin(dict(path='followup-risk-20261110/code/gate_worker_v001.py',sha256=args.producer_sha256))
    a.pin(dict(path='followup-risk-20261110/code/risk_audit_v001.py',sha256=SHARED_SHA))
    directory,summary=pin_summary(a,args.fit_dir,args.fit_sha256)
    a.require(summary['status']=='GATE_MODELS_PREDICTIONS_LOCKED','Completed fit-predict stage')
    a.require(summary['source_sha256']==args.producer_sha256 and summary['config_sha256']==args.config_sha256,
              'Fit producer/config source identity')
    a.same(summary['methods'],METHODS,'Fit exact method panel')
    conf=read(directory/'configuration.json')
    a.require(conf['mode']=='fit-predict' and conf['source_sha256']==args.producer_sha256 and conf['config_sha256']==args.config_sha256,'Fit configuration identity')
    modelpath=a.pin(summary['model_lock']);predpath=a.pin(summary['prediction_lock'])
    a.require(modelpath==directory/'model_lock.json' and predpath==directory/'prediction_lock.json','Model/prediction lock location')
    mlock,plock=read(modelpath),read(predpath)
    for value,name,status in ((mlock,'model','MODELS_LOCKED'),(plock,'prediction','ALL_PREDICTIONS_LOCKED')):
        a.require(value['schema']=='c17-gate-'+name+'-lock-v001' and value['status']==status,'Qualified '+name+' lock')
        a.require(value['source_sha256']==args.producer_sha256 and value['config_sha256']==args.config_sha256,'Lock pinned identity')
        a.require(value['evaluation_truth_accessed'] is False and value['calibration_truth_accessed'] is True,'CAL-only fitting/target boundary')
        a.same(value['methods'],METHODS,'Lock exact method coverage')
    a.require(plock['independent_confirmation'] is False and plock['all_evaluation_previously_exposed'] is True,'Exploratory exposure disclosure')
    a.require(a.pin(plock['model_lock'])==modelpath,'Prediction binds exact model lock')
    required={'configuration.json','transform.json','calibration_statistics.npz','models.json'}
    a.require(set(mlock['files'])==required,'Exact four model-locked artifacts')
    for name,digest in mlock['files'].items():
        a.require(summary['files'][name]==digest,'Model file also bound by stage')
        a.pin(dict(path=str(directory/name),sha256=digest))
    all_input_pins=read(directory/'input_pins.json')
    before=mlock['input_hashes_before_evaluation_features']
    for name,digest in all_input_pins.items():
        a.pin(dict(path=name,sha256=digest))
    for name,digest in before.items():
        a.require(all_input_pins.get(name)==digest,'Before-evaluation input pin preserved')
    forbidden={str(location(r['truth']['path'])) for role in risk['role_inputs'] if role['role']!=risk['calibration_role'] for r in role['records']}
    a.check(not forbidden.intersection(str(location(name)) for name in all_input_pins),'No held evaluation truth read in fit receipt')
    scale=transform_reference(a,risk)
    a.same(read(directory/'transform.json'),scale,'Independent FIT-only feature/target scalers')
    source=sources(a,risk,scale)
    calrole=next(r for r in risk['role_inputs'] if r['role']==risk['calibration_role'])
    calrecords=[source[(calrole['role'],p)] for p in calrole['ids']]
    for rec in calrecords:rec['y']=target(a,rec)
    saved_stats=a.arrays(directory/'calibration_statistics.npz')
    statistics=verify_statistics(a,saved_stats,calrecords,scale)
    models=read(directory/'models.json')
    a.require(models['schema']=='c17-gate-models-v001' and models['dataset']==cfg['dataset'],'Models schema/dataset')
    a.same(models['methods'],METHODS,'Model seven-method identity')
    a.same(models['calibration_ids'],calrole['ids'],'Model exact CAL role IDs')
    a.same(models['calibration_role'],calrole['role'],'Model CAL role identity')
    a.same([x['held_out'] for x in models['lopo']],calrole['ids'],'Complete ordered CAL-LOPO fits')
    qualifications=[dict(kind='final',**verify_model(a,models['final'],calrecords,statistics,'Final'))]
    for fold in models['lopo']:
        rows=[r for r in calrecords if r['id']!=fold['held_out']]
        qualifications.append(dict(kind='calibration_lopo',held_out=fold['held_out'],
            **verify_model(a,fold['model'],rows,statistics,'LOPO/'+str(fold['held_out']))))
    arrays,diagnostics=verify_predictions(a,plock,risk,source,models)
    diagnostic_path=a.pin(plock['weight_diagnostics'])
    a.require(diagnostic_path==directory/'weight_diagnostics.json','Label-free diagnostics location')
    a.same(read(diagnostic_path),[diagnostics[(e['role'],e['profile'])] for e in plock['predictions']],
           'All label-free collapse diagnostics')
    a.same(summary['models'],len(calrecords)+1,'All CAL-LOPO and final model count')
    a.same(summary['predictions'],len(arrays),'Every role prediction count')
    a.results=dict(dataset=cfg['dataset'],model_qualifications=qualifications,
        all_role_prediction_count=len(arrays),methods=METHODS,independent_confirmation=False)
    if args.evaluation_dir:
        evaluation,evaluated=pin_summary(a,args.evaluation_dir,args.evaluation_sha256)
        a.require(evaluated['status']=='EXPLORATORY_GATE_EVALUATION_COMPLETE' and
            evaluated['source_sha256']==args.producer_sha256 and evaluated['config_sha256']==args.config_sha256,'Evaluation identity/status')
        a.require(a.pin(evaluated['prediction_lock'])==predpath,'Evaluation binds audited prediction lock')
        a.require(evaluated['independent_confirmation'] is False and evaluated['all_evaluation_previously_exposed'] is True and
                  evaluated['model_selection_performed'] is False,'Evaluation scope/no new selection')
        evalconf=read(evaluation/'configuration.json')
        a.require(evalconf['mode']=='evaluate' and evalconf['source_sha256']==args.producer_sha256 and
                  evalconf['config_sha256']==args.config_sha256,'Evaluation config/source')
        a.pin(dict(path='followup-risk-20261110/code/risk_screen_analysis_v001.py',sha256=evaluated['metrics_source_sha256']))
        metrics=verify_metrics(a,shared_audit,read(evaluation/'metrics.json'),risk,scale,source,arrays)
        roles=['calibration_lopo']+[role['role'] for role in risk['role_inputs']]
        a.same(evaluated['roles'],roles,'Complete role comparison order')
        comparisons=comparison_reference(metrics,roles)
        a.same(read(evaluation/'comparison.json'),comparisons,'All comparisons and diagnostic flags',atol=1e-8,rtol=1e-10)
        a.results.update(metrics=metrics,comparisons=comparisons,evaluation_audited=True)
    else:
        a.results['evaluation_audited']=False
    a.replayed=dict(fit_transform='all FIT300::5',sufficient_statistics='all CAL rows',
        convex_gate='all255 independent bordered KKT faces plus selected FW certificate, every final/LOPO model',
        scalar='independent convex derivative bisection plus objective/subgradient replay',
        predictions='all seven methods/all role endpoints',metrics='all roles if evaluation supplied',
        base_models='not retrained; exact previously audited prediction lineage only',
        shared_independent_dependency='risk_audit_v001 generic integrity and metric functions only')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--fit-dir',required=True);parser.add_argument('--fit-sha256',required=True)
    parser.add_argument('--evaluation-dir');parser.add_argument('--evaluation-sha256')
    parser.add_argument('--producer-sha256',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--max-seconds',type=float,default=600.)
    args=parser.parse_args()
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=600:
        raise ValueError('Audit cap must be finite in (0,600]')
    if bool(args.evaluation_dir)!=bool(args.evaluation_sha256):
        raise ValueError('Both optional evaluation arguments are required together')
    out=location(args.output_dir)
    if FOLLOWUP not in out.parents:raise ValueError('New audit output must be inside followup')
    out.mkdir(parents=True,exist_ok=False)
    dependency=shared();a=dependency.Audit(args.max_seconds)
    error=None
    try:
        execute_audit(a,args,dependency)
        a.finish_pins()
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        a.failures.append(dict(check='Audit completed all required checks',detail=error))
    report=dict(schema='c17-independent-gate-audit-v001',status='PASS' if not a.failures else 'FAIL',
        source_sha256=sha(__file__),producer_sha256=args.producer_sha256,
        shared_independent_audit_sha256=SHARED_SHA,config_sha256=args.config_sha256,
        checks=a.checks,failures=a.failures,inputs=a.inputs,replayed=a.replayed,results=a.results,
        elapsed_seconds=time.monotonic()-a.start,error=error,
        no_scientific_worker_import=True,no_base_model_refit=True)
    raw=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)
    if len(raw.encode('utf-8'))>100000000:raise RuntimeError('Audit report cap')
    with (out/'report.json').open('x',encoding='utf-8') as handle:handle.write(raw)
    print(json.dumps(dict(status=report['status'],checks=a.checks,failures=len(a.failures),elapsed_seconds=report['elapsed_seconds'])))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
