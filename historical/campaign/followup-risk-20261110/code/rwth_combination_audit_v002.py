"""Independent RWTH scalar-mixture, saved-panel and confirmation-result audit.

No scientific worker import, fitting, model prediction, new hypothesis or
secondary significance test. CONF rotor is decoded only after all23 saved
predictions, fixed-model audits and consumed root admission are validated.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import time
import zipfile
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PREDICTION_AUDIT_SHA="4ed549266d04c60b10edfc5afc5699df441ec5a89e36d7bda5c36d3fe2cb9f1a"
DECISION_ORACLE_SHA="59d2143474064b87575dfcfff5b66d9665568c467a85facd143972782f727e9b"
LEFT="direct9_fitcv_selected"
PAIRS=dict(c15="positive_rc_insample_boost",direct200="direct_boost",linear33="linear33_insample_boost",
           free25="unconstrained_rc_insample_boost",positive_prior="positive_rc")
RULES=["left","right","half","cal_macro_rmse"]
SEEDS=[2026111701,2026111702,2026111703]
MLP_METHODS=["mlp__"+str(s) for s in SEEDS]
METHODS=[p+"__"+r for p in PAIRS for r in RULES]+MLP_METHODS
PRIMARY="c15__cal_macro_rmse"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()


def module(filename,name,expected):
    p=Path(__file__).with_name(filename)
    if sha(p)!=expected:raise ValueError("Independent audit dependency changed: "+filename)
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


F=module("rwth_prediction_audit_v002.py","rwth_independent_prediction_binding",PREDICTION_AUDIT_SHA)
M=F.M;P=F.P;I=F.I;H=F.H


def stage_ref(a,entry,folder,summary):
    return F.stage_ref(a,entry,folder,summary["files"])


def stage(a,args):
    folder=M.local(args.input_dir);a.require(not (folder/"failure.json").exists(),"No combination stage failure")
    M.pin(a,dict(path=str(Path(__file__).with_name("rwth_combination_v002.py")),sha256=args.producer_sha256))
    summary=I.read(M.pin(a,dict(path=str(folder/"summary.json"),sha256=args.input_sha256)))
    a.require(summary["source_sha256"]==args.producer_sha256 and summary["dataset"]=="rwth48v","Same completed producer identity")
    statuses={"calibrate":["RWTH_CAL_COMBINATIONS_LOCKED"],"predict":["RWTH_CONF_23_PREDICTIONS_LOCKED"],
              "evaluate":["RWTH_CONFIRMATION_EVALUATION_COMPLETE","RWTH_CALIBRATION_DIAGNOSTICS_COMPLETE"]}
    a.require(summary["status"] in statuses[args.mode],"Completed exact combination mode")
    for name,digest in summary["files"].items():M.pin(a,dict(path=str(I.path_in(folder,name)),sha256=digest))
    config=I.read(folder/"configuration.json")
    a.require(config["mode"]==args.mode and config["source_sha256"]==args.producer_sha256,"Actual combination invocation")
    a.require(summary["protocol"]["sha256"]==P.PROTOCOL_SHA and summary["partition"]["sha256"]==P.PARTITION_SHA,"Frozen scientific protocol")
    M.pin(a,summary["protocol"]);M.pin(a,summary["partition"])
    a.same(summary["methods"],METHODS,"Fixed23-method output panel")
    a.require(summary["primary"]==PRIMARY,"Frozen primary identity")
    return folder,summary


def own_pass(a,entry,mode):
    report=I.read(M.pin(a,entry))
    a.require(report["status"]=="PASS" and not report["failures"] and report["stage"]==mode
              and report["auditor_sha256"]==sha(__file__),"Complete current independent combination "+mode+" PASS")
    return report


def model_lock(a,entry,args,calibration_report=None):
    path=M.pin(a,entry);lock=I.read(path)
    a.require(not (path.parent/"failure.json").exists(),"No calibration producer failure")
    a.require(lock["schema"]=="rwth-combination-model-lock-v002" and lock["status"]=="CAL_MODELS_LOCKED"
              and lock["source_sha256"]==args.producer_sha256 and lock["dataset"]=="rwth48v"
              and lock["confirmation_accessed"] is False,"Immutable preCONF CAL model")
    a.require(lock["primary"]==PRIMARY and lock["methods"]==METHODS,"Fixed model output panel")
    a.require(lock["protocol"]["sha256"]==P.PROTOCOL_SHA and lock["partition"]["sha256"]==P.PARTITION_SHA,"CAL model protocol")
    a.same(lock["calibration_ids"],P.ROLES["calibration"],"All fixed CAL IDs")
    required={"configuration.json","models.json","fit_threshold.json","component-inputs.json"}
    a.require(set(lock["files"])==required,"Complete four-file CAL model lock")
    a.require(lock["scalar_dependency"]["sha256"]=="d73b6eb70db5c6e57f79a5334c4000d373189d3e49aaaf7371daee9b9a08e25b","Frozen scalar scientific source identity")
    M.pin(a,lock["scalar_dependency"])
    if calibration_report is not None:
        M.binding(a,calibration_report,path,entry["sha256"],"Prior independent calibration PASS binds exact model lock")
        a.same(calibration_report["results"]["model_lock"],entry,"Audited calibration model identity")
    for name,digest in lock["files"].items():
        artifact=M.pin(a,dict(path=str(path.parent/name),sha256=digest))
        if calibration_report is not None:M.binding(a,calibration_report,artifact,digest,"Current CAL artifact independently audited")
    models=I.read(path.parent/"models.json")
    a.require(models["schema"]=="rwth-combination-models-v002" and models["dataset"]=="rwth48v"
              and models["primary"]==PRIMARY and models["left_key"]==LEFT,"Exact scalar model schema")
    for key,value in (("calibration_ids",P.ROLES["calibration"]),("pair_order",list(PAIRS)),("rules",RULES),("methods",METHODS)):
        a.same(models[key],value,"Scalar model fixed "+key)
    a.require(set(models["pairs"])==set(PAIRS),"Five exact expert pairs")
    return path,lock,models


def audited_components(a,entry,audit_ref):
    path=M.pin(a,entry);bundle=I.read(path)
    a.require(bundle["schema"]=="rwth-combination-input-v002" and bundle["dataset"]=="rwth48v","Exact component schema")
    role=bundle["role"];a.require(role in ("calibration","confirmation"),"Exact component role")
    audits=I.read(M.pin(a,audit_ref))
    a.require(audits["schema"]=="rwth-prediction-audits-v002" and audits["role"]==role and len(audits["mlp"])==3,
              "Complete role-specific independent source-audit bundle")
    held=H.pass_report(a,audits["feature_audit"],"held_features","rwth_held_prep_audit_v002.py",F.HELD_AUDIT_SHA)
    a.require(held["results"]["role"]==role,"Independent held audit role")
    cache=I.read(F.audited_ref(a,held,bundle["cache_manifest"],"Independent source audit binds exact role cache"))
    a.same(cache["roles"],P.ROLES,"Unchanged role IDs")
    a.same(bundle["protocol"],cache["protocol"],"Component protocol")
    a.same(bundle["partition"],cache["partition"],"Component partition")
    reports=[H.pass_report(a,e,"base_predictions","rwth_prediction_audit_v002.py",PREDICTION_AUDIT_SHA)
             for e in [audits["cpu"]]+audits["mlp"]]
    refs=[bundle["cpu_prediction_lock"]]+bundle["mlp_prediction_locks"]
    a.require(len(refs)==4,"Exactly CPU plus3 neural component locks")
    locks=[]
    for ordinal,(ref,report) in enumerate(zip(refs,reports)):
        a.require(report["results"]["role"]==role and report["results"]["kind"]==("cpu" if ordinal==0 else "mlp"),"Correct independent component forward audit")
        lockpath=F.audited_ref(a,report,ref,"Every component prediction lock independently replayed")
        a.same(report["results"]["prediction_lock"],ref,"Exact component lock in audit result")
        a.same(report["results"]["cache_manifest"],bundle["cache_manifest"],"All components share one observed feature cache")
        value=I.read(lockpath);a.require(not (lockpath.parent/"failure.json").exists(),"No component prediction failure")
        if ordinal:a.require(value["seed"]==SEEDS[ordinal-1]==report["results"]["seed"],"Three fixed ordered MLP seeds")
        locks.append(value)
    base_locks=dict(cpu=locks[0]["model_lock"],mlp=[v["model_lock"] for v in locks[1:]])
    a.same(held["results"]["base_model_locks"],[base_locks["cpu"]]+base_locks["mlp"],"Held source locked after same four models")
    if role=="confirmation":
        H.consumed_admission(a,cache["confirmation_admission"],cache,[base_locks["cpu"]]+base_locks["mlp"])
        for value in locks:a.same(value["confirmation_admission"],cache["confirmation_admission"],"Common consumed component admission")
    entries=[e for e in cache["records"] if e["role"]==role]
    a.require([e["id"] for e in entries]==P.ROLES[role],"All role profiles in source order")
    cpurows=locks[0]["records"];mlprows=[v["predictions"] for v in locks[1:]]
    for values in [cpurows]+mlprows:a.require([e["profile"] for e in values]==P.ROLES[role],"Complete component row coverage")
    rows={}
    for ordinal,source in enumerate(entries):
        pid=source["id"];n=source["rows"]
        saved=a.arrays(F.audited_ref(a,reports[0],cpurows[ordinal],"Exact audited CPU saved predictions"))
        a.require(set(saved)=={"profile_id","index","source_row","time"}|{"prediction__"+k for k in F.COLUMNS},"Exact eight CPU component columns")
        values={k:saved["prediction__"+k] for k in F.COLUMNS}
        for key in ("index","source_row"):
            a.require(saved[key].dtype==np.int64 and np.array_equal(saved[key],np.arange(n,dtype=np.int64)),"Every source row retained")
        for j,seed in enumerate(SEEDS):
            current=a.arrays(F.audited_ref(a,reports[j+1],mlprows[j][ordinal],"Exact independently replayed MLP predictions"))
            for key in ("index","source_row","time"):P.close_array(a,current[key],saved[key],"Exact shared row alignment",exact=True)
            values["mlp__"+str(seed)]=current["prediction"]
        rows[pid]=dict(id=pid,index=saved["index"],source_row=saved["source_row"],time=saved["time"],values=values,source=source)
    return bundle,cache,rows,base_locks,held,audits


def calibration_truth(a,rows,held_report):
    truth={}
    for pid,row in rows.items():
        values=a.arrays(F.audited_ref(a,held_report,row["source"]["truth"],"CAL truth independently checked after all model locks"))
        H.identity_arrays(a,values,pid,len(row["index"]),row["time"],"CAL truth",True)
        truth[pid]=values["target"]
    return truth


def scalar_objective(records,w):
    return float(np.mean([np.sqrt(np.mean((r["d"]+w*(r["h"]-r["d"])-r["y"])**2)) for r in records]))


def scalar_gradient(records,w):
    intervals=[]
    for r in records:
        difference=r["h"]-r["d"];error=r["d"]+w*difference-r["y"]
        norm=float(np.sqrt(np.mean(error**2)))
        if norm:
            slope=float(np.mean(difference*error))/norm;intervals.append((slope,slope))
        else:
            bound=float(np.sqrt(np.mean(difference**2)));intervals.append((-bound,bound))
    return np.mean(intervals,axis=0)


def scalar_oracle(records):
    left,right=0.,1.
    for _ in range(100):
        middle=(left+right)/2.;low,high=scalar_gradient(records,middle)
        if low<=0<=high:left=right=middle;break
        if high<0:left=middle
        else:right=middle
    return min((scalar_objective(records,w),w) for w in (0.,1.,left,right,(left+right)/2.))


def verify_scalar(a,value,records,tag):
    weight=value["weight"]
    a.require(isinstance(weight,(int,float)) and math.isfinite(weight) and 0<=weight<=1,"Convex scalar feasibility "+tag)
    objective=scalar_objective(records,weight);oracle,ow=scalar_oracle(records)
    a.same(value["objective"],objective,"Scalar macro objective "+tag,atol=1e-11,rtol=1e-11)
    a.require(objective-oracle<=1e-8*(1+abs(objective)),"Independent derivative-bisection optimum "+tag)
    endpoints=[dict(weight=w,objective=scalar_objective(records,w)) for w in (0.,1.)]
    a.same(value["endpoint_objectives"],endpoints,"Both explicit scalar endpoints "+tag,atol=1e-11,rtol=1e-11)
    gradient=scalar_gradient(records,weight);slope=float(np.clip(0.,gradient[0],gradient[1]))
    gap=max(0.,slope*weight-min(0.,slope))
    P.close_array(a,value["subgradient_interval"],gradient,"Independent scalar subgradient "+tag,atol=1e-11,rtol=1e-10)
    a.same(value["convex_gap"],gap,"Independent convex gap "+tag,atol=1e-11,rtol=1e-9)
    optimizer=value["optimizer"]
    a.require(optimizer["success"] is True and 0<=optimizer["x"]<=1 and optimizer["xatol"]==1e-12
              and optimizer["maxiter"]==500 and isinstance(optimizer["nfev"],int) and optimizer["nfev"]>0,
              "Fixed bounded scalar solver contract "+tag)
    a.same(optimizer["objective"],scalar_objective(records,optimizer["x"]),"Optimizer candidate objective",atol=1e-11,rtol=1e-11)
    selected=min((scalar_objective(records,w),w) for w in (0.,1.,optimizer["x"]))
    a.require(weight==selected[1],"Exact endpoint/optimizer candidate choice including lower-weight ties "+tag)
    a.results.setdefault("scalar_checks",[]).append(dict(tag=tag,weight=weight,objective=objective,oracle_objective=oracle,oracle_weight=ow))


def verify_models(a,models,rows,truth):
    for pair,right in PAIRS.items():
        group=models["pairs"][pair]
        a.require(group["right_key"]==right,"Fixed right expert "+pair)
        records=[dict(id=pid,d=row["values"][LEFT],h=row["values"][right],y=truth[pid]) for pid,row in rows.items()]
        a.same(group["final"]["fit_ids"],P.ROLES["calibration"],"Scalar full CAL8")
        verify_scalar(a,group["final"]["cal_macro_rmse"],records,pair+"/final")
        a.require([r["held_out"] for r in group["lopo"]]==P.ROLES["calibration"],"All8 scalar LOPO diagnostics")
        for fold in group["lopo"]:
            held=fold["held_out"];subset=[r for r in records if r["id"]!=held]
            a.same(fold["model"]["fit_ids"],[r["id"] for r in subset],"Scalar LOPO excludes held CAL label")
            verify_scalar(a,fold["model"]["cal_macro_rmse"],subset,pair+"/lopo/"+str(held))


def panel_values(row,models,lopo=False):
    result={};pid=row["id"]
    for pair,right in PAIRS.items():
        group=models["pairs"][pair]
        choice=next(f["model"] for f in group["lopo"] if f["held_out"]==pid) if lopo else group["final"]
        d,h=row["values"][LEFT],row["values"][right];w=choice["cal_macro_rmse"]["weight"]
        result.update({pair+"__left":d,pair+"__right":h,pair+"__half":(d+h)/2.,pair+"__cal_macro_rmse":d+w*(h-d)})
    result.update({k:row["values"][k] for k in MLP_METHODS})
    return result


def locked_panel(a,entry,producer_sha,rows,models,cache,model_ref,bundle_ref,cache_ref,prior_audit=None):
    path=M.pin(a,entry);lock=I.read(path);role="calibration" if list(rows)==P.ROLES["calibration"] else "confirmation"
    a.require(not (path.parent/"failure.json").exists(),"No panel producer failure")
    a.require(lock["schema"]=="rwth-combination-prediction-lock-v002" and lock["status"]=="ALL_23_PREDICTIONS_LOCKED"
              and lock["source_sha256"]==producer_sha and lock["role"]==role and lock["confirmation_truth_accessed"] is False,
              "All23 fixed predictions locked before CONF targets")
    a.same(lock["methods"],METHODS,"No omitted or substituted method")
    a.require(lock["primary"]==PRIMARY,"No changed primary candidate")
    a.same(lock["model_lock"],model_ref,"Panel exact CAL model lock")
    a.same(lock["component_inputs"],bundle_ref,"Panel exact source component bundle")
    a.same(lock["cache_manifest"],cache_ref,"Panel exact role feature manifest")
    a.same(lock["protocol"],cache["protocol"],"Panel frozen protocol")
    a.same(lock["partition"],cache["partition"],"Panel frozen partition")
    a.require(lock["calibration_truth_accessed"] is (role=="calibration"),"Panel target usage declaration")
    if prior_audit is not None:
        M.binding(a,prior_audit,path,entry["sha256"],"Every current saved panel independently checked before evaluation")
        a.same(prior_audit["results"]["prediction_lock"],entry,"Prior panel audit exact lock")
    if role=="confirmation":
        a.same(lock["confirmation_admission"],cache["confirmation_admission"],"Same consumed panel admission")
        gate=H.consumed_admission(a,cache["confirmation_admission"],cache,cache["base_model_locks"])
        key=M.local(model_ref["path"]).relative_to(ROOT).as_posix()
        a.require(gate["bindings"].get(key)==model_ref["sha256"],"Root admission binds preCONF scalar model")
    else:a.require(lock["confirmation_admission"] is None,"CAL no confirmation consumption")
    roles=["calibration_lopo","calibration"] if role=="calibration" else ["confirmation"]
    expected=[(r,pid) for r in roles for pid in rows]
    a.require([(e["role"],e["profile"]) for e in lock["predictions"]]==expected,"Exact23 methods and complete fixed profile-role coverage")
    output={}
    for record in lock["predictions"]:
        pid=record["profile"];row=rows[pid];n=len(row["index"])
        a.require(record["source_role"]==role and record["rows"]==n,"Panel original role and full rows")
        a.same(record["source_features"],row["source"]["features"],"Panel original common feature bytes")
        a.same(record["truth_ref"],row["source"]["truth"],"Panel target separation identity")
        pp=M.pin(a,record)
        if prior_audit is not None:M.binding(a,prior_audit,pp,record["sha256"],"All saved23 predictions bound by prior audit")
        values=a.arrays(pp)
        a.require(set(values)=={"profile_id","index","source_row","time"}|{"prediction__"+m for m in METHODS},"Exact saved23 panel columns")
        a.require(values["profile_id"].dtype==np.int64 and values["profile_id"].shape==() and int(values["profile_id"])==pid,"Panel profile identity")
        for key in ("index","source_row","time"):
            a.require(values[key].dtype==row[key].dtype,"Panel source precision")
            P.close_array(a,values[key],row[key],"Panel exact complete source alignment",exact=True)
        oracle=panel_values(row,models,record["role"]=="calibration_lopo")
        for name in METHODS:
            a.require(values["prediction__"+name].dtype==np.float64,"Panel FP64 outputs")
            P.close_array(a,values["prediction__"+name],oracle[name],"Exact fixed mixture saved prediction "+name,exact=True)
        output[(record["role"],pid)]=values
    return lock,output


def conf_truth(a,folder,summary,cache,rows,panel_ref):
    # Caller has finished validating every model, gate and all23 predictions.
    marker=I.read(folder/"all_predictions_verified.json")
    a.require(marker["status"]=="ALL23_PREDICTIONS_VERIFIED_BEFORE_TARGETS" and marker["methods"]==METHODS
              and marker["prediction_records"]==10,"Producer all23 pre-target checkpoint")
    a.same(marker["prediction_lock"],panel_ref,"Pre-target checkpoint exact prediction lock")
    a.same(marker["confirmation_admission"],cache["confirmation_admission"],"Pre-target checkpoint same consumed admission")
    receipt=I.read(folder/"truth_receipt.json")
    a.require(receipt["schema"]=="rwth-confirmation-truth-receipt-v002" and receipt["rotor_access_after_all23_prediction_lock"] is True
              and receipt["stator_numeric_values_parsed"] is False,"Post-prediction rotor-only truth receipt")
    a.same(receipt["archive"],cache["archive"],"Unchanged fresh archive")
    a.require([e["profile"] for e in receipt["records"]]==P.ROLES["confirmation"],"Every predefined CONF truth record")
    truth={};archive_path=M.pin(a,cache["archive"])
    with zipfile.ZipFile(archive_path) as archive:
        for entry in receipt["records"]:
            pid=entry["profile"];row=rows[pid];source=row["source"];member=f"final/{pid:03d}.csv"
            a.require(entry["member"]==source["member"]==member,"Exact admitted CONF member")
            raw=archive.read(member)
            a.require(hashlib.sha256(raw).hexdigest()==entry["csv_sha256"]==source["csv_sha256"],"Unchanged CONF member bytes")
            reader=csv.reader(io.StringIO(raw.decode("utf-8-sig")));a.require(next(reader,None)==P.HEADERS,"Independent raw rotor column mapping")
            times=[];targets=[]
            for j,fields in enumerate(reader):
                if j%2000==0:a.budget()
                a.require(len(fields)==8 and float(fields[7])==pid,"Original raw profile identity")
                times.append(float(fields[0]));targets.append(float(fields[5]))
            # No stator or measured operating channels are converted here.
            y=np.asarray(targets,dtype=np.float64);t=np.asarray(times,dtype=np.float64)
            a.require(y.shape==row["index"].shape and np.isfinite(y).all(),"Complete finite raw CONF rotor")
            P.close_array(a,t,row["time"],"Raw truth timestamp equals all locked predictions",exact=True)
            values=a.arrays(stage_ref(a,entry,folder,summary))
            H.identity_arrays(a,values,pid,len(y),t,"raw CONF truth",True)
            a.require(values["target"].dtype==np.float64,"Original FP64 CONF rotor")
            P.close_array(a,values["target"],y,"Independent raw CONF rotor lineage",exact=True)
            truth[pid]=y
    return truth


def run(a,args):
    folder,summary=stage(a,args)
    if args.mode=="calibrate":
        model_ref=summary["model_lock"];mpath,lock,models=model_lock(a,model_ref,args)
        bundle_ref=lock["calibration_inputs"]
    else:
        a.require(args.calibration_audit and args.calibration_audit_sha256,"Existing independent CAL model audit required")
        previous=own_pass(a,dict(path=args.calibration_audit,sha256=args.calibration_audit_sha256),"calibrate")
        model_ref=summary["model_lock"];mpath,lock,models=model_lock(a,model_ref,args,previous)
        panel=I.read(M.pin(a,summary["prediction_lock"]));bundle_ref=panel["component_inputs"]
    bundle,cache,rows,bases,held,audits=audited_components(a,bundle_ref,dict(path=args.audit_inputs,sha256=args.audit_inputs_sha256))
    role=bundle["role"]
    a.same(lock["base_model_locks"],bases,"Same frozen base models for all combinations")
    a.same(lock["protocol"],cache["protocol"],"Same scalar/cache protocol")
    a.same(lock["partition"],cache["partition"],"Same scalar/cache partition")
    if args.mode=="calibrate":
        a.require(role=="calibration","Scalar fitting only on CAL")
        a.same(I.read(folder/"component-inputs.json"),bundle,"Exact calibration component manifest copy")
        truth=calibration_truth(a,rows,held)
        verify_models(a,models,rows,truth)
        a.require(args.prep_audit and args.prep_audit_sha256,"Original independent FIT preparation required for threshold audit")
        _,fitrecords=M.fit_cache(a,lock["fit_manifest"],dict(path=args.prep_audit,sha256=args.prep_audit_sha256))
        threshold=float(np.quantile(np.concatenate([fitrecords[p]["target"][::5] for p in P.ROLES["fit"]]),.95))
        a.same(I.read(folder/"fit_threshold.json"),dict(high_temperature_threshold_c=threshold,fit_manifest=lock["fit_manifest"]),"Independent FIT-only high threshold")
        a.same(summary["high_temperature_threshold_c"],threshold,"CAL summary threshold")
        locked_panel(a,summary["prediction_lock"],args.producer_sha256,rows,models,cache,model_ref,bundle_ref,bundle["cache_manifest"])
    elif args.mode=="predict":
        a.require(role=="confirmation","Prediction-only followup is CONF")
        locked_panel(a,summary["prediction_lock"],args.producer_sha256,rows,models,cache,model_ref,bundle_ref,bundle["cache_manifest"])
        a.require(summary["confirmation_truth_accessed"] is False,"CONF predictions precede rotor access")
    else:
        a.require(args.prediction_audit and args.prediction_audit_sha256,"Whole23 prediction audit before result evaluation")
        prior_panel=own_pass(a,dict(path=args.prediction_audit,sha256=args.prediction_audit_sha256),"predict" if role=="confirmation" else "calibrate")
        pl,values=locked_panel(a,summary["prediction_lock"],args.producer_sha256,rows,models,cache,model_ref,bundle_ref,bundle["cache_manifest"],prior_panel)
        truth=conf_truth(a,folder,summary,cache,rows,summary["prediction_lock"]) if role=="confirmation" else calibration_truth(a,rows,held)
        threshold=I.read(mpath.parent/"fit_threshold.json")["high_temperature_threshold_c"]
        roles=["confirmation"] if role=="confirmation" else ["calibration_lopo","calibration"]
        computed={method:{r:I.group_metrics([(pid,truth[pid],values[(r,pid)]["prediction__"+method],rows[pid]["index"]) for pid in rows],None,threshold)
                          for r in roles} for method in METHODS}
        expected=dict(dataset="rwth48v",high_temperature_threshold_c=threshold,methods=computed,
                      time_unit_independently_verified=False,duration_seconds_reported=False)
        a.same(I.read(folder/"metrics.json"),expected,"Independent complete23 metrics, zeros and high-temperature masks",atol=1e-9,rtol=1e-10)
        decision=module("rwth_decision_oracle_v002.py","rwth_independent_fixed_decision",DECISION_ORACLE_SHA)
        M.pin(a,dict(path=str(Path(__file__).with_name("rwth_decision_oracle_v002.py")),sha256=DECISION_ORACLE_SHA))
        means={r:decision.metric_mean(computed,r) for r in roles}
        comparisons=[]
        for r in roles:
            comparisons.extend(decision.contrast(computed[m][r],computed["c15__left"][r],m,"c15__left",r) for m in METHODS)
            comparisons.append(decision.contrast(means[r],computed["c15__left"][r],"mlp_three_seed_metric_mean","c15__left",r))
        a.same(I.read(folder/"mlp_seed_metric_mean.json"),means,"Independent mean of all3 seed metrics",atol=1e-9,rtol=1e-10)
        a.same(I.read(folder/"comparison.json"),comparisons,"All descriptive fixed method comparisons",atol=1e-9,rtol=1e-10)
        if role=="confirmation":
            verdict=decision.formal_decision(computed,means[role],models)
            a.same(I.read(folder/"decision.json"),verdict,"Independent exact sign and all strict contribution conditions",atol=1e-10,rtol=1e-10)
            a.results["decision"]=verdict
        a.results.update(metrics=expected,seed_metric_means=means)
    inputpins=I.read(folder/"input_pins.json")
    for path,digest in inputpins.items():M.pin(a,dict(path=path,sha256=digest))
    a.results.update(stage=args.mode,dataset="rwth48v",role=role,primary=PRIMARY,methods=METHODS,
                     model_lock=model_ref,prediction_lock=summary["prediction_lock"],cache_manifest=bundle["cache_manifest"],
                     confirmation_admission=cache.get("confirmation_admission"),new_tests_beyond_frozen_primary=False,
                     confirmation_rotor_decoded=args.mode=="evaluate" and role=="confirmation",stator_numerically_parsed=False)
    return folder


def synthetic_test():
    data=[dict(id=1,d=np.asarray([0.,2.]),h=np.asarray([2.,4.]),y=np.asarray([1.,3.]))]
    objective,weight=scalar_oracle(data)
    assert objective==0 and weight==.5
    same=[dict(id=1,d=np.asarray([0.,2.]),h=np.asarray([0.,2.]),y=np.asarray([1.,3.]))]
    assert scalar_oracle(same)==(1.,0.)
    print(json.dumps(dict(status="PASS_SYNTHETIC_ONLY",checks=["interior_scalar_optimum","constant_experts_lower_weight_tie"])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["calibrate","predict","evaluate"])
    for key in ("input-dir","input-sha256","producer-sha256","audit-inputs","audit-inputs-sha256","output-dir",
                "calibration-audit","calibration-audit-sha256","prediction-audit","prediction-audit-sha256","prep-audit","prep-audit-sha256"):
        p.add_argument("--"+key)
    p.add_argument("--max-seconds",type=float,default=1200.);p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:synthetic_test();return 0
    if not all((args.mode,args.input_dir,args.input_sha256,args.producer_sha256,args.audit_inputs,args.audit_inputs_sha256,args.output_dir)):
        p.error("Mode, producer summary, source hash, independent input audit bundle and new output required")
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=1200:p.error("Bounded audit cap")
    out=M.local(args.output_dir);out.mkdir(parents=True,exist_ok=False);a=I.Audit(args.max_seconds);status="HOLD"
    try:
        for path,digest in ((Path(__file__),sha(__file__)),(Path(__file__).with_name("rwth_prediction_audit_v002.py"),PREDICTION_AUDIT_SHA)):
            M.pin(a,dict(path=str(path),sha256=digest))
        folder=run(a,args);a.finish_pins();a.require(not (folder/"failure.json").exists(),"No final combination stage failure")
        status="PASS" if not a.failures else "HOLD"
    except Exception as exc:a.failures.append(dict(check="Audit completed without exception",detail=repr(exc)))
    report=dict(status=status,stage=args.mode,auditor_sha256=sha(__file__),producer_sha256=args.producer_sha256,
        checks=a.checks,failures=a.failures,inputs=a.inputs,results=a.results,elapsed_seconds=time.monotonic()-a.start,
        scope="Independent scalar oracle, exact saved23 panel, raw truth/metrics and prospectively fixed decision; no training or secondary hypothesis test")
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(status=status,checks=a.checks,failures=len(a.failures))))
    return 0 if status=="PASS" else 2


if __name__=="__main__":raise SystemExit(main())
