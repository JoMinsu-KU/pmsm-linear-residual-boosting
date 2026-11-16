"""Independent raw-tree/NumPy forward audit of locked RWTH base predictions.

Only target-free role feature caches are decoded. All raw lineage and model
training certificates must already have separate complete independent PASS.
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
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CPU_AUDIT_SHA="0fba461b3bc8a5a9a034732061abff9eba1e13635dc325fe72f1bee3992d0794"
HELD_AUDIT_SHA="ddb4ebfd645650d18a2482c6fe0fe033c66086a4e0a5fe2a3a1ae70b12e8f8db"
COLUMNS=["direct9_fitcv_selected","direct_boost","positive_rc","unconstrained25_prior","linear33_prior",
         "positive_rc_insample_boost","unconstrained_rc_insample_boost","linear33_insample_boost"]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()


def module(filename,name,expected):
    p=Path(__file__).with_name(filename)
    if sha(p)!=expected:raise ValueError("Independent audit dependency changed")
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


C=module("rwth_cpu_audit_v002.py","rwth_independent_cpu_forward",CPU_AUDIT_SHA)
H=module("rwth_held_prep_audit_v002.py","rwth_independent_held_binding",HELD_AUDIT_SHA)
M=C.M;P=C.P;I=C.I


def audited_ref(a,report,entry,label):
    path=M.pin(a,entry);M.binding(a,report,path,entry["sha256"],label);return path


def stage_ref(a,entry,folder,files):
    path=M.local(entry["path"])
    a.require(path.is_relative_to(folder) and files.get(path.relative_to(folder).as_posix())==entry["sha256"],"Prediction completion binding")
    return M.pin(a,entry)


def load(a,args):
    folder=M.local(args.input_dir);a.require(not (folder/"failure.json").exists(),"No prediction failure marker")
    if args.kind=="mlp":
        M.pin(a,dict(path=str(Path(__file__).with_name("rwth_mlp_v002.py")),sha256=args.producer_sha256))
        summary,stage=M.completed_stage(a,folder,args.input_sha256,args.producer_sha256)
        a.require(summary["status"]=="RWTH_MLP_PREDICTIONS_LOCKED","Complete MLP role predictions")
        files=stage["files"]
    else:
        M.pin(a,dict(path=str(Path(__file__).with_name("rwth_cpu_predict_v002.py")),sha256=args.producer_sha256))
        summary=I.read(M.pin(a,dict(path=str(folder/"summary.json"),sha256=args.input_sha256)))
        a.require(summary["schema"]=="rwth-cpu-prediction-stage-v002" and summary["status"]=="RWTH_CPU_PREDICTIONS_LOCKED"
                  and summary["source_sha256"]==args.producer_sha256 and summary["core_sha256"]==C.CORE_SHA,"Complete CPU prediction stage")
        files=summary["files"]
        for name,digest in files.items():M.pin(a,dict(path=str(I.path_in(folder,name)),sha256=digest))
    role=summary["role"];a.require(role in ("calibration","confirmation"),"Exact held role")
    lock=I.read(stage_ref(a,summary["prediction_lock"],folder,files))
    expected_schema="rwth-mlp-prediction-lock-v002" if args.kind=="mlp" else "rwth-cpu-prediction-lock-v002"
    expected_status="ALL_ROLE_PREDICTIONS_LOCKED" if args.kind=="mlp" else "CPU_PREDICTIONS_LOCKED"
    a.require(lock["schema"]==expected_schema and lock["status"]==expected_status and lock["role"]==role
              and lock["source_sha256"]==args.producer_sha256 and lock["dataset"]=="rwth48v","Whole-role prediction lock")
    target_flag="role_targets_accessed" if args.kind=="mlp" else "truth_accessed"
    a.require(lock[target_flag] is False and summary[target_flag] is False,"No held targets used for prediction")
    a.same(lock["cache_manifest"],summary["cache_manifest"],"Prediction source feature manifest")
    a.same(lock["model_lock"],summary["model_lock"],"Prediction source model")
    feature_audit=H.pass_report(a,dict(path=args.feature_audit,sha256=args.feature_audit_sha256),"held_features",
                                 "rwth_held_prep_audit_v002.py",HELD_AUDIT_SHA)
    a.require(feature_audit["results"]["role"]==role,"Same independently audited feature role")
    manifest=I.read(audited_ref(a,feature_audit,lock["cache_manifest"],"Feature audit binds current role manifest"))
    a.same(feature_audit["results"]["cache_manifest"],lock["cache_manifest"],"Exact audited feature manifest identity")
    a.same(manifest["roles"],P.ROLES,"Frozen role partition")
    a.require(manifest["protocol"]["sha256"]==P.PROTOCOL_SHA and manifest["partition"]["sha256"]==P.PARTITION_SHA,"Frozen protocol identity")
    a.same(lock["protocol"],manifest["protocol"],"Prediction protocol")
    a.same(lock["partition"],manifest["partition"],"Prediction partition")
    auditfile="rwth_mlp_audit_v002.py" if args.kind=="mlp" else "rwth_cpu_audit_v002.py"
    auditsha=C.MLP_AUDITOR_SHA if args.kind=="mlp" else CPU_AUDIT_SHA
    model_audit=H.pass_report(a,dict(path=args.model_audit,sha256=args.model_audit_sha256),
                            "mlp_train" if args.kind=="mlp" else "final",auditfile,auditsha)
    modelpath=audited_ref(a,model_audit,lock["model_lock"],"Full model audit binds exact current model")
    a.same(model_audit["results"]["model_lock"],lock["model_lock"],"Audited model identity")
    a.require(lock["model_lock"] in feature_audit["results"]["base_model_locks"],"Feature source opened after same current base model")
    model_lock=I.read(modelpath)
    a.require(not (modelpath.parent/"failure.json").exists(),"No model failure marker")
    for name,digest in model_lock["files"].items():
        audited_ref(a,model_audit,dict(path=str(I.path_in(modelpath.parent,name)),sha256=digest),"Current audited model artifact")
    if role=="confirmation":
        a.same(lock["confirmation_admission"],manifest["confirmation_admission"],"Same consumed CONF admission")
        H.consumed_admission(a,lock["confirmation_admission"],manifest,feature_audit["results"]["base_model_locks"])
    else:a.require(lock["confirmation_admission"] is None,"CAL has no CONF consumption")
    a.same(summary["confirmation_admission"],lock["confirmation_admission"],"Summary admission")
    entries=[e for e in manifest["records"] if e["role"]==role]
    a.require([e["id"] for e in entries]==P.ROLES[role],"All role source profiles")
    saved=lock["predictions"] if args.kind=="mlp" else lock["records"]
    a.require([e["profile"] for e in saved]==P.ROLES[role],"All role predictions in exact order")
    return folder,files,summary,lock,manifest,entries,saved,feature_audit,model_audit,modelpath,model_lock


def mlp_forward(a,values,record,model,weights,tag):
    tr=model["transform"]
    quantized=((record["X33"]-np.asarray(tr["input_mean"]))/np.asarray(tr["input_std"])).astype(np.float32)
    a.require(values["features"].dtype==np.float32,"Saved quantized MLP feature precision")
    P.close_array(a,values["features"],quantized,"Exact MLP standardized observed inputs",exact=True)
    prediction=values["prediction"]
    a.require(prediction.dtype==np.float64 and prediction.shape==(len(quantized),),"Full FP64 MLP predictions")
    maximum_error=maximum_bound=maximum_ratio=0.
    for start in range(0,len(quantized),4096):
        a.budget();raw,bound=I.forward_oracle(a,weights,quantized[start:start+4096])
        oracle=tr["y_mid"]+tr["y_scale"]*raw
        bound=abs(tr["y_scale"])*bound+1e-9
        difference=np.abs(prediction[start:start+4096]-oracle)
        a.require(np.isfinite(oracle).all() and np.isfinite(bound).all() and np.all(difference<=bound),
                  "Independent MLP NumPy/FP32 roundoff bound "+tag)
        maximum_error=max(maximum_error,float(difference.max()));maximum_bound=max(maximum_bound,float(bound.max()))
        maximum_ratio=max(maximum_ratio,float(np.max(difference/bound)))
    a.replayed[tag]=dict(rows=len(quantized),all_rows=True,max_absolute_error_c=maximum_error,
                        max_forward_bound_c=maximum_bound,max_error_bound_ratio=maximum_ratio)


def run(a,args):
    folder,files,summary,lock,manifest,entries,saved,fa,ma,modelpath,model_lock=load(a,args)
    role=summary["role"]
    if args.kind=="cpu":
        a.same(lock["columns"],COLUMNS,"All eight fixed CPU experts")
        priors=I.read(audited_ref(a,ma,model_lock["priors"],"Audited current prior coefficients"))
        boosters={}
        for name,entry in model_lock["boosters"].items():
            metadata=I.read(audited_ref(a,ma,entry["metadata"],"Audited current booster metadata"))
            treepath=audited_ref(a,ma,entry["trees"],"Audited current raw trees")
            boosters[name]=(metadata,C.numeric_nodes(a,treepath),entry["prior_family"])
    else:
        a.require(lock["seed"]==summary["seed"]==model_lock["seed"] and lock["loss"]=="mse","Same fixed MLP seed/loss")
        model=I.read(audited_ref(a,ma,model_lock["model"],"Audited model transform"))
        weights=a.arrays(modelpath.parent/"weights_final.npz");I.weight_geometry(a,weights,"current final")
        runtime=I.read(folder/"runtime.json")
        a.require(runtime["tf32_matmul_allowed"] is False and runtime["tf32_cudnn_allowed"] is False
                  and runtime["deterministic_algorithms"] is True,"Prediction FP32 numerical contract")
    for entry,predentry in zip(entries,saved):
        pid=entry["id"];n=entry["rows"]
        a.require(predentry["role"]==role and predentry["rows"]==n,"Prediction role/row count")
        a.same(predentry["source_features"],entry["features"],"Prediction common feature source")
        record=a.arrays(audited_ref(a,fa,entry["features"],"Independently audited exact feature arrays"))
        H.identity_arrays(a,record,pid,n,record["time"],"source features")
        values=a.arrays(stage_ref(a,dict(path=predentry["path"],sha256=predentry["sha256"]),folder,files))
        expected={"index","source_row","time","features","prediction"} if args.kind=="mlp" else {"profile_id","index","source_row","time"}|{"prediction__"+k for k in COLUMNS}
        a.require(set(values)==expected,"Exact target-free saved prediction fields")
        for key in ("index","source_row","time"):
            a.require(values[key].dtype==record[key].dtype,"Exact prediction alignment precision")
            P.close_array(a,values[key],record[key],"Full original prediction alignment "+key,exact=True)
        if args.kind=="mlp":
            a.require(predentry["prediction_key"]=="prediction" and predentry["index_key"]=="index", "MLP prediction schema")
            a.require(predentry["arrays_sha256"]==I.array_digest(values),"Prediction full-array digest")
            mlp_forward(a,values,record,model,weights,f"{role}/{pid}/seed{lock['seed']}")
        else:
            a.require(values["profile_id"].dtype==np.int64 and values["profile_id"].shape==() and int(values["profile_id"])==pid,"CPU profile identity")
            oracle={}
            for name,(metadata,trees,family) in boosters.items():
                oracle[name]=C.held_boost_prediction(a,metadata,trees,record,f"{role}/{pid}/{name}",priors[family] if family else None)
            for alias,base in model_lock["aliases"].items():oracle[alias]=oracle[base]
            for name,family in (("positive_rc","positive25"),("unconstrained25_prior","free25"),("linear33_prior","linear33")):
                oracle[name]=C.prior_prediction(priors[family],record)
            a.require(set(oracle)==set(COLUMNS),"Complete CPU oracle panel")
            for name in COLUMNS:
                a.require(values["prediction__"+name].dtype==np.float64,"FP64 CPU prediction")
                P.close_array(a,values["prediction__"+name],oracle[name],"Every raw-tree/prior output "+name,atol=1e-8,rtol=2e-11)
    inputs=I.read(folder/"input_pins.json")
    for path,digest in inputs.items():M.pin(a,dict(path=path,sha256=digest))
    # No target file is required for prediction; prove none entered the ledger.
    target_paths={M.local(e["truth"]["path"]) for e in entries if e.get("truth")}
    a.require(not target_paths.intersection({M.local(k) for k in inputs}),"No held truth file in prediction input receipt")
    a.results.update(stage="base_predictions",kind=args.kind,dataset="rwth48v",role=role,
        prediction_lock=summary["prediction_lock"],model_lock=summary["model_lock"],cache_manifest=summary["cache_manifest"],
        profiles=P.ROLES[role],rows=sum(e["rows"] for e in entries),seed=lock.get("seed"),
        columns=COLUMNS if args.kind=="cpu" else ["mlp__"+str(lock["seed"])],
        all_rows_forward_checked=True,held_targets_decoded=False,confirmation_admission=lock["confirmation_admission"])
    return folder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind",choices=["cpu","mlp"],required=True)
    for key in ("input-dir","input-sha256","producer-sha256","model-audit","model-audit-sha256","feature-audit","feature-audit-sha256","output-dir"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--max-seconds",type=float,default=1200.);args=p.parse_args()
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=1200:raise ValueError("Finite bounded audit time")
    out=M.local(args.output_dir);out.mkdir(parents=True,exist_ok=False);a=I.Audit(args.max_seconds);status="HOLD"
    try:
        for file,digest in ((Path(__file__),sha(__file__)),(Path(__file__).with_name("rwth_cpu_audit_v002.py"),CPU_AUDIT_SHA),
                            (Path(__file__).with_name("rwth_held_prep_audit_v002.py"),HELD_AUDIT_SHA)):
            M.pin(a,dict(path=str(file),sha256=digest))
        folder=run(a,args);a.finish_pins();a.require(not (folder/"failure.json").exists(),"No final prediction failure marker")
        status="PASS" if not a.failures else "HOLD"
    except Exception as exc:a.failures.append(dict(check="Audit completed without exception",detail=repr(exc)))
    report=dict(status=status,stage="base_predictions",auditor_sha256=sha(__file__),producer_sha256=args.producer_sha256,
        checks=a.checks,failures=a.failures,inputs=a.inputs,results=a.results,replayed=a.replayed,
        elapsed_seconds=time.monotonic()-a.start,scope="Independent current-model raw-tree/NumPy forward on all saved role rows; no held target decode")
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(status=status,checks=a.checks,failures=len(a.failures))))
    return 0 if status=="PASS" else 2


if __name__=="__main__":raise SystemExit(main())
