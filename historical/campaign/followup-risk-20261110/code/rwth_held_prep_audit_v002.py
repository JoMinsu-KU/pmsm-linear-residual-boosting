"""Independent role-specific held-cache audit after all fixed models are locked.

CAL target is checked only after four model locks and independent audits.
CONF checks observed inputs/features only, after consumed root admission.
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
import zipfile
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
MLP_AUDIT_SHA="0aedfdeb6c8e1c22f38fa7231dbeb84f91a63cc145e9c73cdd7ca3cf02344708"
CPU_PRODUCER_SHA="a22c12eb6962b7ce9a83f956a6cd4d0cf377a940903593a3ed1e0d1ac2abb57a"
MLP_PRODUCER_SHA="4005577d76b689515256e933017c75239335a81c076034e17eedad4ef79d1113"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()


def helper():
    p=Path(__file__).with_name("rwth_mlp_audit_v002.py")
    if sha(p)!=MLP_AUDIT_SHA:raise ValueError("Frozen independent helper changed")
    spec=importlib.util.spec_from_file_location("rwth_held_independent_helpers",p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


M=helper();P=M.P;I=M.I


def pass_report(a,entry,stage,source_file,expected_sha=None):
    source=Path(__file__).with_name(source_file)
    current=sha(source)
    if expected_sha is not None:a.require(current==expected_sha,"Frozen independent auditor source")
    M.pin(a,dict(path=str(source),sha256=current))
    report=I.read(M.pin(a,entry))
    a.require(report["status"]=="PASS" and report["stage"]==stage and not report["failures"]
              and report["auditor_sha256"]==current,"Complete current independent model audit")
    return report


def base_models(a,manifest,audit_bundle_ref):
    bundle=I.read(M.pin(a,manifest["base_model_bundle"]))
    a.require(bundle["schema"]=="rwth-base-model-bundle-v002" and len(bundle["mlp"])==3,"All four fixed model locks")
    a.same(bundle["protocol"],manifest["protocol"],"Base models frozen protocol")
    a.same(bundle["partition"],manifest["partition"],"Base models frozen partition")
    a.same(bundle["fit_manifest"],manifest["fit_manifest"],"Base models original FIT cache")
    audits=I.read(M.pin(a,audit_bundle_ref))
    a.require(audits["schema"]=="rwth-base-model-audits-v002" and len(audits["mlp"])==3,
              "All four independent base-model PASS reports")
    refs=[bundle["cpu"]]+bundle["mlp"]
    a.same(manifest["base_model_locks"],refs,"Exact base locks before held input access")
    before={M.local(k):v for k,v in manifest["input_hashes_before_held_members"].items()}
    cpu=pass_report(a,audits["cpu"],"final","rwth_cpu_audit_v002.py")
    reports=[cpu]+[pass_report(a,e,"mlp_train","rwth_mlp_audit_v002.py",MLP_AUDIT_SHA) for e in audits["mlp"]]
    seeds=[]
    for pos,(entry,report) in enumerate(zip(refs,reports)):
        path=M.pin(a,entry);lock=I.read(path)
        a.require(not (path.parent/"failure.json").exists(),"No model producer failure")
        a.require(before.get(path)==entry["sha256"],"Base model identity pinned before held members")
        M.binding(a,report,path,entry["sha256"],"Independent audit binds actual current model lock")
        a.same(report["results"]["model_lock"],entry,"Audit result model-lock identity")
        a.same(lock["protocol"],manifest["protocol"],"Model protocol")
        a.same(lock["partition"],manifest["partition"],"Model partition")
        if pos==0:
            a.require(lock["schema"]=="rwth-cpu-model-lock-v002" and lock["status"]=="CPU_MODELS_LOCKED"
                      and lock["source_sha256"]==CPU_PRODUCER_SHA,"Fixed CPU model identity")
            a.same(lock["fit_manifest"],manifest["fit_manifest"],"CPU FIT cache")
        else:
            a.require(lock["schema"]=="rwth-mlp-model-lock-v002" and lock["status"]=="MODEL_LOCKED"
                      and lock["source_sha256"]==MLP_PRODUCER_SHA and lock["completed_comparator"] is True,"Fixed completed neural model")
            a.same(lock["cache_manifest"],manifest["fit_manifest"],"MLP FIT cache")
            seeds.append(lock["seed"])
            a.require(report["results"]["seed"]==lock["seed"],"Audited seed matches actual lock")
        for name,expected in lock["files"].items():
            artifact=I.path_in(path.parent,name)
            a.require(before.get(artifact)==expected,"Every model artifact locked before held input")
            M.pin(a,dict(path=str(artifact),sha256=expected))
            M.binding(a,report,artifact,expected,"Every current model artifact independently audited")
    a.same(seeds,M.SEEDS,"All three fixed ordered seeds")
    return refs


def consumed_admission(a,entry,manifest,models):
    a.require(isinstance(entry,dict),"CONF requires root admission before raw member decode")
    path=M.pin(a,entry);gate=I.read(path)
    expected=dict(schema="rwth-confirmation-admission-v002",status="ADMITTED_CONSUMED_NO_RESET",
        confirmation_id="C17_RWTH_FIXED_CONFIRMATION_02",candidate_id="C17",alpha=.025,allowed_roles=["CONF"],
        protocol=manifest["protocol"],partition=manifest["partition"],conf_ids=P.ROLES["confirmation"])
    for key,value in expected.items():a.same(gate[key],value,"Consumed admission "+key)
    for ref in models:
        name=M.local(ref["path"]).relative_to(ROOT).as_posix()
        a.require(gate["bindings"].get(name)==ref["sha256"],"CONF admission binds every current base model")
    for name,digest in gate["bindings"].items():M.pin(a,dict(path=name,sha256=digest))
    before={M.local(k):v for k,v in manifest["input_hashes_before_held_members"].items()}
    a.require(before.get(path)==entry["sha256"],"Root consumption checked before held members")
    return gate


def stage_entry(a,ref,folder,summary):
    path=M.local(ref["path"])
    a.require(path.is_relative_to(folder) and summary["files"].get(path.relative_to(folder).as_posix())==ref["sha256"],
              "Exact artifact bound by completed held stage")
    return M.pin(a,ref)


def identity_arrays(a,values,pid,n,time_values,label,target=False):
    needed={"profile_id","index","source_row","time","target"} if target else {"profile_id","index","source_row","time","operating","X33"}
    a.require(set(values)==needed,"Exact target-separated "+label+" keys")
    a.require(values["profile_id"].dtype==np.int64 and values["profile_id"].shape==() and int(values["profile_id"])==pid,
              "Original held identity")
    for key in ("index","source_row"):
        a.require(values[key].dtype==np.int64,"Explicit held int64 indices")
        P.close_array(a,values[key],np.arange(n,dtype=np.int64),"All held chronological rows "+key,exact=True)
    a.require(values["time"].dtype==np.float64,"Original FP64 timestamp")
    P.close_array(a,values["time"],time_values,"Exact original held timestamp",exact=True)


def run(a,args):
    folder=M.local(args.input_dir)
    a.require(not (folder/"failure.json").exists(),"No held-preparation failure")
    M.pin(a,dict(path=str(Path(__file__).with_name("rwth_prepare_held_v002.py")),sha256=args.producer_sha256))
    summary=I.read(M.pin(a,dict(path=str(folder/"summary.json"),sha256=args.input_sha256)))
    a.require(summary["schema"]=="rwth-held-preparation-summary-v002" and summary["status"]=="RWTH_HELD_FEATURE_CACHE_READY"
              and summary["source_sha256"]==args.producer_sha256,"Completed held preparation")
    role=summary["role"];a.require(role in ("calibration","confirmation"),"One exact held role")
    for name,digest in summary["files"].items():M.pin(a,dict(path=str(I.path_in(folder,name)),sha256=digest))
    manifest=I.read(stage_entry(a,summary["manifest"],folder,summary))
    a.require(manifest["schema"]=="rwth-cache-manifest-v002" and manifest["dataset"]=="rwth48v"
              and manifest["source_sha256"]==args.producer_sha256
              and manifest["status"]==("READY_FIT_CAL" if role=="calibration" else "READY_CONFIRMATION"),"Role cache identity")
    fit,unused=M.fit_cache(a,manifest["fit_manifest"],dict(path=args.prep_audit,sha256=args.prep_audit_sha256))
    del unused
    for key in ("protocol","partition","archive","roles","dt_ref","feature_names","operating_columns","spans","warmup","train_stride","evaluation_stride"):
        a.same(manifest[key],fit[key],"Unchanged FIT feature specification "+key)
    models=base_models(a,manifest,dict(path=args.model_audits,sha256=args.model_audits_sha256))
    if role=="confirmation":consumed_admission(a,manifest["confirmation_admission"],manifest,models)
    else:a.require(manifest["confirmation_admission"] is None,"No CONF admission used for CAL")
    a.same(summary["confirmation_admission"],manifest["confirmation_admission"],"Role admission binding")
    a.same(summary["base_model_bundle"],manifest["base_model_bundle"],"Completed base bundle binding")
    a.same(manifest["roles"],P.ROLES,"Frozen role partition")
    entries=[r for r in manifest["records"] if r["role"]==role]
    a.require([r["id"] for r in entries]==P.ROLES[role],"Exact complete held role and order")
    a.same(manifest["records"],fit["records"]+entries if role=="calibration" else entries,"No extra/moved/excluded role records")
    for key,expected in (("confirmation_rotor_values_parsed",False),("stator_numeric_values_parsed",False),
                         ("calibration_member_contents_opened",role=="calibration"),
                         ("confirmation_member_contents_opened",role=="confirmation"),
                         ("confirmation_attempt_consumed",role=="confirmation"),("fitted_model_count",0)):
        a.same(manifest[key],expected,"Held preparation role flag "+key)
    archive_path=M.pin(a,manifest["archive"]);total=0;opened=[]
    with zipfile.ZipFile(archive_path) as archive:
        a.require(len(archive.namelist())==len(set(archive.namelist())),"Unique raw archive member names")
        for entry in entries:
            pid=entry["id"];member=f"final/{pid:03d}.csv"
            a.require(entry["member"]==member and entry["target_key"] is None,"Original held member; no target in features")
            raw=P.raw_profile(a,archive,member,pid,role=="calibration");n=raw["rows"];total+=n;opened.append(member)
            a.same(entry["rows"],n,"Full held source rows")
            a.same(entry["csv_sha256"],raw["csv_sha256"],"Exact raw held CSV bytes")
            values=a.arrays(stage_entry(a,entry["features"],folder,summary))
            identity_arrays(a,values,pid,n,raw["time"],"feature")
            a.require(values["operating"].dtype==np.float64 and values["X33"].dtype==np.float64,"Held FP64 observed feature arrays")
            P.close_array(a,values["operating"],raw["operating"],"Exact four held observed channels",exact=True)
            P.close_array(a,values["X33"],P.causal_features(raw["time"],raw["operating"],fit["dt_ref"]),"Independent held causal feature reconstruction")
            if role=="calibration":
                target=a.arrays(stage_entry(a,entry["truth"],folder,summary))
                identity_arrays(a,target,pid,n,raw["time"],"truth",True)
                a.require(target["target"].dtype==np.float64,"FP64 separate CAL truth")
                P.close_array(a,target["target"],raw["target"],"Independent CAL rotor lineage after all model locks",exact=True)
            else:a.require(entry["truth"] is None,"CONF rotor remains unparsed and unsaved")
    a.same(manifest["opened_csv_members"],opened,"Only the admitted role members opened")
    a.same(summary["profiles"],len(entries),"Full held profile count")
    a.same(summary["rows"],total,"Full held row count")
    inputs=I.read(folder/"input_pins.json");a.same(inputs,manifest["inputs"],"All held preparation input pins")
    for path,digest in inputs.items():M.pin(a,dict(path=path,sha256=digest))
    a.results.update(stage="held_features",dataset="rwth48v",role=role,cache_manifest=summary["manifest"],
        base_model_locks=models,profiles=[e["id"] for e in entries],rows=total,raw_members_decoded=opened,
        calibration_rotor_decoded=role=="calibration",confirmation_rotor_decoded=False,stator_numerically_parsed=False,
        confirmation_admission=manifest["confirmation_admission"])
    return folder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("input-dir","input-sha256","producer-sha256","prep-audit","prep-audit-sha256","model-audits","model-audits-sha256","output-dir"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--max-seconds",type=float,default=600.)
    args=p.parse_args()
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=1200:raise ValueError("Bounded audit time required")
    out=M.local(args.output_dir);out.mkdir(parents=True,exist_ok=False);a=I.Audit(args.max_seconds);status="HOLD"
    try:
        for file,digest in ((Path(__file__),sha(__file__)),(Path(__file__).with_name("rwth_mlp_audit_v002.py"),MLP_AUDIT_SHA),
                            (Path(__file__).with_name("rwth_prep_audit_v002.py"),M.PREP_AUDITOR_SHA),(Path(__file__).with_name("risk_audit_v001.py"),P.AUDIT_DEP_SHA)):
            M.pin(a,dict(path=str(file),sha256=digest))
        folder=run(a,args);a.finish_pins();a.require(not (folder/"failure.json").exists(),"No final preparation failure")
        status="PASS" if not a.failures else "HOLD"
    except Exception as exc:a.failures.append(dict(check="Audit completed without exception",detail=repr(exc)))
    report=dict(status=status,stage="held_features",auditor_sha256=sha(__file__),producer_sha256=args.producer_sha256,
        checks=a.checks,failures=a.failures,inputs=a.inputs,results=a.results,elapsed_seconds=time.monotonic()-a.start,
        scope="Independent held observed-input/feature lineage; CAL truth after all model locks; CONF rotor never decoded")
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(status=status,checks=a.checks,failures=len(a.failures))))
    return 0 if status=="PASS" else 2


if __name__=="__main__":raise SystemExit(main())
