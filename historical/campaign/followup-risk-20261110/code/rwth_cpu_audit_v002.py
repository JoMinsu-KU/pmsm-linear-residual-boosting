"""Independent RWTH CPU saved-model/CV audit. No sklearn or worker import."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MLP_AUDITOR_SHA = "0aedfdeb6c8e1c22f38fa7231dbeb84f91a63cc145e9c73cdd7ca3cf02344708"
CORE_SHA = "5c7962dae652958bc032267277dcec429d2f104a94b054c33914b543010bb164"
FAMILIES = ["positive25", "free25", "linear33"]
LAMBDAS = [.0001, .01, 1.]
GRID = [dict(max_leaf_nodes=l,max_iter=n,l2_regularization=p) for l,n,p in
        [(15,200,.1),(15,500,.1),(15,1000,.1),(31,200,.1),(31,500,.1),(31,1000,.1),(7,200,.1),(7,500,.1),(15,100,1.)]]
BASE = dict(learning_rate=.05,min_samples_leaf=20,max_bins=255,loss="squared_error",early_stopping=False,random_state=2026111201)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def dependency():
    p = Path(__file__).with_name("rwth_mlp_audit_v002.py")
    if sha(p) != MLP_AUDITOR_SHA:
        raise ValueError("Independent FIT audit helpers changed")
    spec = importlib.util.spec_from_file_location("rwth_independent_mlp_helpers", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


M = dependency(); P = M.P; I = M.I


def array_sha(value):
    x = np.ascontiguousarray(value)
    return hashlib.sha256(x.dtype.str.encode()+str(x.shape).encode()+x.tobytes()).hexdigest()


def scale_for(records, ids):
    ref = P.fit_statistics(records, ids)
    return dict(fit_ids=list(ids), x_mean=ref["input_mean"].tolist(), x_std=ref["input_std"].tolist(),
                y_mid=ref["y_mid"], y_scale=ref["y_scale"], drive_rms=ref["qscale"].tolist()), ref


def design_for(record, scale, family):
    x = record["X33"]
    if family == "linear33":
        return np.column_stack((np.ones(len(x)), (x-np.asarray(scale["x_mean"]))/np.asarray(scale["x_std"])))
    return np.column_stack(((x[:,8:13]-scale["y_mid"])/scale["y_scale"],
                            x[:,13:33]/np.repeat(np.asarray(scale["drive_rms"]), 5)))


def prior_prediction(model, record):
    return model["scale"]["y_mid"]+model["scale"]["y_scale"]*(design_for(record, model["scale"], model["family"])@np.asarray(model["coef"]))


def verify_prior(a, model, records, ids, family, lam, tag):
    a.require(family in FAMILIES and lam in LAMBDAS, "Frozen prior family/lambda " + tag)
    a.require(model["family"] == family and model["lam"] == lam, "Prior identity " + tag)
    a.same(model["fit_ids"], ids, "Fold-exclusive fitting IDs " + tag)
    scale, ref = scale_for(records, ids)
    a.same(model["scale"], scale, "Independent fold-only prior scales " + tag, atol=1e-10, rtol=2e-12)
    design = np.concatenate([design_for(records[p], scale, family)[::5] for p in ids])
    u = (ref["y"]-ref["y_mid"])/ref["y_scale"]
    w = ref["w"]/len(u); penalty = np.ones(design.shape[1])
    if family == "linear33": penalty[0] = 0.
    q = 2*(design.T@(design*w[:,None])+lam*np.diag(penalty))
    r = 2*design.T@(w*u); c = float(np.dot(w,u*u))
    for key, expected in (("Q",q),("r",r),("penalty",penalty)):
        P.close_array(a, np.asarray(model[key]), expected, "Independent objective " + tag + "/" + key, atol=1e-10, rtol=2e-12)
    a.same(model["constant"], c, "Independent objective constant " + tag, atol=1e-10, rtol=2e-12)
    for key, values in (("training_design_sha256",design),("training_target_sha256",u),("training_weights_sha256",w)):
        a.require(model[key] == array_sha(values), "Independent training objective digest " + tag + "/" + key)
    coef = np.asarray(model["coef"], dtype=np.float64)
    a.require(coef.shape == r.shape and np.isfinite(coef).all(), "Finite correct prior coefficients " + tag)
    np.linalg.cholesky(q)
    cert = model["certificate"]; positive = family == "positive25"
    a.require(cert["cholesky_pass"] is True and cert["positive_constraints"] is positive, "QP qualification type " + tag)
    a.require(isinstance(cert["iterations"], int) and 1 <= cert["iterations"] <= 1000, "Bounded QP iterations " + tag)
    e = np.r_[np.ones(5),np.zeros(len(coef)-5)] if positive else np.zeros(len(coef))
    nu = float(cert["nu"]); mu = np.asarray(cert["mu"], dtype=np.float64)
    a.require(math.isfinite(nu) and mu.shape == coef.shape and np.isfinite(mu).all(), "Finite dual certificate " + tag)
    gradient = q@coef-r+nu*e
    if positive:
        a.require(np.min(coef) >= -1e-10 and abs(e@coef-1.) <= 1e-10, "Positive simplex primal feasibility " + tag)
        P.close_array(a, mu, np.maximum(gradient,0.), "Dual feasible multiplier construction " + tag, atol=1e-10, rtol=2e-12)
        a.require(np.min(mu) >= 0., "Dual nonnegative feasibility " + tag)
    else:
        a.require(nu == 0. and np.count_nonzero(mu) == 0, "Unconstrained prior has no multipliers " + tag)
        P.close_array(a, coef, np.linalg.solve(q,r), "Independent unconstrained optimum " + tag, atol=1e-9, rtol=1e-9)
    b = r-nu*e+mu; z = np.linalg.solve(q,b)
    primal = float(.5*coef@q@coef-r@coef+c)
    dual = float(c-nu-.5*b@z)
    gap = float(.5*(coef-z)@q@(coef-z)+mu@coef+nu*(1-e@coef))
    tol = 1e-8*(1+abs(primal))
    a.require(gap >= -1e-10 and gap <= tol and abs(primal-dual-gap) <= tol,
              "Independent stable primal-dual gap qualification " + tag)
    expected = dict(cholesky_pass=True, iterations=cert["iterations"], primal=primal, dual=dual,
                    direct_gap=primal-dual,stable_gap=gap,gap_tolerance=tol,
                    equality_residual=abs(float(e@coef)-1) if positive else 0., min_coefficient=float(coef.min()),
                    nu=nu,mu=mu.tolist(),stationarity=float(np.max(np.abs(gradient-mu))),
                    complementarity=float(np.max(np.abs(mu*coef))),positive_constraints=positive)
    a.same(cert, expected, "Independent full optimality certificate " + tag, atol=1e-10, rtol=1e-8)
    return model


def numeric_nodes(a, path):
    a.require(path in a._pins, "Tree archive pinned before decoding")
    with np.load(path,allow_pickle=False) as saved:
        return {k:saved[k].copy() for k in saved.files}


def traverse(a, nodes, x, tag, count_reference=False):
    required = {"value","count","feature_idx","num_threshold","left","right","is_leaf","is_categorical"}
    a.require(nodes.ndim == 1 and len(nodes)>0 and required <= set(nodes.dtype.names or []), "Numerical tree schema " + tag)
    a.require(np.all(nodes["is_categorical"]==0) and np.isfinite(nodes["value"]).all()
              and np.isfinite(nodes["num_threshold"]).all(), "Finite numeric split tree " + tag)
    leaf = nodes["is_leaf"].astype(bool); state = np.zeros(len(x),dtype=np.int64)
    visits = np.zeros(len(nodes),dtype=np.int64); visits[0] = len(x)
    for _ in range(len(nodes)+1):
        a.budget(); moving = np.flatnonzero(~leaf[state])
        if not len(moving): break
        at = state[moving]; col = nodes["feature_idx"][at].astype(np.int64)
        a.require(np.all((col>=0)&(col<33)), "Tree feature indices " + tag)
        nxt = np.where(x[moving,col]<=nodes["num_threshold"][at],nodes["left"][at],nodes["right"][at]).astype(np.int64)
        a.require(np.all(nxt>at) and np.all(nxt<len(nodes)), "Acyclic forward tree structure " + tag)
        state[moving] = nxt
        if count_reference: visits += np.bincount(nxt,minlength=len(nodes))
    else: raise ValueError("Nonterminating raw tree")
    if count_reference:
        P.close_array(a,nodes["count"],visits,"Independent training node counts " + tag,exact=True)
    return nodes["value"][state]


def replay(a, trees, x, tag):
    prediction = np.full(len(x),float(trees["baseline"].item()),dtype=np.float64)
    for j in range(int(trees["n_iter"][0])):
        prediction += traverse(a,trees[f"tree_{j:04d}"],x,tag+"/"+str(j))
    a.replayed[tag] = dict(rows=len(x),trees=int(trees["n_iter"][0]),all_rows_all_trees=True)
    return prediction


def verify_booster(a, folder, records, ids, grid, tag, prior=None):
    metadata = I.read(M.pin(a,{"path":str(folder/"model.json"),"sha256":sha(folder/"model.json")}))
    a.require(metadata["schema"] == "rwth-hgb-model-v002" and metadata["grid_id"] == grid, "Fixed booster identity " + tag)
    a.same(metadata["fit_ids"],ids,"Fold-exclusive booster IDs " + tag)
    a.same(metadata["parameters"],dict(BASE,**GRID[grid]),"Fixed HGB parameters " + tag)
    scale,ref = scale_for(records,ids)
    a.same(metadata["scale"],scale,"Independent fold-only HGB scales " + tag,atol=1e-10,rtol=2e-12)
    x = (ref["x"]-ref["input_mean"])/ref["input_std"]
    y = np.concatenate([records[p]["target"][::5]-(prior_prediction(prior,records[p])[::5] if prior else 0.) for p in ids])
    w = ref["w"]
    a.require(metadata["training_rows"] == len(y), "Full FIT training rows " + tag)
    for key,values in (("training_X_sha256",x),("training_target_sha256",y),("training_weight_sha256",w)):
        a.require(metadata[key] == array_sha(values), "Independent HGB input digest " + tag + "/" + key)
    a.require(metadata["target_kind"] == ("prior_residual" if prior else "rotor_celsius")
              and metadata["prior_family"] == (prior["family"] if prior else None), "Exact target and residual source " + tag)
    a.require(metadata["histogram_subsample"] is None and len(y)<200000, "All FIT rows used for bin construction " + tag)
    a.require({"trees.npz","training.npz"} <= set(metadata["files"]) <= {"trees.npz","training.npz","model.pkl"}, "Complete raw model artifacts " + tag)
    for name,value in metadata["files"].items(): M.pin(a,{"path":str(folder/name),"sha256":value})
    training = a.arrays(folder/"training.npz")
    expected = dict(profile_id=ref["profile_ids"],index=ref["index"],source_row=ref["index"],target=y,weight=w)
    a.require(set(training) == set(expected), "Exact booster training schema " + tag)
    for key,value in expected.items():
        a.require(training[key].dtype == value.dtype, "Explicit booster training dtype " + tag + "/" + key)
        P.close_array(a,training[key],value,"Independent booster training " + tag + "/" + key,exact=True)
    trees = numeric_nodes(a,folder/"trees.npz"); n = GRID[grid]["max_iter"]
    required = {"baseline","n_iter","n_bins_non_missing"}|{f"tree_{j:04d}" for j in range(n)}|{f"bin_thresholds_{j:02d}" for j in range(33)}
    a.require(set(trees) == required and trees["baseline"].shape == (1,1)
              and np.isfinite(trees["baseline"]).all() and np.array_equal(trees["n_iter"],np.asarray([n],dtype=np.int64)), "Complete fixed-iteration tree archive " + tag)
    P.close_array(a,trees["baseline"],np.asarray([[np.average(y,weights=w)]]),"Independent weighted baseline " + tag,atol=1e-9,rtol=2e-12)
    expected_counts = []
    for col in range(33):
        values = np.sort(x[:,col]); distinct = np.unique(values)
        if len(distinct)<=255:
            thresholds = .5*(distinct[:-1]+distinct[1:])
        else:
            positions = (np.linspace(0,100,256)[1:-1]/100)*(len(values)-1)
            thresholds = .5*(values[np.floor(positions).astype(np.int64)]+values[np.ceil(positions).astype(np.int64)])
        bins = trees[f"bin_thresholds_{col:02d}"]
        P.close_array(a,bins,thresholds,"Independent FIT midpoint bins " + tag + "/" + str(col),atol=1e-10,rtol=2e-12)
        expected_counts.append(len(bins)+1)
    P.close_array(a,trees["n_bins_non_missing"],np.asarray(expected_counts),"Bin counts " + tag,exact=True)
    for j in range(n):
        nodes = trees[f"tree_{j:04d}"]
        a.require(np.count_nonzero(nodes["is_leaf"]) <= GRID[grid]["max_leaf_nodes"], "Registered leaf capacity " + tag)
    # All held rows/all trees are replayed below; selected training-tree counts
    # are a separate structural check, not an optimizer reconstruction.
    for j in sorted({0,n//2,n-1}): traverse(a,trees[f"tree_{j:04d}"],x,tag+"/FIT/"+str(j),True)
    return metadata,trees


def held_boost_prediction(a, metadata, trees, record, tag, prior=None):
    scale = metadata["scale"]
    x = (record["X33"]-np.asarray(scale["x_mean"]))/np.asarray(scale["x_std"])
    pred = replay(a,trees,x,tag)
    return pred+prior_prediction(prior,record) if prior else pred


def stage_ref(a, entry, folder, summary, label):
    path = M.local(entry["path"])
    a.require(path.is_relative_to(folder) and summary["files"].get(path.relative_to(folder).as_posix()) == entry["sha256"],
              "Completion receipt binds " + label)
    return M.pin(a,entry)


def read_stage(a, entry, expected_mode, producer_sha):
    path = M.pin(a,entry); folder = path.parent
    a.require(path.name == "summary.json" and not (folder/"failure.json").exists(), "Complete nonfailed CPU stage")
    summary = I.read(path)
    statuses = {"cv-direct":"RWTH_DIRECT_CV_COMPLETE","cv-linear":"RWTH_LINEAR_CV_COMPLETE","final":"RWTH_CPU_FINAL_MODELS_LOCKED"}
    a.require(summary["schema"] == "rwth-cpu-stage-v002" and summary["mode"] == expected_mode
              and summary["status"] == statuses[expected_mode] and summary["source_sha256"] == producer_sha
              and summary["core_sha256"] == CORE_SHA, "Frozen completed CPU stage identity")
    for name,digest in summary["files"].items():
        M.pin(a,{"path":str(I.path_in(folder,name)),"sha256":digest})
    configuration = I.read(folder/"configuration.json")
    a.require(configuration["mode"] == expected_mode and configuration["source_sha256"] == producer_sha
              and configuration["core_sha256"] == CORE_SHA, "Actual CPU mode invocation")
    a.require(M.local(configuration["manifest"]) == M.local(summary["manifest"]["path"])
              and configuration["manifest_sha256"] == summary["manifest"]["sha256"], "Exact invocation FIT manifest")
    return folder,summary


def held_array(a, entry, folder, summary, pid, record, pred, tag):
    path = stage_ref(a,entry,folder,summary,"held prediction " + tag)
    values = a.arrays(path); n = len(record["target"])
    a.require(set(values) == {"profile_id","index","source_row","prediction"}, "Exact saved held schema " + tag)
    a.require(values["profile_id"].dtype == np.int64 and values["profile_id"].shape == ()
              and int(values["profile_id"]) == pid, "Held profile identity " + tag)
    for key in ("index","source_row"):
        a.require(values[key].dtype == np.int64, "Held explicit int64 indices " + tag)
        P.close_array(a,values[key],np.arange(n,dtype=np.int64),"Every held original row " + tag,exact=True)
    a.require(values["prediction"].dtype == np.float64, "FP64 held predictions " + tag)
    P.close_array(a,values["prediction"],pred,"Independent saved-model forward " + tag,atol=1e-8,rtol=2e-11)
    # Selection must reproduce the producer's saved prediction metric exactly;
    # independent model arithmetic is checked separately immediately above.
    return float(np.sqrt(np.mean((values["prediction"]-record["target"])**2)))


def audit_direct_cv(a,args,folder,summary,records):
    table = I.read(stage_ref(a,summary["cv"],folder,summary,"direct CV table"))
    grid = table["grid_id"]; ids = P.ROLES["fit"]
    a.require(grid in range(9) and summary["grid_id"] == grid and table["schema"] == "rwth-direct-cv-v002", "Fixed direct CV grid")
    a.same(table["ordered_ids"],ids,"Full LOPO identity order")
    a.same(table["parameters"],dict(BASE,**GRID[grid]),"Exact direct grid parameters")
    a.require([r["held_out"] for r in table["folds"]] == ids and summary["folds"] == 20, "All20 complete direct folds")
    errors = []
    for row in table["folds"]:
        held = row["held_out"]; train = [pid for pid in ids if pid != held]
        a.same(row["fit_ids"],train,"Held profile absent from training")
        metadata_path = stage_ref(a,row["model_metadata"],folder,summary,"fold model metadata")
        meta,trees = verify_booster(a,metadata_path.parent,records,train,grid,f"g{grid}/held{held}")
        a.require("model.pkl" not in meta["files"], "CV requires no pickle serialization")
        pred = held_boost_prediction(a,meta,trees,records[held],f"g{grid}/held{held}")
        rmse = held_array(a,row["prediction"],folder,summary,held,records[held],pred,f"g{grid}/held{held}")
        a.same(row["rmse"],rmse,"Held all-row RMSE",atol=1e-12,rtol=1e-12); errors.append(rmse)
    macro = float(np.mean(errors))
    a.same(table["macro_rmse"],macro,"Equal-profile20-fold macro RMSE",atol=1e-12,rtol=1e-12)
    return dict(grid_id=grid,profile_rmse=errors,macro_rmse=macro,cv=summary["cv"])


def audit_linear_cv(a,args,folder,summary,records):
    table = I.read(stage_ref(a,summary["cv"],folder,summary,"prior CV table")); ids = P.ROLES["fit"]
    a.require(table["schema"] == "rwth-linear-cv-v002" and set(table["families"]) == set(FAMILIES)
              and summary["folds"] == 180, "All three prior families and180 fits")
    a.same(table["ordered_ids"],ids,"Full prior LOPO identity order")
    results = {}
    for family in FAMILIES:
        item = table["families"][family]
        a.require([(r["lam"],r["held_out"]) for r in item["folds"]] == [(lam,pid) for lam in LAMBDAS for pid in ids],
                  "Exact three lambdas by20 folds " + family)
        tables = []
        for lam in LAMBDAS:
            errors = []
            for row in [r for r in item["folds"] if r["lam"] == lam]:
                held = row["held_out"]; train = [pid for pid in ids if pid != held]
                model = verify_prior(a,row["model"],records,train,family,lam,f"{family}/{lam}/{held}")
                pred = prior_prediction(model,records[held])
                error = held_array(a,row["prediction"],folder,summary,held,records[held],pred,f"{family}/{lam}/{held}")
                a.same(row["rmse"],error,"Prior held full-row RMSE",atol=1e-12,rtol=1e-12); errors.append(error)
            tables.append(dict(lam=lam,profile_rmse=errors,macro_rmse=float(np.mean(errors))))
        a.same(item["cv"],tables,"Independent lambda selection table " + family,atol=1e-12,rtol=1e-12)
        selected = min(tables,key=lambda r:(r["macro_rmse"],r["lam"]))["lam"]
        a.same(item["selected_lambda"],selected,"Fixed best lambda/tie rule " + family)
        results[family] = dict(selected_lambda=selected,cv=tables)
    return dict(families=results,selected_lambdas={f:results[f]["selected_lambda"] for f in FAMILIES},cv=summary["cv"])


def audit_prior_report(a, entry, input_summary, stage, grid=None):
    report = I.read(M.pin(a,entry))
    a.require(report["status"] == "PASS" and not report["failures"] and report["stage"] == stage
              and report["auditor_sha256"] == sha(__file__), "Current complete independent CV PASS")
    M.binding(a,report,M.local(input_summary["path"]),input_summary["sha256"],"CV PASS binds exact producer summary")
    if grid is not None: a.require(report["results"]["grid_id"] == grid,"Ordered audited direct grid")
    return report


def audit_final(a,args,folder,summary,records):
    a.require(bool(args.cv_audits and args.cv_audits_sha256), "Final models require every CV audit")
    audit_bundle = I.read(M.pin(a,{"path":args.cv_audits,"sha256":args.cv_audits_sha256}))
    a.require(audit_bundle["schema"] == "rwth-cpu-cv-audit-inputs-v002" and len(audit_bundle["direct"]) == 9,
              "Complete nine-grid plus linear CV audit manifest")
    lock = I.read(stage_ref(a,summary["model_lock"],folder,summary,"final model lock"))
    a.require(lock["schema"] == "rwth-cpu-model-lock-v002" and lock["status"] == "CPU_MODELS_LOCKED"
              and lock["dataset"] == "rwth48v" and lock["source_sha256"] == args.producer_sha256
              and lock["core_sha256"] == CORE_SHA, "Immutable final CPU models")
    a.same(lock["fit_ids"],P.ROLES["fit"],"Final fullFIT20")
    a.same(lock["fit_manifest"],summary["manifest"],"Final cache identity")
    a.require(lock["protocol"]["sha256"] == P.PROTOCOL_SHA and lock["partition"]["sha256"] == P.PARTITION_SHA,
              "Final frozen protocol/partition")
    for key in ("calibration_values_accessed","confirmation_values_accessed"):
        a.require(lock[key] is False and summary[key] is False,"No held values during final fit")
    for name,digest in lock["files"].items():
        a.require(summary["files"].get(name) == digest,"Final completion/model-lock common hash")
        M.pin(a,{"path":str(I.path_in(folder,name)),"sha256":digest})
    selection = I.read(stage_ref(a,lock["selection"],folder,summary,"final selection"))
    a.require(selection["schema"] == "rwth-cpu-selection-v002", "Final selection schema")
    cv_bundle = I.read(M.pin(a,selection["cv_bundle"]))
    a.require(cv_bundle["schema"] == "rwth-cpu-cv-inputs-v002" and len(cv_bundle["direct"]) == 9,
              "All nine fixed CV stages")
    a.same(cv_bundle["manifest"],summary["manifest"],"CV/final common FIT source")
    independent_direct = []
    for grid,(cv_entry,audit_entry) in enumerate(zip(cv_bundle["direct"],audit_bundle["direct"])):
        _,cv_summary = read_stage(a,cv_entry,"cv-direct",args.producer_sha256)
        report = audit_prior_report(a,audit_entry,cv_entry,"cv-direct",grid)
        a.same(cv_summary["manifest"],summary["manifest"],"CV/final same FIT manifest")
        independent_direct.append(dict(grid_id=grid,macro_rmse=report["results"]["macro_rmse"]))
    _,linear_summary = read_stage(a,cv_bundle["linear"],"cv-linear",args.producer_sha256)
    linear_report = audit_prior_report(a,audit_bundle["linear"],cv_bundle["linear"],"cv-linear")
    a.same(linear_summary["manifest"],summary["manifest"],"Linear/final same FIT manifest")
    best = min(independent_direct,key=lambda r:(r["macro_rmse"],r["grid_id"]))["grid_id"]
    lambdas = linear_report["results"]["selected_lambdas"]
    a.same(selection["direct_cv"],independent_direct,"Selection from independently audited direct predictions",atol=1e-12,rtol=1e-12)
    a.same(selection["selected_direct_grid"],best,"Exact grid argmin/tie rule")
    a.same(selection["selected_lambdas"],lambdas,"Exact independently audited prior lambda selection")
    a.same(summary["selected_direct_grid"],best,"Final summary selection")
    priors = I.read(stage_ref(a,lock["priors"],folder,summary,"final priors"))
    a.require(set(priors) == set(FAMILIES),"Three final prior families")
    for family in FAMILIES:
        verify_prior(a,priors[family],records,P.ROLES["fit"],family,lambdas[family],"final/"+family)
    jobs = {"direct_boost":(0,None),"positive_rc_insample_boost":(0,"positive25"),
            "unconstrained_rc_insample_boost":(0,"free25"),"linear33_insample_boost":(0,"linear33")}
    if best != 0: jobs["direct9_fitcv_selected"] = (best,None)
    a.require(set(lock["boosters"]) == set(jobs),"All fixed final booster controls")
    a.same(lock["aliases"],{"direct9_fitcv_selected":"direct_boost"} if best == 0 else {},"Only exact grid0 D9 alias")
    for name,(grid,family) in jobs.items():
        entry = lock["boosters"][name]
        a.require(entry["prior_family"] == family,"Final booster prior identity")
        metadata_path = stage_ref(a,entry["metadata"],folder,summary,"final booster metadata")
        meta,_ = verify_booster(a,metadata_path.parent,records,P.ROLES["fit"],grid,"final/"+name,
                                priors[family] if family else None)
        for filename,key in (("trees.npz","trees"),("model.pkl","pickle")):
            path = stage_ref(a,entry[key],folder,summary,"final exported "+filename)
            a.require(path == metadata_path.parent/filename and entry[key]["sha256"] == meta["files"][filename],
                      "Exact final artifact identity")
    expected_columns = ["direct9_fitcv_selected","direct_boost","positive_rc","unconstrained25_prior","linear33_prior",
                        "positive_rc_insample_boost","unconstrained_rc_insample_boost","linear33_insample_boost"]
    a.same(lock["prediction_columns"],expected_columns,"All eight logical CPU outputs")
    a.same(summary["actual_boosters"],len(jobs),"Actual final model count")
    a.same(summary["logical_boosters"],5,"Five logical final boosters")
    a.same(summary["priors"],3,"Three final priors")
    return dict(selected_direct_grid=best,selected_lambdas=lambdas,model_lock=summary["model_lock"],
                actual_boosters=len(jobs),logical_boosters=5,priors=3,
                saved_model_forward_scope="CV all held rows/all trees; final structural/scaler/bin/target validation, role forward deferred")


def synthetic_test():
    a = I.Audit(30)
    dtype = [("value","f8"),("count","u4"),("feature_idx","u4"),("num_threshold","f8"),
             ("left","u4"),("right","u4"),("is_leaf","u1"),("is_categorical","u1")]
    nodes = np.zeros(3,dtype=dtype);nodes[0]["count"]=3;nodes[0]["left"]=1;nodes[0]["right"]=2
    nodes[1]["is_leaf"]=nodes[2]["is_leaf"]=1;nodes[1]["value"]=-2.;nodes[2]["value"]=3.
    nodes[1]["count"]=2;nodes[2]["count"]=1
    x = np.zeros((3,33));x[:,0]=[-1,0,1]
    assert np.array_equal(traverse(a,nodes,x,"synthetic",True),[-2.,-2.,3.])
    assert not a.failures
    print(json.dumps(dict(status="PASS_SYNTHETIC_ONLY",checks=a.checks,scope="saved numeric tree equality direction and node counts")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["cv-direct","cv-linear","final"])
    for key in ("input-dir","input-sha256","producer-sha256","prep-audit","prep-audit-sha256","output-dir",
                "cv-audits","cv-audits-sha256"):
        p.add_argument("--"+key)
    p.add_argument("--max-seconds",type=float,default=1200.)
    p.add_argument("--self-test",action="store_true")
    args = p.parse_args()
    if args.self_test: synthetic_test();return 0
    if not all((args.mode,args.input_dir,args.input_sha256,args.producer_sha256,args.prep_audit,args.prep_audit_sha256,args.output_dir)):
        p.error("Mode, complete stage summary, producer hash, prep PASS and new output required")
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=1200:
        p.error("Finite audit cap in(0,1200] required")
    out = M.local(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    a = I.Audit(args.max_seconds);status="HOLD"
    try:
        for filename,expected in (("rwth_cpu_core_v002.py",CORE_SHA),("rwth_cpu_fit_v002.py",args.producer_sha256),
                                  ("rwth_mlp_audit_v002.py",MLP_AUDITOR_SHA),("rwth_prep_audit_v002.py",M.PREP_AUDITOR_SHA),
                                  ("risk_audit_v001.py",P.AUDIT_DEP_SHA)):
            M.pin(a,{"path":str(Path(__file__).with_name(filename)),"sha256":expected})
        M.pin(a,{"path":str(Path(__file__)),"sha256":sha(__file__)})
        input_summary = dict(path=str(M.local(args.input_dir)/"summary.json"),sha256=args.input_sha256)
        folder,summary = read_stage(a,input_summary,args.mode,args.producer_sha256)
        manifest,records = M.fit_cache(a,summary["manifest"],dict(path=args.prep_audit,sha256=args.prep_audit_sha256))
        a.same(summary["protocol"],manifest["protocol"],"Stage frozen protocol")
        fn = {"cv-direct":audit_direct_cv,"cv-linear":audit_linear_cv,"final":audit_final}[args.mode]
        a.results.update(fn(a,args,folder,summary,records),dataset="rwth48v",stage=args.mode,input_summary=input_summary)
        inputs = I.read(folder/"input_pins.json")
        for path,digest in inputs.items(): M.pin(a,{"path":path,"sha256":digest})
        a.finish_pins();a.require(not (folder/"failure.json").exists(),"No final producer failure marker")
        status = "PASS" if not a.failures else "HOLD"
    except Exception as exc:
        a.failures.append(dict(check="Audit completed without exception",detail=repr(exc)))
    report = dict(status=status,stage=args.mode,auditor_sha256=sha(__file__),producer_sha256=args.producer_sha256,
                  checks=a.checks,failures=a.failures,inputs=a.inputs,results=a.results,replayed=a.replayed,
                  elapsed_seconds=time.monotonic()-a.start,
                  scope="Independent saved-artifact, raw-node forward, prior optimality and FIT-only CV audit; no optimizer replay, pickle loading or held CAL/CONF access")
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(status=status,checks=a.checks,failures=len(a.failures))))
    return 0 if status=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
