"""Run the v013 retrospective equal-candidate-budget comparison from pinned raw files."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import time

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from pmsm_repro.data import acquire, raw_records
from pmsm_repro.features import boost_features, make_record, training, train_index, evaluation_index
from pmsm_repro.metrics import profile_metrics, aggregate_profiles
from pmsm_repro.models import fit_prior, predict_prior

PROTOCOL = ROOT / "configs/equal_budget_v013.json"
STUDY = ROOT / "configs/study.json"
def digest(p):
    return sha256(p.read_bytes()).hexdigest()

def dump(path, x):
    path.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def savecsv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def fit_data(records, ids, regularization):
    x, y, w, scale, ix = training(records, ids)
    prior = fit_prior(records, ids, "linear", regularization, scale)
    p = np.concatenate([predict_prior(prior, records[i])[ix[i]] for i in ids])
    return x, y, w, scale, prior, p

def inner(dataset, arm, records, spec, protocol, grid):
    heldrows = []
    fit_ids = sorted(spec["roles"]["fit"])
    for f in range(3):
        val = fit_ids[f::3]
        fit = [i for i in fit_ids if i not in val]
        assert not set(val).intersection(fit)
        x, y, w, scale, prior, p = fit_data(records, fit, protocol["ridge_lambda"])
        offsets = [0]
        xx, yy, pp = [], [], []
        for i in val:
            r = records[i]
            ix = train_index(r)
            xx.append(boost_features(r, scale)[ix])
            yy.append(r["y"][ix])
            pp.append(predict_prior(prior, r)[ix])
            offsets.append(offsets[-1] + len(ix))
        vx, vy, vp = np.concatenate(xx), np.concatenate(yy), np.concatenate(pp)
        for family in protocol["families"]:
            residual = family == "linear_residual"
            for leaves in sorted({a for a, _ in grid}):
                checkpoints = {n: j for j,(a,n) in enumerate(grid) if a == leaves}
                # With disabled early stopping, prefixes are the exact fixed-N learner
                # for each smaller N; no extra candidate is scored by this implementation.
                booster = HistGradientBoostingRegressor(**protocol["shared"],
                            max_leaf_nodes=leaves, max_iter=max(checkpoints)).fit(
                                x, y-p if residual else y, sample_weight=w)
                for n, pred in enumerate(booster.staged_predict(vx), 1):
                    if n not in checkpoints:
                        continue
                    error = (pred + vp if residual else pred) - vy
                    for j, i in enumerate(val):
                        e = error[offsets[j]:offsets[j+1]]
                        heldrows.append(dict(dataset=dataset, arm=arm, family=family,
                            fold=f+1, profile=i, candidate=checkpoints[n]+1, leaves=leaves,
                            iterations=n, rows=len(e), rmse=float(np.sqrt(np.mean(e*e)))))
        print(dataset, arm, "completed inner fold", f+1, flush=True)
    choices = {}
    selected_rows = []
    for family in protocol["families"]:
        scores = []
        for j, (leaves, n) in enumerate(grid, 1):
            rows = [r for r in heldrows if r["family"] == family and r["candidate"] == j]
            assert len(rows) == len(fit_ids) and {r["profile"] for r in rows} == set(fit_ids)
            scores.append(float(np.mean([r["rmse"] for r in rows])))
        best = int(np.argmin(scores))
        choices[family] = grid[best]
        for k, ((leaves, n), score) in enumerate(zip(grid, scores)):
            selected_rows.append(dict(dataset=dataset, arm=arm, family=family, candidate=k+1,
                leaves=leaves, iterations=n, oof_macro_rmse=score, selected=int(k==best)))
    return heldrows, selected_rows, choices

def arm_run(args):
    ds, arm, data_root = args
    protocol = json.loads(PROTOCOL.read_text())
    spec = json.loads(STUDY.read_text())["datasets"][ds]
    grid = protocol[ds+"_grid"]
    with threadpool_limits(limits=1):
        raw = raw_records(data_root, ds, spec)
        # Only independent within-profile causal history, no learned normalizations here.
        records = {i: make_record(ds, r, arm) for i,r in raw.items()}
        del raw
        for i,r in records.items():
            if len(r["y"]) != spec["raw_rows_by_profile"][str(i)]:
                raise ValueError(f"Unexpected raw record length for {ds}/{i}")
            r.pop("ewma", None)
        heldrows, selections, choices = inner(ds, arm, records, spec, protocol, grid)
        x, y, w, scale, prior, p = fit_data(records, spec["roles"]["fit"], protocol["ridge_lambda"])
        profile_rows, cohorts = [], []
        for family, (leaves, n) in choices.items():
            residual = family == "linear_residual"
            model = HistGradientBoostingRegressor(**protocol["shared"], max_leaf_nodes=leaves,
                                                  max_iter=n).fit(x, y-p if residual else y, sample_weight=w)
            for role, ids in spec["roles"].items():
                if role == "fit":
                    continue
                assert not set(ids).intersection(spec["roles"]["fit"])
                role_rows = []
                for i in ids:
                    rec = records[i]
                    ix = evaluation_index(rec)
                    pred = model.predict(boost_features(rec, scale)[ix])
                    if residual:
                        pred += predict_prior(prior, rec)[ix]
                    row = profile_metrics(rec["y"][ix], pred)
                    role_rows.append(row)
                    profile_rows.append(dict(dataset=ds, arm=arm, role=role, family=family, profile=i, **row))
                cohorts.append(dict(dataset=ds, arm=arm, role=role, family=family, **aggregate_profiles(role_rows)))
        print(ds, arm, "refits and evaluation complete", flush=True)
        return heldrows, selections, profile_rows, cohorts

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT/"data/raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1,5), default=2,
                        help="Number of dataset/initialization processes (1-4); each uses one CPU thread")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new or empty output directory")
    proto = json.loads(PROTOCOL.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    start_clock, start = time.monotonic(), datetime.now(timezone.utc).isoformat()
    receipt = {"status":"protocol recorded before fitting","started_utc":start,
        "protocol_sha256":digest(PROTOCOL),"study_sha256":digest(STUDY),
        "driver_sha256":digest(Path(__file__)),"python":platform.python_version(),"numpy":np.__version__,
        "scipy":scipy.__version__,"sklearn":sklearn.__version__, "workers":args.workers, "threads_per_worker":1}
    dump(args.output/"run.json", receipt)
    receipt["pinned_raw_files_verified"] = [r["dataset"] for r in acquire(args.data_root, proto["datasets"])]
    jobs = [(d,a,args.data_root) for d in proto["datasets"] for a in proto["arms"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(arm_run,jobs))
    files = ["inner_profiles.csv","selection.csv","profiles.csv","cohorts.csv"]
    for k,name in enumerate(files):
        savecsv(args.output/name,[r for result in results for r in result[k]])
    receipt.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-start_clock,sha256={k:digest(args.output/k) for k in files})
    dump(args.output/"run.json", receipt)

if __name__ == "__main__":
    main()
