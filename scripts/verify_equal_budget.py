"""Independently audit v013 training-only selection and paired cohorts (standard library)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

def require(cond, label):
    if not cond:
        raise ValueError(label)

def close(a, b):
    return math.isclose(float(a), float(b), abs_tol=1e-10, rel_tol=1e-11)

def check(folder):
    def read(name):
        with (folder/name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    protocol_file = ROOT/"configs/equal_budget_v013.json"
    study_file = ROOT/"configs/study.json"
    protocol = json.loads(protocol_file.read_text())
    study = json.loads(study_file.read_text())
    receipt = json.loads((folder/"run.json").read_text())
    require(receipt["status"] == "complete" and set(receipt["pinned_raw_files_verified"]) == set(protocol["datasets"]), "incomplete or unverified run")
    for key,path in [("protocol_sha256",protocol_file),("study_sha256",study_file),("driver_sha256",ROOT/"scripts/run_equal_budget.py")]:
        require(receipt[key] == digest(path), "input hash: " + key)
    names = ["inner_profiles.csv","selection.csv","profiles.csv","cohorts.csv"]
    require(set(receipt["sha256"]) == set(names), "receipt inventory")
    for name in names:
        require(digest(folder/name) == receipt["sha256"][name], name)
    inner, sel, profiles, cohorts = [read(n) for n in names]
    def rows(data, **keys):
        return [r for r in data if all(r[k] == str(v) for k,v in keys.items())]
    total = [0]*4
    summary = []
    for ds in protocol["datasets"]:
        cfg = study["datasets"][ds]
        roles = cfg["roles"]
        fit = sorted(roles["fit"])
        grid = protocol[ds+"_grid"]
        for arm in protocol["arms"]:
            for family in protocol["families"]:
                criteria = dict(dataset=ds, arm=arm, family=family)
                scores, winners = [], []
                for j,(leaves,iters) in enumerate(grid,1):
                    keys = dict(**criteria, candidate=j, leaves=leaves, iterations=iters)
                    oof = rows(inner,**keys)
                    require(sorted(int(r["profile"]) for r in oof)==fit,"OOF inventory")
                    for r in oof:
                        pid = int(r["profile"])
                        require(int(r["fold"]) == 1+fit.index(pid)%3, "OOF group")
                        require(int(r["rows"]) == len(range(300,cfg["raw_rows_by_profile"][str(pid)],5)), "OOF endpoints")
                    selection = rows(sel, **keys)
                    require(len(selection)==1, "candidate missing/duplicated")
                    score = mean(float(r["rmse"]) for r in oof)
                    require(close(score, selection[0]["oof_macro_rmse"]), "OOF score")
                    scores.append(score)
                    if selection[0]["selected"] == "1":
                        winners.append(j)
                    total[0]+=len(oof)
                    total[1]+=1
                require(winners == [scores.index(min(scores))+1], "selected candidate not first OOF minimizer")
                for role,ids in roles.items():
                    if role=="fit":
                        continue
                    group = rows(profiles, **criteria, role=role)
                    as_rows = rows(cohorts,**criteria,role=role)
                    require(len(as_rows)==1, "cohort missing/duplicated")
                    aggregate = as_rows[0]
                    require(sorted(int(r["profile"]) for r in group)==sorted(ids), "held IDs")
                    for r in group:
                        require(int(r["n"])==cfg["raw_rows_by_profile"][str(r["profile"])]-300, "evaluation endpoints")
                        require(close(float(r["rmse"])**2, r["mse"]), "rmse/mse")
                    n = sum(int(r["n"]) for r in group)
                    require(n==int(aggregate["n"]) and len(ids)==int(aggregate["profiles"]), "n")
                    for key in ["rmse","mae","bias","under_rmse","over_mae","max_under","max_over","max_absolute_error"]:
                        require(close(mean(float(r[key]) for r in group),aggregate["macro_"+key]), key)
                    require(close(math.sqrt(sum(int(r["n"])*float(r["mse"]) for r in group)/n),aggregate["pooled_rmse"]), "pooled")
                    total[2]+=len(group)
                    total[3]+=1
            for role,ids in roles.items():
                if role == "fit":
                    continue
                a,b = [rows(cohorts,dataset=ds,arm=arm,family=f,role=role)[0] for f in protocol["families"]]
                q = [{int(r["profile"]):float(r["rmse"]) for r in rows(profiles,dataset=ds,arm=arm,family=f,role=role)} for f in protocol["families"]]
                deltas = [q[1][i]-q[0][i] for i in ids]
                summary.append(dict(dataset=ds,arm=arm,role=role,n=len(ids),
                    direct=float(a["macro_rmse"]),residual=float(b["macro_rmse"]),
                    delta=mean(deltas),improvement_pct=-100*mean(deltas)/float(a["macro_rmse"]),
                    residual_wins=sum(x<0 for x in deltas), ties=sum(x==0 for x in deltas),
                    pooled_delta=float(b["pooled_rmse"])-float(a["pooled_rmse"]),
                    macro_under_delta=float(b["macro_under_rmse"])-float(a["macro_under_rmse"])))
    require(total==list(map(len,[inner,sel,profiles,cohorts])), "extra or missing records")
    return {"checks":"pass","records":dict(zip(names,total)),"contrasts":summary}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=ROOT/"reference/v013/equal-budget")
    print(json.dumps(check(p.parse_args().input),indent=2,allow_nan=False))
