"""Single, resumable current-study pipeline; historical results never fill gaps."""
from pathlib import Path
import argparse,csv,importlib.util,importlib.metadata,json,os,platform,time
import numpy as np
from threadpoolctl import threadpool_limits
from . import data,features,models,neural,evaluation
from .io import read_json,dump_json,sha,key_of,stage,verify_stage,runtime

REPOSITORY=Path(__file__).resolve().parents[2]

def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)

class Run:
    def __init__(self,args):
        self.args=args;self.work=args.workdir.resolve();self.work.mkdir(parents=True,exist_ok=True)
        self.config=read_json(REPOSITORY/"configs/study.json")
        self.datasets=args.datasets;self.smoke=args.mode=="smoke";self.start=time.monotonic()
        self.lock=self.work/".active.lock"
        source_files=[p for folder in ("src","scripts","configs","data") for p in (REPOSITORY/folder).rglob("*")
                      if p.is_file() and p.suffix in (".py",".json") and "raw" not in p.relative_to(REPOSITORY).parts]
        numerical_runtime=runtime()
        version_identity={k:numerical_runtime[k] for k in ("python","platform","machine","numpy","scipy","scikit_learn","joblib","threadpoolctl")}
        for package in ("torch","matplotlib"):
            try:version_identity[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:version_identity[package]=None
        self.binding={"schema":"portable-study-run-v2","mode":args.mode,"datasets":self.datasets,
            "numerical_runtime":version_identity,
            "device":args.device,"files":{p.relative_to(REPOSITORY).as_posix():sha(p) for p in sorted(source_files)}}
        self.key=key_of(self.binding)
        if (self.work/"run.json").exists():
            saved=read_json(self.work/"run.json")
            if saved["key"]!=self.key:raise ValueError("Run inputs/code/settings changed; choose a new --workdir. Existing evidence remains intact.")
        else:
            dump_json(self.work/"run.json",{"key":self.key,**self.binding,
                "runtime":runtime(),"independent_confirmation":False})
        self.lock_fd=os.open(self.lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        os.write(self.lock_fd,json.dumps({"pid":os.getpid(),"key":self.key}).encode());os.close(self.lock_fd)
        self.last_storage_check=0.
    def close(self):
        if self.lock.exists() and read_json(self.lock)["pid"]==os.getpid():self.lock.unlink()
    def check(self):
        if time.monotonic()-self.start>self.args.max_hours*3600:raise TimeoutError("Invocation time limit; completed stages are resumable.")
        if time.monotonic()-self.last_storage_check>15:
            size=sum(p.stat().st_size for p in self.work.rglob("*") if p.is_file())
            self.last_storage_check=time.monotonic()
            if size>self.args.max_work_gb*1_000_000_000:raise RuntimeError("Work-directory storage limit reached; existing evidence preserved.")
    def stage(self,name,fn,extra=None):
        key=key_of({"run":self.key,"stage":name,"extra":extra})
        return stage(self.work/name,key,fn,self.check)
    def existing(self,name,extra=None):
        path=self.work/name
        verify_stage(path,key_of({"run":self.key,"stage":name,"extra":extra}))
        return path
    def prepare(self):
        receipts=data.acquire(self.args.data_root,self.datasets,download=self.args.download)
        for ds in self.datasets:
            spec=self.config["datasets"][ds]
            self.stage("cache/"+ds,lambda out,ds=ds,spec=spec:data.prepare(
                self.args.data_root,ds,spec,out,smoke=self.smoke,check=self.check))
        dump_json(self.work/"data-receipt.json",receipts)
    def records(self,ds):
        folder=self.existing("cache/"+ds)
        return data.load_raw_cache(folder)
    def train(self):
        if not self.smoke and not self.args.device.startswith("cuda"):
            raise ValueError("Full current-study MLP training requires --device cuda. Use --mode smoke for CPU integration tests.")
        # Check CUDA/dependencies before expensive fitting; no silent reduced run.
        neural._torch_runtime(self.args.device)
        for ds in self.datasets:
            raw,meta=self.records(ds);spec=self.config["datasets"][ds];fit_ids=spec["roles"]["fit"]
            for arm in spec["arms"]:
                self.check();records=data.make_arm(raw,meta,arm)
                settings=dict(spec["cpu"])
                if arm=="first_load":settings["fit_components"]=["selected_direct","residual"]
                def cpu_operation(out):
                    bundle=models.fit_cpu(records,fit_ids,ds,settings,smoke=self.smoke,check=self.check)
                    models.save_cpu(bundle,out/"cpu")
                cp=self.stage(f"models/{ds}/{arm}/cpu",cpu_operation)
                bundle=models.load_cpu(cp/"cpu")
                if arm=="first_load":continue
                indices={pid:features.train_index(records[pid]) for pid in fit_ids}
                x=np.concatenate([features.neural_features(records[p],bundle["scale"])[indices[p]] for p in fit_ids])
                y=np.concatenate([records[p]["target"][indices[p]] for p in fit_ids])
                profile_ids=np.concatenate([np.full(len(indices[p]),p,dtype=np.int64) for p in fit_ids])
                row_indices=np.concatenate([indices[p] for p in fit_ids])
                for seed in spec["neural"]["seeds"]:
                    self.check()
                    def neural_operation(out,seed=seed):
                        expected=None
                        if arm=="own_boundary":
                            paired=self.existing(f"models/{ds}/original/neural-{seed}")
                            expected=neural.load_neural(paired/"neural")["metadata"]["initial_array_sha256"]
                        artifact=neural.fit_neural(x,y,profile_ids,seed=int(seed),dataset=ds,
                            device=self.args.device,epochs=2 if self.smoke else 200,smoke=self.smoke,
                            rc_scale=bundle["scale"] if ds!="rwth" else None,
                            profile_order=fit_ids,row_indices=row_indices,check=self.check,
                            expected_initial_digest=expected)
                        neural.save_neural(artifact,out/"neural")
                    self.stage(f"models/{ds}/{arm}/neural-{seed}",neural_operation)
    def fitted(self,ds,arm):
        cp=self.existing(f"models/{ds}/{arm}/cpu")
        bundle=models.load_cpu(cp/"cpu")
        networks={}
        if arm!="first_load":
            for seed in self.config["datasets"][ds]["neural"]["seeds"]:
                path=self.existing(f"models/{ds}/{arm}/neural-{seed}")
                networks[int(seed)]=neural.load_neural(path/"neural")
        return bundle,networks
    def evaluate(self):
        for ds in self.datasets:
            raw,meta=self.records(ds);spec=self.config["datasets"][ds]
            for arm in spec["arms"]:
                self.check();records=data.make_arm(raw,meta,arm);bundle,networks=self.fitted(ds,arm)
                self.stage(f"evaluation/{ds}/{arm}",lambda out:evaluation.evaluate(
                    records,bundle,networks,spec["roles"],ds,arm,out,self.check,self.args.device))
    def benchmark(self):
        from .benchmark import benchmark
        for panel in self.config["timing"]["panels"]:
            ds,arm=panel["dataset"],panel["arm"]
            if ds not in self.datasets:continue
            raw,meta=self.records(ds);pid=int(panel["profile_id"])
            bundle,networks=self.fitted(ds,arm)
            record=features.make_record(ds,raw[pid],arm,meta["dt_ref"])
            evaluated=self.existing(f"evaluation/{ds}/{arm}")/panel["role"]/f"profile_{pid}.npz"
            self.stage(f"timing/{ds}/{arm}",lambda out:benchmark(raw[pid],record,bundle,
                networks[int(self.config["timing"]["neural_seed"])],panel,meta,
                evaluated,self.work/"models"/ds/arm,out,self.smoke,self.check))
    def report(self):
        from .reporting import generate_report,REQUIRED_SETTING_LABELS
        accuracy=[];profiles=[];timing=[];diagnostics=[];segments=[]
        for ds in self.datasets:
            for arm in self.config["datasets"][ds]["arms"]:
                document=read_json(self.existing(f"evaluation/{ds}/{arm}")/"evaluation.json")
                accuracy+=document["accuracy"];profiles+=document["profiles"]
                diagnostics+=document["diagnostics"];segments+=document["temporal_fifths"]
        for panel in self.config["timing"]["panels"]:
            if panel["dataset"] in self.datasets:
                timing+=read_json(self.existing(f'timing/{panel["dataset"]}/{panel["arm"]}')/"timing.json")["rows"]
        fitted_settings={}
        for ds in self.datasets:
            bundle,_=self.fitted(ds,"original")
            fitted_settings[ds]=bundle["metadata"]["configurations"]
        metadata={"execution_mode":"SMOKE TEST — reduced rows, trees and epochs; not paper results" if self.smoke else "fresh retraining and evaluation",
            "datasets":{ds:{"training_profiles":spec["roles"]["fit"],"total_profiles":sum(map(len,spec["roles"].values())),
                "warmup_exclusion":spec["warmup"]} for ds,spec in self.config["datasets"].items()},
            "expected_neural_seed_count":3,"timed_neural_seed":self.config["timing"]["neural_seed"],
            "model_settings":dict(zip(REQUIRED_SETTING_LABELS,[
                "33 features plus intercept; training-only standardization",
                "; ".join(ds+": "+str(self.config["datasets"][ds]["cpu"]["lambda"]) for ds in self.datasets),
                "; ".join(ds+": "+str(v["residual"]) for ds,v in fitted_settings.items()),
                "; ".join(ds+": "+str(v["matched_direct"]) for ds,v in fitted_settings.items()),
                "Fixed squared-error histogram boosting; no early stopping; see run settings JSON",
                "; ".join(ds+": "+str(v["residual"]["l2_regularization"]) for ds,v in fitted_settings.items()),
                "Equal weights of selected direct and complete linear residual predictors"])),
            "measured_runtime":runtime(),"run_key":self.key,"configuration":self.config,
            "timing_scope":"New local measurement; source/runtime-specific latency and file sizes are not copied from the manuscript."}
        def emit(out):
            from .analysis import summarize_diagnostics
            write_csv(out/"accuracy.csv",accuracy);write_csv(out/"profiles.csv",profiles);write_csv(out/"timing.csv",timing)
            write_csv(out/"temporal_fifths.csv",segments)
            dump_json(out/"diagnostics.json",diagnostics);dump_json(out/"metadata.json",metadata)
            dump_json(out/"diagnostic-summary.json",summarize_diagnostics(profiles,segments,diagnostics))
            generated=generate_report(out/"paper",accuracy,profiles,timing,metadata,mode="generated",
                                      strict=set(self.datasets)==set(self.config["datasets"]))
            dump_json(out/"report.json",generated)
        self.stage("report",emit)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["prepare","train","evaluate","benchmark","report","all"])
    p.add_argument("--workdir",type=Path,default=REPOSITORY/"work"/"paper")
    p.add_argument("--data-root",type=Path,default=REPOSITORY/"data"/"raw")
    p.add_argument("--datasets",nargs="+",choices=["ipmsm","paderborn","rwth"],default=["ipmsm","paderborn","rwth"])
    p.add_argument("--mode",choices=["paper","smoke"],default="paper")
    p.add_argument("--device",default="cuda",help="Explicit cuda or cuda:0 for paper; cpu permitted only for smoke.")
    p.add_argument("--download",action="store_true",help="Permit pinned raw-data acquisition during prepare/all.")
    p.add_argument("--max-hours",type=float,default=24.)
    p.add_argument("--max-work-gb",type=float,default=20.)
    return p

def main(argv=None):
    args=parser().parse_args(argv)
    if len(set(args.datasets))!=len(args.datasets):raise ValueError("Duplicate dataset selection")
    if not 0<args.max_hours<=120 or not 0<args.max_work_gb<=500:raise ValueError("Finite resource caps required")
    with threadpool_limits(limits=1):
        run=Run(args)
        try:
            steps=["prepare","train","evaluate","benchmark","report"] if args.stage=="all" else [args.stage]
            for name in steps:
                print(json.dumps({"phase":name,"mode":args.mode,"status":"starting"}),flush=True)
                getattr(run,name)()
            print(json.dumps({"status":"COMPLETE","mode":args.mode,"workdir":str(run.work),"run_key":run.key}),flush=True)
        finally:run.close()
    return 0
