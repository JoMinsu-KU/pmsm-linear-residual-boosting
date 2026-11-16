"""Fresh scalar CPU measurements, parity-qualified before the timer is read."""
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from .streaming import FeatureState,legacy_x
from .io import dump_json,sha,runtime

TERMS={
    "d9":[("selected_direct",1.)],
    "linear_prior":[("linear",1.)],
    "linear_residual":[("linear",1.),("residual",1.)],
    "linear_half":[("selected_direct",.5),("linear",.5),("residual",.5)],
    "direct200":[("matched_direct",1.)],
    "capacity_direct":[("capacity_direct",1.)],
    "free_residual":[("history",1.),("history_residual",1.)],
    "free_half":[("selected_direct",.5),("history",.5),("history_residual",.5)],
}

class Predictor:
    def __init__(self,bundle,network):
        self.bundle=bundle;self.network=network;self.scale=bundle["scale"]
        self.kind="rwth" if bundle["dataset"]=="rwth" else "legacy"
    def component(self,name,packet):
        if name=="neural":
            tr=self.network["transform"]
            x=legacy_x(packet,tr["rc_scale"]) if self.kind=="legacy" else packet
            x=((x-np.asarray(tr["input_mean"]))/np.asarray(tr["input_std"])).astype(np.float32)
            for layer in (0,2,4):
                x=self.network["weights"][str(layer)+".weight"]@x+self.network["weights"][str(layer)+".bias"]
                if layer!=4:x=np.maximum(x,np.float32(0.))
            return float(tr["y_mid"]+tr["y_scale"]*float(x[0]))
        if name in ("linear","history"):
            model=self.bundle["priors"][name];s=model["scale"]
            if self.kind=="legacy":
                if name=="linear":
                    x=(legacy_x(packet,s)-np.asarray(s["linear33_mean"]))/np.asarray(s["linear33_scale"])
                else:
                    x=np.concatenate(((packet[0]-s["y_mid"])/s["y_scale"],packet[1]/np.asarray(s["qscale"])))
                design=np.r_[1.,x]
            elif name=="linear":
                design=np.r_[1.,(packet-np.asarray(s["x_mean"]))/np.asarray(s["x_std"])]
            else:
                design=np.r_[(packet[8:13]-s["y_mid"])/s["y_scale"],
                             packet[13:33]/np.repeat(np.asarray(s["drive_rms"]),5)]
            return float(s["y_mid"]+s["y_scale"]*(design@np.asarray(model["coef"])))
        x=legacy_x(packet,self.scale) if self.kind=="legacy" else (packet-np.asarray(self.scale["x_mean"]))/np.asarray(self.scale["x_std"])
        return float(self.bundle["boosters"][name].predict(x[None,:])[0])
    def predict(self,terms,packet):
        value=0.
        for component,weight in terms:value+=weight*self.component(component,packet)
        return value

def endpoints(n,warmup,smoke):
    width=min(8 if smoke else 256,n-warmup)
    if width<=0:raise ValueError("No benchmark endpoints")
    starts=(warmup,warmup+max(0,(n-warmup-width)//2),n-width)
    return np.unique(np.concatenate([np.arange(s,s+width,dtype=np.int64) for s in starts]))

def benchmark(raw,record,bundle,network,panel,cache_meta,prediction_file,model_root,out,smoke=False,check=lambda:None):
    out=Path(out);ds=panel["dataset"];arm=panel["arm"];kind="rwth" if ds=="rwth" else "legacy"
    op=raw["operating"];timestamps=raw.get("time",np.arange(len(op),dtype=np.float64))
    ix=endpoints(len(op),0 if ds=="rwth" else 300,smoke)
    methods=dict(TERMS);seed=int(network["metadata"]["seed"])
    methods[f"mse_mlp_{seed}"]=[("neural",1.)]
    predictor=Predictor(bundle,network)
    state=FeatureState(kind,arm,cache_meta.get("dt_ref"))
    snapshots={};packets={};selected=set(ix.tolist())
    for row in range(int(ix[-1])+1):
        if row in selected:snapshots[row]=state.snapshot()
        packet=state.step(op[row],timestamps[row])
        if row in selected:packets[row]=tuple(v.copy() for v in packet) if kind=="legacy" else packet.copy()
        if row%2048==0:check()
    feature_errors={}
    for j,key in enumerate(["X33"] if kind=="rwth" else ["boundary","heat","instant"]):
        actual=np.asarray([packets[int(i)] if kind=="rwth" else packets[int(i)][j] for i in ix])
        expected=record[key][ix]
        if not np.allclose(actual,expected,atol=1e-8,rtol=2e-12):raise ArithmeticError("Streaming feature parity failed: "+key)
        feature_errors[key]=float(np.max(np.abs(actual-expected)))
    parity={}
    # Do not access the target array while measuring or qualifying predictions.
    with np.load(prediction_file,allow_pickle=False) as saved:
        index=saved["index"];locations=np.searchsorted(index,ix)
        if np.any(locations>=len(index)) or not np.array_equal(index[locations],ix):raise ValueError("Stored prediction endpoint alignment")
        for name,terms in methods.items():
            actual=np.asarray([predictor.predict(terms,packets[int(i)]) for i in ix])
            expected=saved["prediction__"+name][locations]
            tolerance=1e-3 if name.startswith("mse_mlp_") else 1e-7
            difference=float(np.max(np.abs(actual-expected)))
            if not np.allclose(actual,expected,rtol=0.,atol=tolerance):raise ArithmeticError("Scalar prediction parity failed: "+name)
            parity[name]={"max_absolute_difference_c":difference,"tolerance_c":tolerance}
    repetitions=1 if smoke else 5
    timings={name:{"resident_ns":[],"model_only_ns":[]} for name in methods}
    names=list(methods)
    with threadpool_limits(limits=1):
        for repeat in range(-1,repetitions):
            order=names[max(repeat,0)%len(names):]+names[:max(repeat,0)%len(names)]
            for name in order:
                state=FeatureState(kind,arm,cache_meta.get("dt_ref"));previous=None
                resident=[];model_only=[]
                for i in ix:
                    i=int(i)
                    if previous is None or i!=previous+1:state.restore(snapshots[i])
                    if repeat%2==0:
                        start=time.perf_counter_ns();second=predictor.predict(methods[name],packets[i]);stop=time.perf_counter_ns()
                    begin=time.perf_counter_ns();packet=state.step(op[i],timestamps[i]);value=predictor.predict(methods[name],packet);end=time.perf_counter_ns()
                    if repeat%2:
                        start=time.perf_counter_ns();second=predictor.predict(methods[name],packet);stop=time.perf_counter_ns()
                    if value!=second or not np.isfinite(value):raise ArithmeticError("Unstable timed prediction")
                    resident.append(end-begin);model_only.append(stop-start);previous=i
                if repeat>=0:
                    timings[name]["resident_ns"].append(resident);timings[name]["model_only_ns"].append(model_only)
                check()
    root=Path(model_root);cpu=root/"cpu"/"cpu";nn=root/f"neural-{seed}"/"neural"
    rows=[]
    for name,terms in methods.items():
        files=set();payload=0
        for component,_ in terms:
            if component=="neural":
                files.update([nn/"model.json",nn/"weights_final.npz"])
                payload+=sum(v.nbytes for v in network["weights"].values())
            else:
                files.add(cpu/"model.json")
                if component in ("linear","history"):payload+=np.asarray(bundle["priors"][component]["coef"]).nbytes
                else:
                    files.add(cpu/(component+".pkl"));model=bundle["boosters"][component]
                    payload+=model._baseline_prediction.nbytes+sum(tree.nodes.nbytes for row in model._predictors for tree in row)
        if any(not p.is_file() for p in files):raise FileNotFoundError("Required measured artifact missing")
        values=np.asarray(timings[name]["resident_ns"],dtype=np.float64)
        model_values=np.asarray(timings[name]["model_only_ns"],dtype=np.float64)
        rows.append({"dataset":ds,"arm":arm,"role":panel["role"],"method":name,"profile_id":panel["profile_id"],
            "median_us":float(np.median(values)/1000),"p95_us":float(np.quantile(values,.95)/1000),
            "model_only_median_us":float(np.median(model_values)/1000),
            "per_repetition_median_us":[float(np.median(v)/1000) for v in values],
            "artifact_bytes":sum(p.stat().st_size for p in files),"numeric_payload_bytes":int(payload),
            "artifact_files":[{"path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(files)],
            "feature_state_numeric_bytes":200,"endpoint_count":len(ix),"repetitions":repetitions,
            "smoke_only":smoke})
    np.savez_compressed(out/"timings.npz",index=ix,**{name+"__"+mode:np.asarray(v,dtype=np.int64)
                        for name,modes in timings.items() for mode,v in modes.items()})
    dump_json(out/"timing.json",{"rows":rows,"parity":parity,"feature_parity":feature_errors,
        "runtime":runtime(),"source":"Fresh local scalar timings; no historical latency values used.",
        "serialization_scope":"Logical prediction-component file union, excluding integrity receipts; not a self-contained deployment bundle. The generic loader verifies the complete training bundle. Sizes differ from historical packaging.",
        "resident_scope":"Operating row -> causal required features -> transforms -> predictors -> blend; excludes I/O, loading and prefix replay.",
        "state_initialization":"Untimed exact preceding-state replay at each disjoint sampled block.",
        "concurrency":"Sequential local pipeline; does not guarantee exclusivity against unrelated OS processes.",
        "target_values_read":False,"smoke_only":smoke})
