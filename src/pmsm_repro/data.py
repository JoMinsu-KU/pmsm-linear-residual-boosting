"""Pinned raw measurements -> unchanged row order -> causal per-profile caches."""
from pathlib import Path
import csv,importlib.util,io,json,zipfile
import numpy as np
from .io import arrays,dump_json,read_json,sha

REPOSITORY=Path(__file__).resolve().parents[2]

def acquisition_module():
    path=REPOSITORY/"scripts/acquire_data.py"
    spec=importlib.util.spec_from_file_location("pmsm_acquisition",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def acquire(data_root,datasets,download=False):
    module=acquisition_module();registry=read_json(REPOSITORY/"data/registry.json")
    receipts=[]
    for dataset in datasets:
        receipts.append(module.process_dataset(dataset,registry["datasets"][dataset],
                        Path(data_root)/dataset,"download" if download else "verify_only"))
    return receipts

def matrix(path, columns, exact_header=None):
    skiprows=1
    if isinstance(path,bytes):
        text=path.decode("utf-8-sig");header=next(csv.reader(io.StringIO(text)))
        source=io.StringIO(text)
    elif hasattr(path,"read"):
        first=path.readline()
        if isinstance(first,bytes):first=first.decode("utf-8-sig")
        header=next(csv.reader([first]));source=path;skiprows=0
    else:
        with Path(path).open(encoding="utf-8-sig",newline="") as stream:header=next(csv.reader(stream))
        source=Path(path)
    if exact_header and header!=exact_header:raise ValueError("Raw CSV header changed")
    if not set(columns)<=set(header):raise ValueError("Required sensor columns missing")
    indices=[header.index(c) for c in columns]
    values=np.loadtxt(source,delimiter=",",skiprows=skiprows,usecols=indices,dtype=np.float64,ndmin=2,encoding="utf-8-sig")
    if not np.isfinite(values).all():raise ValueError("Non-finite source values")
    return values

def raw_records(data_root,dataset,spec,check=lambda:None):
    root=Path(data_root)/dataset
    ids=[int(p) for role in spec["roles"].values() for p in role]
    if len(ids)!=len(set(ids)):raise ValueError("Dataset roles overlap")
    columns=spec["operating_columns"]
    target=spec["target_column"]
    records={}
    if dataset=="ipmsm":
        for pid in ids:
            check();v=matrix(root/f"id_{pid}.csv",columns+[target])
            records[pid]={"id":pid,"dataset":dataset,"operating":v[:,:7].copy(),"y":v[:,7].copy()}
    elif dataset=="paderborn":
        if (root/"measures_v2.csv").is_file():
            v=matrix(root/"measures_v2.csv",columns+[target,"profile_id"])
        else:
            with zipfile.ZipFile(root/"measures_v2.csv.zip") as z:
                with z.open("measures_v2.csv") as stream:
                    v=matrix(stream,columns+[target,"profile_id"])
        observed=np.unique(v[:,-1])
        if set(observed)!=set(ids):raise ValueError("Exact Paderborn profile inventory differs")
        for pid in ids:
            check();rows=v[v[:,-1]==pid]
            records[pid]={"id":pid,"dataset":dataset,"operating":rows[:,:7].copy(),"y":rows[:,7].copy()}
    elif dataset=="rwth":
        archive=zipfile.ZipFile(root/"DriveCycles.zip") if (root/"DriveCycles.zip").exists() else None
        try:
            for pid in ids:
                check();name=f"final/{pid:03d}.csv"
                source=archive.read(name) if archive else root/name
                v=matrix(source,["time",*columns,target,"profile_id"],
                         ["time","i","torque","motor_speed","coolant","rotor","stator","profile_id"])
                if not np.all(v[:,-1]==pid) or not np.all(np.diff(v[:,0])>0):raise ValueError("RWTH profile/time order changed")
                records[pid]={"id":pid,"dataset":dataset,"operating":v[:,1:5].copy(),"y":v[:,5].copy(),"time":v[:,0].copy()}
        finally:
            if archive:archive.close()
    else:raise ValueError(dataset)
    return records

def prepare(data_root,dataset,spec,out,smoke=False,check=lambda:None):
    out=Path(out);records=raw_records(data_root,dataset,spec,check)
    # The physical time scale is estimated from the complete FIT histories only,
    # even when the smoke path subsequently truncates rows.
    dt_ref=float(np.median(np.concatenate([np.diff(records[int(i)]["time"]) for i in spec["roles"]["fit"]]))) if dataset=="rwth" else None
    descriptions=[]
    for pid,raw in records.items():
        check();n=len(raw["y"])
        stop=min(n,700 if dataset!="rwth" else 256) if smoke else n
        values={k:v[:stop] for k,v in raw.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(out/f"profile_{pid}.npz",**values)
        descriptions.append({"id":pid,"source_rows":n,"retained_raw_rows":stop,"path":f"profile_{pid}.npz"})
    dump_json(out/"dataset.json",{"dataset":dataset,"roles":spec["roles"],"dt_ref":dt_ref,
        "smoke":smoke,"profiles":descriptions,"raw_data_identity":"verified against registry before preparation",
        "evaluation_exposure":"Previously exposed historical records; not independent confirmation."})

def load_raw_cache(folder):
    folder=Path(folder);metadata=read_json(folder/"dataset.json")
    records={}
    for item in metadata["profiles"]:
        pid=int(item["id"]);v=arrays(folder/item["path"])
        records[pid]={"id":pid,"dataset":metadata["dataset"],**v}
    return records,metadata

def make_arm(raw,metadata,arm):
    from .features import make_record
    return {pid:make_record(metadata["dataset"],r,arm=arm,dt_ref=metadata["dt_ref"]) for pid,r in raw.items()}
