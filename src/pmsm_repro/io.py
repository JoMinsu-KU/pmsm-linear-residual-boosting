"""Hash-bound resumable stage outputs. All paths are relative to the run."""
from pathlib import Path
import hashlib,json,os,platform,sys,time,uuid
import numpy as np

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def dump_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def safe(root, name):
    root=Path(root).resolve();path=(root/name).resolve()
    if not path.is_relative_to(root) or path==root:raise ValueError("Path outside output root")
    return path

def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def verify_stage(folder, expected_key):
    folder=Path(folder)
    receipt=read_json(folder/"complete.json")
    if receipt["input_key"]!=expected_key:raise ValueError("Stage belongs to different inputs/settings: "+str(folder))
    names=[entry["path"] for entry in receipt["files"]]
    actual={p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p.name!="complete.json"}
    if len(names)!=len(set(names)) or set(names)!=actual:
        raise ValueError("Completed stage manifest must cover its exact file inventory")
    for entry in receipt["files"]:
        p=safe(folder,entry["path"])
        if p.is_symlink() or not p.is_file() or sha(p)!=entry["sha256"]:raise ValueError("Completed artifact changed: "+str(p))
    return receipt

def stage(folder, key, operation, check=lambda:None):
    folder=Path(folder)
    if folder.exists():
        verify_stage(folder,key)
        print(json.dumps({"stage":folder.name,"status":"reused_verified"}),flush=True)
        return folder
    check()
    attempt=folder.parent/(folder.name+".attempt-"+uuid.uuid4().hex[:12])
    attempt.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    try:
        operation(attempt)
        check()
        files=[{"path":p.relative_to(attempt).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size}
               for p in sorted(attempt.rglob("*")) if p.is_file()]
        dump_json(attempt/"complete.json",{"input_key":key,"elapsed_seconds":time.monotonic()-start,"files":files})
        if folder.exists():raise FileExistsError("Concurrent output creation refused")
        attempt.rename(folder)
        print(json.dumps({"stage":folder.name,"status":"completed","seconds":time.monotonic()-start}),flush=True)
    except BaseException as exc:
        dump_json(attempt/"failure.json",{"type":type(exc).__name__,"message":str(exc)})
        raise
    return folder

def key_of(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def runtime():
    import scipy,sklearn,joblib,threadpoolctl
    result={"python":platform.python_version(),"platform":platform.platform(),"machine":platform.machine(),
        "cpu":platform.processor(),"logical_cpus":os.cpu_count(),"numpy":np.__version__,
        "scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"joblib":joblib.__version__,
        "threadpoolctl":threadpoolctl.__version__,"threadpools":threadpoolctl.threadpool_info()}
    return result
