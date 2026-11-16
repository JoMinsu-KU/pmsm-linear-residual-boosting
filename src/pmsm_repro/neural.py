"""Current study MSE comparator, with explicit CUDA training and NumPy inference.

Inputs are sampled training neural features: legacy normalized-heat X33 or
RWTH raw X33. Feature extraction, endpoint selection and held-role evaluation
are separate operations. No historical module is imported.
"""
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np

SEEDS = (2026111701, 2026111702, 2026111703)
WEIGHT_SHAPES = {"0.weight": (64, 33), "0.bias": (64,), "2.weight": (32, 64),
                 "2.bias": (32,), "4.weight": (1, 32), "4.bias": (1,)}


def settings(dataset):
    if dataset not in ("ipmsm", "paderborn", "rwth"):
        raise ValueError("Unknown study dataset")
    return dict(architecture=[33, 64, 32, 1], parameters=4289, activation="ReLU",
                loss="mse", epochs=200, seeds=list(SEEDS), optimizer="AdamW",
                learning_rate=.001, betas=[.9, .999], eps=1e-8, weight_decay=.0001,
                batch_size=4096, inference_batch_size=4096, gradient_norm_cap=None if dataset == "rwth" else 1.,
                warmup=0 if dataset == "rwth" else 300, label_stride=5,
                feature_std_floor=1e-8, target_iqr_floor=1.,
                scheduler=None, early_stopping=False, checkpoint="last epoch 200",
                seed_aggregation="mean of individual seed metrics; no seed selection or prediction ensemble")


def array_digest(values):
    digest = hashlib.sha256()
    for key in sorted(values):
        value = np.ascontiguousarray(values[key])
        digest.update(json.dumps([key, value.dtype.str, list(value.shape)], separators=(",", ":")).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _features(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 33 or not np.isfinite(x).all():
        raise ValueError("Finite n-by-33 neural features required")
    return x


def training_transform(x, y, profiles, *, profile_order=None, rc_scale=None, dataset="ipmsm"):
    x = _features(x)
    y = np.asarray(y, dtype=np.float64)
    profiles = np.asarray(profiles)
    if len(x) == 0 or y.shape != (len(x),) or profiles.shape != y.shape or not np.isfinite(y).all():
        raise ValueError("Nonempty aligned finite training arrays required")
    if not np.issubdtype(profiles.dtype, np.integer):
        raise ValueError("Integer profile identities required")
    profiles = profiles.astype(np.int64)
    order = list(dict.fromkeys(profiles.tolist())) if profile_order is None else list(profile_order)
    if len(order) != len(set(order)) or set(order) != set(profiles.tolist()):
        raise ValueError("Profile order must cover training profiles exactly once")
    mean = x.mean(0)
    std = np.maximum(x.std(0), 1e-8)
    mid = float(np.median(y))
    scale = max(float(np.percentile(y, 75) - np.percentile(y, 25)), 1.)
    transform = dict(input_mean=mean.tolist(), input_std=std.tolist(), y_mid=mid,
                     y_scale=scale, fit_ids=order, warmup=settings(dataset)["warmup"], label_stride=5)
    if rc_scale is not None:
        transform["rc_scale"] = rc_scale
    weights = np.empty(len(y), dtype=np.float64)
    for pid in order:
        selected = profiles == pid
        weights[selected] = len(y) / (len(order) * int(selected.sum()))
    arrays = dict(x=((x-mean)/std).astype(np.float32),
                  y_normalized=((y-mid)/scale).astype(np.float32),
                  w=weights.astype(np.float32), profile_ids=profiles)
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Nonfinite transformed training values")
    return transform, arrays


def _torch_runtime(device):
    if not (str(device) == "cpu" or str(device).startswith("cuda")):
        raise ValueError("Device must be cpu or an explicit CUDA device")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is needed for training/torch inference; NumPy inference remains available") from exc
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Explicit CUDA requested but unavailable; no CPU fallback")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch


def new_model(seed, *, device="cpu"):
    torch = _torch_runtime(device)
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    model = torch.nn.Sequential(torch.nn.Linear(33,64,dtype=torch.float32), torch.nn.ReLU(),
                                torch.nn.Linear(64,32,dtype=torch.float32), torch.nn.ReLU(),
                                torch.nn.Linear(32,1,dtype=torch.float32)).to(device)
    if sum(p.numel() for p in model.parameters()) != 4289:
        raise AssertionError("Network parameter count differs")
    return model


def _weights(model):
    return {k: v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}


def _validate_weights(weights):
    if set(weights) != set(WEIGHT_SHAPES):
        raise ValueError("Exactly six affine weight/bias arrays required")
    for key, shape in WEIGHT_SHAPES.items():
        a = np.asarray(weights[key])
        if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError("Invalid FP32 weight array: " + key)


def fit_neural(x_fit, y_fit, profile_ids, *, seed, dataset, device="cuda", epochs=200,
               smoke=False, rc_scale=None, profile_order=None, row_indices=None,
               check=None, expected_initial_digest=None):
    """Fit fixed MSE recipe. CPU and shortened fits are explicitly smoke-only.

    check is an optional no-argument budget/cancellation callback, called before
    fitting, each epoch, each minibatch and after fitting. No held targets enter.
    """
    config = settings(dataset)
    if seed not in SEEDS:
        raise ValueError("Seed must be one of the three recorded study seeds")
    if smoke:
        if not 1 <= epochs <= 2:
            raise ValueError("Smoke fits are limited to one or two epochs")
    elif epochs != 200 or not str(device).startswith("cuda"):
        raise ValueError("Full study neural fitting requires explicit CUDA and exactly 200 epochs")
    if check:
        check()
    transform, arrays = training_transform(x_fit,y_fit,profile_ids,profile_order=profile_order,
                                            rc_scale=rc_scale,dataset=dataset)
    if row_indices is not None:
        idx = np.asarray(row_indices)
        if idx.shape != arrays["y_normalized"].shape or not np.issubdtype(idx.dtype,np.integer):
            raise ValueError("Aligned integer source-row indices required")
        idx = idx.astype(np.int64)
        if np.any(idx < config["warmup"]) or np.any((idx-config["warmup"]) % 5):
            raise ValueError("Training rows violate the dataset endpoint mask")
        arrays["index"] = idx
    model = new_model(seed,device=device)
    torch = _torch_runtime(device)
    initial = _weights(model)
    initial_digest = array_digest(initial)
    if expected_initial_digest is not None and initial_digest != expected_initial_digest:
        raise ValueError("Seeded initial weights differ from the paired reference")
    xt = torch.as_tensor(arrays["x"],device=device)
    yt = torch.as_tensor(arrays["y_normalized"],device=device)
    wt = torch.as_tensor(arrays["w"],device=device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
    trace = []
    for epoch in range(1,epochs+1):
        if check:
            check()
        start = time.monotonic()
        model.train()
        permutation = np.random.Generator(np.random.PCG64(seed+epoch-1)).permutation(len(yt)).astype(np.int64)
        total, max_grad = 0., 0.
        for lo in range(0,len(yt),4096):
            if check:
                check()
            batch = torch.as_tensor(permutation[lo:lo+4096],dtype=torch.int64,device=device)
            optimizer.zero_grad(set_to_none=True)
            error = model(xt[batch]).reshape(-1)-yt[batch]
            loss = (wt[batch]*error.square()).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite weighted MSE")
            loss.backward()
            if config["gradient_norm_cap"] is not None:
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            else:
                grad = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad,2)
                                                for p in model.parameters() if p.grad is not None]),2)
                if not torch.isfinite(grad):
                    raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step()
            total += float(loss.detach().cpu())*len(batch)
            max_grad = max(max_grad,float(grad.detach().cpu()))
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        trace.append(dict(epoch=epoch,online_weighted_loss=total/len(yt),max_gradient_norm=max_grad,
                          seconds=time.monotonic()-start,
                          permutation_sha256=array_digest(dict(order=permutation))))
    if check:
        check()
    final = _weights(model)
    _validate_weights(final)
    metadata = dict(schema="pmsm-study-neural-v1",dataset=dataset,seed=int(seed),epochs=int(epochs),
                    settings=config,smoke_only=bool(smoke),completed_paper_fit=not smoke,
                    fit_ids=transform["fit_ids"],training_rows=len(yt),
                    training_profile_counts={str(p):int(np.sum(arrays["profile_ids"]==p)) for p in transform["fit_ids"]},
                    initial_array_sha256=initial_digest,final_array_sha256=array_digest(final),
                    training_arrays_sha256=array_digest(arrays),held_targets_accessed=False,
                    runtime=dict(torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
                                 device=str(device),deterministic_algorithms=True,tf32=False,
                                 cublas_workspace_config=":4096:8"),
                    training_seconds=sum(t["seconds"] for t in trace))
    return dict(metadata=metadata,transform=transform,weights=final,initial_weights=initial,
                training_arrays=arrays,trace=trace)


def predict_scalar_numpy(artifact, row):
    """Recorded CPU scalar FP32 affine/ReLU path, output restored in FP64."""
    tr = artifact["transform"]
    x = (np.asarray(row,dtype=np.float64)-np.asarray(tr["input_mean"]))/np.asarray(tr["input_std"])
    if x.shape != (33,) or not np.isfinite(x).all():
        raise ValueError("Finite 33-component row required")
    x = x.astype(np.float32)
    for layer in (0,2,4):
        x = artifact["weights"][str(layer)+".weight"] @ x + artifact["weights"][str(layer)+".bias"]
        if layer != 4:
            x = np.maximum(x,np.float32(0.))
    return float(tr["y_mid"]+tr["y_scale"]*float(x[0]))


def predict_numpy(artifact, x, *, scalar=False):
    x = _features(x)
    _validate_weights(artifact["weights"])
    if scalar:
        return np.asarray([predict_scalar_numpy(artifact,row) for row in x],dtype=np.float64)
    tr = artifact["transform"]
    z = ((x-np.asarray(tr["input_mean"]))/np.asarray(tr["input_std"])).astype(np.float32)
    for layer in (0,2,4):
        z = z @ artifact["weights"][str(layer)+".weight"].T + artifact["weights"][str(layer)+".bias"]
        if layer != 4:
            z = np.maximum(z,np.float32(0.))
    return tr["y_mid"]+tr["y_scale"]*z.reshape(-1).astype(np.float64)


def predict_neural(artifact,x,*,device="cpu",batch_size=4096,backend="numpy"):
    if backend == "numpy":
        if device != "cpu":
            raise ValueError("NumPy inference is explicitly CPU-only; choose backend='torch' for CUDA")
        return predict_numpy(artifact,x)
    if backend != "torch" or batch_size < 1:
        raise ValueError("Unsupported inference backend or batch size")
    x = _features(x)
    _validate_weights(artifact["weights"])
    tr = artifact["transform"]
    z = ((x-np.asarray(tr["input_mean"]))/np.asarray(tr["input_std"])).astype(np.float32)
    model = new_model(artifact.get("metadata",{}).get("seed",SEEDS[0]),device=device)
    torch = _torch_runtime(device)
    model.load_state_dict({k:torch.as_tensor(v,device=device) for k,v in artifact["weights"].items()},strict=True)
    model.eval()
    outputs = []
    with torch.no_grad():
        for lo in range(0,len(z),batch_size):
            outputs.append(model(torch.as_tensor(z[lo:lo+batch_size],device=device)).reshape(-1).cpu().numpy().astype(np.float64))
    return tr["y_mid"]+tr["y_scale"]*(np.concatenate(outputs) if outputs else np.empty(0,dtype=np.float64))


def save_neural(artifact,directory):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    files = ["model.json","weights_final.npz","weights_initial.npz","training_arrays.npz","epoch_trace.json"]
    if any((directory/name).exists() for name in files):
        raise FileExistsError("Neural artifacts are never overwritten")
    _validate_weights(artifact["weights"])
    doc = dict(artifact["metadata"],transform=artifact["transform"])
    (directory/"model.json").write_text(json.dumps(doc,indent=2,allow_nan=False),encoding="utf-8")
    (directory/"epoch_trace.json").write_text(json.dumps(artifact.get("trace",[]),indent=2,allow_nan=False),encoding="utf-8")
    for name,key in [("weights_final.npz","weights"),("weights_initial.npz","initial_weights"),("training_arrays.npz","training_arrays")]:
        with (directory/name).open("xb") as handle:
            np.savez_compressed(handle,**artifact.get(key,{}))
    return directory


def load_neural(directory):
    directory = Path(directory)
    metadata = json.loads((directory/"model.json").read_text(encoding="utf-8"))
    with np.load(directory/"weights_final.npz",allow_pickle=False) as data:
        weights = {k:data[k].copy() for k in data.files}
    _validate_weights(weights)
    if metadata.get("final_array_sha256") is not None and array_digest(weights) != metadata["final_array_sha256"]:
        raise ValueError("Saved neural weight digest mismatch")
    return dict(metadata=metadata,transform=metadata["transform"],weights=weights,
                trace=json.loads((directory/"epoch_trace.json").read_text(encoding="utf-8")) if (directory/"epoch_trace.json").exists() else [])
