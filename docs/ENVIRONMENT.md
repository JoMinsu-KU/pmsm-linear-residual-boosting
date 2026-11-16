# Recorded environments and dependency limits

These observations come from the copied historical runtime JSON files. They are not measurements of the machine preparing this archive and are not an installation prescription.

| Path group beneath `campaign/linear-followup-20261111/runs/` | Recorded observations |
|---|---|
| Five `streaming-*-main-v003/runtime.json` files | Linux 5.15.167.4 under WSL2, glibc 2.35; Python 3.10.12; NumPy 1.26.4; isolated SciPy 1.15.3, scikit-learn 1.5.1, joblib 1.4.2, threadpoolctl 3.5.0 |
| Six `{ipmsm,paderborn}-own-controls-mlp-{2026111701,2026111702,2026111703}-v001/runtime.json` files | PyTorch 2.7.1+cu126; reported CUDA runtime 12.6; NumPy 1.26.4; NVIDIA GeForce RTX 4090, one device; TF32 disabled; deterministic algorithms enabled; one PyTorch CPU thread |

The five cost environment files are byte-identical. They record `x86_64` and 32 logical CPUs, not a per-run physical CPU model. OpenBLAS dispatch labels `Prescott` and `Haswell` are not CPU model identities. NumPy's OpenBLAS is recorded as 0.3.23.dev and SciPy's as 0.3.28; detected BLAS/OpenMP pools each have one thread. This is not a claim of exclusive-core scheduling or fixed affinity.

The timing clock is `clock_gettime(CLOCK_MONOTONIC)` through `perf_counter_ns`, monotonic and nonadjustable, with reported resolution 1 ns. Resolution does not establish measurement accuracy. Absolute machine roots in five runtime snapshots have been replaced with /recorded-workspace/campaign; version fields, library filenames, flags and numerical values are unchanged.

AST import inspection identifies six third-party top-level modules: `numpy`, `scipy`, `sklearn`, `joblib`, `threadpoolctl`, and `torch`. Their compiled libraries, CUDA driver/toolkit, Python runtime, and operating-system scheduler are not bundled. The PyTorch wheel suffix and runtime field are not evidence of a separately installed full CUDA toolkit. No `pip`, environment installation, or scientific import was performed by the archive verifier.

The neural cost path is NumPy FP32 evaluation of saved network weights; it is not the CUDA training path. The tree path uses the saved sklearn HGB model. Matching version strings alone does not certify numerical parity; the original measurement also required saved reference arrays and tolerances, which are not supplied in this text archive.
