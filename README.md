# Linear Residual Boosting for Rotor-Temperature Estimation

Research code for **Linear Residual Boosting for Rotor-Temperature Estimation in Permanent-Magnet Synchronous Motors: Accuracy, Initialization, and Computational Cost**.

Version 0.2.0 provides a portable pipeline that prepares pinned public measurements, fits the configured models, evaluates their predictions, measures scalar CPU inference cost, and generates the manuscript's tables and figures. Results are computed from that run's artifacts; missing values are never replaced with saved manuscript results. This release reruns the frozen, previously selected settings in [configs/study.json](configs/study.json). It does not repeat the original hyperparameter search.

All configured evaluation profiles have already been examined during the study. A fresh execution is a computational reproduction, **not independent confirmation**. Numerical identity across hardware and software environments is not assumed, and latency must be measured on the reader's machine.

## Install

Use Linux or WSL with Python 3.10. Full study training requires an NVIDIA GPU accessible through CUDA; the smoke run also supports CPU. Run the following from a terminal:

```bash
git clone https://github.com/JoMinsu-KU/pmsm-linear-residual-boosting.git
cd pmsm-linear-residual-boosting
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-cuda.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The pinned CUDA requirements install PyTorch 2.7.1 from the CUDA 12.6 wheel index documented by [PyTorch](https://pytorch.org/get-started/previous-versions/#v271). The other direct dependencies are pinned in `requirements.txt`. A working NVIDIA driver is required for CUDA execution. For a CPU-only smoke environment, replace the CUDA requirements installation with:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

## Obtain the data

Raw measurements are not included. [DATA_SOURCES.md](DATA_SOURCES.md) gives source versions, attribution, expected paths, and manual download instructions. Preview the pinned downloads before enabling network access:

```bash
python scripts/acquire_data.py --dataset all
python scripts/acquire_data.py --dataset all --download
python scripts/acquire_data.py --dataset all --verify-only
```

The default destination is `data/raw/`. Every required file is checked against the byte count and SHA-256 in [data/registry.json](data/registry.json). The default acquisition command is a dry run; network access requires `--download`. Existing mismatched files are preserved and cause an error.

## Run the study

First exercise the pipeline with reduced rows, trees, and neural epochs:

```bash
python scripts/reproduce.py all --mode smoke --device cpu --workdir work/smoke
```

The smoke run uses the actual verified data and all configured profile splits, but truncates each profile, reduces boosting to three iterations, and trains each neural seed for two epochs. Its figures and tables are marked as smoke results and must not be used as paper results. For a smaller integration check, add `--datasets ipmsm`; omitted datasets remain explicitly unavailable in the report.

Run the complete frozen study on CUDA:

```bash
python scripts/reproduce.py all --mode paper --device cuda --workdir work/paper
```

Add `--download` to `all` or `prepare` to allow acquisition during preparation. To use another input root, pass `--data-root /path/to/raw`; it must contain the three dataset subdirectories described in `DATA_SOURCES.md`. Keep input data outside the work directory.

The five stages can also be invoked separately, in this order, using identical mode, dataset selection, device, and work directory:

```bash
python scripts/reproduce.py prepare --mode paper --device cuda --workdir work/paper
python scripts/reproduce.py train --mode paper --device cuda --workdir work/paper
python scripts/reproduce.py evaluate --mode paper --device cuda --workdir work/paper
python scripts/reproduce.py benchmark --mode paper --device cuda --workdir work/paper
python scripts/reproduce.py report --mode paper --device cuda --workdir work/paper
```

An individual stage requires completed artifacts from its prerequisites. `all` runs them in sequence. No model or saved-result download is required.

Each invocation defaults to a **24-hour wall-time limit** and a **20 GB work-directory limit**, configurable with `--max-hours` and `--max-work-gb`. The supervisor enforces the time limit; storage is checked periodically and can exceed its threshold between checks. These limits cover generated work, not input data stored separately, and are limits rather than estimates of runtime or memory needs. Input downloads have separate pinned byte bounds.

To resume, rerun the same command. Completed stages are reused only after their exact file inventory and SHA-256 hashes pass verification. The run is bound to source/configuration hashes, dataset selection, mode, device, and recorded numerical runtime versions. A changed binding requires a new work directory. An interrupted stage starts again; unfinished neural epochs are **not** resumed. Failed attempt directories remain available for inspection and count toward the work-directory limit.

## Generated artifacts

The selected work directory contains `run.json`, a data receipt, prepared caches, fitted models, per-profile predictions, evaluation diagnostics, timing receipts, and stage manifests. The final report is under `report/`:

| Output | Contents |
| --- | --- |
| `accuracy.csv`, `profiles.csv`, `timing.csv` | Newly computed aggregate metrics, profile metrics, and local timing measurements |
| `temporal_fifths.csv`, `diagnostics.json`, `metadata.json` | Within-profile segments, error decompositions, and run settings |
| `diagnostic-summary.json` | Equal-profile temporal contrasts and paired initialization associations, with Pearson and historical-compatible Spearman explicitly distinguished |
| `paper/tables/table1` through `table7` | Seven main tables, each in CSV and LaTeX |
| `paper/figures/figure1` through `figure4` | Four named figures, each in PDF and PNG; plotted numeric data for Figures 2–4 |
| `paper/tables/S1` through `S5` | Five named supplementary CSVs covering accuracy, profiles, timing, initialization, and profile deletion |
| `paper/report-manifest.json` | Input digest, output hashes, provenance, and completeness status |

The output supplies table and figure assets; it does not build the manuscript PDF. Full reports require the configured evidence. Subset reports mark absent panels explicitly. The reporting API also accepts a caller-supplied `reference_snapshot` mode, which labels historical inputs; the normal pipeline always uses newly generated artifacts.

## Checks and saved reference

Verify the listed release-file hashes and run the focused tests:

```bash
python scripts/verify_repository.py
python scripts/run_tests.py
```

The hash verifier uses only the Python standard library. It checks the files listed in `provenance/release-manifest.json`; raw inputs and generated outputs have separate manifests. The tests cover numerical behavior, artifacts, reporting, and pipeline checks. Three optional historical-parity tests require the authors' separate archive through `PMSM_HISTORICAL_ROOT` and are skipped when it is absent. That archive is not needed to execute the reproduction pipeline.

During package validation, all **31 tests passed in WSL**, including the three optional archive checks. An end-to-end smoke run completed all three datasets and configured initialization arms, including 15 neural fits of two epochs each. It generated 193 aggregate accuracy rows, 1,440 profile-metric rows, 45 timing rows, seven main tables, and four figures. **A full paper-scale rerun was not executed in this package validation.** Smoke success verifies pipeline integration, not paper accuracy or timing.

The small CSV snapshot in `reference/` preserves the saved manuscript results independently of newly generated runs. To reproduce the corresponding table and figure assets without data acquisition or model execution:

```bash
python scripts/render_reference.py --output work/reference
```

`work/reference` is also the default output path and must not already exist. The command verifies the snapshot's hashes and marks outputs as `reference_snapshot`. It performs no new training, inference, or timing. The snapshot renderer was checked against 63 numerical cells in the manuscript's main accuracy and underestimation tables. These saved observations never fill gaps in a fresh run.

## Scientific protocol

The target is contemporaneous rotor temperature. A 33-feature causal representation combines operating variables with recursive boundary and heating histories. Each profile resets its filter states. The full-feature ridge predictor is fitted on training profiles, and a histogram gradient booster learns its training residuals. An equal-weight ensemble averages this completed predictor with the previously selected direct booster. Controls include ridge alone, direct boosting with matched or expanded settings, history-only priors, and a feedforward neural network. Models are fitted separately for each dataset; no cross-dataset transfer is claimed.

| Dataset | Training profiles | Other configured profiles | Initial samples excluded |
| --- | ---: | --- | ---: |
| IPMSM | 8 | 3 calibration, 3 pilot, 2 historical evaluation | 300 per profile |
| Paderborn | 41 | 14 development, 14 historical evaluation | 300 per profile |
| RWTH | 20 | 8 calibration, 10 historical evaluation | 0 |

Exact profile IDs, row strides, feature ordering, initialization interventions, selected tree settings, and neural seeds are in `configs/study.json`. Training statistics use training profiles only. IPMSM and Paderborn compare original boundary initialization with channel-wise initialization and a separate first-observation heating intervention. RWTH uses its configured original initialization and timestamp-adjusted filters.

The neural control has architecture 33–64–32–1 with ReLU, AdamW, three fixed seeds, and 200 epochs per seed in paper mode. It uses the final epoch without early stopping. Reported neural summaries average individual seed metrics, not predictions. The timed neural model is one specified seed.

The primary accuracy measure is the unweighted mean of profile RMSEs; pooled RMSE weights all retained endpoints. Underestimation, overestimation, paired profile differences, initialization effects, and descriptive profile-deletion results retain adverse outcomes. Profile deletion does not retrain a model or establish statistical significance.

Timing measures resident-input scalar CPU filter updates, feature preparation, normalization, and prediction. It excludes disk I/O and GPU training. Selected first/middle/final endpoints, warmup, repetitions, and profile IDs are fixed in the configuration. Do not run training or other substantial workloads concurrently with benchmarking. Measured latency and serialized model size are environment-dependent; they are not embedded-device or worst-case execution-time guarantees.

Generated byte counts are the logical union of metadata and weight files used by each prediction path. They exclude integrity receipts and are not a self-contained deployment bundle: the general model loader verifies the complete training bundle. This portable serialization differs from the historical file arrangement; original byte counts remain available only in the explicitly labeled reference snapshot.

The study evaluates a particular sensor budget and previously exposed public profiles. It does not establish universal model superiority, unseen-machine generalization, a new residual-learning theorem, or deployment safety.

## Citation and rights

Use [CITATION.cff](CITATION.cff) for this versioned research package. The citation does not assert journal acceptance or an assigned article DOI. [RIGHTS.md](RIGHTS.md) describes the project's current licensing status and distinguishes external dataset terms from code rights.
