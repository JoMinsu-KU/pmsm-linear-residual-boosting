# Linear Residual Boosting for Rotor-Temperature Estimation

Research code for **Linear Residual Boosting for Rotor-Temperature Estimation in Permanent-Magnet Synchronous Motors: Accuracy, Initialization, and Computational Cost**.

Version **0.5.0** accompanies manuscript v013. It adds the comparison of initialization-dependent model rankings and a separate equal-search-budget refitting experiment. The six saved figures and the original measurements remain in the [public v0.4.0 snapshot](https://github.com/JoMinsu-KU/pmsm-linear-residual-boosting/tree/5da91166ca91b0b7c12f9741c74cb3d541d0c893). The figure revision was v011 and the biography-only revision was v012. The archival v0.2.0 validation described below pertains to a reduced training run, not to these new analyses.

The full pipeline prepares pinned public measurements, fits the configured models, evaluates predictions, measures scalar CPU inference cost, and generates table and figure assets. Two additional scripts analyze residual-correction failures, operating ranges, and paired initialization effects. Results are computed from the named run's artifacts; missing values are never replaced with saved manuscript results. The original pipeline reruns the previously selected settings in [configs/study.json](configs/study.json); the v013 budget experiment below is a new, separately defined search.

All configured evaluation profiles have already been examined during the study. A fresh execution is a computational reproduction, **not independent confirmation**. Numerical identity across hardware and software environments is not assumed, and latency must be measured on the reader's machine.

## Reproduce the v013 analyses

Clone the tagged version used for manuscript v013:

```bash
git clone --branch v0.5.0 --depth 1 https://github.com/JoMinsu-KU/pmsm-linear-residual-boosting.git
cd pmsm-linear-residual-boosting
```

The checkout includes the original figure assets and the v013 experiment; no supplementary ZIP is required. From the repository root, verify the files and regenerate the fixed-setting comparison without downloading data or retraining:

```bash
python scripts/verify_repository.py
python scripts/analyze_initialization_contrast.py --check
python scripts/verify_equal_budget.py
python scripts/analyze_initialization_contrast.py --output work/v013-contrast
```

`cohorts.csv` gives the two models' errors and contrasts for both initializations, the paired change, the count of profile-level changes below zero, and the range after omitting one profile at a time. `profiles.csv` keeps every pair; `receipt.json` identifies the source file by hash. Negative change means that the residual model's relative position improved after channel-wise initialization **and refitting**. This is a descriptive comparison of the historical fixed settings; it is not evidence about an initialization switch with weights held fixed. For a newly executed full study, use `--source work/paper-failure/failure-profile-corrections.csv --output work/fresh-contrast` after the failure analysis instead.

The separate equal-budget experiment uses the [frozen search protocol](configs/equal_budget_v013.json). After installing the study dependencies and acquiring and verifying IPMSM and Paderborn as described below, run:

```bash
python scripts/run_equal_budget.py --data-root data/raw --output work/equal-budget
```

The [archived equal-budget outputs](reference/v013/equal-budget/) include inner-fold profile scores, family-specific selections, held-profile metrics, cohort summaries, and an execution receipt. The standard-library verifier above recomputes selections and aggregates, including all dataset/role contrasts. The search uses three profile-grouped folds within the training split, three candidate settings per family per arm, and a fixed residual-family ridge penalty of 0.001. This freezes the ridge penalty as well as the tree grid before the new fits; it is not a replay of the historical, dataset-specific ridge choices. Inspect `run.json` for the actual package versions. The later evaluation cohorts were already exposed during the earlier campaign. An equal number of candidate settings does not mean equal training time, tree count, or inference latency; timing from the fixed-setting experiment must not be assigned to newly selected models.

## Render the six saved v011 figures

This route uses the included figure data and needs no dataset download, model training, CUDA, or model weights. Use a dedicated Python 3.10 environment: `requirements-figures.txt` pins Matplotlib **3.9.2**, while the fresh-study `requirements.txt` pins **3.10.9**. Keep the two requirement sets in separate virtual environments. On Linux or WSL:

```bash
git clone https://github.com/JoMinsu-KU/pmsm-linear-residual-boosting.git
cd pmsm-linear-residual-boosting
python3.10 -m venv .venv-figures
source .venv-figures/bin/activate
python -m pip install -r requirements-figures.txt
python scripts/verify_v011_reference.py
python scripts/render_reference_v011.py --output work/reference-v011
```

For Windows PowerShell, replace the environment creation and activation lines with `py -3.10 -m venv .venv-figures` and `.\.venv-figures\Scripts\Activate.ps1`.

The renderer rebuilds all six manuscript figures from the saved sources in `reference/v011/data/`. The command writes PDF, PNG, and SVG figures to `work/reference-v011/figures/`, and JSON verification receipts to `work/reference-v011/review/`. The output directory must not already exist; choose a new `--output` path for each rendering. This is a rendering of historical observations, not a new fit, evaluation, or latency measurement. Use the archived files when the exact manuscript figure files are needed; rerendered bytes or pixels can vary with the rendering environment.

The v011 panel labels require a licensed installation of **Times New Roman**. No font files are bundled. If the renderer cannot find the font, pass the regular font file path explicitly. For example, in Windows PowerShell, use this command in place of the rendering command above:

```powershell
python scripts/render_reference_v011.py --output work/reference-v011 --font-path "C:\Windows\Fonts\times.ttf"
```

On Linux or WSL, install the font under its applicable terms or supply a licensed font file:

```bash
python scripts/render_reference_v011.py --output work/reference-v011 --font-path /path/to/licensed/times.ttf
```

The saved renderer also accepts the regular font file path through `TIMES_NEW_ROMAN_FONT`; an explicit `--font-path` takes precedence. It requires Times New Roman and fails if the font is unavailable. The standard-library-only `verify_v011_reference.py` checks the saved source hashes, data consistency, and manuscript figure mapping without a font or plotting dependencies.

Fresh study reports use the v011 line styles, markers, and panel labels. Their optional font override is `PMSM_TIMES_FONT`, and they use **DejaVu Serif** for panel labels if Times New Roman is unavailable. This fallback is recorded in the output and does not reproduce the manuscript typography.

## Install for a fresh study run

Use Linux or WSL with Python 3.10 and a separate `.venv` for the fresh-study dependencies. Full study training requires an NVIDIA GPU accessible through CUDA; the smoke run also supports CPU. Run the following from a terminal. If the repository is already cloned, start from its root and omit the first two lines:

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

Complete raw datasets are acquired separately; the included v011 plot records are insufficient for training or evaluation. [DATA_SOURCES.md](DATA_SOURCES.md) gives source versions, attribution, expected paths, and manual download instructions. Preview the pinned downloads before enabling network access:

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

Then generate the additional analyses used by v011 from that completed run:

```bash
python scripts/analyze_failure.py --run-dir work/paper --output work/paper-failure
python scripts/analyze_initialization.py --run-dir work/paper --output work/paper-initialization
```

Both analysis commands require `--run-dir` and `--output`. They verify the completed run's stage inventories and SHA-256 bindings and require IPMSM and Paderborn with their original and channel-wise initialization arms. The failure analysis also requires the completed report and cached input stages. Use new output directories **outside** `work/paper`; verified run artifacts must remain immutable. These scripts neither train models nor load historical results as a fallback.

For analysis integration checks, replace `work/paper` with a complete all-dataset smoke run and use new output paths, such as `work/smoke-failure` and `work/smoke-initialization`. Smoke labels remain in the outputs. A smoke run can have no qualifying adverse trajectory; the failure script then records the missing example instead of substituting a historical figure.

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

Each core pipeline invocation defaults to a **24-hour wall-time limit** and a **20 GB work-directory limit**, configurable with `--max-hours` and `--max-work-gb`. The supervisor enforces the time limit; storage is checked periodically and can exceed its threshold between checks. These limits cover generated work, not input data stored separately, and are limits rather than estimates of runtime or memory needs. Input downloads have separate pinned byte bounds.

To resume, rerun the same command. Completed stages are reused only after their exact file inventory and SHA-256 hashes pass verification. The run is bound to source/configuration hashes, dataset selection, mode, device, and recorded numerical runtime versions. A changed binding requires a new work directory. An interrupted stage starts again; unfinished neural epochs are **not** resumed. Failed attempt directories remain available for inspection and count toward the work-directory limit.

The analysis commands can read an already completed v0.2.0 run after verifying its original manifests. Updating the repository changes the core run binding, so use a new work directory for core training after upgrading; do not rewrite an older run's manifest to resume it.

## Generated artifacts

The selected work directory contains `run.json`, a data receipt, prepared caches, fitted models, per-profile predictions, evaluation diagnostics, timing receipts, and stage manifests. The final report is under `report/`:

| Output | Contents |
| --- | --- |
| `accuracy.csv`, `profiles.csv`, `timing.csv` | Newly computed aggregate metrics, profile metrics, and local timing measurements |
| `temporal_fifths.csv`, `diagnostics.json`, `metadata.json` | Within-profile segments, error decompositions, and run settings |
| `diagnostic-summary.json` | Equal-profile temporal contrasts and paired initialization associations, with Pearson and historical-compatible Spearman explicitly distinguished |
| `paper/tables/table1.csv` through `table7.csv`, with matching `.tex` files | Seven core tables; see the v011 numbering below |
| `paper/figures/figure1-method-overview`, `figure2-accuracy-latency`, `figure3-paired-profile-errors`, `figure4-initialization-effect` | Four core figures, each in PDF and PNG; numeric data in `figure2-data.csv` through `figure4-data.csv` |
| `paper/tables/S1` through `S5` | Five named supplementary CSVs covering accuracy, profiles, timing, initialization, and profile deletion |
| `paper/report-manifest.json` | Input digest, output hashes, provenance, and completeness status |

The output supplies table and figure assets; it does not build the manuscript PDF. Full reports require the configured evidence. Subset reports mark absent panels explicitly. The reporting API also accepts a caller-supplied `reference_snapshot` mode, which labels historical inputs; the normal pipeline always uses newly generated artifacts.

The additional commands write these outputs separately:

| Output directory | Contents |
| --- | --- |
| `work/paper-failure/` | `failure-profile-corrections.csv`, `failure-cohort-corrections.csv`, `failure-operating-ranges.csv`, `failure-profile-support.csv`, and JSON analysis records |
| `work/paper-failure/figures/` | `figure5-temperature-trajectories.pdf` and `.png`, plus locally generated plotted trajectory data, when qualifying records exist |
| `work/paper-initialization/analysis/` | `initialization-profiles.csv`, `initialization-paired.csv`, `initialization-summary.csv`, `initialization-whole-profile.csv`, `initialization-whole-summary.csv`, and JSON analysis records |
| `work/paper-initialization/figures/` | `figure6-initialization-segments.pdf` and `.png` |

The initialization analysis compares refitted prediction procedures; it does not estimate an unknown physical starting rotor temperature. Retained-row fifths divide rows, not matched physical durations. Operating ranges and exceedances are marginal descriptions, not causal explanations or multivariate out-of-distribution tests.

## v011 manuscript mapping

The manuscript has **nine main tables and six figures**. Established asset filenames are retained, so their numbers do not always match the manuscript. Paths below assume the commands above; core table files also have matching `.tex` versions and figures have matching `.png` versions.

| v011 table | Source and assembly |
| --- | --- |
| I — methodological precedents | Manually curated literature comparison; the core `report/paper/tables/table1.csv` is the earlier comparison, not the revised manuscript table |
| II — data and evaluation roles | `work/paper/report/paper/tables/table2.csv` |
| III — observed marginal ranges | Assemble the displayed ranges from `work/paper-failure/failure-operating-ranges.csv`; supporting exceedances are in `failure-profile-support.csv` |
| IV — earlier confirmation attempts | Manually assembled from [reference/v007/history/confirmations.json](reference/v007/history/confirmations.json), which preserves the earlier criteria and results; no new hypothesis test is performed |
| V — predictor structures | `work/paper/report/paper/tables/table3.csv` |
| VI — channel-wise initialization accuracy | `work/paper/report/paper/tables/table4.csv` |
| VII — error and CPU cost comparison | `work/paper/report/paper/tables/table5.csv` |
| VIII — feature-vector structure | `work/paper/report/paper/tables/table6.csv` |
| IX — fixed settings | `work/paper/report/paper/tables/table7.csv` |

| v011 figure | Fresh-run generated PDF |
| --- | --- |
| 1 — method overview | `work/paper/report/paper/figures/figure1-method-overview.pdf` |
| 2 — accuracy and latency | `work/paper/report/paper/figures/figure2-accuracy-latency.pdf` |
| 3 — paired profile errors | `work/paper/report/paper/figures/figure3-paired-profile-errors.pdf` |
| 4 — illustrative temperature trajectories | `work/paper-failure/figures/figure5-temperature-trajectories.pdf`, when qualifying records exist |
| 5 — aggregate initialization effect | `work/paper/report/paper/figures/figure4-initialization-effect.pdf` |
| 6 — within-profile initialization contrasts | `work/paper-initialization/figures/figure6-initialization-segments.pdf` |

The core report retains its earlier table captions and numbering. Use this mapping when assembling v011; the commands do not automatically produce its revised prose, literature table, or historical decision table. Newly reproduced values and selected illustrative profiles can differ from the archived manuscript observations. Fresh figures follow the v011 presentation conventions, but an independently executed run is not expected to reproduce the saved figure pixels.

## Checks and saved reference

Verify the listed release-file hashes and run the focused tests:

```bash
python scripts/verify_repository.py
python scripts/verify_v011_reference.py
python scripts/verify_review_reference.py
python scripts/analyze_initialization_contrast.py --check
python scripts/run_tests.py
```

The hash verifier uses only the Python standard library. It checks the files listed in `provenance/release-manifest.json`; raw inputs and generated outputs have separate manifests. The tests cover numerical behavior, artifacts, reporting, and pipeline checks. Three optional historical-parity tests require the authors' separate archive through `PMSM_HISTORICAL_ROOT` and are skipped when it is absent. That archive is not needed to execute the reproduction pipeline.

The `verify_v011_reference.py` command uses only the Python standard library. It checks the v011 reference hashes and figure mapping, recalculates saved trajectory RMSE/bias, and verifies the initialization difference grid. The separate `verify_review_reference.py` command also uses only the standard library and retains its historical scope: it verifies the v007 reference manifest and independently recalculates aggregate consistency checks from the saved summary CSVs. Neither command runs models or creates new evaluation evidence.

The v0.4.0 checks and their scope are recorded in [provenance/validation-v011.json](provenance/validation-v011.json). Earlier checks remain in the [v0.3.0 receipt](provenance/validation-v007.json). The [v0.2.0 validation receipt](provenance/validation.json) records **31 passing tests in WSL**, including the three optional archive checks, and an all-dataset smoke run with 15 neural fits of two epochs each. That run generated 193 aggregate accuracy rows, 1,440 profile-metric rows, 45 timing rows, seven core tables, and four core figures; all 38 completed stages were reused after hash checks during its resume test. These are historical v0.2.0 checks, not a new v0.4.0 full-training result. **That validation did not execute a full paper-scale rerun.** Smoke success verifies pipeline integration, not paper accuracy or timing.

The [v011 reference](reference/v011/) contains the final six figures in PDF, PNG, and SVG; saved JSON sources and per-profile summaries; 15,339 selected per-row trajectory records in `data/figure5-plotted-data.csv`; and 75 saved initialization plot cells in `data/initialization-summary.csv`. Its [manifest](reference/v011/manifest.json) records the file hashes and manuscript figure mapping. These data support figure inspection and rendering. They do not include the complete raw datasets or trained model weights. The trajectory records are selected derived plot data, with upstream attribution and applicable terms retained in [DATA_SOURCES.md](DATA_SOURCES.md) and [RIGHTS.md](RIGHTS.md). The package does not redistribute the full manuscript, author photographs, publisher assets, or private papers.

The earlier small CSV snapshot in `reference/` preserves saved core results independently of newly generated runs. The following **legacy renderer** produces its earlier core tables and four figures; it does **not** produce the complete v011 figure set:

```bash
python scripts/render_reference.py --output work/reference
```

`work/reference` is also the default output path and must not already exist. The command verifies the snapshot's hashes and marks outputs as `reference_snapshot`. It performs no new training, inference, or timing. The snapshot renderer was checked against 63 numerical cells in the manuscript's main accuracy and underestimation tables. These saved observations never fill gaps in a fresh run.

The additional [reference/v007/](reference/v007/) directory contains the archived full-run analysis summaries, publication PDF/PNG figures, quantitative history, and provenance records. Its [manifest](reference/v007/manifest.json) records the included files. These files describe the original exposed-profile study; they are not outputs of your reproduction. Figure 6's saved values are the cohort/method/fifth `macro_difference` values in `reference/v007/analysis/initialization-summary.csv`. Per-profile and paired CSVs support independent recalculation of those equal-profile means.

The v007 reference omits raw inputs, model binaries, per-row temperature/prediction trajectory CSVs, manuscript PDFs, author photographs, and exploration documents. That historical exclusion still describes `reference/v007/`; the new `reference/v011/` includes selected trajectory records for figure transparency. For fresh-run trajectory CSVs, use `analyze_failure.py` with locally acquired inputs. The saved publication figure contains only selected illustrative records, while the v007 summary CSVs retain all eligible profiles. Historical receipts can name hashes of the original analysis programs; those hashes document the versions checked at the time, while the release manifest and current validation receipt identify the integrated portable scripts.

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
