# Linear Residual Boosting for Rotor-Temperature Estimation in Permanent-Magnet Synchronous Motors

**Accuracy, Initialization, and Computational Cost**

Research companion, version 0.1.0 — November 16, 2026.

**Authors:** Minsu Jo, Namhyun Yoo, Beomseok Kim, Hoansuk Choi, and Jinhong Yang.

This package contains the historical implementation, saved experimental results, and four final research figures for a controlled comparison of linear residual boosting. A ridge predictor first estimates rotor temperature from causal operating-input histories; a histogram gradient-boosting model then learns its in-sample residuals. A separate experiment evaluates equal-weight averaging with a direct boosted predictor. The study also tests how input-filter initialization changes the refitted models.

The contribution is a conditional assessment of existing components under specified inputs, settings and computational measurements. It does not introduce a new boosting algorithm or identify physical thermal-network parameters.

## Main findings

With channel-wise boundary-filter initialization, linear residual boosting reduced Paderborn profile-macro RMSE by **26.61% on development** and **26.05% on historical evaluation**, relative to direct boosting with the same tree settings. Its measured median scalar CPU latency was **3.34% higher** than that matched direct comparator. The error advantage persisted against a larger direct-tree control.

These results have material limits. On the interior-motor historical evaluation, linear residual boosting instead increased error by **17.51%** against matched direct boosting. One neural-network seed also outperformed it on Paderborn development. Adding the separate averaging branch improved Paderborn historical-evaluation macro error by only **0.57%** beyond linear residual boosting, while worsening pooled and underestimation error and requiring **5.75 times** its median latency.

**All evaluation records were previously exposed.** The evidence is exploratory and conditional on the recorded fitting procedure. Historical filenames such as `confirmation` retain their original provenance meaning; they do not denote a fresh independent test in this study. The package reports no new experiments, universal superiority, embedded-device timing guarantee, or independent confirmation.

## Quick start: verify the saved evidence

Use Python **3.10 or later**. From the repository root, run:

```bash
python -X utf8 -I -B scripts/verify_repository.py
python -X utf8 -I -B scripts/verify_results.py
```

Both commands use only the Python standard library. No package installation, GPU, network access or dataset download is required. Read each command's reported status; an error is not a successful verification.

Repository verification checks the packaged files and their recorded identities. Result verification checks saved-file integrity and arithmetic such as profile-macro/pooled error, paired profile differences, descriptive single-profile deletions and artifact-byte unions. It **does not** retrain models, regenerate predictions, deserialize model binaries, or rerun timings. See [Reproducibility scope](docs/REPRODUCIBILITY.md) for the exact distinction.

## Contents

| Location | Contents |
|---|---|
| `scripts/` | Portable repository and saved-result verification utilities |
| `results/analysis/` | Saved evaluation, influence, condition and cost summaries |
| `results/tables/` | Four CSV tables covering all retained cohorts and comparisons |
| `results/reproducibility/` | Stored artifact-file accounting |
| `historical/campaign/` | All 100 historical source/configuration/protocol/runtime files |
| `provenance/file-lineage.json` | Original and delivered file hashes, with declared transformations |
| `figures/` | Four final PDF figures: method overview, accuracy/latency, profile errors and initialization effects |

All **40 historical Python sources are byte-identical** to their archived originals. All 100 historical files are retained: 40 Python sources, 37 configurations, 12 protocols and 11 runtime records. Absolute machine paths were sanitized in two configurations and five runtime records. Private review-attachment metadata was removed from the artifact inventory. Numerical result values were preserved. Original and package hashes identify these transformations explicitly; changed metadata files are not represented as unchanged originals.

The original source hierarchy and execution guards remain intact. The historical files depend on omitted raw inputs, prepared arrays, model/scaler binaries, locks, runtime components and campaign state. Their deadline and admission guards were not disabled. **This is not a certified standalone retraining package.** Do not treat historical training or timing entry points as the public quick-start. See the [source map](docs/SOURCE_MAP.md) and [recorded environments](docs/ENVIRONMENT.md).

## Data, figures and citation

Raw datasets are external and are not bundled. The [data guide](data/README.md), [source and rights attribution](docs/DATA_ATTRIBUTION.md), and [machine-readable registry](data/registry.json) identify the pinned IPMSM, Paderborn Electric Motor Temperature and RWTH drive-cycle inputs, expected sizes and hashes. Their reuse terms remain separate from this package.

Preview the acquisition plan for all datasets without downloading or writing files:

```bash
python -X utf8 -I -B scripts/acquire_data.py
```

Downloading requires an explicit option. For example, acquire and verify the pinned RWTH input:

```bash
python -X utf8 -I -B scripts/acquire_data.py --dataset rwth --download
```

To check already available RWTH files without network access or extraction:

```bash
python -X utf8 -I -B scripts/acquire_data.py --dataset rwth --verify-only
```

Acquisition checks the registry's pinned sizes and SHA-256 values and preserves existing files rather than overwriting them. Obtaining raw inputs does not provide the missing trained artifacts or turn the historical workers into a certified retraining pipeline. See the data guide for all dataset choices, local-directory use and storage requirements.

Full trained-model artifacts, private review attachments, manuscript author photographs and publisher template files are not included. The four final figure PDFs are included.

[CITATION.cff](CITATION.cff) supplies the five-author package citation. It does not claim a journal publication, acceptance or DOI. Read [RIGHTS.md](RIGHTS.md) before redistribution or reuse; no blanket open-source license is granted by this package.
