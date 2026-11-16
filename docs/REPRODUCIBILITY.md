# What this package reproduces

The supported portable operation is verification of delivered files and saved-result arithmetic. Packaging these materials did not generate a new model, prediction, timing measurement or independent test result.

## Verification commands

Use Python 3.10 or later from the repository root:

```bash
python -X utf8 -I -B scripts/verify_repository.py
python -X utf8 -I -B scripts/verify_results.py
```

These scripts use only Python's standard library. They do not install dependencies, contact dataset services, import the historical scientific workers, train models, deserialize trained predictors or benchmark the processor.

The repository check concerns delivered-file identities and provenance records. The result check concerns the hash-bound saved snapshots and arithmetic implemented by the utility: profile-macro and pooled error, table consistency, paired profile differences, wins/losses, descriptive single-profile deletion means and stored artifact-byte unions. A successful check means those checks passed for those saved inputs. It is not evidence that the original training or measurement was independently rerun.

| Level | Supported by this package |
|---|---|
| Inspect historical numerical source, settings and protocols | Yes |
| Verify delivered-file hashes and declared transformations | Yes |
| Recompute the implemented summaries from saved profile metrics | Yes |
| Independently regenerate predictions from trained models | No; full model/scaler/reference artifacts are not bundled |
| Retrain the complete current-study comparison from raw data | Not certified; no validated standalone training command is supplied |
| Reproduce the measured wall-clock latency on a new host | No; saved timing arithmetic and a new timing run are different tasks |
| Independently confirm generalization on unexposed data | No; all evaluated records were already exposed |

## Scientific interpretation

Evaluation is grouped by complete recorded profile. Profile-macro RMSE averages profile RMSEs equally; pooled RMSE weights retained rows before taking a square root. These measures can favor different predictors. The neural summary averages three seed metrics rather than their predictions. Single-profile deletion ranges are descriptive sensitivity summaries without refitting; they are not confidence intervals or cross-validation.

All retained cohorts, including adverse interior-motor outcomes, remain in the saved result tables. Names such as `confirmation` are historical labels preserved for traceability. They do not restore an untouched evaluation set. The focal linear residual predictor includes its fitted linear branch and residual booster; the averaging branch is assessed separately against that complete predictor.

Timing records concern scalar CPU execution under the documented implementation and environment. File-size totals count required stored files and shared-file unions, not process resident memory. Saved latency values must not be presented as new measurements on the machine running these verification commands.

## Why historical execution is not a quick-start

The historical workers retain their original source-hash pins, active-attempt requirements, output restrictions, admission checks and expired campaign deadlines. They also depend on original acquisition manifests, prepared feature arrays, model/scaler binaries, neural weights, prediction locks, reference arrays and stage receipts that are not fully distributed here. The final Linux timing worker expects an isolated package directory as well as exact recorded versions. Installing a few dependencies does not restore that execution context.

Do not disable guards or overwrite old evidence to make a historical command succeed. A future standalone training implementation would require a separately validated execution layer that preserves feature recurrences, initialization, preprocessing, weights, fitting order, profile partitions, endpoints, fixed control settings, seeds and the exposed-evaluation history. No such validation is claimed here.

The earlier portable constrained-prior experiment shares some numerical helpers but has a different primary model, experimental panel, neural seed selection and timing scope. It is not a substitute for current-study reproduction.

## External inputs and recorded environment

Raw data are not included. Historical configurations preserve acquisition identifiers and partitions for the [author-provided IPMSM source](https://github.com/Zirui24/lptn_informed_LSTM), the [Paderborn Electric Motor Temperature version record](https://doi.org/10.34740/KAGGLE/DSV/2161054), and the [RWTH drive-cycle resource](https://doi.org/10.5281/zenodo.15599966). The original source identities must be checked before any future regeneration. Data access and reuse terms are separate from this repository's verification scope.

The recorded final timing runtime uses Python 3.10.12, NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.5.1, joblib 1.4.2 and threadpoolctl 3.5.0 under Linux/WSL2. Neural training records PyTorch 2.7.1+cu126 and an RTX 4090. These are historical observations, not installation requirements for the standard-library verifiers or a guarantee of identical results. See [ENVIRONMENT.md](ENVIRONMENT.md).

## Integrity and privacy

All 100 historical files remain represented, including all 40 byte-identical Python sources. Location-only edits affect two configurations and five runtime records; private attachment metadata was removed from the artifact inventory. `provenance/file-lineage.json` separates original hashes from package hashes and describes changes. The original absolute machine locations and private attachments are not required by the public verifiers.

Do not convert line endings or edit pinned files in place while expecting their original identities to remain valid. Preserve original and derived artifacts separately, record changes explicitly, and rerun the package checks after any authorized packaging update.
