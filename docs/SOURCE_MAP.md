# Historical source and evidence map

The historical tree contains all 100 selected files from the recorded research campaign: 40 Python sources, 37 configurations, 12 protocols and 11 runtime records. Dated directory names, source versions and abbreviated result keys are retained as provenance identifiers, not proposed scientific method names.

## Keep the source hierarchy intact

```text
historical/campaign/
  code/
  configs/
  protocols/
  followup-risk-20261110/
    code/
    configs/
    protocols/
  linear-followup-20261111/
    code/
    configs/
    protocols/
    runs/*/runtime.json
```

Follow-up modules derive their campaign root from `Path(__file__).resolve().parents[2]`. Dynamic imports use exact relative paths and source hashes. The shared numerical core loads its sibling helper by filename and verifies that helper's hash. Flattening folders, renaming files or automatically converting line endings would break these identities. All 40 historical Python files preserve their original bytes.

## Reading order

Paths in this table are relative to `historical/campaign/`. They are a reading map, not instructions to execute expired stages.

| Research component | Main source and supporting files |
|---|---|
| Causal feature construction and linear/direct fitting | `code/combination_c12_fit_rc_v001.py`, `code/c15_paper_core_v001.py`, `code/c15_paper_suite_v001.py` |
| Filter-initialization refitting | `linear-followup-20261111/code/initialization_intervention_v002.py`, `linear_diagnostic_v002.py` |
| Matched-tree, expanded-tree, history-only and neural controls | `linear-followup-20261111/code/c18_own_controls_v001.py`; shared neural helpers in `code/` and `followup-risk-20261110/code/risk_mlp_v001.py` |
| Supporting RWTH representation | `followup-risk-20261110/code/rwth_prepare_v002.py`, `rwth_cpu_core_v002.py`, `rwth_cpu_fit_v002.py`, `rwth_cpu_predict_v002.py`, `rwth_mlp_v002.py` |
| Reported scalar timing implementation | `linear-followup-20261111/code/c18_streaming_cost_v003.py` and five `configs/streaming-*-main-v003.json` panels |
| Historical independent checks | The corresponding `*_audit_*.py` files and frozen protocol documents |

Earlier source versions are retained where they belong to the historical dependency chain. Their presence does not make them the reported final timing implementation. Source availability also does not imply that every historical comparison or candidate is a contribution of the current paper.

## Result files and terminology

| File | Saved content |
|---|---|
| `results/tables/S1-own-accuracy-all-methods.csv` | 70 aggregate rows: five cohorts and 14 method/seed entries per cohort |
| `results/tables/S2-own-profile-all-methods.csv` | 504 rows: 36 retained profiles and 14 method/seed entries |
| `results/tables/S3-all-streaming-cost.csv` | 45 method/panel rows of stored scalar timing and file-size summaries |
| `results/tables/S4-original-initialization-all-cohorts.csv` | 51 rows under the original initialization procedures |
| `results/analysis/` | Underlying saved evaluation, condition, cost and paired-profile influence summaries |
| `results/reproducibility/deployment-artifacts-v001.json` | Required-file unions and stored byte accounting |

Selected internal keys map to these scientific terms:

| Internal key | Meaning |
|---|---|
| `linear_residual` / `L` | Full-feature linear residual boosting |
| `linear_half` / `H` | Equal-weight average of the complete linear residual predictor and selected direct predictor |
| `d9` / `D9` / `D` | Previously selected direct boosted predictor |
| `direct200` | Direct boosting with matched tree settings |
| `capacity_direct` | Expanded direct-tree control |
| `free_residual` | History-only residual construction |
| `free_half` | History-only residual/direct average |
| `own_boundary` / `own` | Channel-wise boundary-filter initialization |
| `first_load` | First-observation heating-filter initialization intervention |

The three neural seed metrics are separate saved entries. Their arithmetic mean is a summary of seed metrics, not a further trained or deployed ensemble.

## Provenance transformations

`provenance/file-lineage.json` lists original and package SHA-256 values. All historical Python source bytes are unchanged. Two input-lock configurations and five runtime snapshots have location-only sanitization. The artifact inventory omits private review-attachment metadata. Preserve this distinction when referring to original evidence: a public metadata derivative has its own digest.

Original stage deadlines, source pins, admission checks and evaluation-history labels remain in the historical code. No private workspace root is needed to inspect this repository or run its portable saved-result verifiers. A complete execution environment and many scientific runtime inputs remain absent; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
