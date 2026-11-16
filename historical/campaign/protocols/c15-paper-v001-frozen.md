# C15 paper-stage protocol v001

Frozen prospectively on 2026-11-10 before any Paderborn DEV/CONF measurements or original reserved profiles are decoded. The detailed source and input hashes will accompany each immutable execution registration. This document is not a claim that experiments have run.

## Authorization and scope

The user's active goal authorizes the paper stage: necessary full experiments and design, an evidence-based IEEE Access draft using the official template, and a GitHub-ready package for the user to push. This supersedes the earlier stop-at-topic-selection instruction. No external submission, GitHub push, paid service or remote execution is authorized here. Local RTX 4090 and CPU are used. The original deadline 2026-11-11T00:23:59Z, cumulative 600 GPU-hour/100 GB download/500 GB storage caps, prior attempts and confirmation error allocation remain unchanged. Four hours before the deadline are reserved for audit and writing; new exploratory training must stop by 2026-11-10T20:23:59Z.

## Question and limits

For present-time PMSM rotor-temperature estimation using electrical inputs and external boundary temperatures, does a fixed multiscale nonnegative thermal-response prior plus residual histogram gradient boosting improve held-profile accuracy and computational cost compared with equally informed direct and unconstrained alternatives? This is a controlled assessment of an existing-method combination. It does not claim the first physics/ML hybrid, identified physical resistances/capacitances, forecasting, a physical guarantee on the corrected output, or universal superiority. Both beneficial and harmful profiles will be reported. A failed confirmation will remain a failure and cannot be renamed successful confirmation of a different question.

## Data and exposure

Original public LPTN/LSTM source: frozen commit 98e4566b5fb7c70499996fda18dd73179ec16509; FIT [0,2,3,5,7,8,11,14]; previously exposed CAL [1,6,12] and PILOT [4,9,13]; reserved [10,15]. CAL/PILOT are development diagnostics, including their known negative findings. FIT LOPO is internal validation, not fresh confirmation. Reserved profiles are a small, separate secondary evaluation.

Paderborn version 3: original CSV SHA256 78f3d150f0f2ad9c5dc7ff24dd12c00d386ad530f48c1589ad24fcd88867d3ad. The previously frozen split is unchanged: 41 FIT, 14 DEV, 14 CONF. Only FIT data have been decoded at this freeze. DEV is descriptive development evaluation; CONF is the sole primary independent evaluation. The 69 profiles come from one motor/bench, not 69 independent motors. Procedures are refitted per dataset; this is not zero-shot transfer. Dataset input and target sensor differences must be documented. No internal stator temperatures, torque or target-initialization information enters a predictor.

## Shared representation and primary method

Preserve the C12/C15 causal features and primary procedure exactly: spans [16,64,256,1024,4096] in sample counts; 10 boundary, 15 heating and 8 instantaneous features; boundary states initialized to the first ambient value and heating states to zero. Target/scaling calculations use FIT endpoints at indices 300::5. Evaluation uses all samples 300 onward. Original physical sample interval remains unverified; do not translate its spans into seconds. Profile-weighted training gives every FIT profile equal total weight.

Primary model name: positive_rc_insample_boost. Nonnegative 25-coefficient prior with boundary sum one, no intercept, and ridge lambda selected from [0.0001,0.01,1] using FIT-only leave-one-profile-out macro RMSE; ties follow grid order. Residual HGB uses all 33 features, squared-error loss, learning rate .05, 200 iterations, 15 leaves, min leaf 20, L2 .1, no early stopping, seed 2026111201. Residuals are computed in-sample from the full FIT prior. Do not replace this model with the old OOF variant if that looks better. The residual learner has its ordinary nonzero mean-residual initial prediction, so the whole predictor is not physically constrained.

## CPU comparison panel

Preserve the original 15 entries and their meanings. Add the following, sharing exact artifacts when two entries are identical rather than counting them as independent repetitions.

1. Direct HGB nine-setting grid: existing six (15 or 31 leaves, 200/500/1000 iterations, L2 .1), plus (7,200,.1), (7,500,.1), (15,100,1). Other parameters stay fixed. FIT LOPO chooses `direct9_fitcv_selected`, the primary comparator; ties follow the listed order. Existing `direct_boost_fitcv_selected` continues to mean selection from the old six. The best observed test setting is descriptive only.
2. Free linear prior on all 33 standardized common features plus an unpenalized intercept; prior alone and residual HGB. Its scaling is learned within each FIT fold and lambda selected using the same three-value grid.
3. A factorial constraint ablation on the same 25 columns with no intercept: nonnegative plus boundary-sum-one, nonnegative only, boundary-sum-one only, unconstrained. Report each prior and its residual HGB at common lambda .0001, and separately at FIT-CV-selected lambda. Do not confound this comparison with the legacy free prior's intercept. Positive common-lambda entries may alias the primary if the selected lambda matches.
4. Retain the mean, EWMA linear and legacy free 25-plus-intercept baselines and OOF residual variants. For external OOF models, each row's prior prediction must come from a fit excluding its profile; distinguish the legacy globally selected-lambda construction from a fully nested estimator.

All prior-lambda and direct-grid choices use complete FIT LOPO for each dataset. Additional hybrid outer LOPO is optional diagnostic and must select lambda within each outer training fold; it cannot use the full-FIT chosen lambda as if it were fully nested.

## Matched small neural comparison

Predeclare a 33-64-32-1 ReLU MLP with 4,289 parameters, trained either directly or on the positive-prior residual. These are matched small ANN comparators, not reproductions of any cited method. Standardize features with FIT-endpoint mean/std (floor 1e-8); standardize temperature with FIT median/IQR (floor 1 degree C). Residual output is added to the same FIT-only positive prior. AdamW lr .001, betas (.9,.999), eps 1e-8, decay .0001, batch size 4096, profile-balanced mean squared loss, gradient norm cap 1. Epoch candidates [50,100,200], deterministic seed 2026111501 for selection. Select separately for pure and residual MLP by old FIT8 LOPO, with the prior's lambda selected only within the seven training profiles in each outer fold. Refit on each dataset's complete FIT using these frozen epoch choices and paired seeds [2026111501,2026111502,2026111503]. Report all seeds and their mean performance; do not choose a seed on evaluation data. External FIT receives the old-FIT-selected epoch counts, an explicit limitation. No architecture search after DEV/CONF exposure.

The previously prepared LSTM is optional and must remain a separate comparator, with its own prospective registration and complete training if reported. A cost probe is not a completed neural experiment.

## Evaluation, uncertainty and multiplicity

Primary contrast: C15 versus `direct9_fitcv_selected` on Paderborn CONF14. Primary effect is the paired mean of per-profile RMSE(C15)-RMSE(comparator), in degrees C; negative favors C15. Report relative reduction using the comparator macro RMSE and distinguish percentage from percentage points.

Also report profile-level RMSE, MAE, pooled sample RMSE, maximum absolute error and maximum positive (truth-prediction) underestimation, with nonnegative clipping for the latter. Display all profiles, not only successes. Paired profile bootstrap: 10,000 draws with NumPy PCG64 seed 2026111601, percentile 95% interval using linear quantiles; resample profiles, never correlated time rows. This interval is approximate with 14 correlated operating profiles and does not establish cross-motor inference. Report an exact one-sided sign test for probability of an improved profile, excluding absolute RMSE differences <=1e-10 as numerical ties. This sign test concerns direction frequency, not the mean difference.

Consume one of the campaign's at-most-two confirmation attempts and its one-sided alpha .025 allocation when CONF is first decoded. A directional confirmation claim requires both negative primary macro difference and sign-test p <=.025. Do not invent a 3% requirement. Report small effects and uncertainty even if the directional criterion is unmet. Additional comparators, ablations, neural seeds, original reserved profiles, warmup windows and cost outcomes are secondary/descriptive; no unadjusted family of superiority claims. No reallocation of unused alpha or further CONF-driven tuning.

## Timing and persistence

Benchmark whole feature construction, prior and correction inference as well as prediction-only cost on the same host/thread count and inputs, with warmup and repeated timings. Training/selection cost, parameter/tree counts, serialized model bytes and total feature/model state are separate quantities. Do not compare an old cold fit with a new warm fit or infer real-controller performance from desktop Python. Optional initialization sensitivity must use only external boundary inputs and fixed predefined perturbations, with results marked secondary.

Each training job uses an immutable registration, source/dependency hashes, bounded wall time and cumulative accounting. Save coefficients/certificates, tree arrays or neural weights, model-selection records, prediction arrays, truth in separate files, roles and profile indices. Independent audit recomputes metrics and selected predictions from saved artifacts without importing the scientific worker. Raw lineage and role exposure must pass before manuscript claims. A failure or source repair receives a new version/attempt; do not overwrite scientific outputs.

## Confirmation gate and deliverables

Freeze all CPU and neural model choices, selected epochs, final weights, sources and evaluator hashes before decoding CONF. Review DEV and independent audit first; DEV may diagnose bugs, but any substantive scientific model change requires a new prospectively labeled auxiliary method and cannot replace the fixed primary. Gate records the model hashes, unchanged contrast, one confirmation attempt and previous data exposure. Original reserved and external CONF remain separate datasets in all tables.

Deliver an English IEEE Access draft, bibliography verified against original sources, figures/tables generated from audited results, limitations and AI-assistance disclosure, a Korean results/adjudication report, and a portable GitHub-ready package with acquisition instructions, pinned dependencies, scripts, tests of scientific invariants, a reproducibility ledger and push instructions. No acceptance guarantee and no fabricated authors/affiliations/funding. If evidence does not support a useful contribution, state that clearly and prepare a transparent results draft instead of presenting a submission-ready positive paper.
