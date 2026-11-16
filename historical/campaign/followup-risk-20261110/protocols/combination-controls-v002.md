# C17 v002: controls for the observed fixed-combination gain

Frozen for the next adaptive exploratory analysis on 2026-11-10. The C17 v001 result is retained unchanged: on exposed Paderborn evaluation the D9/C15 gate improves on direct9 substantially, but most improvement is already obtained by its calibration-selected constant combination; gate's incremental gain is small. Original-source final gate/constant combination select direct9 alone. This motivates examining the simpler combination and whether the physics-informed component is necessary. It is not fresh confirmation or a newly invented ensemble method.

## Fixed components and candidate/control identities

No models are refitted. Use exact audited existing prediction columns. Every pair uses `prediction__direct9_fitcv_selected` as left component. The primary right component remains `prediction__positive_rc_insample_boost` (C15). The four right-component controls are, in this fixed order:

1. `prediction__direct_boost`: same feature direct200, for a second direct model rather than the constrained prior.
2. `prediction__linear33_insample_boost`: a learned unconstrained linear33 prior plus residual.
3. `prediction__unconstrained_rc_insample_boost`: same25 RC basis without coefficient constraints plus residual.
4. `prediction__positive_rc`: constrained prior alone, without residual learning.

Compare all five predetermined pairs. Do not select a different right component using evaluation and relabel it as the primary physical contribution. If controls explain the C15 gain, report that weakness. Continuous optimal CAL weights are not globally optimal future weights. Original D9/C15 v001 results must reproduce numerically with the same weights/roles.

## Common five prediction rules per pair

Exactly w=0, w=1, w=.5, CAL-profile-balanced MSE-optimal w in[0,1], and CAL-profile-macroRMSE-optimal continuous w in[0,1]. Use the same closed-form MSE objective, scalar bounded minimization plus explicit endpoints, xatol1e-12/maxiter500/lower-w tie break as the frozen C17 v001. Main combination is the continuous macroRMSE selector. No input gate, parameter search, shifts or new training. Refit these same scalar rules in each CAL-LOPO fold for descriptive stability and on fullCAL for each existing exploratory evaluation role. Every pair gets exactly the same CAL and fit budget.

Keep FIT/CAL/evaluation role membership, full300: indices, source-row identities and all inherited high-temperature/risk metric definitions unchanged. Use the original FIT95 threshold solely for reporting. Load held evaluation targets only after saving every pair/rule prediction and its scalar-model identity. Shared C17 scalar helper and shared C16 scientific metric helper may be imported by the producer only when their exact frozen hashes are pinned. Independent audit must reconstruct scalar optima/weights, predictions and metrics without importing that producer.

## Readout and contribution boundary

Primary metric is profile-macroRMSE. Report every pair/rule's RMSE, under-RMS, overMAE and per-profile differences, not just the favorable method. Reuse the explicit C17 diagnostic risk flags without safety claims. For the primary pair, also report centered error correlation and uncentered mean error-product per profile as exploratory descriptions of error complementarity; no causal explanation is proved by correlation.

The main question is whether C15+direct9 gives a useful combination beyond direct9 alone and direct/linear/unconstrained alternatives at the same observation budget. Both experts must be evaluated at inference; no speed/memory advantage is claimed. Tiny consistent differences can support further work, but a favorable mean caused by one profile, pure-direct collapse on a source, or equivalent simpler mixtures must remain visible. No new significance test, best-evaluation pair selection, novelty declaration or fresh-confirmation success is allowed here. Any fresh-source experiment needs a separately frozen single primary candidate and strong controls before its confirmation target access.

CPU-only fit/predict360s (internal300), evaluation180s (internal120), audit600s; output cap200MB per stage. Register new C17/r011 attempts under unchanged campaign deadline/aggregate limits. No new downloads/GPU needed, and no old results, code or public repository are overwritten.
