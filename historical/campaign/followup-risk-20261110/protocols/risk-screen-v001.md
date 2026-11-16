# C16 risk-oriented PMSM estimation: prospective exploratory screen v001

Frozen before the new fits or risk summaries on 2026-11-10. This follows the failed C15 superiority study. All original 16 and Paderborn 69 records have previously been available to the research campaign. Reusing them is development, not fresh confirmation. Old C15 predictions, manuscript and public staged repository remain unchanged.

## Question and hypotheses

With causal operating inputs and the same small prediction model, can asymmetric training reduce profile-macro RMS underestimation beyond calibration-only constant upshift and quantile regression, without an excessive RMSE/overprediction penalty? Let e=y-prediction, u=max(e,0), o=max(-e,0). The primary risk summary is mean across profiles of sqrt(mean(u^2)); all times, including zero underestimation, enter each record's mean. This is an estimation-error study, not a thermal-protection guarantee.

H1: asymmetric squared-error learning offers a better observed risk/error tradeoff than simple calibration of an MSE predictor. H2 (alternative C17, separately registered before implementation): causal input-based weighting of direct and hybrid predictors improves on a fixed blend. A favorable maximum error, a single profile, or one chosen seed cannot establish either contribution.

## Data roles, all exploratory

Keep the prior FIT roles for model training. IPMSM FIT8 trains; prior CAL3 calibrates/selects; prior PILOT3 and prior reserved2 are two separate exploratory evaluation groups. Paderborn FIT41 trains; prior DEV14 calibrates/selects; prior CONF14 is an exploratory evaluation group. Do not pool datasets, rename old CONF as new confirmation, or use evaluation metrics to choose a saved policy.

Training endpoints remain 300::5 and evaluation/calibration endpoints 300:. Causal input features remain the same 33 columns: boundary responses10, heating responses15 normalized with training-only current/speed RMS scales, instantaneous8. Warmup/initialization unchanged. Input mean/std and target median/IQR use training endpoints only. Dataset high-temperature threshold is the pooled training-endpoint target 95th percentile. This is a descriptive dataset threshold, not an engineering safety limit. High-temperature metrics exclude profiles with no qualifying times and list those IDs/counts; whole-profile metrics retain everyone.

## Fixed small neural panel

33-64-32-1 ReLU, 4289 parameters, 200 epochs; AdamW lr .001, betas .9/.999, eps1e-8, weight decay .0001, batch4096, gradient clipping1, profile-balanced row weights. Paired seeds 2026111701/02/03, same initial weights and sample permutations per seed. Three losses on normalized target error: (1) MSE, (2) asymmetric MSE 2/(4+1)*(4*u^2+o^2), (3) pinball .9*u+.1*o. Alpha4 corresponds to an expectile objective, not a coverage guarantee. No new epoch/architecture search. Seed1701 is the initial screen; the remaining two paired seeds are run before any promising-method conclusion. A seed is not chosen based on evaluation results.

## Direct tree and calibration controls

Use freshly fitted direct HGB with squared-error or quantile(.90) loss, 15 leaves/200 iterations, learning rate .05, min leaf20, L2 .1, no early stopping, seed2026111701, same 33 features and profile weights. Retain the prior FIT-selected direct9 and C15/linear33 residual saved predictions as contextual controls with their original exact identities; they are not retrained or selected using new evaluation scores.

For each freshly trained MSE backbone, compare calibrated constant upshift against convex mixtures of its prediction with its paired asymmetric or quantile prediction. The fixed mix grid is [0,.25,.5,.75,1]. Constant shifts are [0,.1,.2,.3,.4,.5,.75,1,1.25,1.5,2,2.5,3,4,5,7.5,10] degrees C plus max(0,q) of profile-balanced calibration residuals at quantiles [.5,.6,.7,.8,.9,.95]. No held-evaluation residual enters a shift or blend selection.

For each family, select on calibration the lowest primary risk among feasible settings, with ties broken by lower RMSE, lower overMAE, then declared setting order. Primary screen budgets: macro RMSE <=1.05 times unadjusted MSE baseline, macro overMAE <=1.25 times its unadjusted baseline. Also report the predeclared full grid RMSE ratios [1,1.02,1.05,1.10] by overMAE ratios [1,1.10,1.25,1.50] and all unselected settings. These are diagnostic tradeoff budgets, not externally justified practical success margins. Unchanged baseline ensures at least one feasible setting. If overMAE=0, use exact zero rather than a hidden floor. Report budget violations on evaluation explicitly; calibration feasibility does not guarantee evaluation feasibility.

## Metrics and interpretation

Per profile and macro: RMSE, MAE, RMS underestimation, underMAE, overMAE, positive-error q95/q99 (zeros included), maximum under/over error, and longest consecutive run with undererror>5 degrees C. Report samples for both datasets, and seconds only for Paderborn's verified .5s interval. High-temperature conditional metrics use y>=the train95 threshold and separately report coverage counts. Pooled RMSE is secondary. Exact source row indices are retained; never treat time rows/seeds as independent motors. All panel settings and seeds, favorable or unfavorable, remain visible. No new p-values or confirmation success in this screen.

Continue only if a candidate's benefit survives simple upshift and quantile controls, both tradeoff axes and paired seeds without relying on one adverse record. If simple controls dominate the candidate panel or selected policies repeatedly violate their budgets, stop that formulation. If uncertainty is material, mark INCONCLUSIVE rather than proving impossibility. Small consistent gains may matter; no 3% minimum is imposed. Current literature already includes hybrid residual and conformal methods, so numerical gains alone do not certify novelty.

## Execution and persistence

User explicitly authorized continued experiments on 2026-11-10. Local CPU and at most one RTX4090 only; remote gate remains CLOSED. Original cumulative600 GPUh,100GB transfer,500GB storage and deadline2026-11-11T00:23:59Z remain, with new exploration stopping20:23:59Z. Existing GPU use5.346326318888889h is not reset. Prior total network bytes have an unmetered component; record any new payload exactly and do not report independently verified global download compliance. No paid calls or publication.

Each actual preparation/training/evaluation is bounded and registered in the existing attempt ledger with new C16/r011 IDs. Two operational repair attempts maximum for an identical blocker. Save exact inputs/roles/config/code hashes, transforms, weights, checkpoint/loss history, locked predictions, targets and row identifiers. Independently reconstruct metrics, calibration choices and neural forward passes from saved arrays. Full optimizer replay is a different claim.

Fresh-source metadata may be investigated separately without decoding prospective confirmation values. A new confirmation requires an explicit source-specific split and question before access, and the unspent campaign confirmation allocation; old confirmation is not reset. Stop this topic only with a bounded evidence-based failure/inconclusive/resource verdict, not a claim that no conceivable method can work.
