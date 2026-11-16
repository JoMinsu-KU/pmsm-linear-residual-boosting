# C17: observable operating-state gate for two frozen predictors

2026-11-10. **Root frozen before C17 fitting or numeric gating summaries.** Only this review note was created. No scientific arrays, new outcomes, raw data, base models, campaign state, or public repository were read or changed for this design.

## Question and scope

Can a small gate based on observable current, speed and causal load history combine the already fitted direct9 and C15 predictors more usefully than the best calibration-selected **constant** combination? This is a controlled test of conditional combination, not a new base regressor, new physics model, new gating theorem, or fresh confirmation. A lower error than an individual expert is insufficient if an optimized constant blend explains the gain.

One learned gate is proposed, with one fixed regularization value and no architecture/hyperparameter search. A simple single-threshold rule is a required control, not a second proposed contribution. The frozen base learners remain unchanged.

All original16 and Paderborn69 profiles have already been exposed in the campaign. Preserve the same roles: IPMSM FIT8 supplies scaling and base training provenance, CAL3 trains the gate, PILOT3 and former reserved2 are distinct exploratory evaluation groups; Paderborn FIT41 supplies scaling and base training provenance, former DEV14 trains the gate, former CONF14 is exploratory evaluation. Do not rename any of these evaluations fresh confirmation. The new RWTH archive is excluded from this screen.

## Exact inherited inputs

Use hash-pinned, previously audited C15 prediction artifacts and their truth/feature locks from the current C16 dataset configs. The two components are:

- `d`: the actual FIT-selected direct9 predictor for that dataset, with the original selected job/model identity. Do not substitute direct200 or reselect direct9 on CAL.
- `h`: `positive_rc_insample_boost`, the original C15 primary, with its original selected prior and200-tree residual model.

Check the exact prediction-key names against the pinned source/lock before implementation; this note does not authorize selecting a similarly named method. Indices remain300: for all gate-training/calibration/evaluation roles. FIT-only scaling uses300::5. Keep full source-row identifiers. No base weights, coefficients or feature histories are refitted.

### Three observable gate features

The audited `instant` geometry is current magnitude at column0 and absolute speed at column3. The `heat` columns are triples `(I², omega², I*omega)` for spans16,64,256,1024,4096 in that order. All are causal and available without rotor labels.

From FIT endpoints only, calculate `I_rms=max(sqrt(mean(I²)),1e-8)` and `omega_rms=max(sqrt(mean(omega²)),1e-8)`, exactly as existing source scaling. Define:

1. `v1=log1p(I/I_rms)`;
2. `v2=log1p(omega/omega_rms)`;
3. `v3=(heat[:,0]-heat[:,9])/I_rms²`, the span16 versus span1024 current²-history difference.

Compute pooled FIT means and standard deviations of these three values; standard-deviation floor1e-8. Set `z=(v-mu_fit)/sd_fit` for every later role. These are operating/load-history proxies, not measured heating rates or identified physical losses. No measured rotor/stator temperature, true error, profile ID, time to profile end or future input is a gate feature. Current and speed must be finite/nonnegative according to the existing audited geometry; do not silently take another absolute value to repair corrupt input.

For normalization of the gate's squared-error objective use the existing FIT-only target scale `s_y=max(IQR(y_fit),1 degree C)`. This target statistic comes from base FIT, never a held gate-validation or evaluation target. Since the base predictors are fixed, using the same FIT scaling in all CAL leave-one-profile-out folds is legitimate; CAL is not the source of those scalers.

## Strong fixed controls

Report exactly these seven methods, without selecting the best one on evaluation:

1. `direct9`: d.
2. `C15`: h.
3. `fixed_half`: (d+h)/2.
4. `fixed_cal_mse`: d+w_mse*(h-d), with closed-form scalar `w_mse=clip(sum_p mean((h-d)*(y-d))/sum_p mean((h-d)^2),0,1)` over CAL. If the denominator is exactly0, select0.
5. `fixed_cal_macro_rmse`: continuous scalar minimizer of CAL profile-macro RMSE over w in[0,1]. Each profile's RMSE is a norm of an affine function of w, so the scalar objective is convex. Use bounded1D minimization plus explicit endpoint comparisons; deterministic lower-w tie-break. Save objective/derivative-bracketing or convex qualification and the chosen w. This is the **main combination comparator**, not a coarse mix grid.
6. `load_threshold`: choose among the four rules w=0, w=1, w=1{v3>=0}, w=1{v3<0}, in that order, using CAL macro RMSE and that order for ties. Threshold0 and featurev3 are fixed, with no threshold search. This may legitimately select a constant expert; record that identity instead of claiming conditional behavior.
7. `observable_simplex_gate`: the single proposed learned gate below.

The continuous fixed blend must not be called globally optimal for future profiles; it is optimal for its stated CAL objective. Keep the MSE-optimal and RMSE-optimal constant controls separate, because macro RMSE is the reporting estimand while the learned gate uses a quadratic fit objective.

## Single learned gate: convex combination of fixed operating ramps

Define `r_j(z)=clip(0.5+0.25*z_j,0,1)` for j=1,2,3. Use the eight fixed basis functions

`phi(x)=[0,1,r1,1-r1,r2,1-r2,r3,1-r3]`.

Let `theta>=0` and `sum(theta)=1`. The gate and prediction are

`w_theta(x)=phi(x) @ theta`,

`prediction=d+(h-d)*w_theta(x)`.

This guarantees a convex combination of the two experts even outside the CAL range; the gate cannot invent an arbitrary upward shift or an unconstrained residual. Basis dependence does not make the numerical fit singular because the fixed strictly positive ridge term below makes the coefficient objective strongly convex. Coefficients need not have a unique physical interpretation; this is a small empirical operating gate.

Fit theta only on CAL by minimizing

`mean_profiles mean_rows [((y-d)-(h-d)*phi@theta)/s_y]^2 + lambda*||theta-theta_ref||²`,

subject to the simplex, with **lambda=0.01 fixed before outcomes** and

`theta_ref=[1-w_mse,w_mse,0,0,0,0,0,0]`.

Thus shrinkage favors the classical MSE-optimal constant blend. No parameter is chosen to maximize a held evaluation result. For each profile p, accumulate its8x8 Gram matrix,8-vector and scalar constant with row weight1/n_p, then average these sufficient statistics across CAL profiles. The total profile weight is equal regardless of record duration.

Writing `A=(h-d)*phi/s_y` and `b=(y-d)/s_y`, the QP has `Q=2*(mean_p(A_p.T A_p/n_p)+lambda*I)`, `r=2*(mean_p(A_p.T b_p/n_p)+lambda*theta_ref)`, and objective `.5 theta.T Q theta-r.T theta+constant`. All operations are float64. This is only an8-variable convex fit, not a neural optimizer.

Implementation can enumerate the255 nonempty active supports and solve each equality-constrained SPD face, retaining feasible candidates and choosing the lowest objective deterministically. These faces are **numerical solver steps of one fixed model**, not255 experimental settings. Alternatively use a small deterministic active-set solver. Require Cholesky/SPD, finite values, simplex feasibility and a convex optimality certificate. For feasible theta, `g=Q@theta-r`; the Frank-Wolfe gap `g@theta-min(g)` upper-bounds objective suboptimality. Record it with a predeclared numerical tolerance, proposed `1e-8*(1+abs(objective))`, plus feasibility residuals. Do not silently count a failed fit as scientific evidence against the method or change tolerances after seeing results.

## CAL leave-one-profile-out is a stability diagnostic, not a selection search

For each held CAL profile, refit all three CAL-trained comparators (both optimized fixed blends, the threshold rule) and the learned gate using only the other CAL profiles. Recompute w_mse and theta_ref within each fold; retain unchanged FIT-only feature/target scalers. The constant individual experts and0.5 remain unchanged. Predict the entire held CAL profile; save every fold and objective/certificate.

Report the resulting pooled held-profile collection as **CAL-LOPO** with macro metrics and profile IDs, not as a second new independent test. Do not choose lambda, a subset of gate inputs or the better of several gates using those outcomes. Refit the prescribed models on all CAL for the separate exploratory evaluation groups. All final prediction arrays and gate-model identities are locked before the evaluator opens held evaluation truth.

Original CAL3 means each diagnostic gate sees only2 calibration profiles. Many time rows do not replace independent operating episodes. A large gate advantage on one original profile is weak evidence and cannot support a motor-wide generalization claim. Paderborn CAL14 is still one physical motor. Large fold-to-fold theta/weight changes should be reported as instability, not repaired by selecting favorable folds.

## Metrics, decisive falsifier and interpretation

Primary: profile-macro RMSE in degrees C, preserving every evaluation profile. Main contrasts are learned gate minus `fixed_cal_macro_rmse`, and minus direct9. Also report all seven methods' RMS-under, underMAE, overMAE, zero-included positive-error q95/q99, maximum under/over errors and consecutive>5-degree undererror runs, using the same already frozen C16 definitions. High-temperature strata keep the same FIT95 threshold and list excluded/eligible profiles.

Because C17 optimizes symmetric error, a smaller RMSE is not by itself improved underestimation control. Report ratios of under-RMS and overMAE against direct9 and the main fixed-blend comparator; apply C16's1.05/1.25 ratios only as explicitly labeled diagnostic tradeoff flags, not engineering safety limits or a new probability guarantee. Do not shift predictions or retune gates after these flags are known.

For weight-collapse diagnosis report per-profile min/mean/max/std of w and the magnitude of `prediction_gate-prediction_fixed_cal_mse`, all label-free. If the fitted gate is practically constant, describe it as another constant blend rather than a successful conditional mechanism. Also preserve each rule/fold's selected constant identities and full theta.

The cheap decisive intervention is the comparison of the **same frozen d/h predictions** under constant versus operating-dependent weights. No base-model training is involved. If fixed blend or the single-threshold control matches/dominates the learned gate, or any apparent improvement depends on one isolated evaluation profile, the conditional-gating rationale is not supported by this screen. Small consistent gains are allowed; no3% cutoff is imposed. A favorable result is only a reason to consider a separately frozen fresh-source study, not a confirmation-success or novelty declaration. Uncertainty with only3/2 original evaluation profiles should be reported as such.

No new significance testing is proposed. Do not choose a different primary after seeing which of the seven methods wins. Do not claim a specific failure cause of C15, thermal protection, energy savings or real-time deployment from this offline screen.

## Source implementation and bounded execution proposal

Root freezes a new C17 protocol/config and exact two-expert input locks before implementation/execution. Proposed worker modes:

- `fit_predict`: load FIT features/targets for the stated scalers; load CAL predictions/features/truth; solve the fixed controls, all CAL-LOPO diagnostics and final gate; then read only evaluation features/predictions to save all seven predictions and a prediction lock. Never load held evaluation truth in this mode.
- `evaluate`: consume the fixed model/prediction lock and separately pinned old exploratory evaluation truths; calculate all predeclared metrics. No model or policy selection here.

Use standard NumPy/SciPy only, no GPU/network/installation. Proposed internal cap300 s and root cap360 s for fit_predict; evaluation120 s; output100MB; independent audit reserve300 s. These are upper proposals pending root resource admission, not measured runtime. An operational cap failure preserves artifacts and cannot be retried as a fresh scientific selection.

Save source/config/input hashes, exact roles/indices/source rows, FIT scalers, per-profile sufficient statistics, fixed scalar-blend solutions, all threshold-rule CAL scores, all LOPO and final theta/feasibility/gap receipts, actual wall time, all seven per-profile predictions and a final prediction lock. Independent audit reconstructs the three features and ramps, weighted8x8 systems, simplex certificates, scalar controls, every prediction and every metric; it must not import the worker or claim base-model retraining.

**Novelty status: UNCONFIRMED.** Mixture-of-experts, input gating, convex aggregation and ridge shrinkage are established. The potential contribution is the bounded empirical answer about when operating-dependent combination of these two thermal estimators helps beyond strong constant and simple state rules, at identical sensor/base-fit budgets. It is not established merely by this design.


## Root binding choices, 2026-11-10

This section resolves the proposal's implementation alternatives. Exactly seven methods above are admitted. Direct9 key is `prediction__direct9_fitcv_selected`; C15 key is `prediction__positive_rc_insample_boost`. The simplex solver enumerates all255 nonempty supports, solves the equality-constrained positive-definite face, rejects negative coordinates below -1e-12, clips only numerical residual negativity and renormalizes; retain minimum objective with support-mask order tie-break. Require feasibility1e-10 and Frank-Wolfe gap<=1e-8*(1+abs(objective)). The scalar CAL-macroRMSE minimizer uses scipy bounded minimization on[0,1], xatol1e-12,maxiter500, plus exact endpoints; lower w breaks exact objective ties. Coefficient regularization lambda .01, all features/bases, scope and diagnostics remain as above.

No C17 scientific arrays were opened to choose this design. The already observed C16 tree control scores do not choose any C17 feature or parameter. These are different, explicitly exploratory questions. Existing C15 history and failed confirmation remain unchanged. Fit/predict stage cap360s (internal300), evaluate180s (internal120), independent audit600s. No GPU/network/new data. Output cap100MB. Configs bind original C16 dataset manifests rather than cloning or replacing their roles. Remote gate stays closed; no publication.
