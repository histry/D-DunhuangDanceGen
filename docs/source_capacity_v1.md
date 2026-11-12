# Frozen source capacity and local correspondence audit

Baseline6fb5bdb and every existing source/config file are frozen. This study adds only research/source_capacity and its own config, launcher, release and tests. No optimizer, new neural branch, normalization fit, production integration, E1-W rollout, Root/SQP or G3 work is performed.

## R1: recover real trajectories, then measure

Pin source_tasks_02 public input/model hashes and its exact4 exposed Val cases. Replay C0-C3 A/B (32 samples), preserving seed20261113, actual50-step DDIM, observed visibility, checkpoint and original Native. Verify normalized/native/FK arrays and physical JSON EXACTLY against prior saved outputs before interpreting instrumentation. Capture every clipped denoised x0, not noisy x_t as an SO(3) pose. Record actual C2-A forward inputs at999/499/19; no clean/noisy state is fabricated from final motion.

At each selected common C2-A state compare both sources for all4 weights; save actual153D,128D,512D and8 decoder output differences. Source response, direction and each latent layer RMS are separate diagnostics, not interchangeable units.

For C2 test JVP and target-cotangent VJP of the raw conditional predictor projected to actual common provided upper8 rotation matrices. The unit target cotangent is the reference B-minus-A matrix chord. Adjoint equality and centered FD at h=.25/.125/.0625 must agree. Use SDPA MATH only inside derivative/probe contexts, keeping historical trajectory replay on its original backend. Require adjoint relative error<=1e-4 and >=2 FD scales with relative error<=.1 and cosine>=.99. Failed/unsupported autodiff closes qualification, not a claim of zero physical capacity.

The learned source residual direction is normalized to1% of the base cond_encoder output RMS; actual unscaled residual differences are also reported. Equal-norm random and target-dependent VJP oracle controls distinguish an ineffective learned direction from available local capacity. Oracle reads true targets and is not deployable source control. Frobenius derivative ratios are relative protocol measurements, not universal physical gain.

## R2: train-only injection sensitivity

Choose4 Train music groups/pairs by stableSHA before any model evaluation from the190 qualified real Train pairs. Probe q(X_A) at the same999/499/19 with common pre-registered Gaussian, not an assertion that low-noise states support joint two-target CF training. Compare the identical numerical512D direction at cond_encoder output and decoder outputs1/3/6, plus decoder3 gated to actual provided frames. Layer coordinate spaces differ, so a larger numerical response alone is not semantic equivalence or usable control.

All controls have matched full-tensor RMS; gated active-frame peak amplitudes differ and are not equal-peak comparisons. The legal time map for these same-clock paired recordings is identity. I3 phase alignment therefore equals I2 gating exactly; no arbitrary warp is introduced to manufacture a benefit.

Record finite +/- perturbation directional preferences, learned JVP-target cosine, FD/adjoint reliability, random and oracle effects. A site passes local directional evidence only if EVERY fixed Train pair/time has reliable numerics, both directions>.001rad and positive cosine. An improved site additionally needs>=1.1x cond response at every paired context. No selection uses Val. Even a passing local probe does not establish free-generation/Native/music qualification, so lightweight training is NOT registered this round.

Compute C2 CF/MSE/following gradient norms and pairwise cosines on output/source encoder/fusion blocks. Mark middle/low CF gradients as outside its registered common-q training support; they are conflict diagnostics, not new admissible training labels. Frozen backbone/weights are checked and no .step is possible.

## R3: freeze encoders and change only alignment

Use all4 saved15/30/dual/dual_shuffle encoders and the exact same Normalizer and offline A2-V3 candidates. Re-extract fixed Val tokens on a common120-frame30Hz grid (15Hz tokens use the existing repeat operator). Reproduce original S0 scores exactly using the old scorer before new paths. No Train/test selection or feature fitting.

S0: fixed-position cosine.
S1: bounded hard DTW covering both complete sequences, allowing repeated matches, band.5s, clock cost.2*distance/.5; score1-cost/120.
S2: existing strict ONE-TO-ONE soft partial_path recurrence, NULL=.5, epsilon=.05; score=(all-drop cost-soft cost)/120. This is NOT full Drop-DTW, and S1/S2 differ in both repeat/drop semantics and soft entropy, so changes cannot solely be attributed to NULL.
S3: same S2 with zero content and only physical clock. On equal full grids this ties candidates; pessimistic R1=0 is not random chance. Analytic uniform1/K and frozen nuisance controls remain visible.

For S0/S1/S2 retain identical global cosine +.5 local weight, and separately report global-only/local-only. A fixed motion-token time permutation preserves token values and marginal statistics while breaking correspondence; labels are held as a counterfactual diagnostic, not invented semantic truth. Compare joint S2 against S0, global-only, nuisance, scrambled-content and frozen shuffled encoder; group-level paired R1/MRR with1000 bootstraps/seed20261113. Only6 known exposed Val groups exist, with no new blind or online retrieval claim. Full DTW fails closed for incomplete grids instead of silently filtering candidates.

Vectorized DTW is a bounded recurrence; partial matching reuses the existing solver. Torch validation checks diagonal-only DTW, synthetic shifts, content-independent S3 and existing exact partial-path agreement at low temperature. No new OT/encoder is introduced.

## Execution

Push first; verify serverSHA; run scripts/run_source_capacity.sh with fresh study/receipt roots in tmux. Independent fresh processes repeat all generations, derivatives, captured arrays, token extraction, path scores and metrics. Optimizer construction is forbidden. Retain any failed attempt. Seal only after launcherexit0; publish aggregates and hashes, keep raw identities/weights/activations on server, clear only temporary SSH helpers created this run.

Literature boundary: supplied MusicInfuser/ReMoDiffuse CVF URLs returned403 in this access. The study is independently specified from inspected source, not a claimed reproduction of inaccessible paper implementations.
