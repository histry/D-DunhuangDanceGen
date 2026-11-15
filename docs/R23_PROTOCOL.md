# R23: bounded guidance comparison and continuous full-body representation
Baseline c6bebeb1603f2b3d1cf35170d7d3de044fadbb31. Production, R13–R22, weights, Scaler and original Native remain frozen.
No operational promotion, new diffusion training, generated-history or whole-song experiment is authorized by this research result.

## A. Frozen guidance
G0/G2 × original eight probes × original layouts0/2 × CFG1/2 =64 conditions.
32 CFG1 trajectories are new; CFG2 reuses28 sealed trajectories and reproduces four canonical case0/24 trajectories with full actual-state diagnostics. Replay repeats36 sampler calls, with no optimizer.
Same R22 seed, 50-step grid, stochastic updates, clip and hard observations. CFG1 retains music. All CFG2 reproduced samples and noise must exactly match R22.
Actual branch error cosine/Pearson, raw CFG error identity, difference energy, clip geometry and saturation are measured at999/519/119/19. Unknown C4 excluded.
Nine actual pre snapshots (three arms ×0/599/1199) provide weighted reconstruction/physical gradient dot products, including per-term values; norm alone is not conflict evidence.
The fixed DEV rule requires each source in both layouts to preserve pose/Root within1.1×CFG2, activity/range ≥0.8×source, interior jerk ≤1.1×CFG2, drift ≤1.1×CFG2, and improve crossing max and foot-skate P95 by5%. No Native-pass regression. At least3/4 sources per arm is a research contrast, not full quality.
No guidance scan or DEV selection. APG has no registered setting this round; a positive result may motivate a separately registered experiment.

## B. Complete source representation upper bound
Reuse the six R22 mechanism-selected windows unchanged (five Train/four groups and exposed DEV case24). No repeat290 search, no additional selection.
Fixed cubic B-spline basis with48 coefficients,147 metric root/column-6D channels, full24 rotations; geometry projects valid 6D to SO(3). Not a learned INR or a reproduction of unavailable NINB code.
Full-target least squares fits are per-instance latent upper bounds, never conditional generation/generalization. Shared decoder has zero trained parameters.
Compare original, unconstrained decoded, boundary8/8 and boundary24/16. Boundary decoder combines observed context cubic splines, a quintic C2 lift and a cubic residual basis with zero endpoint value/first/second derivatives. No decode-then-graft.
Native, rootRMSE≤0.01m, mean pose≤0.02rad, source activity/range ratios in[0.95,1.05] all required. All15 Train decoded conditions must pass for a limited representation mechanism positive; DEV diagnostic cannot select basis/capacity.
Known context is extra input/storage. C4 remains unknown. C2 does not imply discrete Jerk/physical acceptance.
No latent conditional diffusion starts: five Train windows do not provide motion-prior coverage, even if representation passes.

## C. External contracts
Keep original Lodge9+0 strictly loaded; perturb the FK projection and test its graph reachability without changing weights or adding layers. Report actual normalize/unnormalize distortion and clipping.
Retarget known22 rotations to the fixed Native geometry and infer terminal22/23 positions only after parent/order/axis and original-source FK parity checks. Unknown terminal rotations stay unknown; full Native remains NOT_EVALUABLE.
PHC requires a verified simulator, source/policy/body and upright-heading mapping. Export Case24 source/G2 pair in native and explicit SMPL Z-up coordinates, inventory installed server environments. An export is not a simulation/teacher pair. Missing runtime yields NOT_EXECUTED, not a tracking result. Do not download licensed bodies or replace physics with smoothing.

## Reproduction
scripts/run_r23.sh FRESH_STUDY FRESH_RECEIPTS runs regressions, each stage and optimizer-free fresh-process replay, then scripts/verify_r23.py.
Stage failure preserves original files and exit receipt; publish any fix before a new run/recovery. Do not modify R22 scientific results.
Source references:
- [SciPy B-spline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html)
- [SciPy C2 interpolant](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html)
- [PHC runtime and coordinate requirements](https://github.com/ZhengyiLuo/PHC)
- [Pinned Lodge](https://github.com/li-ronghui/LODGE/tree/f38642ad67bc3402444f5f914010a911e5bd5c89)
