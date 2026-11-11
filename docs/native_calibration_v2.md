# Native reliability and Shadow calibration

Legacy Native remains the operational kinematic/contact plausibility contract. Its original31 limits, support population, FK, physical.json and pass/fail are preserved. physical_protocol.json adds code/config/skeleton identities and native_layers.json adds separately versioned continuous/uncertainty reports. Shadow judgments cannot route the existing replanner.

## Three layers

1. Numerical integrity: shape, finite values, nondegenerate rotations, exact skeleton/ground/clock, optional official-FK tolerance1e-5m. These are mathematical/runtime validity checks.
2. Continuous measurements: exact Legacy contact responsibility, separately evidence-conditioned static/sliding populations, counts and coverage, raw/diagnostically filtered jerk, violation duration/rate, PFC, fixed-plane mesh metrics and Root task extents. No single weighted physical score is introduced.
3. Contracts: immutable Native-Legacy-v1 plus Native-Calib-v2 in unqualified Shadow mode. No Legacy threshold is changed or represented as a universal paper threshold.

configs/native_calibration_v2.json registers every actual limit, its source, units, category, direction and scope. Root travel/stage limits are task constraints. Collinearity is a conditioning proxy; it is not silently substituted for mathematical singularity.

## Reproducible read-only experiment

Publish code first, pull on the server, use a new directory and tmux:

    PYTHONPATH=src python scripts/native_calibration_experiment.py --assets outputs/aist_features_01/assets.json --benchmark-assets inputs/benchmark_assets.json --training outputs/event_transfer_01/cross/train --prior-target-audit outputs/event_source_quality_01/train_targets --prior-experiment outputs/event_transfer_01 --baseline-mechanism outputs/mechanism_01 --source-ledger outputs/aist_calibration_01/source_ledger.json --out outputs/native_calibration_new

Run with the pinned Native Python and available neutral SMPL model. No licensed model data is published with the code. The experiment:
- Inventories all saved matching-contract Native results and replays each distinct measurement identity, without reinterpreting old EDGE representations.
- Replays all1468 historical cross-recording Train targets against their sealed original receipts, and separately invokes the real physical entry on the latest40 executions plus historical EDGE/Temporal150.
- Decomposes rederived-only, low-height-only and overlapping support observations using the exact frozen operations. Their exceedance ownership is not an independent semantic label.
- Freezes reference floors before comparison. Candidate self-floor, common reference, Native fixed offsets and neutral beta0 mesh are recorded separately; proxy ground is never scene truth.
- Uses existing connected source components for a fixed calibration/holdout split inside historical Train. Reports recording-, choreography- and component-cluster bootstrap intervals with300 fixed-seed draws. Repeated/overlapping windows are not independent observations; few components limit uncertainty estimates.
- Evaluates real-source prefixes, saved long-motion prefixes, fixed millimetre noise, and predeclared synthetic slide/penetration/floating/jitter curves. Perturbations are in-memory copies or new fixtures; source motion and old Native results are untouched.
- Builds an unreviewed contact annotation packet with exact video SHA, source time, motion SHA and UNKNOWN labels.

    PYTHONPATH=src python scripts/verify_native_calibration.py --root outputs/native_calibration_run --out outputs/native_calibration_replay_new.json

The fresh process rechecks source hashes, Native results, contact responsibility, deterministic grouped estimates, generated layers and neutral mesh caches. No optimizer, generation, data filtering or operational threshold activation is permitted.

## Evidence boundaries

Native recomputes contacts from FK; prediction Contact4 is reported separately and cannot clear a foot-slip failure. Low height alone does not establish static support. Without independently reviewed contact labels, real static/sliding states stay UNKNOWN and conditional static speed/drift are NOT_EVALUABLE, never fabricated zero.

Independent labels must bind motion hash,150-frame clock, foot joints, reviewer and a manual-video/force-plate source independent of the measured speed. Quality labels must be independently reviewed, not derived from Legacy pass or PFC. Annotation templates are validated=false and are not calibration truth.

Without these labels, reference-distribution quantiles are descriptive only. False-reject/missed-error rates and proposed operational thresholds remain null. Synthetic metric sensitivity is not real-world classifier accuracy or expert agreement. Even complete optional annotation input produces Shadow proposals; this software does not automatically release or route a new contract.

EDGE PFC is retained with its official displacement-product definition; zero Root acceleration remains undefined. PhysDiff metrics retain the existing Appendix-definition reimplementation,5mm tolerance interpretation and50mm geometric contact choice; author-code parity and leaderboard comparability remain unestablished. New wrappers report explicit conditional-population status without changing historical benchmark values.

The audit is kinematic/contact plausibility. Contact forces, friction cones, mass distribution and dynamic balance are not simulated. Native failure of a trajectory does not blacklist its source event.
