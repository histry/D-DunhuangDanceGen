# Event execution mechanism experiments

This research path follows the supplied mechanism-first review. It does not add a second generator, change Native limits, promote research weights, or infer semantic quality from tensor RMS.

## Frozen information-flow experiment

Run only on a complete strict paired manifest. The runner validates original target, source, music, split and calibration receipts before loading the model. It uses eight distinct historical Val captures (first admitted window per recording; uniformly spaced sorted capture list).

    PYTHONPATH=src python -m dunhuang.research.mechanism --assets outputs/aist_features_01/assets.json --adapter outputs/aist_train_01/adapter.pt --out outputs/mechanism_new

Use a new output directory and tmux. The protocol is written before CUDA work; failures preserve status/error. Fixed seeds 20261113/20261114, denoising times 100/500/900, 50 DDIM reverse steps, maximum 12 free-running generations.

The identity probe holds music and noisy state identical for every event, so target identity cannot enter through the denoising input. Two visibility views fit a fixed ridge probe; the third is held out. Classes are the same eight event identities, so this is a closed-set representation probe, not unseen-source accuracy. Save temporal-bin descriptors, covariance effective rank, Adapter128/512, injected music conditions, and all eight decoder layers.

Denoiser controls keep target/noise, body-group measurement and reference observation policy fixed: original, removed reference, mask-only, phase reversal, different event, different real music, deliberately circularly shifted feature negative, and zero Adapter. Raw prediction MSE/velocity are target-noised completion diagnostics, not generation quality. The invalid circular negative is never a training pair.

The free-running two-music/two-event/two-seed factorial includes four zero-Adapter controls. Samples are unmodified outputs of the existing 50-step sampler with guidance=none. No target pose is pasted into a result. Save matched noise hashes, SO(3) own/other-event pose preference, Native FK/audits, output activity, and the same four-frame jerk templates before and after official assembly. A positive preference margin means outputs prefer their respective source over the counterfactual; merely large output differences are insufficient.

## Independent-reference tasks

source_pairs.py forms Train/Val tasks only when source/target have different capture IDs and raw bytes, but identical official basic routine, music bytes, frame start, offset, FPS and duration. It never uses Test as a condition or fitting target. It preserves the original target and records pose_exact_correspondence=false: an official shared routine does not establish pose-exact cross-performer annotations.

Full source-pair and per-window files are ignored local/server outputs, not GitHub publications. Public reports contain aggregate diagnostics only. Historical validation exposure and EDGE pretraining exposure remain limitations.
