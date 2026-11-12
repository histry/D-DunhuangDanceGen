# Dense source-clock challenge and frozen EDGE causal audit

Baseline: 54c1f9bd45b4529a8a29e8910a24c06fa84475a8. This is a new, exposed-source research challenge. Historical R1 and Native31 results and thresholds remain intact. Generation defaults remain legacy. No RootNet, second diffusion, PFG, bilateral generation or SQP is added.

## A1: source coverage before representation training
Reuse the sealed paired Train/Val population, full raw AIST++ recordings, calibrated video/audio clocks and audio-only/full-motion event sidecars. Require verified origin plus at least three calibration anchors and residual no greater than one frame. Bind audio/motion SHA and actual decoded duration. Compute motion derivatives over complete recordings, then crop real 150-frame windows at stride 15. The inner 120-frame content has actual 15-frame halos. Never pad, wrap or extrapolate source motion/audio. New windows do not constitute new independent recordings or new paired 151D/Jukebox targets.

Every identity group (source association, recording, canonical music, choreography) retains its previous role. No Test and no new-blind claim. The unchanged strict gates require exact event counts, mean 16-quantile distance <=0.15s, four-bin histogram L1 <=0.25, both energy ratios <=1.25, onset density difference <=0.25/s, >=30 queries and >=15 recordings each role, >=3 Train and >=2 Val association components. All original-clock multi-capture positives remain positives; negative windows are at least 120 frames apart.

Compare frozen music TCN30, its even 15Hz samples, and Transformer15 at identical detected event times within the common real support (<=118/30s). Motion Transformer15 is fixed; 16D descriptors and old Train normalization are identical. If the gate passes, four cold 144D projection heads share exact initial states, fixed historical NULL, clock-only fixed paths, the same group-balanced schedule and 400 updates. Fourth arm randomizes positive slots as a destructive supervision control. Trainable matrices and budgets are equal. No Val selection. If either population is insufficient, all four training arms remain NOT_STARTED; no threshold adaptation.

Rebuild full source features and dense nodes in a fresh process, recheck all clocks, then reload saved heads with optimizer constructors forbidden. Require exact node, candidate, clock-template, schedule and prediction arrays and equal scientific metrics. Public summaries omit private source identities, paths, weights and raw tensors.

## G1: frozen event condition causality
Use the initial native upper8 Cross Adapter referenced by the frozen root study protocol, not a newly selected checkpoint. Select the first two distinct music identities from its eight registered validation pairs without observing outcomes; source A is a different recording of the same clock/choreography, source B is a different choreography from that registered list. These are exposed validation examples, not blind generalization or exact cross-capture pose ground truth.

Hold music, x_t and noise fixed at t=100/500/900. Compare source A/B, removed reference, mask-only, dense/completion visibility, corrupted unknown storage and bypass. Trace 153D geometry, 128D encoder/attention, 512D residual, EDGE music/condition and all eight decoder blocks. VJP and centered finite differences at t=500 probe feature-space sensitivity (not a feasible SO(3) repair witness).

Legacy is the default and must reproduce the exact former formula with identical state-dict keys. The explicit observed research policy sanitizes unknown entries, supplies Transformer and MHA key-padding masks and bypasses all-missing event inputs. It is inference diagnosis on legacy-trained weights, not proof of newly trained visibility adaptation. No new training occurs in G1.

Free-run seven conditions per music with the same seed using real frozen EDGE, Jukebox and 50-step DDIM without hard guidance: 14 generations, followed by 14 fresh-process recomputations. Use official stitching/FK and unchanged Native31 with complete continuous diagnostics. Following requires all two cases to have observed-mask two-way source preference >0.001 rad, lower own-reference distance than bypass, and >=90% bypass upper-body angular activity. Native qualification is separate; a feature norm change alone is not following. Failure does not trigger tuning or bilateral/SQP escalation.

The source-only unified primitive contract binds canonical time map, audio identity, observation SHA, up to three actual audio-detected beat/central source keys, and source boundary contexts. It explicitly marks generated neighboring context absent, expression identity and contact semantics UNKNOWN. SHADOW cannot route to production.

## Evidence and limits
Run dependency-appropriate full regression and supplemental EDGE tests under tmux, retain execution commit, logs, UTC timestamps, exit status and SHA manifests. Publish code first. Actual GPU execution, fresh replay, physical gates and semantic qualification are reported separately. No human event correspondence labels are manufactured. Existing historical source windows remain development-exposed.
