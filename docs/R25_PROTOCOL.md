# R25: trained full-event and explicit local-phase adapters
Baseline665cd5a6bb45eb896f50cecf99f1f688b7cdf2c2. Independent research/lodge_adapter_r25; formal generator and R22–R24 remain frozen.
This is a project adaptation inspired by multi-layer zero controls and part/sequence reference encoding, not an OmniControl/MoST model reproduction.

## Data and roles
FineDance real-audio admission fails:207 WAVs under danceRAG are Git LFS pointers, zero real waveforms. Use the fixed28 AIST recording identities from R22:24 Train/12 music groups,4 exposed DEV/4 groups.
Re-decode original SMPL, authenticate source/audio/calibration SHA and VERIFIED_PAIR clocks, crop actual256 frames and real audio, extract official256x35 features twice. If the original225 crop lacks31 frames, shift the same recording crop backward only; never pad. Record every shift and Normalizer distortion.
Native columns/Y-up → Lodge rows22/Y-up; retain original FineDance skeleton and Normalizer. C4 is unknown. Body targets are official normalized/clipped132 rotation coordinates. Raw Root does not become a clipped truth: Root and outside-Core behavior are distilled against the same frozen prediction. DEV music, recording, raw-motion/audio identities are disjoint from this adaptation, but historical exposure, choreography/performer overlap and pretraining exposure remain.
ChangE/new Dunhuang music are never paired training truth.

## Matched actual training
Two models: event and phase. Same seed, parameter count, weights, Train ledger, objective,600 AdamW updates each, batch2,lr0.0002,gradient clipping1. Fixed final checkpoint; no DEV/model selection.
Part encoder sees all22 body joints, including wrists20/21:6D rotation, angular velocity and joint validity, pooled into trunk/two arms/two legs. Temporal convolutions encode ordered local memory and one global event token.
Actual Lodge layers1/4/7 (zero-based of9) receive cross-attention residuals via zero-initialized64→512 heads. Backbone and music parameters are frozen, but gradients flow through them. Do not fix the unused upstream FK refinement stack.
Both models reserve the same reference/target phase projections. Event sets their8 channels to zero; phase reads mapped target clock and four kinematic landmark bumps. Landmarks are speed-change/speed/curvature/low-speed proxies, not cultural phase truth. This finite initial phase mechanism does not claim learned semantics.
References are96 true Core frames at[64,160), with legal monotone view reparameterization,24-frame occlusion and a masked arm/wrist joint. Invalid predecessor velocities are masked. Second row reference dropout every5 steps (120/1200 presentations), independent from official music dropout0.2. At least one reference-present row per update. Masking mitigates a copying shortcut; it does not prove music use.
Loss:1×Core normalized-body MSE +0.1×Core first-difference MSE +0.1×Root teacher preservation +0.25×outside-Core body teacher preservation. Teacher and student share exact xt and music-dropout RNG state. No added contact/foot/Native loss or guidance search.

## Checks and finite evaluation
Initial heads preserve real Lodge output exactly. Empty reference bypasses the branch exactly before/after training. Actual pre/post weights, gradients and layer residuals at0/1/299/599 are saved; zero encoder gradient at first zero-head step is expected.
Curves at0/300/600 are Train only. Optimizer-free replay reconstructs actual saved updates and all curves; it is not a second full training.
ChangE: same R24 source pair, original music35 and seeds24041/24042. New adapters use the official soft sampler with ALL state-write masks zero. Music CFG is unchanged; reference remains in both branches. R2412 outputs stay frozen baselines.
Event-correct12 + phase-correct/phase-removed/phase-wrong/reference-swapped/music-plan60 =72 new ChangE trajectories. Null-reference rows reproduce R24 no-reference outputs exactly.
Paired: fixed probes0/2/4/7 and DEV24/25/26/27, seed25041; eight modes baseline, event/phase correct, phase reference removed/swapped, phase removed/wrong, occluded reference/music-swapped =72 trajectories. All21 batches/144 trajectories receive fresh-process replay with identical Gaussian draws.
Report upper8, upper10 including wrists, wrist rotations, wrist-to-shoulder paths, elbow angle and double-arm relationships, source timing proxy and beat-distance proxy. Nonzero response is not perceptual music adaptation.
Jerk partitions all253 four-frame templates into legacy-cue-neighborhood, unprompted-Core,Core edge and outside. R25 has NO actual writeback cues. Include global high-frequency and foot/Root geometry proxies; contact truth remains UNKNOWN.
Per music×seed group require both events:5% body-source gain over R24 sparse, wrists≤1.1×sparse, body/upper10 prefer assigned reference, global/Core/outside Jerk≤1.1×no-reference. Four groups per trained model; no operational promotion. Explicit phase contrasts are reported separately, never replacing the original R24 gate.
Original139D/22 output cannot enter full Native24; no fabricated terminal rotations/fingers. No whole songs, actual-history experiment or simulator teacher this round.

References: [OmniControl zero controls](https://github.com/neu-3dvi/OmniControl/blob/76336a6e78f074f3ba7059643414b91a1eced9f2/model/cmdm.py), [MoST official](https://github.com/Boeun-Kim/MoST), [pinned Lodge](https://github.com/li-ronghui/LODGE/tree/f38642ad67bc3402444f5f914010a911e5bd5c89).

Before training and DEV reads, register the secondary music-plan contrast: keep canonical full-reference content but map phase clocks using existing E1-W solver and actual native35D beat indices per music. Record all four legal-map certificates and score against corresponding retimed-reference proxies. It is separate from the fixed R24-matched primary comparison; never call the new retimed source paired musical truth. Source phase order compares pairwise order against its own source landmark order, not channel-index order.
