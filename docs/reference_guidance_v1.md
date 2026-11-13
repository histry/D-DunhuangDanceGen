# R4-0 reference-only control and A2 event information audit

Baseline: e531ea7ca8f32299c9a222ef8b6ae681ace79d1e. Production generation,
existing configs, thresholds and frozen model weights are unchanged. All new
logic lives in research/reference_guidance; no unused upstream network is copied.

## Registered R4-0
Four Train pairs are reused exactly from source_capacity_03 (four music groups);
four exposed Val source pairs and their saved C2-A free-sampler states are reused.
Times: 999, 499, 19. Train uses the prior deterministic q(X_A) noise construction.
No new case or time is selected using results.

The actuation space is the existing 512D condition, not a new latent/Root network.
Use the actual CFG2 clipped x0 prediction. B0 is the frozen C2 Adapter direction
for the provided source, B1 is the negative observed-reference SO(3) loss gradient,
B2 weights source-only kinematic phases by 3 and limits injection to their
radius-two visible frames. Detect phases independently in each fully visible
contiguous block with the frozen Data V3 detector; discard block-edge proposals.
No derivative is computed across a missing interval. Legal provided keys validate
the source contract but are not treated as kinematic phase labels. B3 is an equal-norm random direction;
B3_stage additionally uses the B2 support for a fair phase-information control.
Every direction has norm 1% of the base condition RMS over all 150x512 entries.
This is not a line search. The original target-dependent Oracle is excluded.

Controllers receive only source-condition values, visibility and verified mapped
keys. Hidden/full target poses and the other source enter evaluation only.
The registered map in these same-clock probes is identity; this is not evidence
for nontrivial tempo adaptation. Unknown source values are sanitized before SO(3).
Derivative checks use central differences at .25/.125/.0625, at least two with
relative error <=.1 and cosine >=.99, and JVP/VJP adjoint error <=1e-4.

The conservative Train gate requires, for every source/case/time, trustworthy
derivatives, own-reference gain >.001 rad, own-minus-wrong gain >.001 rad,
greater gain than the matched random control, and protection: other-joint mean
change <=.01 rad, Root RMS <=.01m, contact max change <=.05, activity ratio
[.9,1.1], angular acceleration/jerk P95 proxy ratios <=1.1.
These guards are not learned realism. Native is audited on four Val t19 baselines
and 40 controlled x0 predictions (44 audits); none is a full 50-step generated dance.
Full-chain guidance and low-rank mapping training require a separate registration,
even if this local gate passes. Failed local gates stop that progression.

## Registered A2 probe
Reuse all four frozen 15/30/dual/dual_shuffle encoders. Generate music onset/beat
labels independently from the original waveform event sidecars, compare their
independent replay hashes, retain raw-audio identity. Motion onset/speed peak/turn/
hold are full-source upper8 kinematic proxies computed before cropping.
A positive frame is within two frames of the detector event centre.

Canonical physical frames remove overlap duplicates; music is also deduplicated
across recordings. Keep up to 240 frames per audio asset and 96 per motion source,
uniformly covering their available frames; use the context nearest its centre,
with UID as deterministic tie-break. This is a sampled-frame readout, not full
timeline localization. Test is forbidden. Val remains six known exposed music
groups; no new independent semantic labels are created.

Fit fixed-lambda .01 ridge heads on Train only, with source-group/class-balanced
weights, Train-only standardization, fixed .5 threshold. Compare raw 50D/74D,
simple time+signal summaries, all four frozen tokens and within-asset content
permutations. In addition swap Val features under the same fitted head.
Minimum support is 20 Train positive/negative frames, 10 Val positives and two
evaluable Val music groups. Otherwise report NOT_EVALUABLE. Report group AP/F1,
paired group bootstrap intervals (1000 replicates); motion recordings are averaged
before music groups. No best model is selected on Val.
Music50 already contains audio-derived controls and motion74 contains kinematics;
decoding these automatic labels is not independent semantic understanding.
Cross-modal event correspondence and full-timeline localization stay NOT_EVALUABLE.
The actual closed-form fit count is reported; encoder/Adapter optimizer updates
are zero, and the independent replay loads saved heads without refitting.

## Mechanism references and limits
[Guided Motion Diffusion](https://github.com/korrawe/guided-motion-diffusion)
motivates inference-time gradients; this experiment uses local SO(3), not its
spatial trajectory protocol. [OmniControl](https://github.com/neu-3dvi/OmniControl)
motivates separating control and realism; our kinematic guards do not reproduce
its learned realism guidance. No claim of paper reproduction or quality transfer.

## Execution
Push code/config/tests first. Server launcher: scripts/run_reference_guidance.sh
with fresh study and receipt roots, inside tmux. It runs full regression, EDGE
checks, then fresh-process control and event replays. Seal using
scripts/export_reference_guidance.py --seal only after exit_status=0.
Preserve any failed attempt directory and its receipt.
Public outputs are aggregated; private inputs, poses, identities, weights and fits
remain outside Git. SSH credentials and relay channels stay in RAM; no temporary
SSH helper files are needed.

## Preserved execution attempts
Attempt reference_guidance_01 reached full regression 230 passed/1 skipped and
EDGE checks 14 passed, then stopped before any probe: the norm helper attempted
NumPy conversion of a CUDA tensor. The correction computes the equivalent 1%
L2 norm on the tensor device and adds a regression test. No budget, data, gate,
random seed or control amplitude changed; the failed root/receipt remain intact.


Attempt reference_guidance_02 was deliberately interrupted during the first Val
case after static contract review: the inherited keys combine a centre and music
beats, not independently detected source kinematic phases. No metric or parameter
selection was made from that attempt. B2 was corrected to detect phases from
visible contiguous reference blocks only, with unknown-poison invariance tested.
Earlier key-gating outputs are excluded from the corrected registered comparison.
