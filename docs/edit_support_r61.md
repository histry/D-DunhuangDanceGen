# R61 — Intervention Validity and Edit-Support Consistency

Parent `6f72406a11efaaf0013694ef2e587c4e2348782e`. No production routing, Planner expansion, duration grid, new training, stage-detector retuning or physical-qualification changes.

## Frozen pilot before quality scores

The fixed roster has all eight ChangE recordings and FineDance001–004 from the disclosed official @Genre Train diagnostic assets. Reuse R60's authenticated whole-record kinematic evidence and accepted temporal components. FineDance's shared observed temporal support does not imply ChangE quality-mask or Native24 admission. All sources have prior development exposure. The exact six post-hoc diagnostic viewing intervals are excluded; this is a mechanism pilot, not a new unseen-source generalization test.

Select the first chronological eligible event of the registered type per source: three holds, three root-angular spans, six upper-region speed peaks. Holds must have the detector's existing >=30-frame support, root-angular spans >=12 frames (eligibility, not a detector change), peaks use their original three-frame evidence. Require 45 actual context frames before/after the evidence, plus sufficient fixed-gap boundary observations within one accepted component. No model error or result score is consulted; missing rows remain NOT_MATCHABLE.

For P0, A4 contains the entire original raw support plus 45 frames on each side. A5 is a strict right-end crop of A4, retaining the identical left boundary. Peak candidates cut before the original raw evidence and delete the previously present peak center. Holds/root-angular candidates cut at the raw span midpoint and delete the latter half. `added_frames=0`. Length changes are explicit: this is intervention validity, not length-controlled phase benefit. Do not call a root angular span a verified yaw turn, or an automatic evidence deletion a human-confirmed choreographic phase deletion. `verified_phase=UNKNOWN` and `human_reviewed=false` remain.

The source-only viewer shares the absolute source clock. A5 ends at its real last frame; gray subsequent poses are labeled deleted context, not frozen output or padding. An A5-only playback ends at its actual crop length. MP4 previews retain the same original speed and explicit deleted-context display. No paired music is claimed.

## Three distinct outputs

1. INTERVENTION_VALIDITY: original raw/band identity, nested crop, source deletion mask and retained original motion. P0 performs no reconstruction edits (`M_edit=[]`).
2. EDIT_SUPPORT_AUDIT: actual overwritten target frames plus source interpolation reads. `floor(q)` and `ceil(q)` conservatively project evidence for fractional positions. Raw evidence, engineering band and independently verified phase support stay separate. `verified=None` implies intrusion=null, never zero. Crop removal is audited separately. Current real tasks use identity source-to-local clocks; nonidentity transport is contract-tested, not a new music experiment.
3. INBETWEENING_QUALITY: one fixed 48-frame hidden target centered on the selected event and the same four real observations on each side. SLERP/G1 use the same input and target. All outputs retain observations exactly; hidden truth is evaluated only after solving. Full edit-mask overlap with both target-event and all-source proxies must be recorded, including when a center cut appears safe. Intruded contents are Reconstruction, not source-preserved claims.

CLI: `python -m dunhuang.research.edit_support_r61 main`, then `verify_main`. Only after local viewing acceptance bound to `main/result.json` may `fixed` run. `verify` independently rebuilds the fixed targets, numerical outputs and masks with exact result/NPZ replay. Each action must run from the project root in a fresh output directory. Preserve all rejection states and original R48 numerical screen; body22 root-relative diagnostics are not Native24.

Gates: exact kinematic deletion and mask consistency can pass without artificial human labels; semantic/perceptual phase validity remains unestablished. Compare G1−SLERP on the identical target, recording-balanced with source-level uncertainty, not R60's different A4/A5 hidden targets. No positive pilot statistic automatically authorizes new training, planning, co-duration searches or paper novelty claims. The second supplied duration-allocation proposal is deferred.
