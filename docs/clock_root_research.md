# Clock-constrained NULL alignment and joint Root/contact conditions

This research path leaves the EDGE generator, Native31-limit acceptance, default heuristic Planner and old checkpoints intact. It does not reproduce the VTaMo sign-language model.

## Frozen design

The source is [VTaMo](https://arxiv.org/html/2607.09126v1) and its [public OT implementation](https://github.com/junyi2005/vtamo/blob/main/vtamo/ot_sinkhorn.py). We independently implement music/motion matching; no sign/text reordering or T5 code is copied. The old danceRAG paired v3/local/sync reports remain unqualified; their losses and probes are not evidence of new source generalization.

Use the verified AIST manifest (1480 Train,481 Val;457 Test excluded from new feature extraction/evaluation). Every source-origin attestation and associated recording/choreography/music role is checked. Full-audio50D observations are extracted with the frozen legacy implementation; motion74 is recomputed from actual150-frame native observations. Music has a real120-frame core, motion a real150-frame halo; four offsets are noncyclic crops. No shift label enters the model clocks or features.

Freeze the old global SDMR encoder and normalization. All A/B/C arms use the same Train schedule,800 updates,batch12 and fixed final weights. A refits the original sync architecture initialized from the prior frozen sync head; B/C begin with identical identity projections and learned dustbin cost. Different A/B parameter counts and prior sync pretraining are disclosed; B/C match parameters and initialization exactly.

- A: original Sync Head.
- B: clock-banded NULL OT, entropy0.15,80 Sinkhorn iterations,0.5s band, expected-position monotonic regularization.
- C: B plus stop-gradient position-weighted InfoNCE. Only distinct known association components serve as cross-example negatives; same choreography/music captures are never negative solely because dancer identity differs.
- Energy and frozen global retrieval are separate baselines. Unrelated-pair controls require different music/choreography/component and the same start position, preselected by energy correlation. This is pair-identity discrimination, not a claim that a different dance is artistically illegitimate.

NULL capacity comes from both observed populations and is not a fixed20% target. Matched coverage, NULL fractions, marginal residuals, band leakage and monotonic violations are reported. A narrow band is not strict monotonicity; the OT plan is never used to replace the Planner time warp. Clock consistency is prior-biased and not independent action-peak localization truth. Only two Val source components exist, so bootstrap intervals have unestablished coverage. Val is previously exposed development data; no scientific/operational qualification is granted.

The service accepts an optional hash-bound transport checkpoint, computes scores on the same source_positions time map and forces Shadow mode. Existing ranking and generation retain their original authority. Orthogonal calibration and learned attention pooling remain disabled; a fixed even/odd-frame and antialias phase diagnostic is reported without selecting a new pooling method.

## Root/contact path

Root/body responsibility uses the exact vector identity J_world=J_root_translation+J_root_relative. Squared components include their signed cross term; they are not causal percentages. Root-relative still includes orientation, while the additional root-frame measure removes it. Holding translation, the full Root rigid transform, or local body rotations produces diagnostic counterfactuals, never executable repair candidates.

The37D source primitive contains local displacement/velocity, heading/yaw change, geometric contact proxy and masks, source movement saliency, and entry/exit velocity/heading. Independent static-contact confidence stays0. Unknown Root storage cannot become stationary truth. Source-only time-warped descriptors require explicit source_root_observation_v1 evidence: source motion hash,30Hz,m,Y-up,provenance, observed-mask reference and optional geometric-contact-mask reference. Legacy Dunhuang events without this evidence receive zero values plus absent observation masks.

RootContactAdapter adds a zero-initialized condition projection to the existing event/music Adapter. It never overwrites EDGE Root output. The matched full vs masked-condition arms start from the same previously selected cross Adapter, use400 updates with identical batches/noise, and save fixed final weights. Cross-recording official routine/time supervision is not exact pose/Root retiming truth. Six150-frame demonstrations use two music windows, one fixed seed, full/original, full/Root-removed and masked/original. Six samples do not establish quality or generalization. No RootNet or Root SQP is added.

## Commands and receipts

Publish before server execution. Use a new output directory and tmux. Actual paths belong to private runtime assets.

    python scripts/root_responsibility.py --assets outputs/aist_features_01/assets.json --prior outputs/native_calibration_03 --out outputs/root_responsibility_new
    python scripts/clock_alignment_experiment.py --assets outputs/aist_features_01/assets.json --global-checkpoint outputs/sdmr_full_02/retrieval.pt --sync-checkpoint outputs/sync_frozen_01/sync.pt --config configs/clock_alignment.json --out outputs/clock_alignment_new
    python scripts/root_condition_experiment.py --assets outputs/aist_features_01/assets.json --initial-adapter outputs/event_transfer_01/cross/train/adapter.pt --steps 400 --generate --out outputs/root_condition_new

EDGE training/generation uses the pinned EDGE Python; diagnosis/alignment/Native replay use their appropriate pinned runtime.

    python scripts/verify_clock_root.py --assets outputs/aist_features_01/assets.json --clock outputs/clock_alignment_new --root outputs/root_condition_new --diagnosis outputs/root_responsibility_new --out outputs/clock_root_replay_new

The fresh verifier re-extracts observations, recomputes frozen encodings and A/B/C measurements, reconstructs Root/body diagnostics, and invokes the real Native worker on saved new generations. It never retrains or resamples. Keep full logs, input/output hashes, launch/exit receipts and failed attempts. Public Git receives code and aggregate results only; raw source identities, waveform/model data and server paths remain private. Remove only SSH helpers and transfer archives created for this run.
