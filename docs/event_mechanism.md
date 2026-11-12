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


## Cross-recording training and parameter-matched fusion

    PYTHONPATH=src python -m dunhuang.research.legacy.event_transfer --assets outputs/aist_features_01/assets.json --out outputs/event_transfer_preflight_new --preflight-only
    PYTHONPATH=src python -m dunhuang.research.legacy.event_transfer --assets outputs/aist_features_01/assets.json --mode cross --steps 1200 --out outputs/event_transfer_cross_new
    PYTHONPATH=src python -m dunhuang.research.legacy.event_transfer --assets outputs/aist_features_01/assets.json --mode additive --steps 1200 --out outputs/event_transfer_additive_new

Train source and target are different authority-linked captures; full151 supervision always reads the target, while the condition reads only independently observed source upper8 rotations. No Test fitting. Native source features contain absolute/first-observed-relative/adjacent SO(3) columns plus availability and phase (153D). They do not require source rotations to fit the target Scaler. Missing storage cannot enter the encoder.

A single128D event Transformer, projected frozen512D music tokens, one attention layer and a zero-init512D residual form the research Adapter. Cross fusion uses music queries into event keys/values; the additive comparator uses event self-attention plus music projection. Both use every module, have identical parameter counts and identical initial weights. An auxiliary source-rotation reconstruction head preserves event geometry; it is not target supervision.

Registered objective: full151 target MSE +0.1 normalized target-velocity error +0.02 SO(3) chordal rotation error +0.05 observed-source reconstruction. All coefficients, steps, data roles and checkpoint selection are fixed before either run. Selection uses fixed8 cross-recording Val pairs, full151 MSE every100/final steps, maximum1200. Report original, zero Adapter, removed reference, wrong source, reversed phase, alternate music and intentionally shifted music controls. Activity and body-group errors are diagnostic safeguards, not a quality certificate. Historical Val exposure remains explicit.

The mechanism runner also accepts these checkpoints and repeats the same frozen information-flow/factorial protocol. Its same-record conditional denoiser interventions are distinct from the trainer's cross-recording validation and are labelled accordingly.

The existing generate.py/EDGE worker supports native event geometry only with both explicit research flags shown in configs/native_event_adapter.example.json. Default generation and old Adapter schemas remain unchanged. Native-source geometry is never an output target or a pose-pasting constraint. New research checkpoints cannot load through the default qualified Adapter path.


## Registered server sequence

    PYTHONPATH=src python scripts/transfer_experiment.py --paired-assets outputs/aist_features_01/assets.json --generation-assets outputs/event_plan_01/prepare/assets.json --plans outputs/v07_01/plans/index.json --out outputs/transfer_experiment_new

Publish code before running. The script writes the protocol, runs complete regression and source-task preflight, then cross1200, its12-case factorial, two real ChangE/A65 Event Plan cases and fresh validation replay; only then the identical-budget additive ablation. All28 free-generation executions are retained, including duplicated zero-Adapter controls across methods; this is not28 unique motions. A crash preserves its own output directory and exit receipts. No automatic retry, DEV tuning, native threshold change, long-song promotion or learned feasibility probability is performed. Unsuccessful trajectory executions do not blacklist the source event.


## Independent evidence replay and pretraining limits

    python scripts/verify_event_transfer.py --assets outputs/aist_features_01/assets.json --training outputs/event_transfer_run/cross/train --out outputs/event_transfer_cross_replay_new.json
    python scripts/verify_transfer_execution.py --baseline outputs/mechanism_baseline --experiment outputs/event_transfer_run --out outputs/event_transfer_native_replay_new
    python scripts/audit_pretraining_exposure.py --assets outputs/aist_features_01/assets.json --out outputs/pretraining_recipe_membership_new.json

The neural replay uses a fresh process and checks all seven controls, source-pair identity, complete training history, the predeclared best-checkpoint rule and unchanged parameters. Native replay executes the original authority in fresh directories, compares exact motion/FK and full physical receipts, and recomputes saved probes/MSE. Zero-Adapter factorial motions must remain bitwise equal to the original mechanism baseline; new fusion and sampler observation cannot alter that control. No new sampling or optimization is permitted in these verification paths.

The provenance audit intersects current sources with the pinned official EDGE recipe's Train/Test minus ignore lists. This is a list-membership audit, not a reconstruction of the downloaded checkpoint's actual optimizer/data lineage. Even a source absent from these lists is not certified unexposed. The current protocol never claims a fresh blind test or unknown-Dunhuang generalization.


    python scripts/audit_music_response.py --assets outputs/aist_features_01/assets.json --training outputs/event_transfer_run/cross/train --out outputs/event_music_attribution_new.json

This additional read-only audit compares correct/replaced/shifted music with and without Adapter injection under the same noisy targets. It separates inherited frozen-EDGE mismatch sensitivity from any incremental Adapter effect. It does not select weights or tune the training protocol; perturbation sensitivity is not proof of reasonable musical expression.


The original7a1d368 denoiser field named torso covers local rotation channels for joints0..11 (including legs). Later output uses body0_11_rotation to make that scope explicit. Raw historical receipts remain unchanged; public summaries never interpret that field as a torso-only measure. Native and all training body-group calculations are unaffected.


The execution verifier also audits the exact fixed8 fitted Val target motions under unchanged Native limits. Representation support and physical acceptance are separate. These source-quality observations neither filter training data nor reselect checkpoints and do not turn fitted151D values into sensor truth.


7a1d368 research weights bind protocol_sha256 but omit the optional condition_scope/policy descriptive fields; the runner used the fixed completion_upper8 cross-recording task recorded in that bound protocol. These original checkpoint bytes and hashes are preserved. Future saves embed the descriptive scope/music-source/target fields too, and inference audits display an explicit missing-header marker plus protocol hash instead of silently writing null. Qualification flags remain false.


    python scripts/audit_training_targets.py --assets outputs/aist_features_01/assets.json --training outputs/event_transfer_run/cross/train --out outputs/train_target_native_audit_new

Run with the pinned Native Python. This descriptive audit covers all Train targets admitted to the cross-recording task, after the registered comparison. It never loads Val/Test targets, changes data membership, filters recordings, reweights training, selects checkpoints or starts an optimizer. Native failures are contract membership results, not assertions that real dances or their events are invalid. Per-target JSONL stays in ignored outputs.


    python scripts/audit_transfer_pfc.py --baseline outputs/mechanism_baseline --experiment outputs/event_transfer_run --out outputs/event_transfer_pfc_new.json

Reuse the existing EDGE PFC implementation and execute the unmodified pinned official calc_physical_score on in-memory copies of every saved FK. Per-case values must match exactly; zero-root-acceleration remains undefined. PFC uses the upstream displacement-product-times1e4 definition, not m/s or a Native threshold. These small registered pools are not a paper leaderboard, and no motion is modified.
