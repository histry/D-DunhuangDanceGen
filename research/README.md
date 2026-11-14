# Research and baseline boundaries

Production entrypoints remain scripts/generate.py, planning/worker.py and edge_backend.py. Frozen EDGE, Native-Legacy-v1 and the default legacy Adapter visibility are unchanged.

New experiment implementations live under src/dunhuang/research/source_identifiability:
- g2: same-architecture Adapter retraining, exact-source key supervision and corrupted-identity negative control.
- a2: canonical audible-clock coverage and separately registered source-association challenge.

Run these as Python modules with PYTHONPATH=src, not duplicated Python wrappers. Runtime mathematical operators (audio clocks, source-following geodesics and event-condition consumption) are shared and tested separately from the study drivers.

Historical generation trainers have moved from generation/{train,event_transfer,root_experiment}.py into research/legacy/{paired_training,event_transfer,root_experiment}.py. All active callers and their subprocess module names have been updated. generation/targets.py retains only target-coordinate conversion and parameter hashing shared by data preparation and experiments. Historical scientific artifacts and prior failed studies remain available; their exact original code can also be checked out by recorded Git SHA.

Four redundant four-line script wrappers were removed; their launchers now use python -m. No dataset, checkpoint or scientific receipt was deleted. No compatibility layer imports old training drivers into the production generation package.

Observed-trained G2 weights require a matching research runtime. Legacy consumers reject these checkpoints instead of silently applying legacy visibility. The new experiment checkpoints are unqualified and never replace the default generator.

## A2-V3 / G2-M source necessity

The study in `src/dunhuang/research/necessity` reuses frozen operators and runtimes without changing existing source or configuration files. Its separate `configs/source_necessity_v1.json` registers Train-only candidate support, two source-time tasks, exact self-recording training targets, a matched 2x2 supervision/time-step experiment and an occurrence-preserving source-shuffle control.

Task qualification is limited to exposed development mechanism research; it does not promote generic Data V3 training, independent semantic evaluation, default retrieval, generation or G3. Main runs and fresh optimizer-free replays use separate new output roots. See [protocol](../docs/source_necessity_v1.md).

## Paired source and fixed-candidate content tasks

New study drivers are in src/dunhuang/research/source_tasks. They reuse the existing frozen EDGE runtime, sampler, source condition and Native implementation. P0 reads saved generation outputs; P1 binds real paired performances and training-only q support; CF compares shared-state source objectives; the independent A2 comparison shares encoder parameters across rate choices and binds fresh Train-only statistics.

Only one existing production file changes: pipeline.py rejects legacy anchor selection in Dunhuang whole-song requests, so the Planner cannot silently overwrite an earlier learned selection. The research checkpoints remain unqualified and require explicit research consumers. See [registered contract](../docs/source_tasks_v1.md).

## Frozen source capacity and local paths

The no-optimizer study in research/source_capacity replays the frozen C0-C3 trajectories, checks layer responses and JVP/VJP/finite differences, and compares parameter-free injection probes on fixed Train pairs. Its oracle directions use true targets and are not deployable source conditions. The independent local-path study preserves all four content checkpoints, Normalizer and candidate populations, and separates global/temporal content from DTW/NULL freedom. See [contract](../docs/source_capacity_v1.md). No existing baseline source or configuration is modified.

## Reference-only directions and event information

New independent drivers are under src/dunhuang/research/reference_guidance.
The controller reads only provided reference support and source-only kinematic phases;
true hidden poses are evaluation-only. Fixed single-step controls preserve CFG/clipping,
verify derivatives, compare equal-norm random directions and enforce protected-motion
readouts. No full-chain or learned mapping is admitted.
The separate event driver fits only diagnostic Train-only ridge heads over frozen
encoders, binds roles to the declared unseen-song registry and explicitly reports
shared choreography exposure. Fresh replay loads heads without fitting. Continuation
uses hash-frozen successful control artifacts with separate execution commits.
See [contract](../docs/reference_guidance_v1.md) and [results](../reports/参考动作可用引导与事件信息审计报告.md).

## Protected source, interval/pose and Shadow planning

New drivers live under research/protected_guidance, research/event_information and research/planner_integration. They isolate small matrix-free source projections, no-weight-learning information audits and E1-W condition/identity checks from production. Norm-matched shrink/random controls, explicit proxy-label scopes, Train-only statistics, actual Planner consumption and independent replay remain mandatory. See [protocol](../docs/protected_source_v1.md) and [report](../reports/受保护来源控制与事件信息及规划集成报告.md).


## R6 source expression and private blind review

The source_expression package isolates frozen failure attribution, Train-reference-only pose/dynamic scales, five common-dose protected objectives, multi-audio Shadow planning and private review preparation. Historical weights, source/config baseline and qualification thresholds are unchanged. Saved-array verification is independent of GPU replay; human data review remains separate from model admission. The media-only continuation explicitly binds prior successful stages and retains the failed floating-WAV timestamp receipt. See [protocol](../docs/source_expression_r6_v1.md) and [report](../reports/来源动态控制与独立盲审准备报告.md).


## R12 explicit motion-state Bridge

The research-only package src/dunhuang/research/motion_bridge compares matched Train-only
music/motion/Decoder injection directions, then trains two 311,808-parameter observed-state
Bridges with raw-anchor and raw-seam objectives. B3 adds bidirectional attention; B2 uses a
parameter-matched pointwise block. Checkpoints remain unqualified; no production consumer
or default-profile change is introduced. Frozen C2/R11 artifacts remain comparison baselines.
See [protocol](../docs/R12_MOTION_BRIDGE_PROTOCOL.md) and
[results](../reports/R12显式动作Bridge与双侧状态实验报告.md).


## R13 direct mask-conditioned denoising

src/dunhuang/research/masked_denoiser directly conditions motion input and finetunes
5,040,279 selected EDGE motion parameters. Music and unselected layers stay frozen;
this is not a bounded residual Adapter. Tiny-fit admission precedes population training,
with replayable self-policy snapshots and separately reported q/rollout/Native diagnostics.
The pretrained official DanceFusion baseline is audited, not claimed runnable with missing
data modules. Checkpoints remain research-only and do not feed the production pipeline.
See [protocol](../docs/R13_MASKED_DENOISER_PROTOCOL.md) and
[results](../reports/R13条件去噪与双侧Bridge实验报告.md).


## R14 boundary-state representation and frozen input capacity

src/dunhuang/research/boundary_state_r14 isolates immutable-template necessary bounds,
full-step frozen R13 responsibility readouts, selected-parameter gradient diagnostics,
five matched analytical readouts and reference-only input-space capacity probes.
Legacy boundary_state (R9) and masked_denoiser (R13) remain frozen.
A2 passes the registered predictive representation gate, but single-state Native/FK
capacity fails; no new population training, output Root writer or full songs.
Saved models, trajectories and Native arrays have independent exact replay.
See [protocol](../docs/R14_BOUNDARY_STATE_PROTOCOL.md) and
[report](../reports/R14边界状态表示与冻结条件容量报告.md).


## R15 protected Root–Pose capacity

src/dunhuang/research/root_pose_r15 separates strict immutable-template exclusions from
local mechanism qualification. Frozen R14 self-policy states compare Root-only,
Root-plus-lower-limb and world-FK-guided input interventions under the same total budget.
Paired hidden targets are Train-capacity oracles, never inference conditions.
All finite candidates receive actual output trust/protection checks and full Native reports.
The registered local gate fails; no feedback neural branch, population training or full songs.
544 Native/FK arrays and 10 array bundles have exact independent replay.
See [protocol](../docs/R15_ROOT_POSE_PROTOCOL.md) and
[report](../reports/R15联合RootPose容量与物理编辑域报告.md).


## R16 event-synchronous continuous-window research

continuous_r16 creates one global event/motion/music clock, verifies real same-recording
AIST window pairs and source-only ChangE E1-W contracts, then compares frozen C0
30-frame grafts against fully generated 150-frame C1 windows. Source keys remain soft;
all new outer joins and full Native results are measured. Independent exact sampling
replay passes; the joint mechanism gate fails despite positive source response.
No C2 state model or full-song promotion is claimed. Four fixed-world diagnostic
videos compare real paired AIST, C0 and C1 without smoothing or Root recentering.
See [protocol](../docs/R16_CONTINUOUS_PROTOCOL.md) and
[report](../reports/R16连续事件生成与真实窗口对照报告.md).

## R17 global-clock dual-view state research

Independent package: src/dunhuang/research/global_clock_r17. A 225-frame state is read through two 150-frame frozen R13 views. Matched cross-window and self-window heads each train for exactly 240 updates; global Root/FK losses do not supply future Root at FREE/PROGRESSIVE inference. Hard/soft factor traces and full sampler/Native replays are saved separately. Mechanism qualification is 0/4 and Native 0/36; production and R13–R16 remain frozen. See [protocol](../docs/R17_GLOBAL_CLOCK_PROTOCOL.md) and [results](../reports/R17全局时钟双窗口训练与Mask归因报告.md).

## R18 task-matched denoising

Independent package: src/dunhuang/research/task_denoising_r18. Exclusive cross-view keys and disentangled content/clock/phase controls are compared with a matched self-view model. Both selected R13 motion layers and the state head adapt for120 updates on two Train sequences; all three tasks receive ranking supervision. Tiny admission failed, so eight-group training, D3 and new whole songs were skipped. Full22 sampler replays and read-only objective-gradient replays are exact; Native0/22. See [protocol](../docs/R18_TASK_DENOISING_PROTOCOL.md) and [report](../reports/R18任务匹配去噪与独有内容训练报告.md).
