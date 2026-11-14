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
