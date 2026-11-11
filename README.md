# DunhuangDanceGen

当前正式路径：**Event Catalog → Music Timeline → Event Plan → 冻结 EDGE（Temporal V0.5）→ 原 Native 验收**。保留官方 EDGE 基线。已完成 5 秒、15 秒、整曲六例，全部 Native 未通过。学习检索处于 shadow，Adapter 因真实资产不足未训练，尚非已验证的 V1。见[本轮报告](reports/EventPlan实验报告.md)和[机器指标](reports/event_plan_workflow.json)。

## 正式入口

公开 profile 与运行机器资产分离。从 configs/assets.example.json 填入真实绝对路径、SHA256、固定依赖版本，保存到忽略目录 inputs/runtime_assets.json，占位值不能直接运行。目录源清单沿用 source_records/events，raw SMPL 文件名与哈希必须匹配。EDGE / Native 使用各自环境；prepare 导出原权重 Normalizer 后解码事件目录。

```bash
python -m pip install -e '.[test]'
python scripts/prepare.py --assets inputs/runtime_assets.json --out outputs/new_prepare
python scripts/generate.py --config configs/edge_baseline.json --assets outputs/new_prepare/assets.json --out outputs/new_baseline
python scripts/generate.py --config configs/dunhuang.json --assets outputs/new_prepare/assets.json --out outputs/new_plan
python scripts/evaluate.py --run outputs/new_plan --compare outputs/new_baseline
```

省略 --frames 生成整曲，可指定 --frames 150 或 450。--plan /path/event_plan.json 可重放显式计划，引用、时钟和映射必须验证。换音乐先运行 scripts/extract_music.py，把真实 layer66/4800D 缓存和绑定音频的提取收据填入资产清单；--audio 不会让旧特征自动对应新音频。所有输出使用新目录。

代码 push、服务器 pull 后，在项目目录通过 tmux 运行长任务：
```bash
tmux new-session -d -s ddg-generation 'python scripts/generate.py --config configs/dunhuang.json --assets outputs/new_prepare/assets.json --out outputs/new_plan > outputs/new_plan.log 2>&1; echo $? > outputs/new_plan.exit'
```

## 模块与资格

planning/ 实现事件目录、节拍/能量时间线、单调时间映射及 Top5/beam3 全局规划。约 3 秒节拍吸附切分不是人工乐句；计划覆盖和实际控制比例分开报告，当前仅 25 帧上身原语使用 Temporal 控制。

alignment/ 是 50D 音乐 / 74D 动作双编码器与局部同步头。heuristic 默认启用；learned_shadow 需要显式绑定全局/同步 checkpoint、对应编码器和查询收据，只记录两级评分；learned_qualified 还需验证资格。历史 Test 已暴露，只作为开发回归。同歌检索与同歌时段检索分开评估。

generation/adapter.py 提供零初始化、掩码条件 Adapter，冻结 EDGE 主干，复用原 CFG。generation/train.py 提供真实 151D/4800D 配对、来源分离、目标支持域预检；当前没有可用训练权重。以下训练命令需 EDGE 环境：
```bash
python scripts/train.py --assets outputs/new_prepare/assets.json --out outputs/new_motion_training --task motion --steps 300
python scripts/train.py --assets inputs/paired_assets.json --out outputs/new_paired_training --task paired --steps 300
```
配对资产额外给 paired_generation_manifest 哈希引用。清单 rows 每行给 role(train/val/test)、source_group、start_frame、pairing_evidence=real_aligned_audio_motion、fps=30、frames=150，以及 SHA256 绑定的 motion[150,151]、jukebox[150,4800] 文件引用。训练本身不授予 music_qualified。双向两遍生成、225 帧训练和新 PFG/SQP 实验尚未完成。

## 审计与复核

pipeline.py 是正式生成入口。physical.py 保留固定版本 Native 完整绝对阈值，计划阶段始终 UNKNOWN。每次输出保存音频/特征/权重/代码哈希、真实 FK、完整指标、日志和状态。文化文件名标签只是来源粗类别，不是逐事件真值。

```bash
python -m pytest -q
python scripts/verify_event_plan_report.py
```
第二条只读复核保存实验，不训练、不重采样。生成完成、Native 通过、语义风格和人工自然性分别判断。权重发布来源未独立核实，不宣称严格复现官方论文效果。

## 历史研究入口

以下保留旧实验及证据，当前正式流程不调用历史 SDMR 生成编排。research/legacy/sdmr_select.py 仅为旧脚本兼容。

## B2 时序可执行性实验

同采样器对照使用 `custom_none`、`anchored`、`temporal` 三种 mode，默认重叠策略为 `mean`。`temporal` 还需要 `trajectory_radius`、`trajectory_strength` 以及每个锚点的哈希绑定源片段和中心帧。当前实验半径为 12 帧、强度为 0.55。原 `baseline` 始终调用官方 long DDIM。

在已具备首轮实验产物的服务器上，可用全新输出目录运行以下命令。首个脚本先只读复核保存结果，然后从 5 秒开始执行匹配对照；后两个是独立研究案例。
```bash
PY=/home/disk/lsm/conda_envs/dance-rag/bin/python
$PY scripts/b2.py outputs/new_b2
$PY scripts/b2_music.py outputs/new_b2 outputs/new_b2_phrase
$PY scripts/b2_physical.py outputs/new_b2_support
```

[B2 实验报告](reports/B2实验报告.md) 和 [机器指标](reports/b2.json) 记录各时长的完整比较、支持域与物理审核。

## B3 物理责任与原语准备

[B3实验报告](reports/B3实验报告.md) 和 [机器指标](reports/b3.json) 包含B2整曲的来源窗口FK归因、冻结支撑人口Root-only条件能力、六个几何方向及一次受保护续步、106个中心的事件归并和三锚点联合候选。原Native仍是完整验收依据。复现入口为 scripts/b3.py、scripts/b3_primitives.py、scripts/b3_verify.py。

## SDMR 学习式检索

alignment/ 提供50D/74D双编码器、独立同步训练、文件名粗标签、专家偏好弃权接口和候选评分。scripts/sdmr.py先训练完整框架，scripts/sdmr_ablate.py仅在完整框架结束后重训一次去声学32D的消融。生成入口按检查点和资格收据决定启用或回退。见[实验报告](reports/SDMR实验报告.md)和[机器指标](reports/sdmr.json)。本轮真实评价未通过同步资格。

## V0.6 实验分支：连续事件软条件与局部可训练探针

新增 generation/soft_reference.py：按 Event Plan 的单调时间映射做 SO(3) 测地插值，将真实来源 12–19 关节的连续轨迹、来源相位和支持域掩码送入同一个冻结 EDGE 的 Adapter。在重叠的 150 帧窗口中保持条件一致，超出冻结归一化支持域的关节帧显式标记为未知，不裁剪源姿态、不把参考轨迹后贴到生成输出。保存 soft_reference_audit.json。原 V0.5 Temporal 默认路径及原 Native 物理阈值不改变。

由于目前 151D 全目标合格窗数为零，新增任务 upperbody_probe 仅在 48D 上身支持域内开展带遮挡的补全研究，其他通道使用零值作为输入占位，**不是真实完整 151D 动作**。在服务器 EDGE 环境可单独运行以下研究命令：

~~~bash
python scripts/train.py --assets outputs/new_prepare/assets.json --out outputs/upperbody_probe_v06 --task upperbody_probe --steps 300
~~~

此训练不包含真实配对 Jukebox，得到的权重即使训练成功也只有研究意义，必须保留 music_qualified=false、full_motion_qualified=false，不能作为正式 V1 或合格整曲。event_guided 现在还要求 condition_schema=soft_event_reference_v1；没有实际获得配对数据资格时，默认拒绝该权重。显式 research_allow_motion_only_adapter仅用于自担风险的影子推理，不得当作正式质量提升证据。

下一阶段仍需真实 150×4800D 配对音乐、全 151D 可表达目标，以及150 帧单窗与 225 帧双窗的实际 GPU 物理验收；双侧 Inpainting 与局部 SQP 本次没有实现。不使用测试集指标给予检索或生成器发布资格。

可先用 CPU 预检真实训练来源与支持域而不初始化 EDGE/GPU：

~~~bash
python scripts/train.py --assets outputs/new_prepare/assets.json --out outputs/probe_preflight_v06 --task upperbody_probe --preflight-only
~~~

预检只写 asset_audit.json/status.json；PREFLIGHT_READY_RESEARCH_ONLY 不是模型训练成功，也不是完整物理通过。

V0.6 Train/Val 以完整来源组隔离；upperbody_probe 与 motion 使用独立来源保留集，paired 严格使用 manifest 的 train/val 角色。若缺少独立训练/验证来源，状态为 INSUFFICIENT_DISJOINT_TRAIN_VAL_SOURCES 且训练步数为 0。
