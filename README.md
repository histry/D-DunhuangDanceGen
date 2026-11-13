# DunhuangDanceGen

当前默认路径：**Event Catalog → Music Timeline → Event Plan → 冻结 EDGE（Temporal V0.5）→ 原 Native 验收**。学习检索保留shadow。研究路径已完成真实151D配对补全，以及原生SO(3)跨录制条件的Cross/Additive同参数量训练（各1,200步）；事件跟随与完整生成质量仍未合格。

上一轮事件机制实验共40次150帧执行、32个不同动作哈希，Native0/40；两种融合效果近似。对1,468个Train目标的只读Native审计有40窗通过，未据此过滤数据或重选权重。详见[事件机制与跨来源训练报告](reports/事件机制与跨来源训练报告.md)、[机器汇总](reports/event_mechanism_summary.json)及[运行合同](docs/event_mechanism.md)。

新增[时钟约束对齐与Root联合条件实验报告](reports/时钟约束对齐与Root联合条件实验报告.md)及[机器汇总](reports/clock_root_summary.json)：实现带NULL的时钟OT/位置对比、37D来源Root–Contact软条件和责任诊断，完成A/B/C固定预算训练、Root掩码消融、六例150帧生成与独立复核，服务器128项测试通过。时间辨别未建立稳定优势，Root条件收益仍未成立，Native0/6；新路径保持Shadow研究模式，默认检索与31项门限不变。

新增[Native物理评价可靠性报告](reports/Native物理评价可靠性报告.md)与[机器汇总](reports/native_reliability_summary.json)：本轮只读重放1,468个目标和42次历史生成执行，并完成地面/骨架/噪声及23个合成测试；服务器117项回归通过，独立复核通过。旧31项限制与判决不变。Native-Calib-v2仍为未获准的Shadow，真实静态接触、误拒率与专家一致性因缺少独立标注保持未知；本轮没有训练或新采样。

新增 [Data Split V3 / ChangE-E1 离线协议](docs/data_event_v3.md)及[服务器实验报告](reports/数据划分V3与ChangE事件E1实验报告.md)：严格官方曲目划分48/6/6、ChangE源留出8折索引，499个事件峰覆盖288→398，但合法候选对11298→8640。完整回归181通过+1依赖跳过，EDGE补充18通过，全新进程重解码与复验通过。FineDance音频203条仍为LFS指针；零模型训练/生成，旧Catalog、权重及Native门限保持冻结，新协议未获训练或生成资格。

新增 [E1-W 固定原语时间映射协议](docs/primitive_warp_v1.md)及[服务器实验报告](reports/E1W时间映射可行域实验报告.md)：固定E1中心后，合法候选8640→12969、可用事件388→448，保留所有旧可行对；新旧中心的同求解器对照为13188/12969。196项主回归通过+1依赖跳过，EDGE补充33通过，原始来源新进程复验与每进程48902组整数域穷举通过。零训练/生成；默认Planner不变，A2/G2/G3资格不因此改变。

新增 [A2新划分资格与G2来源归因报告](reports/A2新划分资格与G2来源归因实验报告.md)：1,037来源/54音频的独立原始数据重建得到9,199窗，Train合格106、Val仍0，未启动新训练。冻结G2的32次生成及新进程复验完成，来源跟随0/4、Native0/32；补齐自由轨迹观察与提供/隐藏/关键帧诊断。回归202通过+1依赖跳过，EDGE补充19通过。正式路径与门限不变。

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

generation/adapter.py 提供零初始化、掩码条件 Adapter，冻结 EDGE 主干，复用原 CFG。research/legacy/paired_training.py 保留历史真实 151D/4800D 配对、来源分离与支持域预检；scripts/train.py 为历史重放入口，不消费新的 Split V3，也不授予新协议资格。尚无获完整生成资格的训练权重。以下历史命令需 EDGE 环境：
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

## V0.6 研究基线：连续事件软条件与真实训练探针

PR #2 已通过 merge commit 合并，研究基线标签为 v0.6-research；main 默认生成路径仍是 V0.5。soft_reference.py 根据真实 Event Plan 的单调源相位做 SO(3) 测地插值，支持域外关节帧明确未知；重叠窗口条件一致。连续参考只进入冻结 EDGE 的条件 Adapter，不后贴源动作。

本轮服务器完成 300 步48D探针、八例150/225帧生成、两次独立数值重放。42项完整回归通过；原官方/V0.5动作哈希不变；同噪声零 Adapter 与无 Adapter 完全相同。探针验证MSE降低约17.28%，但八例Native全未通过，动作幅度也下降，不能宣称音乐或整体质量提升。详见[V0.6实验报告](reports/V06实验报告.md)及[机器收据](reports/v06.json)。

~~~bash
# 真实条件支持域预检
python scripts/inspect_event_reference.py --plan outputs/event_plan_01/dunhuang_150/event_plan.json --scaler outputs/new_prepare/edge_contract/scaler.npz --out outputs/v06_reference_new
# CPU资产预检，不初始化CUDA
python scripts/train.py --assets outputs/new_prepare/assets.json --task upperbody_probe --preflight-only --out outputs/v06_preflight_new
# 一个固定探针 + 匹配噪声对照；服务器长任务需通过tmux运行
python scripts/v06_experiment.py --assets outputs/event_plan_01/prepare/assets.json --prior-root outputs/event_plan_01 --out outputs/v06_new --steps 300
python scripts/verify_v06.py --root outputs/v06_new --replay
~~~

所有输出目录必须为新目录，路径按实际资产替换。记录实际运行提交，训练/生成版本与随后报告提交分开。研究权重的 music_qualified / full_motion_qualified 始终为 false，正式推理默认拒绝，实验才显式给 research_allow_motion_only_adapter。

其他103通道仅是探针占位，不能当完整真值。完整151D目标与真实配对Jukebox仍缺失；学习式音乐演绎V1、双侧生成及SQP尚未完成，未升级450帧/整曲质量验收。

## V0.7：条件合同与底层诊断

本阶段不重复训练探针。共享 planning/time_map.py 将分段源→音乐时间映射同时用于生成条件和检索；74D节奏特征在真实映射后的SO(3)轨迹上重新求导，旧的均匀缩放同步资格不能自动升级到新执行合同。候选身份绑定来源、时长、时间映射、演法占位和评分模型哈希。当前演法仍是identity，不宣称已学习力度或停顿。

generation/conditions.py 分开来源参考、观测掩码、相位及目标监督，dense_upper8 / completion_upper8 显式命名。匹配的150帧训练/推理条件由同一函数产生；全局窗口原点用于保证重叠掩码一致。历史V0.6 dense推理与completion训练的差异单独入审计，研究权限不授予正式资格。

~~~bash
tmux new-session -d -s ddg-v07 'python scripts/v07_experiment.py --assets outputs/event_plan_01/prepare/assets.json --prior outputs/v06_01 --alignment-request outputs/event_plan_shadow_verified/request.json --out outputs/v07_new > outputs/v07_new.log 2>&1; echo $? > outputs/v07_new.exit'
~~~

输出新目录，保存逐通道支持域、坐标往返、逐层激活、注意力熵、随机VJP、来源/相位/音乐干预、匹配噪声单窗生成及原Native结果。VJP只衡量局部网络敏感性，不是完整物理可控性证明。未来PFG-1D、Music–Event Cross-Attention、双侧两遍生成与SQP仍需各自数据和资格，不由诊断成功自动启用。

## V0.7与论文评价的真实结果

本轮V0.7主实验不训练新模型：568窗支持域、两噪声seed/三时间步的层级响应、八例150帧生成均完成。右踝通道60在全部568窗越域，完整151D支持仍为0；来源替换对现有探针的作用很弱，不能仅凭局部MSE改善宣称事件演绎成功。八例Native仍0/8。详见[V0.7报告](reports/V07实验报告.md)和[机器数据](reports/v07.json)。

新增独立evaluation_benchmarks/：EDGE PFC、网格Penetrate/Float/Skate、敦煌接触状态和事件/窗口边界诊断。17个保存结果、15个唯一动作已只读评价，PFC逐例与冻结EDGE官方脚本一致；统一SMPL网格与smplx参考最大差约2.4e-7m。

PhysDiff仅为附录定义复现：公共地面来自冻结基线网格，Skate接触高度50mm为本项目显式协议选择；官方评价源码未获得，统一neutral网格骨架也与旧Native骨架不同。不能直接与论文表格排名比较。无连续接触样本时Skate/Phys-Err未定义；无独立标签时静止/有意滑动保持UNKNOWN，不自动豁免高速滑动。旧Native判决和阈值不变。

~~~bash
python scripts/evaluate_benchmarks.py --assets inputs/benchmark_assets.json --manifest configs/benchmark_runs.example.json --out outputs/paper_benchmarks_new
python scripts/verify_benchmarks.py --root outputs/paper_benchmarks_01
python scripts/verify_v07.py --root outputs/v07_01
~~~

第二、三条只读复核，不更新动作或优化权重。资产模板为configs/benchmark.example.json。详见[论文指标报告](reports/论文物理指标复核.md)、[完整指标](reports/paper_benchmarks.json)与[CSV](reports/paper_benchmarks.csv)。


## V0.8 Native 反馈闭环

新增只读反馈层与有界控制器：Native 精确回验 → 指标/时间窗/事件关联 → 条件或种子重生成 → 单槽局部重规划。复用原 beam planner，固定未改事件槽。候选按完整上下文缓存，执行错误保持 UNKNOWN，大范围物理失败默认停止；明确的小额诊断模式最多一次重规划。Native 阈值不变，150/225 帧结果不升级成整曲验收。

入口及阶段边界见 [闭环说明](docs/v08_closed_loop.md)，实际结果见 [V0.8 实验报告](reports/V08实验报告.md) 与 [机器核验数据](reports/v08.json)。完整 151D 训练、音乐配对演绎、双侧上下文和 SQP 尚未由这一控制回路解决。

~~~bash
python scripts/generate.py --config configs/closed_loop.example.json --assets outputs/event_plan_01/prepare/assets.json --out outputs/loop_new
python scripts/generate.py --config configs/closed_loop.example.json --assets outputs/event_plan_01/prepare/assets.json --out outputs/loop_new --resume
python scripts/v08_experiment.py --assets outputs/event_plan_01/prepare/assets.json --out outputs/v08_new
python scripts/verify_v08.py --root outputs/v08_new --out outputs/v08_new/readout.json
~~~

首次输出目录必须不存在。完成后的 resume 重新核对 Native 合约、计划和缓存，不启动新生成；中断半成品需使用新目录明确重跑。


## 历史配对训练资产预检（b880337）

新增 AIST++/FineDance 原始资产校验、独立 151D 转换、observed/derived/supervision 掩码、冻结 Scaler 支持域统计、真实 Jukebox 候选提取及严格 paired_generation_manifest_v2。原始数据保存在忽略的 assets/datasets，不随源码发布；保留历史角色和曝光状态。

服务器实际完成 1,566 条原始动作哈希核对、1,564 条解码和 15,665 个 150 帧窗口审计。AIST++ 4,964/5,040 个窗符合当前表示合同；FineDance 0/10,625，且缺少末端手部 22/23 监督。两个不同 Train 关联组的真实 Layer66 Jukebox 提取完成。配对训练因时钟/来源证据不足明确 BLOCKED，训练0步，无新模型质量结论。

详见 [配对资产预检报告](reports/配对资产预检报告.md)、[运行与合同](docs/paired_assets.md)、[支持域数据](reports/paired_asset_support.json)、[独立核验](reports/paired_asset_verification.json)。旧无观测/时钟收据的 paired 清单拒绝使用；单次生成和 V0.8 闭环保持原入口。


## AIST++ 基础舞配对与首轮训练

已利用官方精修视频、CSV和基础编舞工作簿核验1,363条来源：1,166条基础舞通过身份/音频时钟校准；197条高级舞共享编舞身份未明确，仍排除。完整配对清单为2,418个窗（Train1480/Val481/Test457），预检通过后提取127份去重的真实Layer66 Jukebox，再完成冻结EDGE的300步151D Adapter补全训练。

服务器96项测试通过，新进程验证MSE与保存结果完全一致、零权重更新。Val补全MSE较零Adapter约改善1.97%，移除来源参考几乎不变；未证明音乐演绎或Native质量，默认生成入口不变。角色隔离只覆盖已核验官方关联键，不宣称语义独立或新盲测。

详见 [实际配对训练报告](reports/AIST++配对训练报告.md) 和 [运行说明](docs/aist_pairing_pipeline.md)。完整清单与原始回执保留在本地/服务器的忽略输出目录，GitHub仅发布汇总。

## 事件可辨识性研究（Shadow）

新增自动事件、来源可追溯弱标签、严格部分路径、合成插入/删除/合法变速及独立重放。固定预算与指标边界见 [研究协议](docs/事件可辨识性研究协议.md)。历史 A/B/C、Native31、默认生成路径保持冻结。真实事件评价仍需要独立审核。

本轮服务器六组固定预算实验及独立重放已完成；结果与边界见 [事件可辨识性实验报告](reports/event_identifiability_report.md)。合成身份/NULL恢复改善，真实时钟内容判别未超过时间先验，弱对应未显示独立收益，继续保持 Shadow。

## 局部表征可辨识性 R1 / 条件内容 R2（Shadow）

在01c0515上冻结编码器、旧头与默认生成，新增统一网格的信息可访问性探针、Token谱/Hubness、固定路径内容重评分及五类损失梯度审计。严格控制候选人口足够时才执行等参数R2；不足不放宽。见 [预注册协议](docs/局部表征可辨识性研究协议.md)。

R1服务器审计与独立重放已完成，严格条件候选Train/Val均为0，R2按门限未启动。详见 [局部表征可辨识性报告](reports/local_identifiability_report.md) 与脱敏回执；没有新检索模型或生成质量提升声明。


## Dense A1 / G1 research audit (2026-11-12)

See [experiment report](reports/dense_condition_20261112.md) and [registered protocol](docs/DENSE_CONDITION_PROTOCOL.md). Real source coverage: 936 recordings / 8,510 windows; strict Train 18 queries / 10 recordings, Val 0; training was not started. Frozen EDGE audit: 14 real generations plus 14 exact recomputations, Native 0/14 and source-follow gate failed. Explicit research visibility passes the six matched unknown-storage and six bypass invariants; legacy remains the default. No bilateral generation or SQP qualification is claimed.


## G2 / A2 source-identifiability study (2026-11-12)

[Report](reports/source_identifiability_20261112.md) · [receipt](reports/source_id_03_receipt.json) · [code boundaries](research/README.md) · [registered protocol](docs/G2_A2_PROTOCOL.md).

New studies are isolated in research/source_identifiability; historical generation trainers moved to research/legacy, and four duplicate script wrappers were removed. Three matched 357,424-parameter Adapters completed 400 updates each. All model arms failed source-following qualification (0/4); Native 0/52. All44 diffusion outputs and8 nonlearned controls were independently recomputed exactly. New A2 population: Train69 queries/63 recordings/6 groups, Val0; no A2 training. Final regression164 passed +1 dependency skip; supplemental EDGE13 passed. Default generation and Native remain unchanged, and observed-trained checkpoints require a matching research runtime.


## A2-V3 candidate support / G2-M mechanism study (2026-11-13)

See [report](reports/A2V3候选支持域与G2M机制对照报告.md), [registered protocol](docs/source_necessity_v1.md), and [sealed receipt](reports/source_necessity_receipt.json). New separately registered A2 tasks have 1,023/885 Val queries; the historical strict task remains unchanged. Five matched Adapters completed 400 updates each. All source-following arms remain 0/4; Native 11/52 overall (bypass 1/4, learned arms 2/8 each). All52 generations and156 motion/FK arrays were independently replayed exactly with zero optimizer calls. Production retrieval/generation and G3 remain unqualified.


## Real source CF and fixed-candidate content tasks (2026-11-13)

[Report](reports/来源效应与G2CF及A2内容对照报告.md) · [protocol](docs/source_tasks_v1.md) · [receipt](reports/source_tasks_receipt.json). Saved-source response was independently measured; real paired high-noise source qualification and four matched CF arms completed. All CF source-follow gates remain 0/4 (Native9/44). Four equal-parameter15/30/dual/shuffle content models were trained on the same frozen offline A2 candidates and independently replayed. Default retrieval/generation, Native and G3 remain unqualified. The pipeline now rejects legacy anchor selection which a whole-song Planner would overwrite.


## Frozen capacity / local path audit (2026-11-13)

[Report](reports/冻结来源控制容量与局部路径审计报告.md) · [protocol](docs/source_capacity_v1.md) · [receipt](reports/source_capacity_receipt.json). Zero optimizer updates. All72 derivative probes passed numeric checks; no site passed directional control qualification.32 frozen controls and96 motion/FK arrays were exactly regenerated in an independent process. All4 frozen content encoders were compared on fixed candidates using same-frame, banded DTW, partial NULL and content controls, then independently replayed. Group intervals do not support a reliable local-path gain. No production or G3/R4 promotion.

## Reference-only guidance / event information audit (2026-11-13)

[Report](reports/参考动作可用引导与事件信息审计报告.md) · [protocol](docs/reference_guidance_v1.md) · [sealed receipt](reports/reference_guidance_receipt.json).
All96 reference-only direction derivatives passed numeric checks; protected all-case qualification failed, so no full-chain guidance or mapping training started.44 single-step Native readouts and132 arrays were independently replayed exactly. Four frozen encoders received60 Train-only diagnostic linear readouts; all scores/heads were replayed without refitting. Source registry exposes60 shared choreography identities despite valid known-song isolation. Proxy decodability does not qualify cross-modal event semantics, production generation or a new blind benchmark.
