# Event Plan V0.5 架构修正与真实实验

本轮完成 Event Catalog → Music Timeline → Event Plan → Temporal / 冻结 EDGE → 原 Native 验收路径，并在服务器完成 5 秒、15 秒、120.03 秒整曲六例生成。**六例 Native 均未通过。** 学习式同步头未获验证资格，Adapter 因真实训练资产不满足条件未训练。当前交付是 V0.5，不能称为已完成或验证了学习式 V1。

## 版本与证据

六例主实验运行提交为 `9ceee94149eee9d086e7775cc458ef8620d846e0`。后续提交补齐模型绑定、两级 shadow 排序、同曲时段诊断、训练资产合同及输入验证；未把旧动作重命名为新版本生成。每例 request.json 保留实际运行提交。

主实验：`outputs/event_plan_01`；同步头：`outputs/sync_frozen_01`；最终 shadow：`outputs/event_plan_shadow_verified`；修正后的 Adapter 审计：`outputs/adapter_asset_verified`；真实配对预检：`outputs/paired_adapter_preflight`。完整指标见 [event_plan_workflow.json](event_plan_workflow.json)。

服务器使用既有 EDGE / dance-rag 隔离环境；EDGE 固定 `17c3428`，Native 固定 `465f35b`。权重 SHA256 `28ca4ce167bb17c36869b4d021af8762a34c6df034002f61b3bc1c1d0b1b02c7`，发布来源未独立核实。音频 SHA256 `36ce6f03d7a8c57507071fcd74429562b256c03f5523c7e0657f0d1037cd598c`；使用原有真实 layer66 / 4800D Jukebox 缓存，未声称本轮重新提取训练配对数据。

先 push GitHub，服务器 pull 后执行，长任务通过 tmux 启动并保存日志/退出码。主实验、同步训练、最终 shadow、同曲时段评测退出码均为 0。主实验与同步资产共 618 个文件按服务器 SHA256 回传核验。视频是随后本地真实 FK 的衍生产物。

## 已实现机制与边界

- **目录**：从显式资产清单解码 8 个真实 SMPL 来源，导出 499 个事件、源动作引用、74D 特征和掩码。绑定源 SHA、固定 decoder、冻结 scaler。文件名仅提供来源级粗动作/文化类别，不冒充逐事件或音乐语义真值。
- **时间线与规划**：真实波形节拍和能量，约 3 秒切分并向节拍吸附，不是人工乐句分析。候选通过时长和 [.75, 1.33] 局部倍速边界，Top5、beam3，全局累计活动匹配、事件重复、来源复用及 SO3 入口/出口代价。每事件保留严格单调的源→音乐时间映射，完整覆盖时钟。
- **原语**：25 帧片段通过已有 Temporal 路径控制上身 12–19 关节，三角权重两端为零，每事件实际有效 23 帧。约束发生在扩散采样中；计划全覆盖不等于连续整段控制。
- **基线和验收**：官方 EDGE long DDIM 保留，Native 完整原始阈值不变。官方对照包含采样器差异；另与历史 B2 Temporal 检查相同 seed、采样器和噪声指纹。
- **双阶段学习检索**：50D/74D 全局召回 Top5，局部头在五个真实非循环相位对照的共同有效区间评分。全局/同步编码器身份和父 checkpoint 必须匹配。heuristic / learned_shadow / learned_qualified 分离；shadow 不改变选段，资格仅根据验证数据。
- **Adapter**：zero-init 条件残差插入冻结 EDGE cond_encoder，同时影响 pooling/FiLM 与 cross-attention，复用原 CFG。输入仅含已观测值、掩码、相位。训练器支持 motion-only 补全及显式真实 Jukebox 配对，来源划分、测试排除、目标支持域检查完整；没有可用训练权重，默认路径不调用。
- **工程**：正式入口 pipeline.py，公开 profile 与服务器资产分离，提供 prepare/generate/train/evaluate。老 SDMR 编排移到 research/legacy，历史脚本和证据保留。显式 Event Plan 可哈希绑定重放，拒绝音频、时钟、源区间、原语、倍速不一致。

## 六例真实结果

jerk 单位 m/s³，滑步 P95 单位 m/s。Native 使用完整检查，不只表内两项。

| 帧数 | 官方 EDGE jerk / 滑步 | Event Plan + Temporal jerk / 滑步 | 事件 / 来源 | 有效控制比例 |
|---|---:|---:|---:|---:|
| 150 | 4738.18 / 0.6956 | 4766.24 / 0.6677 | 2 / 2 | 30.67% |
| 450 | 4825.21 / 1.0128 | 7901.65 / 0.4165 | 5 / 4 | 25.56% |
| 3601 | 24111.20 / 1.0515 | 30530.54 / 0.9157 | 40 / 8 | 25.55% |

六例全部完成真实推理、转换、FK 和 Native，**0/6 Native 通过**。FK 最大差异不超过 `9.54e-7 m`，EDGE 参数前后哈希一致。计划覆盖率 100%，整曲 40 个不同事件。不同动作哈希/FK 只证明输出改变，不证明风格、自然性或语义正确。

相同 Temporal 采样器与噪声的历史 B2 整曲：jerk 从 32026.35 降至 30530.54（约 4.7%），滑步从 0.9007 升至 0.9157（约恶化 1.7%）。因此不能宣称整体质量提升。短片段相同对照保留在机器报告，未根据结果追加调参网格。

## 同步训练与同曲时段诊断

同步头实际优化 1000 步，原全局编码器前后 SHA 完全一致。Val aligned-vs-shift 胜率：FineDance 约 50.30%，AIST++ 约 53.04%，未过既定资格。最终 shadow 正确绑定模型、计算全局/局部分数，但计划与 heuristic 一致。

新增诊断把“同歌命中”和“同歌正确物理时段”分开，使用已有全局 checkpoint，没有重训或修改资格：

| 划分 / 域 | 查询数 / 来源组 | 同曲时段 Recall@1 | 机会水平 |
|---|---:|---:|---:|
| Val / FineDance | 168 / 35 | 32.14% | 25.00% |
| Val / AIST++ | 1044 / 2 | 13.22% | 20.62% |
| Test / FineDance | 160 / 34 | 37.50% | 25.00% |
| Test / AIST++ | 1072 / 2 | 17.16% | 20.16% |

Test 历史上已暴露，是开发回归而非新盲测；AIST++ 仅两个来源组。配对沿用 release-zero 对齐假设，未经独立逐事件校准，不能据此声称语义对应能力。

## 未执行阶段与原因

修正 Root 原点为首个已观测 XZ（保留高度）后，8 来源、568 个 150 帧窗口：body8 支持 564，手腕支持 0，Root 支持 84，完整 151D 支持 **0**。没有裁剪源目标或用 50D/74D 伪造 4800D 音乐；同时没有真实配对 Jukebox 训练清单，配对预检返回 MISSING_REAL_PAIRED_JUKEBOX。

Adapter 实际训练 **0 步**，没有学习效果、225 帧资格或 V1 整曲推理。双向生成只有已观测边界掩码工具，未完成两遍生成；新 PFG/SQP 未执行。专家逐事件标注、独立风格及人工自然性评测仍缺失。需要先补真实且可表示的配对目标，验证 150 帧补全，再进入更长窗口、两遍上下文和物理修正；整个最终架构尚未完成。

## 复核与产物

```bash
python scripts/verify_event_plan_report.py
python -m pytest -q
```

只读复核六个 native.npy 哈希、Native 指标、生成提交、冻结权重、计划和训练/评测收据，不执行优化。静态测试不替代实际实验。

视频位于 outputs/event_plan_01：comparison_150.mp4、comparison_450.mp4、**comparison_3601_complete.mp4**。整曲修正版为 H264、3601 帧、120.033333 秒，带原音频。之前少尾帧的 comparison_3601.mp4 仅为历史产物。左右分别为 Frozen EDGE baseline / Event Plan + Temporal，使用真实生成 FK，未以源动作替换。

SSH 使用进程内客户端，没有临时含凭据脚本、askpass 或 key 文件落盘；未删除既有 SSH 配置和实验输入。
