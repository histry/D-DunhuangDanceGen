# R6：冻结失败归因、来源动态控制与独立盲审准备

基准 0265485；研究代码独立于正式生成器及历史实验。旧配置、冻结权重与保护阈值不改。先提交 GitHub，再在服务器新目录执行；失败目录和日志保留。没有新的独立人工标签，不开始训练或整曲生成。

## P0：原始失败责任矩阵
保存全部 96 个 PB1/PB2 原始判定及来源／共同增益、保护余量、范数和夹角。只对 Val case0、499/19、双来源 PB1/PB2 做八个固定数值诊断。比较真实 clipped CFG2 与冻结 clip 活跃集的诊断代理，保持 .25/.125/.0625 步长及原阈值。代理不进入生成或覆盖原数值失败。没有新增剂量扫描，幅度不足原因保持 UNKNOWN。

## P1：固定来源表达对照（不训练）
旧四组 Train 只用提供的可见参考姿态及相邻 SO(3) 增量，拟合两个单元尺度（姿态角平方、30Hz 角速度平方），下限 1e-4；无模型预测或 Val 参与。固定单元权重，不根据结果调节。

D0：原 PB1 姿态目标；D1：归一化姿态＋动态；D2：增加来源自动运动学阶段姿态；D3：打乱可见动态和阶段、保留姿态；D4：保护域随机方向。所有组使用同一完整 condition512 支持、rank6、小规模 SLSQP 投影、预算和实际非线性保护。完整投影剂量作为次要读数；主要对照将五组统一缩小至最小范数，绝不放大。保留每组完整／统一剂量数组、数值检查和真实输出。

四组旧 Train＋四组来源／音乐组与原始动作 SHA 均不重合的已知 Train 探针，3 时刻、2 来源、5 组，共 240 次局部投影、480 条读数；独立进程完整重放。新探针仍属于旧 checkpoint 的 Train 暴露人口，不能当成未见来源泛化或新盲测。D4 的统一范数来自共同剂量规则，不声称完全无来源信息。隐藏完整动作仅用于事后评价，不进入目标、支持或方向计算。全部 24 条/组必须数值可信并同时满足来源增益、相对来源增益 >.001rad 和旧保护才算完整通过；报告失败不选择最优组部署。无新的 Val 目标对照，无 Native 评价／全链采样／权重更新。

## Planner：三份真实音频的 Shadow 合同
从来源已核验的 AIST++ Train 人口按固定哈希选三首不同官方音乐身份、不同风格代理及不同完整 PCM 哈希的音频；每首 450 帧。音乐作品别名及预训练暴露未知部分继续披露；它们不是未见模型训练数据。E0、E1、E1-W 共九份计划；验证实际 Planner 消费、E0 默认路径等价、E1-W 整数时间域穷举、源区间、SLERP、支持掩码与条件绑定。无法安排的槽位如实 NOT_EVALUABLE，不替换歌曲。

## 独立人工事件与姿态盲审协议 V1
状态 PREPARATION_ONLY。实际记录为零，human_reviewed=false，semantic_event_truth=UNKNOWN，跨模态逐事件对应 NOT_EVALUABLE。该包来自已使用的 ChangE 来源，仅为流程试点，不是新独立盲测人口。

EVENT 与 POSE 判断 VALID / INVALID / UNSURE；MUSIC_MATCH 判断 COMPATIBLE / INCOMPATIBLE / UNSURE。三任务独立，兼容性不等于逐事件对应真值。先单独完成 EVENT/POSE；音乐兼容任务独立审核。自动 onset、hold、speed peak、turn 不向审核人展示。

管理员在审核前封存盲编号与来源对应表。审核端只含随机 ID、必要媒体和中性任务说明，隐藏算法、自动标签、分数、来源名称及数据角色。至少两名独立审核人互不查看结果；分歧由第三独立审核人处理，保留全部原始投票及裁决引用。不得根据 Val 结果换样本或修改标准。

每条记录保留 review_id、blind_item_id、reviewer_id、reviewer_roster_ref、task_type、source_uid、raw_motion_sha256、event_uid、start_frame、end_frame、fps、media_sha256、含时区 reviewed_at、decision、confidence、reason_code、协议版本及裁决引用。来源身份由管理员在审核后绑定，不进入盲审界面。身份／资质映射及原始数据受控保存，不上传公开仓库。源动作 SHA、半开帧区间、原音频与时间映射、媒体与渲染配置可重建；审核媒体为拟合的 22 关节 Root 相对骨架，不能替代原视频可见性、真实 Root 或接触证据。

UNSURE 原因：VISIBILITY_INSUFFICIENT、BOUNDARY_AMBIGUOUS、SOURCE_INCOMPLETE、CULTURAL_SEMANTICS_UNKNOWN、MUSIC_CONTEXT_INSUFFICIENT、TECHNICAL_FAILURE、OTHER（必须说明）。不删除弃权，不强迫转为正负例。

身份、时间、证据、独立评价与争议处理不全则 PARTIAL_REVIEW；零审核 UNREVIEWED；无法判断 UNSURE；完整可核验一致记录才 HUMAN_REVIEW_VERIFIED。导入时需私有 roster 的 coverage_attested，管理员确认身份、来源、媒体重建、封存选择与角色／曝光覆盖。原始审核不被裁决覆盖。审核结果需按独立来源、音乐组、审核人报告人口、一致性、分歧、弃权原因与缺失；目前没有记录，不计算虚构一致性或置信区间。正式使用前另登记覆盖、质量和性能门槛及按独立来源抽样的统计协议；本试点不授予独立模型评价或训练资格。

## 执行与导入
服务器：bash scripts/run_source_expression.sh outputs/source_expression_01 outputs/source_expression_01_receipts，在 tmux 内启动，独立进程逐阶段重放。验收需退出码 0、每阶段协议／结果 SHA、实际测试日志和独立检查回执。
私有审核导入：python scripts/import_independent_reviews.py --protocol .../review/protocol.json --roster PRIVATE_ROSTER.json --records ORIGINAL_A.json ORIGINAL_B.json --out NEW_PRIVATE_DIR。审核人先登记核实 roster，其内容 SHA 作为 roster 引用；实际身份文件绝不公开。分歧时加 --adjudications，第三方原始审核也须列入 records。人工数据通过不自动授权新训练、检索升级或发布。

本轮不新增 Encoder、RootNet 或正式生成路径，不删除仍被历史冻结实验调用的代码。仅添加必要研究模块、测试与协议；不复制已有可哈希复用的大资产。

## 媒体确定性修正与续跑（工程修正，科研预算不变）
source_expression_03 的四个科研阶段及各自完整重放已完成；浮点 WAV 的 PEAK 块含写入时间戳，导致字节 SHA 不同，但三份 PCM 数组逐样本完全一致。原启动器按规则退出 1，失败目录不修改或删除。

剪辑改为 PCM24，逐样本读回必须与原 FP32 音频完全一致，否则拒绝输出；本次三首原音频均为 PCM24。新增字节与样本双重确定性测试。音乐检测器、动作条件、目标、阈值、投影预算均不变。

configs/source_expression_r6_continue_v1.json 封存四阶段的十四个协议／结果／数组／重放 SHA。续跑在新目录中用符号链接复用已重放产物，原执行提交为 41fd659；启动前重新核对全部 SHA、失败退出码及科研内核和配置的 Git 差异为空。仅重新生成盲审媒体和完整媒体重放。回执分别列出四阶段与媒体阶段的真实执行提交，不将原失败启动器改称成功。

执行：bash scripts/run_source_expression_continuation.sh outputs/source_expression_04 outputs/source_expression_04_receipts。完成后另在新进程运行 scripts/check_source_expression.py --out outputs/source_expression_04，核对保存数组、实际源读数及媒体证据后再封存。
