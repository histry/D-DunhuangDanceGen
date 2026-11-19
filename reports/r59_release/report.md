# R59：完整录制配对资格与R58内部目标审计

先发布代码、服务器tmux构建真实完整固定网格，回传六组真实配对来源/Heuristic/Frozen及六段视频，浏览器验收后才执行其余冻结模型、内容干预、偏移任务。不是新敦煌舞生成。本轮没有神经训练或Planner改权重。

## 数据与任务边界

实际来源：{'AIST++_train': 919, 'AIST++_val': 118, 'FineDance_genre_train_diagnostic': 12}；完整网格3659行、固定元数据查询338条。AIST++原动作/Native/R41旋转/官方时钟/音频SHA逐项核对；全录制150帧、75帧步长加末端，不再沿用正例分箱后的旧特征人口。完整性相对于既有校准源注册表，不宣称全部AIST++公开数据。

FineDance十二条从本地danceRAG真实原315D/音频/标签逐SHA复制，原服务器目录中的LFS占位没有当波形。官方@Genre Train取001–011/013；@Dancer身份另存。统一前22身体关节，row-6D显式转换；原始Root保存但没有新世界坐标或物理准入。完整原音频同50D提取器重提（librosa0.11.0），冻结AIST Train归一化、不在目标域重新拟合。原始共同零点不套用Lodge双侧裁剪偏移；009动作多3帧、011音频多25帧等尾部差异保留，只使用真实共同前缀。首轮在完整数据清单生成前因元数据预检而中止，退出回执单独不能充当完成证据；中断日志与产物保留，没有模型评分或训练。

AIST开放池排除查询同原动作SHA；FineDance没有多录制对应证据，采用包含真实目标窗的同录制时间检索，不能与跨来源泛化混称。候选池先按角色/长度/元数据形成，困难池只按查询音频与候选活动差取64，不查看目标动作活动，也不强行补正例。目标缺失与UNKNOWN都保留，条件召回和目标覆盖分开报告。

AIST Val六音乐组已历史曝光，当前Data V3可能共享编舞；FineDance只是十二条官方Train跨域诊断。完整预训练曝光尚未获证，因此没有新独立音乐或未见编舞结论。真实源时钟不等于唯一艺术正确事件。官方规则见docs/paired_benchmark_r59.md引用的AIST++与FineDance原始来源。

## 实际开放池读数

|数据/角色|方法|R@1|R@5|MRR|目标覆盖|
|---|---|---:|---:|---:|---:|
|AIST++/val|Heuristic|0.05556|0.25000|0.16192|1.00000|
|AIST++/val|Frozen|0.16667|0.50000|0.32783|1.00000|
|AIST++/val|Conditional|0.16667|0.52778|0.33647|1.00000|
|AIST++/val|NoClock|0.16667|0.52778|0.30330|1.00000|
|AIST++/val|Conditional_TeacherRescore|0.27778|0.50000|0.40088|1.00000|
|AIST++/val|NoClock_TeacherRescore|0.30556|0.41667|0.36223|1.00000|
|FineDance/genre_train_diagnostic|Heuristic|0.00000|0.00000|0.01844|1.00000|
|FineDance/genre_train_diagnostic|Frozen|0.00000|0.02778|0.03026|1.00000|
|FineDance/genre_train_diagnostic|Conditional|0.00000|0.00000|0.01986|1.00000|
|FineDance/genre_train_diagnostic|NoClock|0.00000|0.05556|0.02850|1.00000|
|FineDance/genre_train_diagnostic|Conditional_TeacherRescore|0.02778|0.05556|0.05466|1.00000|
|FineDance/genre_train_diagnostic|NoClock_TeacherRescore|0.00000|0.05556|0.03233|1.00000|

公开来源审计只保留提取器合同、采样/时长信息及SHA；逐帧音频派生曲线保留私有原始回执，不入Git。本轮H的录制时钟检索读数不否定R51/R52 Heuristic整曲观感；任务和候选池不同，不据此更换编舞基线。R@K是源音频时钟目标检索，不是艺术正确率。1e-7固定分箱，同分按随机排列的解析期望统计，不以UID先后判对；额外同歌曲/活动困难池、零音乐/错音乐/反转动作Token、所有查询缺失统计见CSV和paired_metrics.json。录制内均衡后按音乐组重采样，区间只描述本次历史人口，不能代替独立泛化。TeacherRescore是Frozen+.25(raw_R55−Frozen)，没有训练新Teacher-Preserved模型。

## 已知非循环偏移恢复

|数据/角色|方法|准确率|MAE帧|
|---|---|---:|---:|
|AIST++/val|Heuristic|0.07222|13.73889|
|AIST++/val|Frozen|0.30000|11.73889|
|AIST++/val|Conditional|0.12222|14.20000|
|AIST++/val|NoClock|0.12778|13.85000|
|AIST++/val|Conditional_TeacherRescore|0.38333|10.47778|
|AIST++/val|NoClock_TeacherRescore|0.36111|10.60000|
|FineDance/genre_train_diagnostic|Heuristic|0.11667|12.55000|
|FineDance/genre_train_diagnostic|Frozen|0.20000|12.75556|
|FineDance/genre_train_diagnostic|Conditional|0.14444|12.43889|
|FineDance/genre_train_diagnostic|NoClock|0.13889|12.38889|
|FineDance/genre_train_diagnostic|Conditional_TeacherRescore|0.15556|13.34444|
|FineDance/genre_train_diagnostic|NoClock_TeacherRescore|0.11667|13.10556|

按音乐/登记音频组的准确率和MAE配对差值95%区间另存paired_clock_differences.json；FineDance音频ID不等于已获证的独立音乐作品。真实连续150帧内注入[-15,-8,0,8,15]帧偏移，共同非循环支持上恢复相反偏移；Heuristic使用音频活动与动作速度相关。FineDance只有官方同ID/共同零点依据，合成标签证明相对扰动，不是对其原始音乐动作相位做AIST同等级波形校准。

## R58目标提高来自何处？

|音乐|启发式差|执行差|长度差|边界差|几何差|事件重复差|来源重复差|总分差|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|dunhuangwu2|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|
|dunhuangwu3|0.605653|0.000382|-0.010000|-0.158926|-0.009700|-0.100000|0.000000|0.327409|
|dunhuangwu4|0.278672|-0.003499|-0.016000|-0.077742|-0.001743|0.100000|0.000000|0.279687|

后两首内部总分提高主要来自活动匹配项改善，同时边界奖励项下降；执行项没有成为稳定主导收益。各项是原冻结目标的数值分解，不是独立艺术评价。

全部C1/C2/C3的内部得分逐项精确重建；B1只按共同目标重评分，不改写旧得分。所有动态阶比均未知：图节点0个，选中节点0个，匀速档案0个。未观测信息的新读出返回UNKNOWN而非低代价；R58封存代码/轨迹保持。是否实际造成选路偏差需依据这些计数，不从一个return0分支直接宣称偏差。

## 资格与独立质量证据

本轮新训练资格保持未通过：历史Val曝光、完整预训练角色谱系未获证、FineDance主划分Train诊断。各新旧权重相对Frozen/Heuristic及正确音乐干预的音乐组差值95%区间完整列入qualification.json，不按Val挑模型/改预算。没有启动Teacher-Preserved训练或FineDance长序列新规划，也不把来源数量当独立音乐数量。

已准备三首音乐的独立纯音频边界/重音/重复/UNSURE入口，以及B0/B1/C1/C2六组合×三音乐共18组同音频/尺度的随机化A/B盲页，方法映射私有。实际人工记录0；至少两独立评价者、保留不同意见，真实乐句/艺术偏好及代理与偏好一致性均NOT_EVALUABLE。评价记录不用于回调本轮权重。

## 复验与交付

721项回归通过、1跳过；完整原来源和FineDance音频重提、新网格/源数组、新进程全部候选重编码分数NPZ及R58分解精确复验。只读候选视图先在原主效果完整JSON/分数NPZ上验证精确一致，再用于完整复验，公式/权重/人口未改。临时启动器的CRLF导致一次准备页未启动和独立主复验目录多一个回车后缀，失败回执保留；目录规范化前后14文件SHA一致，不改变数值。父代988文件逐SHA不变。复验优化器0。所有代码先GitHub再服务器tmux；原始数据/分数数组/媒体/私人映射不入Git。

本机统一入口：D:/DunhuangDanceGen/outputs/paired_benchmark_r59_01/index.html；main/viewer/index.html配对来源与实际检索；main_video/index.html六段含原音乐视频；quality_review/viewer/index.html盲化整曲；quality_review/music/index.html纯音频结构标注准备。

默认EDGE/Lodge与Planner不升级；没有完整新整曲或Native24合格结论。
