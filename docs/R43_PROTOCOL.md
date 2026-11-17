# R43执行登记：几何锚定Core演绎与独立事件检索

基线`5183d2078c034ed0f7cb16e590120ecdd3119dbe`。R42已结束、三组身体门0/4，不重训。先发布代码，再在服务器tmux运行；先交付主候选实际播放，再做干预及消融。代码分别放在`research/music_reperformance_r43`及`research/sdmr_event_rank`。正式EDGE/Lodge、Planner、旧Catalog、R41/R42归档不改。

## Module II

先零训练检验既有Lodge音乐/来源权重的受限编辑。逐帧逐关节Core上身预算0.1rad，下肢Core与Root Core为零；Bridge身体0.06rad、Root0.03m，三帧端点与C2渐入，外端各8帧保护。每侧可见来源速度代理峰±2帧零权限，向外6帧平滑恢复。阶段是可观测权限代理；冻结SegmentedReference的阶段输入仍为零，不冒称已训练神经阶段机制。没有把旧0.6rad放大到Core。

采样积分器复用R38，冻结Source、R38 Context、Lodge、ActionIO、Normalizer；旧R39 RootContinuous作为冻结对照。C4不监督、不作为真实接触条件；Root仅按既有模型坐标解释。每步SO(3)及物理米制投影，零残差精确等于Geometry。

人口：R42冻结数据中八个ChangE录制各最早准入窗，四Train/四适配留出；AIST每个Train音乐组的首个缓存案例及四个适配留出案例；另有两个冻结R37真实Event Plan异源压力案例。所有256帧、两固定种子。单录制窗口是两个可见Core及48帧隐藏Bridge；异源来自既有E1-W编译绑定，不把局部任务称完整事件或整曲。

主候选先执行Geometry/Frozen/Core、音乐A；主观看和两段固定来源MP4完成并实际验收后，才执行另一音乐、音乐置零和错来源。同一来源、几何、时间映射、预算与噪声保持一致；AIST另一音乐为错音乐对照，ChangE没有配对音乐真值。评价Source角度/构形、Core与Bridge的绝对D2/D3/活动量、边界和保护域，配对误差与绝对Jerk分开。不能用差异大小当音乐协调收益。

新训练只可能在Train AIST正确音乐相对置零/错音乐的Gap D0及D3均在至少3个音乐组改善，且至少3个ChangE Train来源的Core活动比[.85,1.15]、绝对Core/Bridge D3均不超过Geometry的1.05倍、全部权限/保护合格后启动。还须有真实跨录制参考、目标时钟和35D/139D合同资格。若这些条件不满足，本轮神经更新0，保留实际失败，不临时添加新头。通过后下一项固定240步，不在曝光留出选预算。整曲0，Native24未具资格，human_reviewed=false。

## SDMR-E2

P0先构建R41 Parent Event之下的90/150/240帧检索子窗口索引（半窗步长，末端补一个合法实窗）。不修改原动作边界，所有导数来自已核验完整录制。按音乐/原始动作SHA/来源组继承Train或曝光Val；Test不读。选择每角色音乐组的元数据排序前两个来源，每长度最多早/中/晚三个查询，固定候选ID后才计算模型分数。

每查询候选排除同录制，自然多正例按真实音频时钟IoU≥.6，错时段须隔至少15帧。三个候选池：同歌错时段、等时长音乐能量代理匹配、跨歌困难候选。候选与查询均相同长度，每池最多4个多正例和20负例，不足时如实报告。筛选只能读取查询音乐与候选动作，不读取隐藏目标动作活动。时钟标签不是艺术适配真值。

冻结R41 522632参数。比较Duration、Train校准Energy、Global、masked Diagonal、有限±15帧Offset、严格一对一Partial Path、Train z校准Global+Diagonal。局部Token实际有效掩码参与评分；每三帧抽一个Token用于全部局部方法，保留帧索引；NULL/覆盖和路径伸缩另外保存。Partial Path复用仓库Drop序列思想，不称完整Drop-DTW。消融置零音乐/双置零/错音乐/错动作使用完全相同候选人口。

先交付真实来源/查询/候选可观看页，再零训练执行P1。P2资格仅由Train预登记Diagonal相对Duration/Energy/Global的音乐组配对bootstrap区间下界均>0，且超过双置零决定；不从Val挑方法或系数。若失败，新排序训练0并保持Shadow；通过后固定240步的同初始化、同数据预算对照另有执行回执。未知ChangE音乐适配不产生伪标签，正式can_route=false。

独立新进程重放人口、数组、条件干预与评分；保存来源/音乐/模型/Normalizer/配置SHA、失败日志、退出码及主观看先于对照的回执。临时SSH、bundle和只读本机服务在科学封存后清理。
