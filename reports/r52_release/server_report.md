# R52 Heuristic强基线与独立完整来源训练

修复了绝对音乐帧与局部短片帧混用；原1008条空分区读数重算为882条有效区域，原计划、轨迹和whole统计不变。高D3不直接解释为不自然或Native失败。

## 两阶段接口

EVENT_PLAN_R52_V1绑定真实来源、完整区间、合法Warp、阶段代理/未知状态、软参考上下文、实际过渡预算和真实音乐35D。SDMR分数保留Shadow元数据，不成为ModuleII训练门槛。

## 完整来源与补间分开

训练FULL_REFERENCE_RECONSTRUCTION可读取完整合法来源，这是身份重建任务；HIDDEN_GAP_INBETWEENING隐藏中心48帧并使用真实连续目标。AIST++真实配对音乐与ChangE零音乐源保持任务各占半批，不伪造敦煌配对重演真值。只训练已有Source/Context，固定Lodge、CanonicalIO、General；Root/C4不入损失。两个240步训练组在主效果观看后才进行零Phase同预算对照，动态项按Train初始梯度校准，NoPhase沿用主组权重。

## 两个完整事件

固定R51 Heuristic的两个真实完整事件、E1-W和48帧几何桥，Core240+240，整段528帧。每个256帧计算窗都能读取同一完整528帧来源记忆；四窗对同一全局噪声轨迹共同去噪。只有整段外端16帧硬保护，Root/C4保留未知模型规范，未生成合格世界Root。来源Core可读480帧，桥区参考未知，不把几何桥当真实重演目标。采样从t119执行50步，仍属温启动局部研究。

|案例|方法|来源Core角误差rad|来源CoreFK误差m|
|---|---|---:|---:|
|fixture0|FullReference|0.363543|0.196731|
|fixture1|FullReference|0.365488|0.207226|

|案例|检查点|角误差rad|FK误差m|全片D3 Q95代理|
|---|---|---:|---:|---:|
|fixture0|Frozen|0.438963|0.201663|317.342|
|fixture1|Frozen|0.440301|0.204772|365.580|
|fixture0|Main240|0.363543|0.196731|791.722|
|fixture1|Main240|0.365488|0.207226|760.807|
|fixture0|NoPhase240|0.359492|0.194706|942.960|
|fixture1|NoPhase240|0.360274|0.204950|1053.291|

|固定fixture0条件|角误差rad|FK误差m|
|---|---:|---:|
|FullReference|0.363543|0.196731|
|NoSource|0.530035|0.212428|
|ZeroPhase|0.404270|0.227386|
|NoLeft|0.365111|0.201272|
|NoRight|0.362747|0.198340|
|WrongSource|0.465272|0.241576|
|WrongMusic|0.369900|0.205733|

几何基线Core直接等于来源，来源误差为0，因此要求神经生成保持来源、改善连接/音乐响应，而不声称超过零误差。所有Core/Bridge/边界/whole动态及相对来源D1/D2/D3残差保留机器摘要；源动作本身的快速变化不等同异常抖动。两首敦煌音乐是非配对历史开发输入，换音乐只能检验响应，不能证明新音乐重演正确。

## 规划对照

Heuristic、SDMR、Hybrid=Heuristic+0.1Full、RhythmCurve=Heuristic+0.1Multi在同一候选/图/预算上比较；重复、来源均衡和同来源前向不重叠事件连接开关完全匹配。方法筛查失败不等价于所有神经重演皆不可能。冻结Heuristic保持原18组选择，不直接晋升SDMR。

## 复验

{'exact_scientific_replays': 7, 'replay_details': [{'main': 'outputs/strong_baseline_r52_01/plans', 'replay': 'outputs/strong_baseline_r52_01/replay/plans', 'exact_arrays': 38, 'non_scientific_metadata_differences': [{'path': 'weights.device', 'main': 'cuda', 'replay': 'cpu', 'scope': 'EXECUTION_METADATA_ONLY_SCORES_FROZEN_NO_NETWORK_RERANK'}]}, {'main': 'outputs/strong_baseline_r52_01/plan_controls', 'replay': 'outputs/strong_baseline_r52_01/replay/plan_controls', 'exact_arrays': 194, 'non_scientific_metadata_differences': []}, {'main': 'outputs/strong_baseline_r52_01/music_controls', 'replay': 'outputs/strong_baseline_r52_01/replay/music_controls', 'exact_arrays': 12, 'non_scientific_metadata_differences': []}, {'main': 'outputs/strong_baseline_r52_01/frozen_full', 'replay': 'outputs/strong_baseline_r52_01/replay/frozen_full', 'exact_arrays': 10, 'non_scientific_metadata_differences': []}, {'main': 'outputs/strong_baseline_r52_01/main', 'replay': 'outputs/strong_baseline_r52_01/replay/main', 'exact_arrays': 10, 'non_scientific_metadata_differences': []}, {'main': 'outputs/strong_baseline_r52_01/controls', 'replay': 'outputs/strong_baseline_r52_01/replay/controls', 'exact_arrays': 24, 'non_scientific_metadata_differences': []}, {'main': 'outputs/strong_baseline_r52_01/no_phase', 'replay': 'outputs/strong_baseline_r52_01/replay/no_phase', 'exact_arrays': 10, 'non_scientific_metadata_differences': []}], 'gradient_snapshot_replays': 6, 'r51_sealed_files_verified': 88, 'unchanged_heuristic_trajectories': 18, 'applied_no_beat_clock_audit': [{'key': 'song0_events2_gap48', 'actual_no_beat_frame_center_maps': [[[2020, 1536], [2267, 1775]], [[2473, 1824], [2708, 2063]]], 'compiled_source_Core_exact': True, 'registered_source_events_retain_original_E1W_metadata': True}, {'key': 'song1_events2_gap48', 'actual_no_beat_frame_center_maps': [[[2020, 1534], [2267, 1773]], [[2473, 1822], [2708, 2061]]], 'compiled_source_Core_exact': True, 'registered_source_events_retain_original_E1W_metadata': True}, {'key': 'song2_events2_gap48', 'actual_no_beat_frame_center_maps': [[[2660, 1536], [2895, 1775]], [[2473, 1824], [2708, 2063]]], 'compiled_source_Core_exact': True, 'registered_source_events_retain_original_E1W_metadata': True}], 'optimizer_calls': 0, 'train_fit_scope': 'FIXED_TRAIN_ROWS_BEFORE_AFTER_NO_SELECTION', 'can_route': False}
656 passed, 1 skipped, 5 warnings in 32.34s
原R51的88文件保持；本轮轨迹、检查点、真实梯度及采样重放保留服务器。完整Native不可评、无整曲、研究不晋升。

## 结果解释

NoSource保留相同几何Anchor与Context，只移除专用Source入口；并非所有来源信息被移除。Context可读取几何桥先验，不能称为真实桥观测。NoPhase零化整个8D位置/运动学代理字段，仍保留独立2D时钟，不是人工语义阶段消融。音乐替换为同长真实片段，不把非目标偏移定义为艺术错误。NoBeat实际线性帧中心图及其Core数组校验保存在独立复验回执；source_events中原E1W是冻结选择登记，不是NoBeat实际图。

本轮固定R51的16事件预算库，每Core实际合法候选10条，并不把它称为完整139候选库的搜索结果。匹配结构对照只使用预先固定的三首音乐、两事件、48帧桥，32种配置共享同一候选及几何预算。匿名页仅是准备，human_reviewed=false。

训练后的角度误差降低不等同整体质量提高，必须同时检查FK、幅度及相对来源动态残差。新增固定Train输入的初末检查点前向评价用于拟合诊断，不用于挑选权重。独立训练复核验证6份实际梯度快照，未宣称从零重训复现全部480次Adam更新。
