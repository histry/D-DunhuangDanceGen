# R51 来源集中归因与真实冻结条件验证

先交付规划与四个真实冻结Lodge案例，再开展注册对照。全部原始动作、投影前proposal和层残差均保留；保护区由显式投影保留。

## 规划选择

完整预算池与原Top4的选择、错误音乐及禁止重复均按固定人口报告。来源增加不等于音乐内容更准确。

|计划|评分/约束|选中来源|重复事件|
|---|---|---|---|
|song0_events2_gap24|Top4|male_meditation → female_FeiTian|False|
|song0_events2_gap24|FullPool|male_meditation → female_FeiTian|False|
|song0_events2_gap24|FullPoolWrongMusic|female_FeiTian → male_meditation|False|
|song0_events2_gap24|Top4WrongMusic|female_FeiTian → male_meditation|False|
|song0_events2_gap24|FullPoolNoExactRepeat|male_meditation → female_FeiTian|False|
|song0_events2_gap24|LocalFullPool|female_FeiTian → male_meditation|False|
|song0_events2_gap24|HeuristicFullPool|male_ribbon_FenHe → male_pipa_2|False|
|song0_events2_gap48|Top4|male_meditation → female_FeiTian|False|
|song0_events2_gap48|FullPool|male_meditation → female_FeiTian|False|
|song0_events2_gap48|FullPoolWrongMusic|male_meditation → female_FeiTian|False|
|song0_events2_gap48|Top4WrongMusic|male_meditation → female_FeiTian|False|
|song0_events2_gap48|FullPoolNoExactRepeat|male_meditation → female_FeiTian|False|
|song0_events2_gap48|LocalFullPool|female_FeiTian → male_meditation|False|
|song0_events2_gap48|HeuristicFullPool|male_ribbon_FenHe → male_pipa_2|False|
|song0_events2_gap72|Top4|male_meditation → female_FeiTian|False|
|song0_events2_gap72|FullPool|male_meditation → female_FeiTian|False|
|song0_events2_gap72|FullPoolWrongMusic|female_FeiTian → male_meditation|False|
|song0_events2_gap72|Top4WrongMusic|female_FeiTian → male_meditation|False|
|song0_events2_gap72|FullPoolNoExactRepeat|male_meditation → female_FeiTian|False|
|song0_events2_gap72|LocalFullPool|female_FeiTian → male_meditation|False|
|song0_events2_gap72|HeuristicFullPool|male_ribbon_FenHe → male_pipa_2|False|
|song0_events3_gap24|Top4|male_meditation → female_FeiTian → male_meditation|True|
|song0_events3_gap24|FullPool|male_meditation → female_FeiTian → male_meditation|True|
|song0_events3_gap24|FullPoolWrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song0_events3_gap24|Top4WrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song0_events3_gap24|FullPoolNoExactRepeat|male_ribbon_FenHe → female_FeiTian → male_meditation|False|
|song0_events3_gap24|LocalFullPool|male_meditation → female_FeiTian → male_meditation|True|
|song0_events3_gap24|HeuristicFullPool|female_FeiTian → male_ribbon_FenHe → male_pipa_2|False|
|song0_events3_gap48|Top4|male_meditation → female_FeiTian → male_meditation|True|
|song0_events3_gap48|FullPool|male_meditation → female_FeiTian → male_meditation|True|
|song0_events3_gap48|FullPoolWrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song0_events3_gap48|Top4WrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song0_events3_gap48|FullPoolNoExactRepeat|female_lotus → female_FeiTian → male_meditation|False|
|song0_events3_gap48|LocalFullPool|male_meditation → female_FeiTian → male_meditation|True|
|song0_events3_gap48|HeuristicFullPool|male_pipa_2 → male_ribbon_FenHe → male_pipa_2|True|
|song0_events3_gap72|Top4|male_meditation → female_FeiTian → male_meditation|True|
|song0_events3_gap72|FullPool|male_meditation → female_FeiTian → male_meditation|True|
|song0_events3_gap72|FullPoolWrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song0_events3_gap72|Top4WrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song0_events3_gap72|FullPoolNoExactRepeat|male_ribbon_FenHe → male_meditation → female_FeiTian|False|
|song0_events3_gap72|LocalFullPool|male_meditation → female_FeiTian → male_meditation|True|
|song0_events3_gap72|HeuristicFullPool|male_pipa_2 → male_ribbon_FenHe → male_pipa_2|True|
|song1_events2_gap24|Top4|female_FeiTian → male_meditation|False|
|song1_events2_gap24|FullPool|female_FeiTian → male_meditation|False|
|song1_events2_gap24|FullPoolWrongMusic|male_meditation → female_FeiTian|False|
|song1_events2_gap24|Top4WrongMusic|male_meditation → female_FeiTian|False|
|song1_events2_gap24|FullPoolNoExactRepeat|female_FeiTian → male_meditation|False|
|song1_events2_gap24|LocalFullPool|male_meditation → female_FeiTian|False|
|song1_events2_gap24|HeuristicFullPool|male_ribbon_FenHe → male_pipa_2|False|
|song1_events2_gap48|Top4|male_meditation → female_FeiTian|False|
|song1_events2_gap48|FullPool|male_meditation → female_FeiTian|False|
|song1_events2_gap48|FullPoolWrongMusic|male_meditation → female_FeiTian|False|
|song1_events2_gap48|Top4WrongMusic|male_meditation → female_FeiTian|False|
|song1_events2_gap48|FullPoolNoExactRepeat|male_meditation → female_FeiTian|False|
|song1_events2_gap48|LocalFullPool|male_meditation → female_FeiTian|False|
|song1_events2_gap48|HeuristicFullPool|male_ribbon_FenHe → male_pipa_2|False|
|song1_events2_gap72|Top4|male_meditation → female_FeiTian|False|
|song1_events2_gap72|FullPool|male_meditation → female_FeiTian|False|
|song1_events2_gap72|FullPoolWrongMusic|female_FeiTian → male_meditation|False|
|song1_events2_gap72|Top4WrongMusic|female_FeiTian → male_meditation|False|
|song1_events2_gap72|FullPoolNoExactRepeat|male_meditation → female_FeiTian|False|
|song1_events2_gap72|LocalFullPool|male_meditation → female_FeiTian|False|
|song1_events2_gap72|HeuristicFullPool|male_ribbon_FenHe → male_pipa_2|False|
|song1_events3_gap24|Top4|male_meditation → female_FeiTian → male_meditation|True|
|song1_events3_gap24|FullPool|male_meditation → female_FeiTian → male_meditation|True|
|song1_events3_gap24|FullPoolWrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song1_events3_gap24|Top4WrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song1_events3_gap24|FullPoolNoExactRepeat|male_ribbon_FenHe → male_meditation → female_FeiTian|False|
|song1_events3_gap24|LocalFullPool|male_meditation → female_FeiTian → male_meditation|True|
|song1_events3_gap24|HeuristicFullPool|female_FeiTian → male_ribbon_FenHe → male_pipa_2|False|
|song1_events3_gap48|Top4|male_meditation → female_FeiTian → male_meditation|True|
|song1_events3_gap48|FullPool|male_meditation → female_FeiTian → male_meditation|True|
|song1_events3_gap48|FullPoolWrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song1_events3_gap48|Top4WrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song1_events3_gap48|FullPoolNoExactRepeat|male_meditation → female_meditation → female_FeiTian|False|
|song1_events3_gap48|LocalFullPool|male_meditation → female_FeiTian → male_meditation|True|
|song1_events3_gap48|HeuristicFullPool|female_FeiTian → male_ribbon_FenHe → male_pipa_2|False|
|song1_events3_gap72|Top4|male_meditation → female_FeiTian → male_meditation|True|
|song1_events3_gap72|FullPool|male_meditation → female_FeiTian → male_meditation|True|
|song1_events3_gap72|FullPoolWrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song1_events3_gap72|Top4WrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song1_events3_gap72|FullPoolNoExactRepeat|male_ribbon_FenHe → male_meditation → female_FeiTian|False|
|song1_events3_gap72|LocalFullPool|male_meditation → female_FeiTian → male_meditation|True|
|song1_events3_gap72|HeuristicFullPool|male_ribbon_FenHe → male_pipa_2 → male_ribbon_FenHe|True|
|song2_events2_gap24|Top4|male_meditation → female_FeiTian|False|
|song2_events2_gap24|FullPool|male_meditation → female_FeiTian|False|
|song2_events2_gap24|FullPoolWrongMusic|male_meditation → female_FeiTian|False|
|song2_events2_gap24|Top4WrongMusic|male_meditation → female_FeiTian|False|
|song2_events2_gap24|FullPoolNoExactRepeat|male_meditation → female_FeiTian|False|
|song2_events2_gap24|LocalFullPool|male_meditation → female_FeiTian|False|
|song2_events2_gap24|HeuristicFullPool|female_FeiTian → male_pipa_2|False|
|song2_events2_gap48|Top4|male_meditation → female_FeiTian|False|
|song2_events2_gap48|FullPool|male_meditation → female_FeiTian|False|
|song2_events2_gap48|FullPoolWrongMusic|male_meditation → female_FeiTian|False|
|song2_events2_gap48|Top4WrongMusic|male_meditation → female_FeiTian|False|
|song2_events2_gap48|FullPoolNoExactRepeat|male_meditation → female_FeiTian|False|
|song2_events2_gap48|LocalFullPool|female_FeiTian → male_meditation|False|
|song2_events2_gap48|HeuristicFullPool|female_FeiTian → male_pipa_2|False|
|song2_events2_gap72|Top4|female_FeiTian → male_meditation|False|
|song2_events2_gap72|FullPool|female_FeiTian → male_meditation|False|
|song2_events2_gap72|FullPoolWrongMusic|male_meditation → female_FeiTian|False|
|song2_events2_gap72|Top4WrongMusic|male_meditation → female_FeiTian|False|
|song2_events2_gap72|FullPoolNoExactRepeat|female_FeiTian → male_meditation|False|
|song2_events2_gap72|LocalFullPool|female_FeiTian → male_meditation|False|
|song2_events2_gap72|HeuristicFullPool|female_FeiTian → male_pipa_2|False|
|song2_events3_gap24|Top4|male_meditation → female_FeiTian → male_meditation|True|
|song2_events3_gap24|FullPool|male_meditation → female_FeiTian → male_meditation|True|
|song2_events3_gap24|FullPoolWrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song2_events3_gap24|Top4WrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song2_events3_gap24|FullPoolNoExactRepeat|male_meditation → male_ribbon_FenHe → female_FeiTian|False|
|song2_events3_gap24|LocalFullPool|male_meditation → female_FeiTian → male_meditation|True|
|song2_events3_gap24|HeuristicFullPool|male_ribbon_FenHe → female_FeiTian → male_pipa_2|False|
|song2_events3_gap48|Top4|male_meditation → female_FeiTian → male_meditation|True|
|song2_events3_gap48|FullPool|male_meditation → female_FeiTian → male_meditation|True|
|song2_events3_gap48|FullPoolWrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song2_events3_gap48|Top4WrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song2_events3_gap48|FullPoolNoExactRepeat|female_lotus → male_meditation → female_FeiTian|False|
|song2_events3_gap48|LocalFullPool|male_meditation → female_FeiTian → male_meditation|True|
|song2_events3_gap48|HeuristicFullPool|male_pipa_2 → female_FeiTian → male_pipa_2|True|
|song2_events3_gap72|Top4|male_meditation → female_FeiTian → male_meditation|True|
|song2_events3_gap72|FullPool|male_meditation → female_FeiTian → male_meditation|True|
|song2_events3_gap72|FullPoolWrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song2_events3_gap72|Top4WrongMusic|male_meditation → female_FeiTian → male_meditation|True|
|song2_events3_gap72|FullPoolNoExactRepeat|female_lotus → male_meditation → female_FeiTian|False|
|song2_events3_gap72|LocalFullPool|male_meditation → female_FeiTian → male_meditation|True|
|song2_events3_gap72|HeuristicFullPool|male_pipa_2 → female_FeiTian → male_pipa_2|True|

各臂汇总：{'FullPool': {'cases': 18, 'admitted': 18, 'unique_sources': 2, 'unique_events': 2, 'repeated_event_cases': 9}, 'FullPoolNoExactRepeat': {'cases': 18, 'admitted': 18, 'unique_sources': 5, 'unique_events': 5, 'repeated_event_cases': 0}, 'FullPoolWrongMusic': {'cases': 18, 'admitted': 18, 'unique_sources': 2, 'unique_events': 2, 'repeated_event_cases': 9}, 'HeuristicFullPool': {'cases': 18, 'admitted': 18, 'unique_sources': 3, 'unique_events': 3, 'repeated_event_cases': 5}, 'LocalFullPool': {'cases': 18, 'admitted': 18, 'unique_sources': 2, 'unique_events': 2, 'repeated_event_cases': 9}, 'Top4': {'cases': 18, 'admitted': 18, 'unique_sources': 2, 'unique_events': 2, 'repeated_event_cases': 9}, 'Top4WrongMusic': {'cases': 18, 'admitted': 18, 'unique_sources': 2, 'unique_events': 2, 'repeated_event_cases': 9}}

规划保存原139事件合法性与分项，连接图仅使用R50两条/来源的16条元数据预算。FullPool取消Top4截断，NoExactRepeat为工程约束，不能作为音乐选择成功证明。

## 真实冻结采样

|来源|条件|Gap角误差rad|投影前Gap角误差|活动度|外侧精确保留|
|---|---|---:|---:|---:|---|
|paired_4|C0Frozen|0.623353|0.487034|0.607919|True|
|paired_4|C1ZeroPhase|0.565666|0.315340|0.495307|True|
|paired_4|C2Phase|0.569876|0.322718|0.503285|True|
|paired_4|C3NoMusic|0.571472|0.324345|0.494465|True|
|paired_4|C4WrongMusic|0.572647|0.327247|0.490788|True|
|paired_4|C5WrongSource|0.569991|0.318565|0.499378|True|
|paired_4|C6NoLeft|0.571854|0.327049|0.494226|True|
|paired_4|C7NoRight|0.569875|0.322715|0.503293|True|
|paired_4|C8WrongPhase|0.569049|0.320363|0.500838|True|
|paired_8|C0Frozen|0.460093|0.391270|1.375076|True|
|paired_8|C1ZeroPhase|0.373767|0.247736|1.300516|True|
|paired_8|C2Phase|0.376188|0.255393|1.297034|True|
|paired_8|C3NoMusic|0.373159|0.250076|1.310981|True|
|paired_8|C4WrongMusic|0.373554|0.250815|1.299080|True|
|paired_8|C5WrongSource|0.384467|0.277541|1.300603|True|
|paired_8|C6NoLeft|0.375228|0.254191|1.298448|True|
|paired_8|C7NoRight|0.376189|0.255395|1.297033|True|
|paired_8|C8WrongPhase|0.375353|0.253095|1.296811|True|
|female_FeiTian_5256|C0Frozen|0.545552|0.489246|0.981659|True|
|female_FeiTian_5256|C1ZeroPhase|0.466922|0.347725|0.975876|True|
|female_FeiTian_5256|C2Phase|0.466274|0.347957|0.978992|True|
|female_FeiTian_5256|C3NoMusic|0.466274|0.347957|0.978992|True|
|female_FeiTian_5256|C4WrongMusic|0.469056|0.356262|0.986090|True|
|female_FeiTian_5256|C5WrongSource|0.466808|0.357414|0.981172|True|
|female_FeiTian_5256|C6NoLeft|0.466376|0.348136|0.979085|True|
|female_FeiTian_5256|C7NoRight|0.466263|0.347947|0.978968|True|
|female_FeiTian_5256|C8WrongPhase|0.466633|0.349016|0.979772|True|
|male_pipa_2_1592|C0Frozen|0.366806|0.486133|0.425969|True|
|male_pipa_2_1592|C1ZeroPhase|0.312414|0.468085|0.490501|True|
|male_pipa_2_1592|C2Phase|0.310556|0.466487|0.488099|True|
|male_pipa_2_1592|C3NoMusic|0.310556|0.466487|0.488099|True|
|male_pipa_2_1592|C4WrongMusic|0.309195|0.451540|0.498828|True|
|male_pipa_2_1592|C5WrongSource|0.311299|0.476571|0.497804|True|
|male_pipa_2_1592|C6NoLeft|0.310569|0.466191|0.488155|True|
|male_pipa_2_1592|C7NoRight|0.310229|0.470166|0.487743|True|
|male_pipa_2_1592|C8WrongPhase|0.310339|0.465486|0.488284|True|

Source仍沿用R38的1-C2_ENVELOPE注入门控，中心处直接Source残差为零；必须通过完整自注意力与Context间接传播，本轮未擅自修改旧门控。阶段来自可见时钟及运动学代理；隐藏缺口不参与条件。Source的双侧观测可能与Context冗余，去单侧Context不能解释为删除了全部单侧信息。ChangE零音乐不提供跨音乐真值，Root只有模型曲线，没有完整世界Root与接触资格。

## 同人口配对排序

### equal_duration_near_motion_energy
summary：{'tasks': 128, 'methods': {'Content': {'R1': 0.3, 'ci95': [0.10000000000000002, 0.5], 'songs': 6}, 'MusicSwap': {'R1': 0.24444444444444446, 'ci95': [0.061111111111111116, 0.4666666666666666], 'songs': 6}, 'Energy': {'R1': 0.08287037037037037, 'ci95': [0.024999999999999998, 0.1486111111111111], 'songs': 6}, 'Duration': {'R1': 0.20000000000000004, 'ci95': [0.20000000000000004, 0.20000000000000004], 'songs': 6}, 'BothZero': {'R1': 0.20000000000000004, 'ci95': [0.20000000000000004, 0.20000000000000004], 'songs': 6}, 'Local': {'R1': 0.3, 'ci95': [0.10000000000000002, 0.5], 'songs': 6}, 'Full': {'R1': 0.3351851851851852, 'ci95': [0.14351851851851852, 0.5185185185185185], 'songs': 6}, 'FullMusicSwap': {'R1': 0.26944444444444443, 'ci95': [0.13333333333333333, 0.4527777777777778], 'songs': 6}}, 'paired_differences': {'Content': {'delta': 0.03518518518518516, 'ci95': [-0.05833333333333337, 0.13518518518518516], 'songs': 6}, 'MusicSwap': {'delta': 0.09074074074074075, 'ci95': [-0.17592592592592593, 0.37409722222222197], 'songs': 6}, 'Energy': {'delta': 0.2523148148148148, 'ci95': [0.029629629629629634, 0.48148148148148145], 'songs': 6}, 'Duration': {'delta': 0.13518518518518513, 'ci95': [-0.0564814814814815, 0.3185185185185184], 'songs': 6}, 'BothZero': {'delta': 0.13518518518518513, 'ci95': [-0.0564814814814815, 0.3185185185185184], 'songs': 6}, 'Local': {'delta': 0.03518518518518516, 'ci95': [-0.05833333333333337, 0.13518518518518516], 'songs': 6}, 'FullMusicSwap': {'delta': 0.06574074074074075, 'ci95': [-0.17592592592592593, 0.31666666666666665], 'songs': 6}}}
source_balanced_summary：{'tasks': 128, 'methods': {'Content': {'R1': 0.30833333333333335, 'ci95': [0.10833333333333334, 0.5083333333333333], 'songs': 6}, 'MusicSwap': {'R1': 0.24444444444444446, 'ci95': [0.061111111111111116, 0.4666666666666666], 'songs': 6}, 'Energy': {'R1': 0.09120370370370372, 'ci95': [0.024999999999999998, 0.1615740740740741], 'songs': 6}, 'Duration': {'R1': 0.20638888888888893, 'ci95': [0.20000000000000004, 0.21583333333333335], 'songs': 6}, 'BothZero': {'R1': 0.20638888888888893, 'ci95': [0.20000000000000004, 0.21583333333333335], 'songs': 6}, 'Local': {'R1': 0.30833333333333335, 'ci95': [0.10833333333333334, 0.5083333333333333], 'songs': 6}, 'Full': {'R1': 0.3351851851851852, 'ci95': [0.14351851851851852, 0.5185185185185185], 'songs': 6}, 'FullMusicSwap': {'R1': 0.27777777777777773, 'ci95': [0.1361111111111111, 0.4527777777777778], 'songs': 6}}, 'paired_differences': {'Content': {'delta': 0.02685185185185183, 'ci95': [-0.05833333333333337, 0.112037037037037], 'songs': 6}, 'MusicSwap': {'delta': 0.09074074074074075, 'ci95': [-0.17592592592592593, 0.37409722222222197], 'songs': 6}, 'Energy': {'delta': 0.24398148148148147, 'ci95': [0.021296296296296296, 0.46666666666666673], 'songs': 6}, 'Duration': {'delta': 0.12879629629629624, 'ci95': [-0.05814814814814817, 0.3121296296296296], 'songs': 6}, 'BothZero': {'delta': 0.12879629629629624, 'ci95': [-0.05814814814814817, 0.3121296296296296], 'songs': 6}, 'Local': {'delta': 0.02685185185185183, 'ci95': [-0.05833333333333337, 0.112037037037037], 'songs': 6}, 'FullMusicSwap': {'delta': 0.057407407407407414, 'ci95': [-0.17592592592592593, 0.3074074074074074], 'songs': 6}}}
### same_song_wrong_time
summary：{'tasks': 20, 'methods': {'Content': {'R1': 0.2, 'ci95': [None, None], 'songs': 1}, 'MusicSwap': {'R1': 0.25, 'ci95': [None, None], 'songs': 1}, 'Energy': {'R1': 0.05, 'ci95': [None, None], 'songs': 1}, 'Duration': {'R1': 0.2, 'ci95': [None, None], 'songs': 1}, 'BothZero': {'R1': 0.2, 'ci95': [None, None], 'songs': 1}, 'Local': {'R1': 0.2, 'ci95': [None, None], 'songs': 1}, 'Full': {'R1': 0.2, 'ci95': [None, None], 'songs': 1}, 'FullMusicSwap': {'R1': 0.15, 'ci95': [None, None], 'songs': 1}}, 'paired_differences': {'Content': {'delta': 0.0, 'ci95': [None, None], 'songs': 1}, 'MusicSwap': {'delta': -0.04999999999999999, 'ci95': [None, None], 'songs': 1}, 'Energy': {'delta': 0.15000000000000002, 'ci95': [None, None], 'songs': 1}, 'Duration': {'delta': 0.0, 'ci95': [None, None], 'songs': 1}, 'BothZero': {'delta': 0.0, 'ci95': [None, None], 'songs': 1}, 'Local': {'delta': 0.0, 'ci95': [None, None], 'songs': 1}, 'FullMusicSwap': {'delta': 0.05000000000000002, 'ci95': [None, None], 'songs': 1}}}
source_balanced_summary：{'tasks': 20, 'methods': {'Content': {'R1': 0.2, 'ci95': [None, None], 'songs': 1}, 'MusicSwap': {'R1': 0.25, 'ci95': [None, None], 'songs': 1}, 'Energy': {'R1': 0.05, 'ci95': [None, None], 'songs': 1}, 'Duration': {'R1': 0.2, 'ci95': [None, None], 'songs': 1}, 'BothZero': {'R1': 0.2, 'ci95': [None, None], 'songs': 1}, 'Local': {'R1': 0.2, 'ci95': [None, None], 'songs': 1}, 'Full': {'R1': 0.2, 'ci95': [None, None], 'songs': 1}, 'FullMusicSwap': {'R1': 0.15, 'ci95': [None, None], 'songs': 1}}, 'paired_differences': {'Content': {'delta': 0.0, 'ci95': [None, None], 'songs': 1}, 'MusicSwap': {'delta': -0.04999999999999999, 'ci95': [None, None], 'songs': 1}, 'Energy': {'delta': 0.15000000000000002, 'ci95': [None, None], 'songs': 1}, 'Duration': {'delta': 0.0, 'ci95': [None, None], 'songs': 1}, 'BothZero': {'delta': 0.0, 'ci95': [None, None], 'songs': 1}, 'Local': {'delta': 0.0, 'ci95': [None, None], 'songs': 1}, 'FullMusicSwap': {'delta': 0.05000000000000002, 'ci95': [None, None], 'songs': 1}}}

配对排序沿用历史曝光Val，不声称新音乐盲测。90帧候选Identity Warp为零，Full与Local的差异来自Multi。来源均衡子集与完整人口分开报告。

## 资格与复验

644 passed, 10 skipped, 5 warnings in 29.66s。五组新进程科学JSON/全部数组精确重算；主效果和对照共同数组完全相同，R50的29个封存文件保持。没有新训练和整曲，Native24仍不可评，SDMR和神经候选保持Shadow。统一EDGE/Lodge公平选型仍需统一帧数、动作表示和监督任务，本轮不把它们混成架构性能结果。
