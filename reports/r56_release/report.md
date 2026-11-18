# R56 整曲音乐结构归因：固定基线与受控对照

完成完整音乐B0/B3主观看与验收后，再执行3个历史音频×4组固定实验；不是新音乐独立盲测，不将自动代理解释为真实乐句或艺术偏好。

## 冻结与改变

B0精确重建原dunhuangwu2：旋转、FK、事件UID一致。13个Core、12×48帧过渡，16个封存单位，原E1-W速度域[.75,1.33]、G0/G1和R48阈值不改。B1用音频谱变化＋停顿代理分配180–300帧Core；B2置乱B1时长排列，保留时长多重集；B3与B1共享完全相同槽位、候选和连接图，额外约束音乐/动作重复结构相似度。Beam128，每布局同一3072次几何上限，实际合法候选/试算数可不同，不能称计算量精确匹配。

B3固定0.25×平均重复核不一致代价；音乐核用Chroma Core均值/标准差，动作核用16单位32相位局部SO3距离。所谓主题只指自动几何相似性，不是文化语义真值。原Heuristic、单位/来源重复代价(.1/.02)与过渡D3权重(.01)不变。未训练SDMR或Diffusion。

边界指标用另一路Chroma novelty代理；重复指标用log-frequency与来源相位FK核；动作重音对比音频spectral flux。它们同源且相关，不能称独立人工评价。真正Phrase Boundary Alignment保持NOT_EVALUABLE；没有人类偏好评分。

## 全部固定结果

|输入|方法|状态|来源/事件|边界代理F1|重复代理rho|整体D3 Q95|
|---|---|---|---:|---:|---:|
|dunhuangwu2|B0|NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW|6/8|0.0952|-0.0986|305.1321|
|dunhuangwu2|B1|NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW|7/9|0.2857|-0.0197|422.0830|
|dunhuangwu2|B2|NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW|6/8|0.0952|-0.1712|425.7726|
|dunhuangwu2|B3|NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW|7/9|0.2857|0.0312|414.6879|
|dunhuangwu3|B0|NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW|6/8|0.1538|0.0868|310.0471|
|dunhuangwu3|B1|NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW|7/9|0.6154|0.1380|424.9508|
|dunhuangwu3|B2|NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW|7/9|0.0769|-0.2325|425.1242|
|dunhuangwu3|B3|NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW|7/9|0.6154|-0.0343|421.1077|
|dunhuangwu4|B0|NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW|6/8|0.1000|0.2076|322.3133|
|dunhuangwu4|B1|NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW|6/8|0.4000|-0.1860|438.8538|
|dunhuangwu4|B2|NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW|7/8|0.0000|-0.1958|451.1466|
|dunhuangwu4|B3|NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW|6/8|0.4000|-0.1860|438.8538|

逐音乐的B1−B0、B1−B2、B3−B1和B3−B0差值见comparison_differences.json，不根据这轮结果调权重或选最优音乐。13个事件的78个事件对不是78个独立音乐样本；此处只做描述，不报告伪独立置信区间。高D3可能同时受活动量和高频误差影响，不能单独认定动作更差或更好。全片/Core/Bridge/边界D1–D3、来源保持误差、每个事件的实际时间图均保存。

## 可观看交付

- 主候选完整音乐视频：`D:/DunhuangDanceGen/outputs/music_structure_r56_02/main_video/B0_vs_B3_full.mp4`。
- 主交互：`D:/DunhuangDanceGen/outputs/music_structure_r56_02/main/viewer/index.html`。
- 全部对照：`D:/DunhuangDanceGen/outputs/music_structure_r56_02/controls/viewer/index.html`，三首完整音乐，共同播放/暂停/倍速/逐帧，音频结构曲线与每方法事件条可点击联动。
- 盲评准备：`outputs/music_structure_r56_02/blind/blind_pair.mp4`，仅显示A/B、同相机/尺度/音乐，隐藏方法名、条件颜色和分段条。admin_mapping.json仅供管理员，未作为公开报告或人类评分。当前PREPARATION_ONLY、人类记录0。

全部显示与输出是Root-relative身体22关节。世界Root/接触UNKNOWN，Native24不可评，不能声称物理合格、敦煌文化正确或感知收益已经确立。

## 失败与复验

01在变长B3连接图遭遇原G0 Ambiguous SO3 chart而退出1，完整失败回执保留。02仅在R56外层拒绝这种已知方法错误，其他ValueError仍抛出；G1未尝试也明确记录，不把该拒绝称为全局物理不可行。音乐算法、时长范围、评分权重、随机种子和几何阈值未根据质量结果重调。

main和controls均由新进程重新读取原音频/动作，重建全部候选图、路径、评分与轨迹，结果及确定性NPZ字节精确一致。运行提交单独记录，唯一单列的非科学字段为binding.code_commit；原提交的每份数值源码SHA均独立核对。模型/神经优化器调用0。完整回归692通过、1跳过；当前没有新的独立音乐作品或独立人工乐句真值，因此不升级正式规划资格。
