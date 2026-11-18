# R45：来源可行阶段时钟与冻结SDMR目标域审计

## 结果与观看

先回传主效果并验收，再运行对照；本轮新训练0、新神经采样0、新整曲0。正式EDGE/Planner/Native及R38–R44父代保持冻结。新增代码在独立research包phase_r45、target_review_r45中，没有删除仍被复验依赖的历史基线。

- `outputs/phase_r45_01/videos/index.html`：预登记飞天、女冥想、琵琶2，实际256帧五栏视频。
- `outputs/phase_r45_01/viewer/index.html`：22案例主效果，共同音乐时钟、逐帧、倍速、来源时钟与诊断。
- `outputs/phase_r45_01/viewer_controls/index.html`：同锚点PWL/C²、错音乐、无音乐、固定时钟换音乐。
- `outputs/review_r45_01/viewer/index.html`：23查询×8来源，共184匿名候选。EVENT/POSE为原始30Hz来源及上下文；M为实际E1-W重定时参考，三任务分开。

这些是保存轨迹的真实播放，来源保持部分包括显式几何保留，不能称为新神经网络学会敦煌舞。

## 阶段时钟

在完整可见来源块上检测速度峰、启动代理、制动代理、转向代理及低速保持；代理不是语义发力真值。来源独立构造最多25种时间图，限制相位±6帧、速度范围和C²端点，再按源运动活动度、D2、D3筛选；正确/错误/无音乐使用完全相同来源可行集合。极低活动保留恒等图。固定评分使用音频spectral-flux/beat代理，无模型训练。Bridge与保护域精确保留。
来源筛选要求活动度0.85–1.15、D2/D3不高于基线1.05；这是构造约束，不是独立物理验收，也未针对琵琶2特判。

|来源|Phase活动度/Geometry|Phase Core D3/Geometry|恒等Core数/2|
|---|---:|---:|---:|
|female_FeiTian_0__music|1.000000|1.000000|2|
|female_lotus_0__music|0.993698|0.992807|1|
|female_meditation_0__music|1.000000|1.000000|2|
|male_meditation_0__music|1.000000|1.000000|2|
|male_pipa_1_0__music|1.000074|0.999302|1|
|male_pipa_2_0__music|0.999949|0.977203|0|
|male_ribbon_0__music|0.988346|0.965529|0|
|male_ribbon_FenHe_0__music|1.022628|0.976461|1|

## 同锚点结构对照

MatchedLinear与MatchedC2使用R44实际相同可行锚点，仅改变时间插值，姿态插值相同。MatchedC2逐元素等于冻结R44C2；NoMusic逐元素等于Geometry；FixedMapMusicB逐元素等于Phase。

|来源|PWL切点D3|C²切点D3|PWL全Core D3|C²全Core D3|
|---|---:|---:|---:|---:|
|female_FeiTian_0__music|119.2725|114.6813|31.5864|31.2686|
|female_lotus_0__music|56.0129|55.4906|35.7375|36.2605|
|female_meditation_0__music|N/A|N/A|12.2678|12.2678|
|male_meditation_0__music|N/A|N/A|8.8217|8.8217|
|male_pipa_1_0__music|19.7807|20.0693|13.3765|13.9721|
|male_pipa_2_0__music|244.8771|216.4676|179.2289|184.1881|
|male_ribbon_0__music|312.6893|317.0115|101.9721|104.8256|
|male_ribbon_FenHe_0__music|100.5033|102.2024|67.6670|68.9611|

C²在切点与内部的效果不同，不能把平滑的时钟曲线当作Jerk必然改善。冥想恒等图没有非平凡内部锚点，切点指标N/A。

## Train-only音乐收益与停止决定

音乐组为抽样单位，8个真实配对Train音乐组、5000次组bootstrap。正值表示Phase时间代理误差低于对应控制。

|独立评价代理／控制|音乐组|改善95%区间（秒）|通过|
|---|---:|---|---|
|rms_timing_seconds/WrongMusic|8|[-0.0022934027777777814, 0.03208506944444441]|False|
|rms_timing_seconds/NoMusic|8|[-0.005694444444444446, 0.023333333333333327]|False|
|wrist_acceleration_timing_seconds/WrongMusic|8|[-0.0225708912037037, -2.6020852139652106e-18]|False|
|wrist_acceleration_timing_seconds/NoMusic|8|[-0.00976860119047619, 0.02143551587301586]|False|

音乐收益筛查=False；ChangE Train来源构造筛查=4/4。四项音乐检验均未通过，训练决定：NO_CREDIBLE_MUSIC_PROXY_BENEFIT。
当前只证明来源路径可按音乐代理选择不同时间图，未证明真实音乐表达更好。RMS与腕加速度对齐均为代理；未使用人工缺失当作停止客观实验的理由。
Train配对资产另检出9062个跨原始录制、同音频、至少90帧重叠的候选对；真实物理时钟不等于跨表演的精确参考—目标姿态/阶段真值，故该训练合同仍未验证。

## 琵琶2与数值归因

R44C2的Core D3为184.1881，Geometry为172.6710（约+6.67%）；新Phase为168.7347（约−2.28%）。未改变来源姿态残差或Bridge。连续链式法则分解使用7点导数、h=0.025来源帧、3/5阶三重Gauss积分，并单独计算139D旋转重投影/量化的增量。保存实际30Hz差分与连续积分残差，不混为同一个精确公式。

|方法|来源速度项|时钟加速度项|时钟Jerk项|积分/求导残差|表示往返增量|3/5阶相对差异|
|---|---:|---:|---:|---:|---:|---:|
|R44C2|3615.788458|68.577670|56.031373|369.576775|-0.003340|0.274134|
|Phase|-959.657473|-200.245456|-47.764523|-136.210022|0.003301|0.261303|

以上为平方D3增量的有符号诊断贡献，含交叉协方差，不能解释为独立因果百分比。3/5阶敏感性约26–27%，残差约5–7%，故只报告有限诊断；不能据此精确断言某一时钟导数就是抖动根因。表示往返增量在本例很小，不足以解释主要变化。

## 冻结SDMR审计与匿名审核准备

从原552候选预先按blind-ID哈希选每查询每来源1个：184项、23查询、8来源，不按模型分数挑选。冻结R41编码器522632参数、24个Train校准组，实际23次编码前向；Global/Diagonal/Offset与旧Native保存分数最大差为0。Debias仅对候选亲和度做Train组内平均后top10组校正，并非完整BeatDance复现。EnergyHeuristic只是归一化音量/活动均值差，并非完整Planner。

|方法|均衡后分数并列均分Top1计数|并列查询|
|---|---|---:|
|Global|{'female_lotus': 5.0, 'female_FeiTian': 12.0, 'male_pipa_2': 4.0, 'male_ribbon_FenHe': 2.0}|0|
|Diagonal|{'male_meditation': 9.0, 'male_ribbon_FenHe': 9.0, 'female_meditation': 1.0, 'male_pipa_2': 4.0}|0|
|Partial|{'female_FeiTian': 2.0, 'female_lotus': 1.0, 'female_meditation': 1.0, 'male_meditation': 1.0, 'male_pipa_1': 1.0, 'male_pipa_2': 15.0, 'male_ribbon': 1.0, 'male_ribbon_FenHe': 1.0}|8|
|Debias|{'female_lotus': 5.0, 'female_FeiTian': 8.0, 'male_pipa_2': 7.0, 'male_pipa_1': 1.0, 'male_ribbon': 1.0, 'male_ribbon_FenHe': 1.0}|0|
|EnergyHeuristic|{'male_ribbon_FenHe': 5.0, 'male_pipa_1': 8.0, 'male_ribbon': 5.0, 'male_pipa_2': 1.0, 'female_FeiTian': 2.0, 'female_lotus': 2.0}|0|

均衡候选仍有来源集中：Global飞天12/23，Partial琵琶2为15/23（包含8个并列查询的均分计数），Debias改变集中分布，不能据此称检索更正确。完整552池、均衡184池的来源可用频次和归一化比例均保存。
真实人工记录=0，状态=NOT_EVALUABLE_NO_VERIFIED_COMPLETE_MUSIC_JUDGMENTS；nDCG、偏好一致性和NULL正确性均不可报告。
导入器要求可核验独立审核人、原始来源/媒体身份、时区时间、独立意见和争议处理；EVENT/POSE/MUSIC_MATCH/QUERY_MATCH各自独立。允许UNSURE及没有合适候选，保留原始意见，不用自动标签冒充人工真值。页面默认演示标记，本轮QA演示记录被排除。三首音乐均历史曝光，新独立音乐数0；不宣称盲测泛化。

## 验证、恢复与边界

完整回归：621 passed, 10 skipped, 5 warnings in 27.64s；最后显示修正后定向测试15 passed。新进程独立复验4组科学输出，资格与官方FK JSON精确一致，审核包也重建一致；9个父代共5900文件逐SHA保持。
官方FK实算396条轨迹，最大误差6.26780093e-07 m。该结果是实现一致性，不是物理合格；完整Native24与真实接触足滑仍不可评。
同来源两种子身体几何相同，不能当独立来源。old_failed/new_plan的Root与接触先验不同，最大Root差约0.0814/0.0657米；页面明确显示该合同，不再断言全部139D相同。
原partial音频末采样与姿态最后中心的边界断言过严：现保留真实EOF且要求最后姿态中心有音频支持，不补零、不改变槽。两个尾部partial-bin样本与初始失败回执保存。原seed合同失败、原页面也保存；仅更正断言与显示，未修改候选/阈值/来源数据。
审核页CSS原先覆盖hidden，EVENT/POSE虽已移除src但仍显示空音频控件；独立派生viewer修正隐藏，原package和replay保持逐字节不变。浏览器实际测试了匿名、双时钟、帧1=1/30秒、音乐切换、共同帧与导出；下载等待曾超时，但真实演示JSON已找到归档，未重复导入。
服务端GitHub fetch遇TLS EOF时使用已推GitHub的同SHA增量bundle，未关闭TLS验证。所有运行经tmux，成功/失败回执、日志和原始输出留服务器；仅返回观看媒体与公开统计、复验/发布回执，无训练权重或原始数据集回传。临时SSH辅助与本机只读HTTP服务由最终清理回执确认。

## 参考与下一步

本轮采用可解释来源阶段/时间图与受控诊断，不宣称实现新的音乐重演网络。MotionBeat官方页面的节拍扰动与结构对齐可作为后续训练启发；其当前GitHub仅README/图像，且引用信息有MusicWeaver串项，未声称复现其训练代码。GTCC的多邻居/背景拒配用于研究依据，未新增该模型。
[MotionBeat官方页面](https://motionbeat2025.github.io/) · [官方仓库](https://github.com/wangxuanchen88/MotionBeat) · [GTCC CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Donahue_Learning_to_Predict_Activity_Progress_by_Self-Supervised_Video_Alignment_CVPR_2024_paper.html)
下一阶段先修正音乐代理与实际运动阶段的可辨识性，保留当前来源几何可行集合；在新的固定Train测试通过之前不扩大训练，不升级SDMR选路或整曲发布。
