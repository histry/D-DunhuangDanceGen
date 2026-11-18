# R46：几何桥责任、完整来源覆盖与Train音乐阶段上限

本轮按附件R46方案，先回传并验收22案例实际轨迹与3条视频，再运行固定时钟集合的Oracle和条件对照。没有新增Lodge/SDMR训练，没有新神经采样或整曲生成。正式EDGE/Planner/Native及11个父代冻结；研究代码独立在bridge_audit_r46。未删除仍被基线复验依赖的历史代码。

## 观看

- `outputs/bridge_audit_r46_01/videos/index.html`：飞天、女冥想、琵琶2四栏实际256帧视频。
- `outputs/bridge_audit_r46_01/viewer/index.html`：22案例，来源／冻结Geometry／仅观察端点重建／冻结Phase，支持117帧跳转、共同音频时钟、逐帧和两种坐标视角。
- `outputs/bridge_audit_r46_01/viewer_oracle/index.html`：8个AIST++ Train音乐组，正确／错音乐／同能量异起音／已知错位／无音乐与事后Oracle。该页是诊断上限，不是新生成模型。

## P0：构形差异的实际责任

仅用101/102/103/152/153/154帧的观测旋转重建原R38五次桥，隐藏48帧NaN与干扰值产生同一结果。所有22例最大旋转重建差如下；原Root/contact先验保持。Geometry与Phase的Bridge一致；不是修改Core时钟可以修复的部位。

|案例|重建最大角差rad|117帧平均角差rad|连续图最大范数rad|局部图域检查|
|---|---:|---:|---:|---|
|female_FeiTian_0__music|7.67e-08|0.2499|0.6017|LOCAL_CHART_ONLY_ELIGIBLE_NOT_NATIVE|
|female_lotus_0__music|6.42e-08|0.0910|1.1103|LOCAL_CHART_ONLY_ELIGIBLE_NOT_NATIVE|
|female_meditation_0__music|6.27e-08|0.0035|0.0326|LOCAL_CHART_ONLY_ELIGIBLE_NOT_NATIVE|
|male_meditation_0__music|8.42e-08|0.0217|0.1787|LOCAL_CHART_ONLY_ELIGIBLE_NOT_NATIVE|
|male_pipa_1_0__music|7.26e-08|0.0523|0.3610|LOCAL_CHART_ONLY_ELIGIBLE_NOT_NATIVE|
|male_pipa_2_0__music|7.1e-08|0.5299|2.2357|LOCAL_CHART_ONLY_ELIGIBLE_NOT_NATIVE|
|male_ribbon_0__music|8.36e-08|0.5263|2.9242|LOCAL_CHART_ONLY_ELIGIBLE_NOT_NATIVE|
|male_ribbon_FenHe_0__music|6.57e-08|0.1424|0.7407|LOCAL_CHART_ONLY_ELIGIBLE_NOT_NATIVE|
|paired_92|7.67e-08|1.0251|5.9104|REJECT_QUINTIC_LOCAL_CHART_EXCURSION|
|paired_48|7e-08|1.4555|8.0246|REJECT_QUINTIC_LOCAL_CHART_EXCURSION|
|paired_32|7.93e-08|1.3197|10.2842|REJECT_QUINTIC_LOCAL_CHART_EXCURSION|
|paired_0|7.66e-08|1.4476|18.7904|REJECT_QUINTIC_LOCAL_CHART_EXCURSION|
|paired_37|6.68e-08|1.5351|12.4624|REJECT_QUINTIC_LOCAL_CHART_EXCURSION|
|paired_26|7.26e-08|0.7261|8.9505|REJECT_QUINTIC_LOCAL_CHART_EXCURSION|
|paired_21|7.18e-08|1.5945|5.7306|REJECT_QUINTIC_LOCAL_CHART_EXCURSION|
|paired_76|7.45e-08|1.5020|6.5819|REJECT_QUINTIC_LOCAL_CHART_EXCURSION|
|paired_4|8.22e-08|1.0947|4.9911|REJECT_QUINTIC_LOCAL_CHART_EXCURSION|
|paired_8|7.67e-08|0.4997|6.2010|REJECT_QUINTIC_LOCAL_CHART_EXCURSION|
|paired_12|6.58e-08|1.5621|8.0564|REJECT_QUINTIC_LOCAL_CHART_EXCURSION|
|paired_16|6.74e-08|1.2958|13.9459|REJECT_QUINTIC_LOCAL_CHART_EXCURSION|
|old_failed|1.36e-07|N/A|8.5114|REJECT_QUINTIC_LOCAL_CHART_EXCURSION|
|new_plan|9.51e-08|N/A|2.6068|LOCAL_CHART_ONLY_ELIGIBLE_NOT_NATIVE|

原端点检查只限制观测相对旋转，没有检查中间五次轨迹。新增独立sidecar将既有π−0.05局部图半径检查延伸到整条曲线；13/22例超域。方法对范数平方多项式的导数求根并检查端点及实根极值；这是解析极值候选加数值求根，不是区间算术严格证明。
这项检查限制当前局部坐标插值的适用域，不能证明对应动作全局不可连接，也不代表图域内的桥自然或物理合格。原桥未覆盖或裁剪，正式路由未升级；不将拒绝样本伪装成修复成功。
同录制缺口20例可评价真实隐藏动作；2个异源压力连接没有唯一真实过渡，117帧源误差N/A。全部保存源SHA、边界log速度/加速度、肩腕/双腕/脚踝/躯干几何、D1/D2/D3、Root范围及骨长误差，避免仅用低Jerk推断自然性。官方FK用于排查坐标与实现差异，不评价艺术质量。

## P3：飞天和冥想为何保持原时间

|案例|左Core原因|右Core原因|左/右可行非恒等图数量|
|---|---|---|---|
|female_FeiTian_0__music|NO_DETECTED_STAGE|NO_DETECTED_STAGE|0/0|
|female_lotus_0__music|NO_DETECTED_STAGE|NONIDENTITY_SELECTED|0/12|
|female_meditation_0__music|NO_DETECTED_STAGE|NO_DETECTED_STAGE|0/0|
|male_meditation_0__music|NO_DETECTED_STAGE|NO_DETECTED_STAGE|0/0|
|male_pipa_1_0__music|NO_DETECTED_STAGE|NONIDENTITY_SELECTED|0/14|
|male_pipa_2_0__music|NONIDENTITY_SELECTED|NONIDENTITY_SELECTED|10/10|
|male_ribbon_0__music|NONIDENTITY_SELECTED|NONIDENTITY_SELECTED|10/8|
|male_ribbon_FenHe_0__music|NONIDENTITY_SELECTED|NO_DETECTED_STAGE|12/0|

NO_DETECTED_STAGE包含低活动和没有通过检测/保护范围的阶段，不能全部解读成动作库没有运动。还独立扫描完整8条Native来源，避免把首256帧当成整条录制的能力。最大活动窗只用于诊断覆盖，不当作完整语义动作、不用于模型选择或新训练。

|来源|总帧|当前256帧覆盖|当前上身平均角速rad/s|完整来源95分位角速rad/s|最高均值256帧区间|最高窗均值rad/s|
|---|---:|---:|---:|---:|---|---:|
|female_FeiTian|11819|2.17%|0.0440|2.0489|[8920, 9176]|1.8353|
|female_lotus|6948|3.68%|0.1226|2.3191|[5037, 5293]|2.0282|
|female_meditation|6114|4.19%|0.0109|1.0318|[5684, 5940]|0.6804|
|male_meditation|6133|4.17%|0.0224|0.8980|[4048, 4304]|0.6616|
|male_pipa_1|3833|6.68%|0.1297|3.1732|[2464, 2720]|2.4339|
|male_pipa_2|3439|7.44%|1.3821|2.7129|[191, 447]|1.7948|
|male_ribbon|2366|10.82%|0.5069|2.9333|[1908, 2164]|1.9063|
|male_ribbon_FenHe|2885|8.87%|0.0916|2.6412|[1752, 2008]|1.6327|

这些字段构成Planner可读取的研究sidecar：来源、当前窗口、阶段证据、可行时间图、幅度/速率范围及接法图域风险。没有让Planner凭速度认定文化表达正确，也不强迫冥想定格产生新运动。

## P1：不扩大时钟集合的Train Oracle

只读取8个已登记AIST++ Train组，重建同一<=25图银行；Correct与冻结Phase逐元素一致，NoMusic与Geometry逐元素一致。OracleCorrect按每Core两个已定义代理误差之和事后选图，OracleWrong在错误音乐代理下选图，最终全部对正确配对音乐评价。代理缺失时不填0，支持与分母完整保留。
EnergyMatched从真实其他Train音乐中按RMS接近选择，并要求起音余弦<0.9且拍点列表不同；若无候选则记不可用。该名称只表示控制构造，不表示BPM/曲式/语义完全相同。已知音频偏移为−15/−8/+8/+15帧，无循环补齐；未观测尾部带mask，不作为负例。这里只测试评分/时钟响应，没有训练偏移分类器或声称恢复准确率。

|比较（正值表示代理误差改善）|音乐组|均值秒|95%组bootstrap区间秒|登记筛查|
|---|---:|---:|---|---|
|OracleCorrect/rms_timing_seconds/Wrong|8|0.033125|[0.020137152777777778, 0.04527951388888887]|True|
|OracleCorrect/rms_timing_seconds/EnergyMatched|8|0.03493055555555556|[0.020555555555555556, 0.05180729166666665]|True|
|OracleCorrect/rms_timing_seconds/NoMusic|8|0.034236111111111106|[0.022498263888888884, 0.04590277777777778]|True|
|OracleCorrect/rms_timing_seconds/OracleWrong|8|0.03777777777777777|[0.02298611111111111, 0.053125000000000006]|True|
|Correct/rms_timing_seconds/Wrong|8|0.007500000000000002|[-0.006874999999999999, 0.02277777777777778]|False|
|Correct/rms_timing_seconds/EnergyMatched|8|0.009305555555555558|[-0.006249999999999999, 0.030000000000000002]|False|
|Correct/rms_timing_seconds/NoMusic|8|0.008611111111111108|[-0.0059722222222222225, 0.023682291666666636]|False|
|OracleCorrect/wrist_acceleration_timing_seconds/Wrong|8|0.025140542328042335|[0.015997023809523815, 0.03602017195767196]|True|
|OracleCorrect/wrist_acceleration_timing_seconds/EnergyMatched|8|0.030646494708994717|[0.019502314814814826, 0.04158399470899471]|True|
|OracleCorrect/wrist_acceleration_timing_seconds/NoMusic|8|0.026284722222222223|[0.012164062500000001, 0.038946759259259264]|True|
|OracleCorrect/wrist_acceleration_timing_seconds/OracleWrong|8|0.026268187830687833|[0.0173941798941799, 0.03477116402116401]|True|
|Correct/wrist_acceleration_timing_seconds/Wrong|8|0.004174933862433863|[-0.01245225694444444, 0.021007564484126977]|False|
|Correct/wrist_acceleration_timing_seconds/EnergyMatched|8|0.009680886243386247|[-0.002719907407407401, 0.022643849206349202]|False|
|Correct/wrist_acceleration_timing_seconds/NoMusic|8|0.005319113756613752|[-0.010450066137566143, 0.021941054894179867]|False|

Oracle必要筛查=True；固定评分筛查=False。
Oracle是读取评价代理后的同样本可达上限，不能当部署收益、独立验证或真实语义对应。即使部分区间正，也不能因此直接训练生成网络。所有方法的完整曲线与候选逐项误差保留，不按曝光Val挑参数或权重。

## P2与训练决定

沿用R45的184项匿名准备包与导入合同，未收到新的真实独立审核文件。真实审核数0，艺术适配nDCG/偏好/拒配正确性保持NOT_EVALUABLE。人工记录缺失不阻断本轮客观P0/P1/P3。三首敦煌音乐仍为历史曝光开发资产。
跨录制候选物理时钟对9062个，经同编舞和实际>=90帧支持筛选剩476个；逐事件阶段关系仍未验证。共享时钟不是精确演绎目标，不通过音频互换制造ChangE真值。
新训练与整曲维持0：没有将Oracle输出当新监督，也没有自动晋升SDMR或Geometry。下一阶段优先处理当前桥局部图域适用性和真实阶段信息；在单事件来源保持、音乐净收益及物理支持共同成立前不扩展整曲。

## 验证与恢复

完整回归：626 passed, 10 skipped, 5 warnings in 26.50s。两包科学数组与JSON新进程精确重放；覆盖侧写、图域检查及官方FK分别重算。11个父代共6346文件逐SHA保持。
- main官方FK实算132轨迹，最大差5.10364771e-07米。
- oracle官方FK实算176轨迹，最大差5.10364771e-07米。

完整Native24与真实接触足滑仍不可评。局部视角移除每帧Root；世界视角不伪造来源Root，生成Root仍为保存先验。
首次导出误将Phase显示时钟标为恒等，现已绑定冻结真实时钟；原首次页面/数组/回执保留。独立检查确认除Phase clock元数据外所有动作与诊断数组不变。首轮截图为旧帧缓存，最终验收在新一轮读取中确认117帧=3.9秒，随后118帧=3.933秒与半速播放；最终截图保留。
新增连续图域极值检查的测试暴露了多项式乘积自动裁短导致的形状错误，已用固定阶卷积修正，合成恒等与超域案例通过；未修改真实动作或阈值来通过测试。
代码先推GitHub，服务器tmux运行；GitHub fetch遇TLS EOF后用同一已发布SHA的Git bundle同步，未关闭证书验证。媒体及公开报告回传本机，完整数组/日志/成功失败回执留服务器。封存与临时SSH/HTTP清理状态以发布回执为准。

## 补充观看：完整来源的活动覆盖

`outputs/source_coverage_r46_01/main/index.html`新增三条无音轨双栏原始来源视频，左为R45首256帧，右为同一录制中上身平均角速最高的256帧。飞天[0,256)为0.0440rad/s，对比[8920,9176)的1.8353rad/s；女冥想[0,256)为0.0109rad/s，对比[5684,5940)的0.6804rad/s。琵琶2也保留相同对照。这个结果解释首窗口的覆盖偏差，不证明高活动窗是完整原语、艺术上更好或适合目标音乐。

视频没有附加音乐，也没有新生成器；固定相机、比例和30FPS，原始帧位置写入画面。三条视频均256帧并全量解码，实际来源数组与元数据由新进程独立重建一致，浏览器播放到8.533333秒。该补充包独立封存，未修改此前194文件的桥/Oracle封存。见`coverage_display.json`中的媒体协议与再次核验回执。
