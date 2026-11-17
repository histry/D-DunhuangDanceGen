# R44：来源姿态保持的音乐时钟与冻结检索审计

本轮先交付并在浏览器验收22案例主候选，再运行后续对照。新增研究与正式EDGE/Planner/Native分开；R38–R43父代均保持封存。新训练0、新神经采样0、新整曲0。

## 观看

- `outputs/reperformance_r44_01/videos/index.html`：飞天、女冥想、琵琶2三个预登记来源，五栏实际256帧视频。
- `outputs/reperformance_r44_01/viewer/index.html`：先交付的主候选，22案例，共同音乐时钟、暂停/倍速/逐帧、Root-relative/世界视角、诊断及时间图。
- `outputs/reperformance_r44_01/viewer_controls/index.html`：加入错音乐时间图、固定图换音乐与无音乐时间图。
- 原时钟真实来源的隐藏缺口只作评价；音乐重定时参考与它不同步的地方不应误读为来源丢失。几何双种子输出相同，不是独立样本。

## 合法时钟与实际动力学

G0是冻结Geometry，G1是旧E1-W分段线性时钟，G2是有解析速度界的C²五次时钟，G4是R43已保存Core失败样本。两种重定时使用同一SO(3) RotationSpline和Root CubicSpline，不新增姿态残差；Bridge及保护域精确保持。G2同时限制相位位移和速度变化，因此G1/G2不是只改变C²单一因素的消融。

|来源|角色|C²活动度/Geometry|C²Core D3/Geometry|动力学筛查|
|---|---|---:|---:|---|
|female_FeiTian_0__music|train|0.9861|0.9855|True|
|female_lotus_0__music|train|1.0075|0.9918|True|
|female_meditation_0__music|held|1.0000|1.0000|True|
|male_meditation_0__music|held|1.0000|1.0000|True|
|male_pipa_1_0__music|train|0.9985|0.9650|True|
|male_pipa_2_0__music|held|1.0213|1.0667|False|
|male_ribbon_0__music|train|1.0001|0.9969|True|
|male_ribbon_FenHe_0__music|held|0.9932|0.9375|True|

ChangE Train筛查 4/4；正确音乐的RMS重音时间代理相对WrongMapB和NoMusicTime的音乐组bootstrap区间分别为 [-0.025001736111111117, 0.024656249999999942]、[-0.024723958333333337, 0.019374999999999993] 秒。总准入 False，决定：STOP_NO_CREDIBLE_TRAIN_ONLY_BENEFIT。
固定时间图换播放音乐时，动作逐元素相同；改变音乐并重新求图才会改变动作时钟。这证明的是显式时间映射，而非神经网络新学到音乐表达。RMS重音和上身速度峰均为代理，不能当作真实逐事件语义。

官方FK核对 308 条轨迹，最大误差 6.27e-07 m。另保存模型Root D1/D2/D3、世界FK和足部速度代理；未知接触不能计算足滑真值，22关节不能通过完整Native24。Root是继承模型先验，非ChangE世界轨迹。

## 冻结SDMR人口与代价

全48 Train音乐组，两个原始录制、每长度90/150/240一个元数据中位窗；原Val任务不变。同一录制候选排除，UNKNOWN不硬标。正例是官方物理时钟关联，不是艺术适配真值。共享编舞与缺失支持完整报告，不宣称未见编舞泛化。
查询 Train=258，Val=90，有效任务=777；不可用查询池=218，缺失查询长度支持=30。

|角色/候选池|Global|Diagonal|Offset|Partial|CalibratedPartial|TrainBankDebias|
|---|---:|---:|---:|---:|---:|---:|
|train/same_music|9.72|53.70|24.07|22.28|24.27|2.78|
|train/energy_matched|30.90|48.26|50.00|31.60|30.56|36.46|
|train/cross_music|30.90|47.95|49.69|31.60|30.56|36.46|
|val/same_music|17.73|10.66|11.92|13.32|13.87|27.45|
|val/energy_matched|36.90|42.11|39.88|39.60|34.84|39.29|
|val/cross_music|36.90|42.11|39.88|38.49|33.73|39.29|

固定路径match=1-sim，双侧drop总计0.6，局部无差异相似度约0.4；这不是全局DP的充分必要条件。原始23×24候选全部与冻结路径评分核对。Train24组负例q95中位数校准阈值=0.334641，drop=0.332680；另外24个Train probe音乐组宏FPR=0.0669，TPR=0.1067，信息性筛查=False。这些probe仍是编码器已训练人口，不是新泛化集。
TrainBankDebias为固定Train查询库K=10的候选亲和度校正，未完整复现BeatDance。仅在配对候选任务评价；没有把该校正未经验证地接到Native选路。候选池含离线构造正例，不能当部署检索总体准确率。

## Native来源集中与拒配

|方法|分数并列均分Top1来源计数|所选候选平均路径覆盖|
|---|---|---:|
|Global|{'female_FeiTian': 23.0}|0.809%|
|Diagonal|{'male_meditation': 9.0, 'male_ribbon_FenHe': 9.0, 'female_meditation': 1.0, 'male_pipa_2': 4.0}|0.292%|
|Offset|{'male_meditation': 9.0, 'male_ribbon_FenHe': 11.0, 'female_meditation': 1.0, 'male_pipa_2': 2.0}|0.214%|
|Partial|{'female_FeiTian': 7.958333333333342, 'female_lotus': 0.29166666666666663, 'female_meditation': 0.29166666666666663, 'male_meditation': 0.29166666666666663, 'male_pipa_1': 0.29166666666666663, 'male_pipa_2': 13.291666666666664, 'male_ribbon': 0.29166666666666663, 'male_ribbon_FenHe': 0.29166666666666663}|1.631%|
|Hybrid|{'male_meditation': 6.0, 'female_FeiTian': 13.0, 'female_meditation': 1.0, 'male_ribbon_FenHe': 2.0, 'male_pipa_2': 1.0}|0.334%|

Train校准移到Native后，平均并列Top1覆盖 8.347%；增加覆盖不等于拒配/语义正确。来源可用频次和Top1/可用候选比例保留在摘要；不能把飞天数量多的候选池与音乐适配能力混为一谈。
`native/blind_items.json`、`admin_map.json`仅为552项独立盲审准备，未产生人工结论；E/P/M任务、弃权和争议处理保持分开。无人工标签不阻断本轮客观实验，但不能声称艺术适配通过。

## 可复核边界

原始/校准路径、完整候选评分、音乐置换/动作置换/双置零及音乐组置信区间均封存。特征、候选人口、Native、主候选和全部后续对照均由新进程复算；复验不优化权重。最终完整回归 606 passed, 10 skipped, 5 warnings in 25.30s；7个父代逐文件SHA复核；发布回执及失败记录保留。
保持正式选路关闭。下一步应先解决音乐代理收益不稳和留出动态风险，不因合同通过增加训练或强跑整曲。

参考机制：[SciPy RotationSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RotationSpline.html)、[BPoly.from_derivatives](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BPoly.from_derivatives.html)、[BeatDance ICMR2024官方实现](https://github.com/XulongT/BeatDance)、[Drop-DTW官方实现](https://github.com/SamsungLabs/Drop-DTW)。本轮路径为严格一对一match/drop，并非完整Drop-DTW；C²不保证Jerk降低。
