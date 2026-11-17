# R43 音乐条件 Core 试验与 SDMR-E2 事件排序报告

本轮先交付实际生成和来源候选观看，再执行固定对照。正式 EDGE、旧 Planner、R41/R42 及历史资产未修改。所有结论限于登记人口。

## 生成侧

22 个案例、两种子：主候选 88 条、三个条件控制 132 条；另有 220 条独立完整采样重放。Lodge、Source 和 Context 全部冻结，新生成器更新 0。保护域和权限合同通过，完整物理资格不等于通过。
Train 正确音乐同时优于置零/错音乐 2/8 组；ChangE 动态保持 0/4。新生成训练未准入。事后工程风险门接受 Core 0/44，回退 Geometry 44/44；回退不是新的预登记模型胜利，也不改变原始失败。

主观看：`outputs/music_reperformance_r43_01/viewer_guarded/index.html`。来源、Frozen、Core、另一音乐、NoMusic、WrongSource 与 Guarded 可切换。两个固定 MP4 不是整曲；阶段代理用于权限，未宣称新神经阶段学习。

## 检索侧

10,107 个 90/150/240 帧子窗口、178 查询、433 个有效任务、101 个不可用查询池；子窗口继承 Parent Event，不重写完整动作边界。同长度、不同录制，时钟标签不是艺术匹配。六 Train 与六历史曝光 Val 音乐组；Test 不读。
P1 Train 准入 True。NoRank/Rank/ShuffledRank 各 240 步，R41 同初始化 522632 参数、相同 Train 顺序及优化器设置；不按 Val 选择权重或预算。

下表为音乐组宏平均 Diagonal R@1（%），不同候选池可能大量重合，不能当成独立复验。

|角色/候选池|Frozen|NoRank|Rank|ShuffledRank|
|---|---:|---:|---:|---:|
|train/same_music|41.67|62.50|90.00|38.33|
|train/energy_matched|81.79|67.11|100.00|56.99|
|train/cross_music|81.79|67.11|100.00|56.99|
|val/same_music|10.66|22.47|16.21|16.79|
|val/energy_matched|42.11|37.95|41.37|43.15|
|val/cross_music|42.11|37.95|41.37|38.99|

曝光 Val 困难检索：Rank 41.37%、ShuffledRank 43.15%、Frozen 42.11%；同歌时段 Rank 16.21%、NoRank 22.47%。未建立学习排序的泛化净收益，不升级选路、不按 Val 追加训练。
已知 Token 时钟扰动 Val：Frozen 30.48%、Rank 28.18%，双置零均为20%。该辅助任务不同于历史 R41 的音频裁剪五偏移评价，不能直接混报。
Energy 标签仅指第0音乐特征均值与274D第264通道均角速度的Train-z标量代理；不是声压能量、艺术匹配或独立语义真值。候选池主动加入能量代理相近的困难负例，结果须按该构造解释。

Rank 相对各基线的音乐组配对 bootstrap 区间、R@5/MRR、内容置零与交换、六目标梯度及已知 Token 时钟扰动结果均保留在机器摘要和原始回执；不将这项辅助时钟任务称为语义事件真值。
三首历史敦煌音乐选择 24 个完整事件位置，其中 23 个有真实支持；其余未替补。每个最多 24 个合法 E1-W 候选，从原生 SO(3) 实际重定时并重新求导。全部适配标签 UNKNOWN，不把旧规划选择标为正例，SDMR 不接管选路。

## 验证与限制

独立进程逐数组重放采样、特征、评分与完整事件；三组训练在 0/120/239 步复验全部六项目标的实际共享梯度和损失，并确认更新后权重改变，复验优化器调用 0。五个冻结父代逐文件 SHA 校验见 independent。

保留首次 viewer 命名冲突、完整库字段错误、FK/Token 显示键冲突及评价中断回执。显示修复增加真实 XYZ，不改变此前评分；缓存优化对 1,732 份冻结评分矩阵逐元素完全一致。页面可播放验收与模型质量判断分开。

完整 Native24、手部和敦煌语义资格仍未取得，ChangE 世界 Root/Contact 真值保持未知；新整曲 0。AIST 生成例来自历史适配缓存，不能称新的 Data V3 Test；检索 Val 为历史曝光组，不是新盲测。

参考机制：[BADM, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Bidirectional_Autoregessive_Diffusion_Model_for_Dance_Generation_CVPR_2024_paper.html)、[Drop-DTW, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/729c68884bd359ade15d5f163166738a-Abstract.html)。本轮复用仓库单主干/严格部分路径机制，不声称完整复现这些论文。

## 实际交付与检查

完整CPU回归：595通过、10依赖跳过；GPU合同测试12通过。12组数组包和5,024个父代文件独立核验通过。两份科学封存共599文件、3,101,970,541字节，封存后由新进程逐文件复核；见 `seal_recheck.json`。

本机已回传生成、冻结检索、23个完整ChangE查询及NoRank/Rank/ShuffledRank三个训练观看页。训练权重、张量与完整原始实验保存在服务器；GitHub发布代码与公开统计。旧空白观看页和失败日志属于诊断归档，完整事件请打开 `viewer_change_v2/index.html`。
