# R55 任务一致的局部残差与可验证时钟恢复

R54历史23/32及75%资格门保持封存。本轮完成新的任务一致资格、真实主候选、先观看后固定预算对照、按音乐组评价与新进程精确复验。

## 资格实验

|项|Held 8候选|Held 32候选|作用|
|---|---:|---:|---|
|S1|81.25%|—|池化候选规模诊断|
|S2|—|98.44%|池化候选规模诊断|
|S3|100.00%|100.00%|与真实任务共用局部残差损失|
|S4|100.00%|100.00%|与真实任务共用局部残差损失|

S1/S2在共同无偏移新随机数据上比较池化候选规模，并非只修改R54候选数的逐项复现（原R54随机裁剪起点不同）。S3按6候选训练、Held8/32各4种子；S4从同一连续随机信号非循环裁剪出已知偏移，所有偏移共用支持。零音乐和反序对照保存在summary.json。真实8任务拟合R@1=100.00%。合成320更新＋真实拟合120更新；不将此结果视为艺术语义成功。

## 真实人口与固定对照

严格R54配对人口48 Train音乐组，预定哈希留出8组只做校准，40组优化；6个Val音乐组已有历史开发曝光。522632总参数／261312可训练参数；各组从原R41初始化，240步同一查询日程，未继承8任务拟合权重。Conditional加入局部时钟辅助；NoClock移除此项；Unconditional移除困难候选限制；Shuffled打乱关联与辅助时钟标签。候选池依赖已知正例分箱，只是离线诊断，不是在线ChangE召回。

主候选先回传并实际浏览器验收，随后才执行另三组。训练提交逐组保存在summary.json；objective代码、初始化与日程逐项一致。V1仍有留组动作被用作负候选，保留960次主模型更新及评价中断证据；启动器退出码即使为0，也不能代替缺失的最终评价和复验产物。V2同时隔离查询与两种候选池的音乐组、原动作SHA和音频SHA；543个有效训练任务覆盖40个优化组，删除651次候选曝光，排除2个无负例任务，隔离交集为空。8个留组只对本次适配隔离，R41预训练已曝光。

|历史Val任务|方法|组均衡R@1|组bootstrap95%区间|
|---|---|---:|---|
|cross_music|Conditional|35.14%|[0.19166666666666665, 0.4777777777777778]|
|cross_music|Heuristic|47.92%|[0.33749999999999997, 0.6055555555555555]|
|cross_music|Frozen|52.22%|[0.2722222222222222, 0.7166666666666667]|
|cross_music|HybridFrozen|32.22%|[0.16666666666666666, 0.4666666666666666]|
|cross_music|wrong_music|33.89%|[0.25555555555555554, 0.42777777777777776]|
|cross_music|music_zero|28.06%|[0.09722222222222221, 0.48055555555555546]|
|cross_music|time_reverse|31.94%|[0.13333333333333333, 0.5055555555555555]|
|cross_music|NoClock|38.61%|[0.2222222222222222, 0.5277777777777778]|
|cross_music|Unconditional|50.00%|[0.25000000000000006, 0.75]|
|cross_music|Shuffled|14.72%|[0.06666666666666667, 0.22784722222222126]|
|cross_music|EnergyCurve|27.64%|[0.13194444444444445, 0.45]|
|same_music|Conditional|49.17%|[0.14166666666666666, 0.8416666666666666]|
|same_music|Heuristic|40.00%|[0.1, 0.7]|
|same_music|Frozen|53.33%|[0.35, 0.7833333333333333]|
|same_music|HybridFrozen|43.33%|[0.16666666666666666, 0.75]|
|same_music|wrong_music|36.67%|[0.1, 0.6333333333333333]|
|same_music|music_zero|32.50%|[0.025, 0.7]|
|same_music|time_reverse|51.67%|[0.15, 0.8833333333333332]|
|same_music|NoClock|39.17%|[0.05, 0.7416666666666666]|
|same_music|Unconditional|53.33%|[0.18333333333333332, 0.8833333333333332]|
|same_music|Shuffled|20.00%|[0.0, 0.4]|
|same_music|EnergyCurve|25.00%|[0.05, 0.45]|

## 已知注入偏移恢复

|模型／控制|准确率|MAE帧|
|---|---:|---:|
|Conditional/ClockOnly|20.00%|9.200|
|Conditional/EnergyCurve|21.67%|13.233|
|Conditional/Frozen|30.00%|11.383|
|Conditional/actual|33.33%|11.433|
|Conditional/both_zero|17.33%|14.160|
|Conditional/music_zero|1.67%|15.283|
|Conditional/time_reverse|20.00%|12.667|
|NoClock/ClockOnly|20.00%|9.200|
|NoClock/EnergyCurve|21.67%|13.233|
|NoClock/Frozen|30.00%|11.383|
|NoClock/actual|40.00%|9.917|
|NoClock/both_zero|17.33%|14.160|
|NoClock/music_zero|20.00%|12.000|
|NoClock/time_reverse|20.00%|13.733|
|Unconditional/ClockOnly|20.00%|9.200|
|Unconditional/EnergyCurve|21.67%|13.233|
|Unconditional/Frozen|30.00%|11.383|
|Unconditional/actual|10.00%|12.783|
|Unconditional/both_zero|17.33%|14.160|
|Unconditional/music_zero|23.33%|12.600|
|Unconditional/time_reverse|23.33%|11.517|
|Shuffled/ClockOnly|20.00%|9.200|
|Shuffled/EnergyCurve|21.67%|13.233|
|Shuffled/Frozen|30.00%|11.383|
|Shuffled/actual|20.00%|10.917|
|Shuffled/both_zero|17.33%|14.160|
|Shuffled/music_zero|20.00%|12.383|
|Shuffled/time_reverse|25.00%|12.083|

时钟标签是人为控制的注入偏移；恢复成功并不能证明音乐重音与动作发力的语义正确。ClockOnly、EnergyCurve、Frozen及零音乐/双零/反序在同一任务中独立报告。Train留组决定置信阈值；Val只按该阈值报告接受覆盖和错误率。阈值=None，Val接受0/60。没有资格时拒答，不向ChangE宣称置信概率。

新增内容增益登记门：False。不升级默认Planner，不宣称敦煌迁移或文化语义成功。完整评价含配对音乐组差值区间及各组原值。

## 复验和交付

资格全部评分精确重现；真实拟合0/60/119及四组0/120/239步损失/梯度精确复验、更新后状态改变；全部最终评分NPZ字节和统计重建一致，优化器调用0。观看页是实际来源回放和检索/偏移读数，不是神经动作生成。服务器保留权重和原始输入，本机回传观看页、统计和回执。

本机观看：`D:/DunhuangDanceGen/outputs/clock_residual_r55_02/main_view/viewer/index.html`。独立整曲实验另见whole_song_geometry_release，未使用本轮SDMR。
