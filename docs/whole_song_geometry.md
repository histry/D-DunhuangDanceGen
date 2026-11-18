# Whole-Song Geometry v1 — first complete-audio engineering baseline

固定dunhuangwu2，复用R51的16个来源单位、R52同一Heuristic公式、原E1-W[.75,1.33]和R50 G0/G1＋R48数值阈值。没有SDMR、Diffusion或训练。不改变旧强基线。

30FPS以实际音频采样数向上取整决定总帧数。选择最接近240帧Core的事件数，精确分配全部Core长度，中间过渡固定48帧。整段来源长度/窗口长度解耦；每个Core仍须独立满足原E1-W合法域。初版不是学习式乐句切分。层间全部异来源候选连接计算G0/G1；128宽有限Beam按Heuristic、原.01过渡D3代价及预定重复惩罚(.1同单位/.02同来源)选整曲路径，失败保存位置与原因，不放宽几何阈值，不声称全局不可能。

P0音乐帧恰好覆盖一次；P1全部连接通过原筛查；P2保存全片、Core、过渡与边界D1/D2/D3及来源SO3、重复和拍点代理；P3 Root/Contact/Native24依然未知或不可评。输出rotations、root-relative 3D、逐帧来源/时间图，world_root_translation用NaN+false观测掩码，不伪造Root=0真值。只称几何整曲研究预览。

先GitHub发布，再服务器tmux run；verify用新进程从原来源重建所有候选、连接、路径与NPZ，禁止优化器。观看页播放完整音乐，支持逐帧、公共时钟、来源事件跳转、3D方位和D3曲线。
