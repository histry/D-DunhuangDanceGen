# R26实际调用与边界

`scripts/run_r26_training.sh` → `reference_r26/data.py`（DataV3原始SMPL/真实音频/verified clocks →102份256帧数据）→ `preflight.py`（结构、FK/几何梯度/读取映射预检）→ `train.py`（C0/C1/C2各1200步及零优化器快照重放）。

训练：同一body目标与音乐 → q_sample（包含C4占位/裁剪Root的139D状态）→ 原Lodge音乐/动作主干。参考96×22×[6D旋转、SO(3)角速度、valid] → C0部位均值或C1/C2固定关节槽MLP →96局部64D＋全局token →主层1/4/7 cross-attention残差。主干全部冻结，梯度穿过后续层回到新支路。C1/C2共同初值，phase通道全零。C2额外由官方Normalizer反归一化→直接矩阵FK→可表示目标掩码上的几何/速度残差反传。

`scripts/run_r26_evaluation.sh` → `compress.py`（冻结memory的Train-only ridge及ChangE往返诊断）→ `sample.py`（原软DDIM与CFG、零写回Mask、166条件及完整独立重采样）→ `evaluate.py`（直接参考输入干预、来源/幅度/区域Jerk、202记录）。

`scripts/release_r26.py`：verify检查数据/PCM/真实更新/重采样/冻结基线；render生成42个固定世界视角视频；collect只发布汇总和哈希。原始权重与张量保留服务器，公众统计省去重复大账本正文但保存其哈希及原result绑定。

相对时间偏置可选实现默认关闭，仅做8帧合法映射的数学读取测试，C3未训练。Lodge本轮没有真实左右生成上下文；正式pipeline/Planner仍是冻结EDGE路径，SDMR资格不随本轮升级。22关节不补零冒充Native24，未知Root/接触/手部字段保持未知；没有整曲或文化语义资格声明。
