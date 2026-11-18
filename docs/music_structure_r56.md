# R56 Whole-Song Music-Structure Attribution

本轮仅纯几何研究。R52/R53、Whole Song v1/v2、SDMR和生产路径不改。13个Core、12×48帧过渡、16个封存单位、原E1-W[.75,1.33]与G0/G1筛查不变；每布局最大3072条几何试算、Beam128，B1/B3共享完全相同候选和连接图。合法候选数量可随时长变化，报告实际试算数，不把相同上限称为相同耗时。

B0用原算法重建dunhuangwu2并要求旋转、FK和事件ID逐值一致。B1按全音轨48维对数频谱变化与停顿提出边界，整数DP分配180–300帧Core，边界定义为Transition中心，精确覆盖整首音乐。B2固定种子置乱B1时长排列，保留时长多重集/事件数/过渡/时间总预算。B3与B1同一槽位图，在原Heuristic、重复与连接代价上加固定0.25的重复结构不一致项；音乐用Chroma Core均值/标准差相似性，动作用源库32相位SO3距离核。不用新神经训练或SDMR。

全部参数先于看视频冻结。不因编舞结果重做音乐切分；主B0/B3先回传并验收，再执行B1/B2和三首已曝光开发音乐固定对照，无新盲测或泛化结论。几何失败保留位置/原因，不放宽时间或物理筛查。

评价区分目标与代理：边界评价用另一路Chroma novelty，重复评价用log-frequency音乐描述与FK动作核，动作峰值对比音频spectral flux；它们仍是相关的自动代理，不是独立人工真值。真实Phrase Boundary Alignment保持NOT_EVALUABLE；未收盲评时human_reviewed=false。不能因B3目标更高就称艺术协调更好。按完整音乐分别报告，不将78个事件对当成78个独立音乐样本。

保留全片/Core/Bridge/边界D1–D3、来源误差、事件重复、完整时间图和拒绝连接；不以Jerk高直接判动作不自然。世界Root/接触UNKNOWN，Native24不可评。主视频与交互页面全120秒、共同音乐/视角/尺度，另可输出隐藏方法名的A/B准备媒体；准备协议不产生人工评价。

CLI：`PYTHONPATH=src python -m dunhuang.research.music_structure_r56 <main|controls|verify|render|blind> --out <fresh-path> [--source <saved-stage>]`。verify新进程读取真实原音频和动作、重建全部图/轨迹，要求评分/数组NPZ字节完全一致；只单独记录运行提交不同，逐项核对原提交科学源码SHA，不放宽数值相等。训练和优化器调用均0。

V2只处理首轮变长图实际触发的原G0 `Ambiguous SO3 chart`异常：在R56外层拒绝该边并保存方法失败原因，G1因原G0异常未执行也明确记录；不吞掉其他ValueError，不修改被冻结的G0/G1算法或阈值。01失败回执保留，02同参数重跑。
