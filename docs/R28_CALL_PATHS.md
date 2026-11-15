# R28 调用与冻结边界

正式生成入口不变：pipeline → Music Timeline / Event Plan → frozen EDGE → original Native。研究路径不进入正式Planner，不把新C1权重交给旧Adapter消费者。

R28输入：
- R26已核验DataV3原始来源、真实音频时钟和冻结裁窗：8条只读，4Train、4适配持出（均历史R26曝光）。
- R27教师及P：冻结正对照，另按固定种子生成4条不训练的新教师。
- danceRAG本机官方归档中的FineDance315D／35D／真实波形：按文件清单SHA复制四序列至inputs/contract_r28；服务器原LFS目录保留。

data.py调用固定LODGE官方函数抽取器与FK，定义一次来源级floor0／高度C4合同；旧raw139/target不动，输出新旧数组和可逆坐标偏移。数据与原函数源哈希绑定，独立新进程重建。第一次数据元数据的clipping字段来自已裁剪输出，audit.py只读补算归一化前支持域；当前data.py已修正未来构建字段，原科学归档不覆盖。

representation.py先解码SO(3)再输出规范行6D；只作用参考，不替换目标或采样状态。factorial.py固定P与C1，比较见过／新来源×原始／规范参考，保存最终生成、memory/注入读出与几何等价证据。readout.py补冻结P真实来源单步预测；official_clipped与affine_unclipped_diagnostic命名分开，后者不进入原采样器或验收。

audio_audit.py从认证FineDance波形重新运行原LODGE35D提取，保存当前librosa版本、共同裁切、各特征族与旧归档差异。旧35D接口相同不视为数值版本相同。四序列真实特征可复建，本轮未加入T0/T1训练。

admit.py封存A结果与T0/T1有限训练合同。train.py仅真实小集两个同起点、同剂量对照，Core C4均无真值损失；evaluate.py固定完整采样和无参考精确恢复，所有预测按共同官方model0坐标评价，不重新贴地。decision.py分开来源、代理物理保留与扩大人口准入；完整Native24仍不可评。

change.py固定历史两原语／两音乐／seed24041，完成未配对ChangE迁移，全部模型输出保留，不按ChangE挑选权重。Source Root未观测，旋转/局部几何与Root/足部真值严格分开。

release_r28.py在新进程核验数组、真实pre/post梯度更新、相同噪声时刻、骨干摘要、失败与恢复回执，再渲染固定29段对照、收集公共统计并封存科学文件清单。raw权重／张量／音频不进Git。29段视频与观看页需要本批明确回传授权；公开报告与统计先发布，服务器最新SHA同步后记录交付结束。临时SSH文件仅清理本轮实际创建项，科学日志和预处理日志是证据而非SSH辅助文件。
