# R54 Residual SDMR：注册协议

起点为19d74d0；R52/R53、生产规划、Native保持冻结。单独研究A3，不把Geometry资格或E1-W分数当音乐标签。

1. 从封存R41配对录制与R43不改边界的子窗建立人口；使用全部Train音乐组（数据具备48组），Val为6组历史曝光诊断，Test不读。FineDance不在当前准入清单，不造配对。
2. 真音频RMS百分位与全录制相对身体FK速度建立Heuristic，等长度且同0.05评分箱内竞争。该箱由真实正例定位，所以是离线条件诊断；不能冒称在线候选选择。
3. 同音频跨录制时钟IoU>=.6为多正例；模糊重叠、同录制、共享编舞跨音乐为UNKNOWN。自动近重复音频指纹排除明显重复，不能保证艺术负例唯一性。错时仅表示物理时钟任务非目标。
4. R41总容量522632不变，训练音乐/动作输入与local分支，其余冻结。Residual=当前local−冻结local；Total=H+0.25 Residual。温度.1，残差L2权重.01。Conditional/Unconditional/Shuffled各240步，同初始化、查询顺序与候选数量，不按Val选择。
5. 先交付Train真实候选预览，然后80步合成正对照及120步固定Train拟合（门.75/.6），通过才训练Conditional。回传其真实检索效果并验收后才跑两个固定对照；全部模型冻结后统一开Val。
6. A3要求Val跨音乐R@1相对H、Frozen、HybridFrozen、Unconditional、Shuffled、WrongMusic的音乐组配对95%区间下界全部>0；至少4组。A4额外要求同歌错时任务相对H下界>0。未通过不追加Warp训练、A5或接管规划。通过只是有限历史人口资格，非跨域文化结论。
7. 新进程重建数据并精确比对NPZ；复验合成评分与训练检查点的实际损失/梯度及更新变化，复验优化器调用0；重算全部正确/错音乐评分与指标。缓存仅冻结推理，固定batch1；不得进入训练梯度路径。

服务器使用tmux、PYTHONPATH=src、OMP/OPENBLAS/MKL线程1及固定GPU。执行入口：`python -m dunhuang.research.residual_sdmr_r54 MODE --out 新目录`，MODE依序为prepare、preview、checks、train --arm Conditional、main_view、train --arm Unconditional/Shuffled、evaluate、verify、release。来源与权重各自SHA绑定，所有成功/失败与无法构造保留。观看通过只表示身份和播放正确，human_reviewed仍false。原始模型/音频/数组不入Git，完成后清理本轮SSH辅助文件。
