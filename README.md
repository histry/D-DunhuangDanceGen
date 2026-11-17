# DunhuangDanceGen

当前默认路径：**Event Catalog → Music Timeline → Event Plan → 冻结 EDGE（Temporal V0.5）→ 原 Native 验收**。学习检索保留shadow。研究路径已完成真实151D配对补全，以及原生SO(3)跨录制条件的Cross/Additive同参数量训练（各1,200步）；事件跟随与完整生成质量仍未合格。


新增 [R40执行合同训练与独立裁剪报告](reports/R40执行合同训练与独立裁剪对照实验报告.md)与[机器摘要](reports/r40_release/summary.json)：先交付[18组实际局部对照](outputs/contract_r40_01/viewer_final/index.html)，再完成同预算Extra。现有Context两组各240步，冻结Lodge、Source、R39连续Root及RTS；Matched未稳定超过Frozen/Extra，不升级整曲。SDMR独立裁剪90帧、绝对原点只构造时钟标签，固定80步后五偏移准确率仍20%，置零/错配也20%，继续Shadow。24个ChangE连续身体/相对Root窗口完成通道预检，5个Root位置超范围不否定身体和导数。两组缓存、梯度检查点与各72个采样数组、22个SDMR评价数组独立重放；官方FK检查通过，567测试通过/8跳过。保留R38/R39封存和全部回执，新整曲0、Native24仍不可评。

新增 [R37内容记忆与共同过渡报告](reports/R37内容记忆与共同过渡实验报告.md)、[指标与回执](reports/r37_release/r37_summary.json)：先交付[16组联动观看页](outputs/transition_r37_02/viewer_final/index.html)，包括三首完整音乐规划、8条完整来源、KF/RTS及两个真实新旧边界；新计划331帧可跳转生成128帧并返回，完整来源默认不循环。固定候选库下，来源帧重用36.49%/26.32%/31.03%降为0%，来源覆盖2条增为6/5/5条，但不以选择指标冒充艺术质量。Joint是几何共同过渡底座上的有界Lodge修改，选定接缝峰值下降，过渡区Jerk仍弱于几何基线；保护主体含显式复制，未证明学会音乐重演。Data V3真实AIST连续单位4632 Train/648 Val、48/6音乐组，姿态/几何274D与共享token重建的80步SDMR主候选完成；Val整体R@1 20.06%、同歌39.86%，仍Shadow。三首规划、双种子生成、全部单位哈希与Train统计、四个单位重建及固定权重评价独立核验；548回归通过/8跳过。保留条件首版和发布标签检查失败回执；Native24不可评、整曲生成0。

新增 [R36完整动作与来源编辑报告](reports/R36完整动作与来源编辑实验报告.md)、[指标](reports/r36_release/r36_summary.json)及[复验](reports/r36_release/r36_checks.json)：保留旧499事件，增加1,955个运动学父级候选与几何家族；三首计划42/40/40区间变为13/15/14，但平均速度降低。单一Lodge上进行160步Context任务修正及SO(3)有界来源编辑，同计划双种子保持更贴近参考，接缝仍未通过，整曲生成0、SDMR不选路、Native24不可评。最终539回归通过/8跳过，训练检查点和生成数组独立复核。先看本地 `outputs/complete_r36_01/viewer_final04/index.html`：三首计划、8来源边界、3组四栏生成；保形包含显式约束，不宣称已学会敦煌音乐重演。

新增 [R35转场图与真实上下文主效果报告](reports/R35转场图与真实上下文主效果实验报告.md)、[汇总](reports/r35_release/r35_summary.json)与[回执](reports/r35_release/r35_receipt.json)：单一 Lodge，冻结 R33 Dyn／General／CanonicalIO，只训练268160参数Context600步。8来源原始first22校核、122槽／119边界的转场图保留同来源多个候选；转场代理降低42%–52%，但事件内速度下降13%–27%，不宣称编舞质量全面改善。历史曝光Holdout4角误差0.503→0.344rad、跨端D2误差13.11→7.49，Core来源保持通过有限主效果门。先看[15组离线动态对照](outputs/transition_r35_01/viewer02/index.html)：新旧完整计划／替代候选、同计划参考／Frozen／Context、诊断与逐帧；19文件已回传并核验，浏览器交互通过，临时本机服务已关闭。4数组包／4更新对、92成功轨迹独立重放一致，复验优化器0；531回归通过／8跳过、专项33通过。原两次入口／噪声合同失败及44失败采样轨迹计数另存，不隐去失败或宣称未落盘的失败数组已复验。左右／错邻居消融延后、完整三事件0、新整曲0、Native24不可评，模型不晋升。

新增 [R34连续覆盖与两阶段观看报告](reports/R34连续覆盖与两阶段观看实验报告.md)、[机器汇总](reports/r34_release/r34_summary.json)及[回执](reports/r34_release/r34_receipt.json)：单一 Lodge Local、固定 CanonicalIO/General，Extra与Coverage同起点各600步。完整256帧连续来源使外围角误差较Extra下降66.4%／62.5%，跨旧Core边缘D2下降97.6%／95.6%；但Core肩腕误差较冻结R33上升26.6%／19.2%，稳定门失败，Context更新0、新消融0、新整曲0。三首完整音乐42／40／40槽计划已重放，并各有同一计划事件的真实Frozen/Coverage两种子短窗。统一[两阶段观看页](outputs/coverage_r34_01/viewer/index.html)已回传本地：整曲计划、候选及E1-W、三栏生成对照、逐帧及诊断联动；Chrome实际交互通过，临时只读本机服务已关闭。9数组包／10实际更新对及76轨迹新进程重放一致，优化器0；525回归通过／8依赖跳过，专项27通过。Native24仍不可评、模型不晋升，保留失败回执与冻结基线。

新增 [R32固定平台与ChangE部分观测适配报告](reports/R32固定平台与ChangE部分观测适配实验报告.md)及[回执](reports/r32_release/r32_receipt.json)：冻结R31 Canonical动作I/O与Lodge，只训练265328个参考参数；两原语raw/memory有限读出均可学，AIST隐藏Root/C4的120步pilot通过。两主臂各600步，均衡16来源各75次配对回放，辅助来源为AIST四录制或两ChangE×两历史音乐；未知Root/C4不拼成完整x0，使用真实访问状态。ChangE角误差.4516→.2453rad、局部肩腕.2633→.1147m，幅度约.963、活动约1.001；但琵琶动态及AIST持出3/4→2/4仍退化，模型不晋升、整曲0。NoRef逐位固定；13数组包/23更新对的合格谱系复验通过，原非均衡账本67–83次的失败首轮与[诊断附录](reports/R32首轮不均衡诊断附录.md)完整保留。实际累计解码2400步、生成2520步、96调用/426轨迹及独立重放，复验优化器0；478回归通过/8依赖跳过，EDGE专项10通过。42段视频及[观看页](outputs/content_r32_02/videos/index.html)已获R32本批授权回传，44文件/13,707,100字节大小与SHA一致，本地42视频全帧及36音频流核验通过，见[媒体交付回执](reports/r32_release/r32_media_delivery_receipt.json)；Native24不可评。见[预登记](docs/R32_PROTOCOL.md)、[调用合同](docs/R32_CALL_PATHS.md)。

新增 [R31原来源几何与动作合同报告](reports/R31原来源几何与动作合同一致性实验报告.md)及[回执](reports/r31_release/r31_receipt.json)：冻结R0/COVER，原Scaler盒内等价row6D审计仅FineDance八窗合格，20条AIST与两条ChangE未通过完整稳定编译，目标-only训练未准入。显式新身体Normalizer＋动作I/O＋噪声/采样合同30/30几何资格通过，保持未知Root/C4掩码；LegacyIO/CanonicalIO同300440参数、同16来源/1200步对照。历史适配持出来源R0/LegacyIO/CanonicalIO为1/4、2/4、3/4，但ChangE构形与Jerk未稳定改善，不晋升、不跑整曲。74数组包、8对实际更新、80条梯度向量及172新轨迹的独立复验通过，重放优化器0；全回归468通过/8依赖跳过，EDGE专项13通过。原常量分量内缩错误、入口导入失败及梯度首次重放失败均保留并独立恢复，未放宽登记容差。33段视频及[观看页](outputs/geometry_r31_02/videos/index.html)已按R31本批授权回传，35文件/10,641,325字节大小与SHA一致，本地33视频全帧解码及27音频流核验通过，见[媒体交付回执](reports/r31_release/r31_media_delivery_receipt.json)；Native24不可评。见[预登记](docs/R31_PROTOCOL.md)、[调用合同](docs/R31_CALL_PATHS.md)。

新增 [R30连续参考读取与来源动力学报告](reports/R30连续参考读取与来源动力学实验报告.md)及[回执](reports/r30_release/r30_receipt.json)：冻结COVER为R0，同起点/16录制/1200步R1只改Core局部插值＋全局门控，Train4/4、适配持出1/4，但动力学未改善。Train校准的R2配对Root/身体物理时序目标降低误差与Jerk，却收缩活动/幅度，来源Train2/4、持出1/4；Root直接增量抑制主要改善Root，未解决身体运动。查明AIST八来源Core身体坐标预归一化越界5.19%–9.52%，固定Normalizer裁剪造成最大约.45m关节偏移；保留合同，不以新指标改写旧失败。16数组包、8对实际更新/4个校准梯度读出、192新轨迹及192独立重放核验通过，复验优化器0；455回归通过、8依赖跳过，专项6通过。27段视频及[观看页](outputs/continuity_r30_01/videos/index.html)已获本批授权回传；29文件/9,022,786字节的大小、SHA及本地27视频流核验通过，见[媒体交付回执](reports/r30_release/r30_media_delivery_receipt.json)。研究候选不晋升，Native24不可评、整曲0。见[预登记](docs/R30_PROTOCOL.md)、[调用合同](docs/R30_CALL_PATHS.md)。

新增 [R29显式时间对应与来源覆盖报告](reports/R29显式时间对应与来源覆盖实验报告.md)及[回执](reports/r29_release/r29_receipt.json)：保留Lodge＋有序C1，265328参数不变，在完整采样中增加Core显式时钟。冻结R28 T1为B0，不重训；同起点/1200步B1的AIST训练来源1/4→4/4，适配持出0/4→1/4，Jerk代价明显。预登记16录制/8组覆盖对照保持同总预算，持出角误差/局部腕改善，但来源仍1/4。FineDance两Train/两持出录制各两窗，F1训练2/4窗口通过、录制0/2，持出0/2。COVER固定ChangE两来源×两音乐的角度/局部腕均4/4胜错时钟及交换，完整物理未通过；不跑整曲。43数组包、9对读出器/12对生成适配器更新、322新轨迹及322独立重放核验通过，复验优化器0；449回归通过、8依赖跳过，专项5通过。42段视频及[观看页](outputs/time_r29_01/videos/index.html)已获本批授权回传；44文件/12,245,647字节的大小、SHA及本地42视频流核验通过，见[媒体交付回执](reports/r29_release/r29_media_delivery_receipt.json)；完整Native24不可评。研究与冻结基线分开，见[预登记](docs/R29_PROTOCOL.md)、[覆盖合同](docs/R29_COVERAGE.md)、[调用说明](docs/R29_CALL_PATHS.md)。

新增 [R28模型输入合同与真实来源适配报告](reports/R28模型输入合同与真实来源适配实验报告.md)及[回执](reports/r28_release/r28_receipt.json)：核对官方floor0/足高C4与Normalizer裁剪，8来源Root往返RMSE .215m→2.72e-8m。冻结P已见教师raw/canonical均4/4，新教师均0/4；T0/T1同C1起点各1200步，T1完整Train来源1/4、适配持出0/4，ChangE组合仍0/2，不扩大训练或跑整曲。真实FineDance四序列315D/35D/波形已认证复制、执行原官方预处理与当前35D重提并精确复验；本轮FineDance训练0。30数组包/8对更新/成功240轨迹及240独立重放一致，失败8轨迹另存；444回归通过、8依赖跳过，专项7通过。29段视频及[观看页](outputs/contract_r28_01/videos/index.html)已获本批授权回传，31文件大小/SHA及本地视频流核验通过；见[媒体回执](reports/r28_release/r28_media_delivery_receipt.json)，完整Native仍不可评。见[预登记](docs/R28_PROTOCOL.md)、[小集合同](docs/R28_TINY_TRAINING.md)、[调用说明](docs/R28_CALL_PATHS.md)。

新增 [R27完整状态与参考学习一致性报告](reports/R27完整状态与参考学习一致性实验报告.md)及[回执](reports/r27_release/r27_receipt.json)：保留Lodge＋C1，记录真实50步逆扩散状态并完成完整139D模型教师正对照。P训练1200步，新噪声来源控制8/8，冻结C1为0/8；真实AIST/ChangE迁移未通过，历史ChangE子集综合门0/2，不跑整曲。13数组包、4对更新、82条新执行轨迹及82条独立重放一致，复验优化器0；437回归通过、8依赖跳过，EDGE专项6通过。未触发容量/缓存续训执行代码已清理；完整Native仍不可评，14段视频及[观看页](outputs/state_r27_01/videos/index.html)已获本批授权回传，16文件大小/SHA与本地视频流核验通过；见[交付回执](reports/r27_release/r27_media_delivery_receipt.json)。见[预登记](docs/R27_PROTOCOL.md)及[固定迁移协议](docs/R27_TRANSFER_PROTOCOL.md)。

新增 [R26关节身份与来源几何报告](reports/R26关节身份与来源几何实验报告.md)及[回执](reports/r26_release/r26_receipt.json)：96Train/24音乐组、6曝光DEV/3组；C0原编码、C1有序关节槽、C2加配对几何，各1200步。强制置换不变性修复，但两候选来源综合门仍0/4；C2腕部距离/Jerk部分降低伴随活动收缩，正确参考腕指标0/8胜交换。113数组包/102PCM/12对更新/166生成条件独立复验，431回归通过、8依赖跳过，专项8通过。42段视频及观看页已获授权回传并核验（outputs/reference_r26_01/videos/index.html）；C3/整曲0，完整Native不可评价。研究与冻结基线分开，见[协议](docs/R26_PROTOCOL.md)。

新增 [R25完整事件与阶段参考适配器报告](reports/R25完整事件与阶段参考适配器实验报告.md)及[回执](reports/r25_release/r25_receipt.json)：冻结Lodge Local，在真实主干三层训练完整事件/阶段参考适配器，各256720参数、600步。135条注册轨迹独立重放，31数组包/21份PCM/147条评价一致；保留失败重放后实际63次采样405轨迹，复验优化器0。阶段模型部分上身/Jerk改善，但来源综合门两模型均0/4、正确阶段仅1/8胜错置，完整Native不可评价、整曲0。423回归通过、8依赖跳过，EDGE专项11通过；40段视频与观看页已获授权回传本地并核验（outputs/lodge_adapter_r25_02/videos/index.html）。研究与冻结基线分开，见[协议](docs/R25_PROTOCOL.md)。

新增 [R24 Lodge敦煌原语与音乐控制报告](reports/R24Lodge敦煌原语与音乐控制实验报告.md)及[回执](reports/r24_release/r24_receipt.json)：主研究转向Lodge Local，真实ChangE飞天／琵琶完整Core与dunhuangwu2/3，两个种子12轨迹及独立重放。未提示来源误差8/8改善，但双向来源与Jerk保留0/4，完整Native不可评价；不替换正式生成器。412回归通过、8依赖跳过，专项15通过，14段服务器视频；训练0、整曲0。

新增 [R23固定引导与连续表示报告](reports/R23固定引导与连续全身表示实验报告.md)：G0/G2 CFG对照均0/4、Native0/64。五Train正例的15解码条件保持保真并通过Native，曝光DEV短边界仍失败。144数组包／88物理报告独立一致，9梯度pre读出。提供18段本地观看对照（outputs/local_r23_01/videos_02/index.html），原媒体失败回执保留。

新增 [R22分支去噪训练与Lodge局部基线实验报告](reports/R22分支去噪训练与Lodge局部基线实验报告.md)及[封存回执](reports/r22_release/r22_receipt.json)：固定R21 B3起点，G0/G1/G2各1,200步，分支拟合改善未转化为固定引导采样收益，两项研究对照均0/4、Native0/72。118数组包、9对实际更新、72条最终轨迹及144份Native/FK数组独立一致；397回归通过、8依赖跳过，EDGE专项18通过。Lodge官方Local实际执行六条轨迹并重放，完整Native24因关节合同不完整保持NOT_EVALUABLE。补充审查发现三条来源硬Root冲突及一条表示域内Native可行见证，不能用来源矛盾解释全部生成失败。提供24段服务器视频；原始归档回传待具体载荷授权，新整曲0。研究与正式路径分开，见[预登记](docs/R22_PROTOCOL.md)。

新增 [R21联合上下文边界与支撑实验报告](reports/R21联合上下文边界与支撑实验报告.md)及[封存回执](reports/r21_release/r21_receipt.json)：12个Train音乐组24段真实225帧动作、4个曝光DEV组，split/joint×混合/分区目标各1,200步，固定B3起点Q/M/F各600步。六项研究对比均未达3/4来源门，Native0/124；M的Train采样收益未迁移至DEV，F未形成稳定足部协调。6,600更新、219数组包、21对实际更新读出、150次采样/缓存重放、248份Native/FK数组及124份物理报告独立一致；379回归通过、8依赖跳过，EDGE专项16通过。32段固定世界视角视频，保留原失败与恢复回执，新整曲0；研究实现与正式路径分离，见[预登记协议](docs/R21_PROTOCOL.md)。

新增 [R20双端联合视野与采样时序实验报告](reports/R20双端联合视野与采样时序实验报告.md)及[封存回执](reports/r20_release/r20_receipt.json)：同初始化/240步比较双150视图与真实225联合前向；直接双端通路成立，但联合结构未满足预登记保留条件。选定共同状态后A0/A1/A2各240步，A3因重建保留门失败未训练。两段Train、两段曝光DEV、双种子最终Native0/40；全轮1200更新，70次评价采样及8条缓存独立重放，90数组包、136份Native/FK数组、68份物理报告、15对实际更新读出一致。363回归通过、8依赖跳过，EDGE专项9通过。核验器路径错误已修复，原失败回执及科学输出保留；无新整曲。研究代码独立于正式路径，见[调用图](docs/R20_CALL_PATHS.md)。

新增 [R19重建优先目标校准与实际更新报告](reports/R19重建优先目标校准与实际更新报告.md)及[封存回执](reports/r19_receipt.json)：两段Train、单任务INBETWEEN，N0/N1/N2及按门准入U各240步。N1基础可学2/2，未知误差降到初始6.8%/4.8%；N2保住重建并改善部分实际采样指标，但内部Jerk仍高、Native0/12、独有内容增量未成立。16对实际更新、12次采样独立复验；354回归通过、8依赖跳过，专项7通过，提供2段并排机制视频。无新整曲。

新增 [R18任务匹配去噪与独有内容训练报告](reports/R18任务匹配去噪与独有内容训练报告.md)及[封存回执](reports/r18_receipt.json)：显式独有区掩码、内容/时钟/阶段干预，动作层5040279参数与状态头97728参数两组各训练120步。实际仅2段Train；Tiny门未通过，未扩展八组人口。首尾补全有局部改善，独有内容未稳定胜过同预算对照、Native0/22。22次采样独立重放、26数组包及44份Native/FK数组一致；347回归通过、8依赖跳过，专项10通过，提供6段Train机制视频及只读目标梯度诊断。新整曲0。

新增 [R17全局时钟双窗口训练与Mask归因报告](reports/R17全局时钟双窗口训练与Mask归因报告.md)及[封存回执](reports/r17_receipt.json)：真实225帧全局条件、硬软Mask因子及同参数量跨窗口/窗口内状态头各训练240步。16对硬软实验raw轨迹及未覆盖输出完全一致；跨窗口机制0/4、Native0/36，未启动ChangE接入或三首整曲。76次采样独立重放、91数组包及360份Native/FK数组一致，337回归通过、8依赖跳过，EDGE专项8通过；提供12段固定世界视角并排视频。

新增 [R16连续事件生成与真实窗口对照报告](reports/R16连续事件生成与真实窗口对照报告.md)及[封存回执](reports/r16_receipt.json)：完成4组真实225帧连续AIST数据、4份E1-W全局条件合同及24次C0/C1采样。C1来源响应4/4、上下文响应3/4，中心Jerk均值降低约86.4%，但新外接缝全失败、整体门0/4、Native0/24；C2训练与新整曲未启动。24次完整独立重采样、96份Native/FK数组与32个数组包一致，329回归通过、8依赖跳过，EDGE专项9通过；提供4段固定世界视角Source/C0/C1视频。视频属于AIST机制诊断，不是敦煌舞风格成果。

新增 [R15联合RootPose容量与物理编辑域报告](reports/R15联合RootPose容量与物理编辑域报告.md)及[封存回执](reports/r15_receipt.json)：12原始窗中6个编辑域外Jerk严格排除，其余UNKNOWN。固定8个Train音乐组16状态，J0–J4同总输入预算、256有限候选全部验收；J3受保护3/16、组资格0/8，Native0/256，不启动反馈分支训练或整曲。544个Native/FK数组、10个数组包及选择记录独立复验一致，320回归通过、8依赖跳过，EDGE专项13通过；路径预检失败回执原样保留。

新增 [R14边界状态表示与冻结条件容量报告](reports/R14边界状态表示与冻结条件容量报告.md)及[封存回执](reports/r14_receipt.json)：冻结400状态责任审计，两个编辑域外Jerk模板构成必要排除，其余UNKNOWN。五组同维解析头对照中A2在8个头持出音乐组Root误差降低6.52%；冻结输入引导仅2/16状态保护全部左右接缝，Native0/16，未准入人口训练或整曲。154关键数组、8条冻结基线轨迹、64份物理读出独立复验一致；313主回归通过、8依赖跳过，EDGE专项6通过。研究代码独立于R13和旧R9。

新增 [R13条件去噪与双侧Bridge实验报告](reports/R13条件去噪与双侧Bridge实验报告.md)及[封存回执](reports/r13_receipt.json)：直接观测/Mask输入并适配5040279个动作参数，四窗240步机制4/4通过，重新初始化的人口400步与自身策略缓存完成。四个适配持出Train组的Root接缝均值0.318→0.212m，但有逐接缝退化、Native0/4；不是新盲测或敦煌风格资格。完整采样/缓存/Native独立重放，主回归307通过、8依赖跳过，EDGE专项4通过；官方DanceFusion数据链不完整，仅审计，三首新整曲0。

新增 [R12显式动作Bridge与双侧状态实验报告](reports/R12显式动作Bridge与双侧状态实验报告.md)及[封存回执](reports/r12_receipt.json)：Train 内匹配幅度通路审查选用 Decoder 条件入口，311808 参数的双侧注意力/逐位置 Bridge 各训练400步并重放。B3接缝Root步长0.359m，弱于冻结C2的0.307m；来源及双侧响应、Native均未合格。72次缓存/生成执行及关键数组独立复验一致，303主回归通过、8依赖跳过，EDGE专项7通过；三首新整曲0，正式路径与冻结基线不变。

新增 [R11状态衔接与双侧条件审计报告](reports/R11状态衔接与双侧条件审计报告.md)及[封存回执](reports/r11_receipt.json)：8例几何Oracle通过，定位到原始x0与真实边界不兼容，Root接缝步长在硬覆盖时由约0.024升至0.307米，后续转换/回接几乎不增偏差。左右软条件正确/错配分别仅1/4胜，未启动新训练、Root联合修正或整曲。16次冻结C2重现/重放、48动作数组、16迹线数组及16 Oracle数组一致。新增本地逐帧检查器，浏览器自动预览受file://策略限制，仅完成脚本/数据/链接检查；正式路径和R10均冻结。

新增 [R10条件依赖与Root足部协调报告](reports/R10条件依赖与Root足部协调实验报告.md)及[封存回执](reports/r10_receipt.json)：定位旧B2空观察/音乐残差与Root坐标接口问题；完成内容差分、SO(3)、真实t/CFG一致性和一次Train单位修正，三组各200步。正确上下文仍未稳定优于错配，固定Train探针中Root辅助有效梯度弱；Root容量局部53/96但Native0/96，连续参考0/24，真实单窗各组仍未通过物理资格。40次首次+40次重放，320份神经数组及194份容量数组一致，16条世界坐标视频；三首新整曲0，研究代码与默认生成分离。

新增 [R9来源解耦与边界条件学习报告](reports/R9来源解耦与边界条件学习实验报告.md)及[封存回执](reports/r9_receipt.json)：相对Root使转向误差0.396→0.174rad，但下肢局部旋转冻结仍不能保护世界足端，来源Native各组0/8。48个C²容量提案研究门0/48；真实DataV3预检获得1833/273窗，已训练341888参数双侧Mask Adapter300步，Val生成仅有限改善，原上下文ROI仍0/4。544份关键数组重放一致；完整回归288通过、4依赖跳过，B2的SO(3)梯度专项通过。新路径均研究独立，三首新整曲0，默认生成与Native不变。

新增 [R8完整事件与双侧窗口实验报告](reports/R8完整事件与双侧窗口实验报告.md)及[封存回执](reports/r8_receipt.json)：完整ChangE Core、E1-W与阶段逐步条件使来源角误差0.590→0.168rad，但S3来源门2/16、Native0/16。四例真实双侧Inpainting中心Jerk改善却引入更大新接缝，连接门0/4。72单事件与8局部生成均重新采样核验，432单事件数组一致；新整曲采样0、训练0，研究与基线分离，正式路径不变。

新增 [R7来源辨识实验报告](reports/R7来源辨识实验报告.md)：固定曝光Train人口中，A2两人口均20/24；零剂量保留原分母，未取得完整来源控制资格。144次投影完整重放及1,704数组一致，独立核验通过。[整曲开发协议](docs/exposed_full_song_r7_v1.md)固定dunhuangwu2/3/4为完整音乐输入，区分全局事件槽与EDGE重叠窗口；不因缺少人工标签阻断客观实验。

新增 [R7整曲开发与物理诊断报告](reports/R7整曲敦煌音乐开发与物理诊断报告.md)及[机器汇总](reports/exposed_full_song_summary.json)：三首完整音乐为3601/3598/3601帧、48/47/48窗，完成18份全曲规划、95个新Jukebox窗口、10条完整动作和逐条重新生成复验。273项回归通过、1项依赖跳过；60动作数组一致，10视频全帧核验通过。原Native0/10；窗口边界附近的Root/Jerk异常需继续处理，学习式来源演绎与正式检索仍未获准。

上一轮事件机制实验共40次150帧执行、32个不同动作哈希，Native0/40；两种融合效果近似。对1,468个Train目标的只读Native审计有40窗通过，未据此过滤数据或重选权重。详见[事件机制与跨来源训练报告](reports/事件机制与跨来源训练报告.md)、[机器汇总](reports/event_mechanism_summary.json)及[运行合同](docs/event_mechanism.md)。

新增[时钟约束对齐与Root联合条件实验报告](reports/时钟约束对齐与Root联合条件实验报告.md)及[机器汇总](reports/clock_root_summary.json)：实现带NULL的时钟OT/位置对比、37D来源Root–Contact软条件和责任诊断，完成A/B/C固定预算训练、Root掩码消融、六例150帧生成与独立复核，服务器128项测试通过。时间辨别未建立稳定优势，Root条件收益仍未成立，Native0/6；新路径保持Shadow研究模式，默认检索与31项门限不变。

新增[Native物理评价可靠性报告](reports/Native物理评价可靠性报告.md)与[机器汇总](reports/native_reliability_summary.json)：本轮只读重放1,468个目标和42次历史生成执行，并完成地面/骨架/噪声及23个合成测试；服务器117项回归通过，独立复核通过。旧31项限制与判决不变。Native-Calib-v2仍为未获准的Shadow，真实静态接触、误拒率与专家一致性因缺少独立标注保持未知；本轮没有训练或新采样。

新增 [Data Split V3 / ChangE-E1 离线协议](docs/data_event_v3.md)及[服务器实验报告](reports/数据划分V3与ChangE事件E1实验报告.md)：严格官方曲目划分48/6/6、ChangE源留出8折索引，499个事件峰覆盖288→398，但合法候选对11298→8640。完整回归181通过+1依赖跳过，EDGE补充18通过，全新进程重解码与复验通过。FineDance音频203条仍为LFS指针；零模型训练/生成，旧Catalog、权重及Native门限保持冻结，新协议未获训练或生成资格。

新增 [E1-W 固定原语时间映射协议](docs/primitive_warp_v1.md)及[服务器实验报告](reports/E1W时间映射可行域实验报告.md)：固定E1中心后，合法候选8640→12969、可用事件388→448，保留所有旧可行对；新旧中心的同求解器对照为13188/12969。196项主回归通过+1依赖跳过，EDGE补充33通过，原始来源新进程复验与每进程48902组整数域穷举通过。零训练/生成；默认Planner不变，A2/G2/G3资格不因此改变。

新增 [A2新划分资格与G2来源归因报告](reports/A2新划分资格与G2来源归因实验报告.md)：1,037来源/54音频的独立原始数据重建得到9,199窗，Train合格106、Val仍0，未启动新训练。冻结G2的32次生成及新进程复验完成，来源跟随0/4、Native0/32；补齐自由轨迹观察与提供/隐藏/关键帧诊断。回归202通过+1依赖跳过，EDGE补充19通过。正式路径与门限不变。

## 来源与切分事件查看

[ChangE 来源／切分事件库](outputs/change_catalog_view_01/index.html)显示8条连续来源、499个实际事件，左右同帧对照、来源切分时间轴、阶段候选、逐帧／倍速和原速片段循环。仅展示原身体旋转的22关节局部姿态，不含生成动作或E1-W重定时；世界Root未知。离线观看文件在忽略目录中。构建入口为 scripts/build_change_catalog_view.py，显式传入原Catalog、E1 sidecar、经校核的common22 rig和全新输出目录；逐项检查来源SHA、499个半开区间和骨长。

## 正式入口

公开 profile 与运行机器资产分离。从 configs/assets.example.json 填入真实绝对路径、SHA256、固定依赖版本，保存到忽略目录 inputs/runtime_assets.json，占位值不能直接运行。目录源清单沿用 source_records/events，raw SMPL 文件名与哈希必须匹配。EDGE / Native 使用各自环境；prepare 导出原权重 Normalizer 后解码事件目录。

```bash
python -m pip install -e '.[test]'
python scripts/prepare.py --assets inputs/runtime_assets.json --out outputs/new_prepare
python scripts/generate.py --config configs/edge_baseline.json --assets outputs/new_prepare/assets.json --out outputs/new_baseline
python scripts/generate.py --config configs/dunhuang.json --assets outputs/new_prepare/assets.json --out outputs/new_plan
python scripts/evaluate.py --run outputs/new_plan --compare outputs/new_baseline
```

省略 --frames 生成整曲，可指定 --frames 150 或 450。--plan /path/event_plan.json 可重放显式计划，引用、时钟和映射必须验证。换音乐先运行 scripts/extract_music.py，把真实 layer66/4800D 缓存和绑定音频的提取收据填入资产清单；--audio 不会让旧特征自动对应新音频。所有输出使用新目录。

代码 push、服务器 pull 后，在项目目录通过 tmux 运行长任务：
```bash
tmux new-session -d -s ddg-generation 'python scripts/generate.py --config configs/dunhuang.json --assets outputs/new_prepare/assets.json --out outputs/new_plan > outputs/new_plan.log 2>&1; echo $? > outputs/new_plan.exit'
```

## 模块与资格

planning/ 实现事件目录、节拍/能量时间线、单调时间映射及 Top5/beam3 全局规划。约 3 秒节拍吸附切分不是人工乐句；计划覆盖和实际控制比例分开报告，当前仅 25 帧上身原语使用 Temporal 控制。

alignment/ 是 50D 音乐 / 74D 动作双编码器与局部同步头。heuristic 默认启用；learned_shadow 需要显式绑定全局/同步 checkpoint、对应编码器和查询收据，只记录两级评分；learned_qualified 还需验证资格。历史 Test 已暴露，只作为开发回归。同歌检索与同歌时段检索分开评估。

generation/adapter.py 提供零初始化、掩码条件 Adapter，冻结 EDGE 主干，复用原 CFG。research/legacy/paired_training.py 保留历史真实 151D/4800D 配对、来源分离与支持域预检；scripts/train.py 为历史重放入口，不消费新的 Split V3，也不授予新协议资格。尚无获完整生成资格的训练权重。以下历史命令需 EDGE 环境：
```bash
python scripts/train.py --assets outputs/new_prepare/assets.json --out outputs/new_motion_training --task motion --steps 300
python scripts/train.py --assets inputs/paired_assets.json --out outputs/new_paired_training --task paired --steps 300
```
配对资产额外给 paired_generation_manifest 哈希引用。清单 rows 每行给 role(train/val/test)、source_group、start_frame、pairing_evidence=real_aligned_audio_motion、fps=30、frames=150，以及 SHA256 绑定的 motion[150,151]、jukebox[150,4800] 文件引用。训练本身不授予 music_qualified。独立 R17 已完成全局 225 帧损失下的双视图状态头最小训练，未获得生成资格；正式双向连续生成及新 PFG/SQP 尚未完成。

## 审计与复核

pipeline.py 是正式生成入口。physical.py 保留固定版本 Native 完整绝对阈值，计划阶段始终 UNKNOWN。每次输出保存音频/特征/权重/代码哈希、真实 FK、完整指标、日志和状态。文化文件名标签只是来源粗类别，不是逐事件真值。

```bash
python -m pytest -q
python scripts/verify_event_plan_report.py
```
第二条只读复核保存实验，不训练、不重采样。生成完成、Native 通过、语义风格和人工自然性分别判断。权重发布来源未独立核实，不宣称严格复现官方论文效果。

## 历史研究入口

以下保留旧实验及证据，当前正式流程不调用历史 SDMR 生成编排。research/legacy/sdmr_select.py 仅为旧脚本兼容。

## B2 时序可执行性实验

同采样器对照使用 `custom_none`、`anchored`、`temporal` 三种 mode，默认重叠策略为 `mean`。`temporal` 还需要 `trajectory_radius`、`trajectory_strength` 以及每个锚点的哈希绑定源片段和中心帧。当前实验半径为 12 帧、强度为 0.55。原 `baseline` 始终调用官方 long DDIM。

在已具备首轮实验产物的服务器上，可用全新输出目录运行以下命令。首个脚本先只读复核保存结果，然后从 5 秒开始执行匹配对照；后两个是独立研究案例。
```bash
PY=/home/disk/lsm/conda_envs/dance-rag/bin/python
$PY scripts/b2.py outputs/new_b2
$PY scripts/b2_music.py outputs/new_b2 outputs/new_b2_phrase
$PY scripts/b2_physical.py outputs/new_b2_support
```

[B2 实验报告](reports/B2实验报告.md) 和 [机器指标](reports/b2.json) 记录各时长的完整比较、支持域与物理审核。

## B3 物理责任与原语准备

[B3实验报告](reports/B3实验报告.md) 和 [机器指标](reports/b3.json) 包含B2整曲的来源窗口FK归因、冻结支撑人口Root-only条件能力、六个几何方向及一次受保护续步、106个中心的事件归并和三锚点联合候选。原Native仍是完整验收依据。复现入口为 scripts/b3.py、scripts/b3_primitives.py、scripts/b3_verify.py。

## SDMR 学习式检索

alignment/ 提供50D/74D双编码器、独立同步训练、文件名粗标签、专家偏好弃权接口和候选评分。scripts/sdmr.py先训练完整框架，scripts/sdmr_ablate.py仅在完整框架结束后重训一次去声学32D的消融。生成入口按检查点和资格收据决定启用或回退。见[实验报告](reports/SDMR实验报告.md)和[机器指标](reports/sdmr.json)。本轮真实评价未通过同步资格。

## V0.6 研究基线：连续事件软条件与真实训练探针

PR #2 已通过 merge commit 合并，研究基线标签为 v0.6-research；main 默认生成路径仍是 V0.5。soft_reference.py 根据真实 Event Plan 的单调源相位做 SO(3) 测地插值，支持域外关节帧明确未知；重叠窗口条件一致。连续参考只进入冻结 EDGE 的条件 Adapter，不后贴源动作。

本轮服务器完成 300 步48D探针、八例150/225帧生成、两次独立数值重放。42项完整回归通过；原官方/V0.5动作哈希不变；同噪声零 Adapter 与无 Adapter 完全相同。探针验证MSE降低约17.28%，但八例Native全未通过，动作幅度也下降，不能宣称音乐或整体质量提升。详见[V0.6实验报告](reports/V06实验报告.md)及[机器收据](reports/v06.json)。

~~~bash
# 真实条件支持域预检
python scripts/inspect_event_reference.py --plan outputs/event_plan_01/dunhuang_150/event_plan.json --scaler outputs/new_prepare/edge_contract/scaler.npz --out outputs/v06_reference_new
# CPU资产预检，不初始化CUDA
python scripts/train.py --assets outputs/new_prepare/assets.json --task upperbody_probe --preflight-only --out outputs/v06_preflight_new
# 一个固定探针 + 匹配噪声对照；服务器长任务需通过tmux运行
python scripts/v06_experiment.py --assets outputs/event_plan_01/prepare/assets.json --prior-root outputs/event_plan_01 --out outputs/v06_new --steps 300
python scripts/verify_v06.py --root outputs/v06_new --replay
~~~

所有输出目录必须为新目录，路径按实际资产替换。记录实际运行提交，训练/生成版本与随后报告提交分开。研究权重的 music_qualified / full_motion_qualified 始终为 false，正式推理默认拒绝，实验才显式给 research_allow_motion_only_adapter。

其他103通道仅是探针占位，不能当完整真值。完整151D目标与真实配对Jukebox仍缺失；学习式音乐演绎V1、双侧生成及SQP尚未完成，未升级450帧/整曲质量验收。

## V0.7：条件合同与底层诊断

本阶段不重复训练探针。共享 planning/time_map.py 将分段源→音乐时间映射同时用于生成条件和检索；74D节奏特征在真实映射后的SO(3)轨迹上重新求导，旧的均匀缩放同步资格不能自动升级到新执行合同。候选身份绑定来源、时长、时间映射、演法占位和评分模型哈希。当前演法仍是identity，不宣称已学习力度或停顿。

generation/conditions.py 分开来源参考、观测掩码、相位及目标监督，dense_upper8 / completion_upper8 显式命名。匹配的150帧训练/推理条件由同一函数产生；全局窗口原点用于保证重叠掩码一致。历史V0.6 dense推理与completion训练的差异单独入审计，研究权限不授予正式资格。

~~~bash
tmux new-session -d -s ddg-v07 'python scripts/v07_experiment.py --assets outputs/event_plan_01/prepare/assets.json --prior outputs/v06_01 --alignment-request outputs/event_plan_shadow_verified/request.json --out outputs/v07_new > outputs/v07_new.log 2>&1; echo $? > outputs/v07_new.exit'
~~~

输出新目录，保存逐通道支持域、坐标往返、逐层激活、注意力熵、随机VJP、来源/相位/音乐干预、匹配噪声单窗生成及原Native结果。VJP只衡量局部网络敏感性，不是完整物理可控性证明。未来PFG-1D、Music–Event Cross-Attention、双侧两遍生成与SQP仍需各自数据和资格，不由诊断成功自动启用。

## V0.7与论文评价的真实结果

本轮V0.7主实验不训练新模型：568窗支持域、两噪声seed/三时间步的层级响应、八例150帧生成均完成。右踝通道60在全部568窗越域，完整151D支持仍为0；来源替换对现有探针的作用很弱，不能仅凭局部MSE改善宣称事件演绎成功。八例Native仍0/8。详见[V0.7报告](reports/V07实验报告.md)和[机器数据](reports/v07.json)。

新增独立evaluation_benchmarks/：EDGE PFC、网格Penetrate/Float/Skate、敦煌接触状态和事件/窗口边界诊断。17个保存结果、15个唯一动作已只读评价，PFC逐例与冻结EDGE官方脚本一致；统一SMPL网格与smplx参考最大差约2.4e-7m。

PhysDiff仅为附录定义复现：公共地面来自冻结基线网格，Skate接触高度50mm为本项目显式协议选择；官方评价源码未获得，统一neutral网格骨架也与旧Native骨架不同。不能直接与论文表格排名比较。无连续接触样本时Skate/Phys-Err未定义；无独立标签时静止/有意滑动保持UNKNOWN，不自动豁免高速滑动。旧Native判决和阈值不变。

~~~bash
python scripts/evaluate_benchmarks.py --assets inputs/benchmark_assets.json --manifest configs/benchmark_runs.example.json --out outputs/paper_benchmarks_new
python scripts/verify_benchmarks.py --root outputs/paper_benchmarks_01
python scripts/verify_v07.py --root outputs/v07_01
~~~

第二、三条只读复核，不更新动作或优化权重。资产模板为configs/benchmark.example.json。详见[论文指标报告](reports/论文物理指标复核.md)、[完整指标](reports/paper_benchmarks.json)与[CSV](reports/paper_benchmarks.csv)。


## V0.8 Native 反馈闭环

新增只读反馈层与有界控制器：Native 精确回验 → 指标/时间窗/事件关联 → 条件或种子重生成 → 单槽局部重规划。复用原 beam planner，固定未改事件槽。候选按完整上下文缓存，执行错误保持 UNKNOWN，大范围物理失败默认停止；明确的小额诊断模式最多一次重规划。Native 阈值不变，150/225 帧结果不升级成整曲验收。

入口及阶段边界见 [闭环说明](docs/v08_closed_loop.md)，实际结果见 [V0.8 实验报告](reports/V08实验报告.md) 与 [机器核验数据](reports/v08.json)。完整 151D 训练、音乐配对演绎、双侧上下文和 SQP 尚未由这一控制回路解决。

~~~bash
python scripts/generate.py --config configs/closed_loop.example.json --assets outputs/event_plan_01/prepare/assets.json --out outputs/loop_new
python scripts/generate.py --config configs/closed_loop.example.json --assets outputs/event_plan_01/prepare/assets.json --out outputs/loop_new --resume
python scripts/v08_experiment.py --assets outputs/event_plan_01/prepare/assets.json --out outputs/v08_new
python scripts/verify_v08.py --root outputs/v08_new --out outputs/v08_new/readout.json
~~~

首次输出目录必须不存在。完成后的 resume 重新核对 Native 合约、计划和缓存，不启动新生成；中断半成品需使用新目录明确重跑。


## 历史配对训练资产预检（b880337）

新增 AIST++/FineDance 原始资产校验、独立 151D 转换、observed/derived/supervision 掩码、冻结 Scaler 支持域统计、真实 Jukebox 候选提取及严格 paired_generation_manifest_v2。原始数据保存在忽略的 assets/datasets，不随源码发布；保留历史角色和曝光状态。

服务器实际完成 1,566 条原始动作哈希核对、1,564 条解码和 15,665 个 150 帧窗口审计。AIST++ 4,964/5,040 个窗符合当前表示合同；FineDance 0/10,625，且缺少末端手部 22/23 监督。两个不同 Train 关联组的真实 Layer66 Jukebox 提取完成。配对训练因时钟/来源证据不足明确 BLOCKED，训练0步，无新模型质量结论。

详见 [配对资产预检报告](reports/配对资产预检报告.md)、[运行与合同](docs/paired_assets.md)、[支持域数据](reports/paired_asset_support.json)、[独立核验](reports/paired_asset_verification.json)。旧无观测/时钟收据的 paired 清单拒绝使用；单次生成和 V0.8 闭环保持原入口。


## AIST++ 基础舞配对与首轮训练

已利用官方精修视频、CSV和基础编舞工作簿核验1,363条来源：1,166条基础舞通过身份/音频时钟校准；197条高级舞共享编舞身份未明确，仍排除。完整配对清单为2,418个窗（Train1480/Val481/Test457），预检通过后提取127份去重的真实Layer66 Jukebox，再完成冻结EDGE的300步151D Adapter补全训练。

服务器96项测试通过，新进程验证MSE与保存结果完全一致、零权重更新。Val补全MSE较零Adapter约改善1.97%，移除来源参考几乎不变；未证明音乐演绎或Native质量，默认生成入口不变。角色隔离只覆盖已核验官方关联键，不宣称语义独立或新盲测。

详见 [实际配对训练报告](reports/AIST++配对训练报告.md) 和 [运行说明](docs/aist_pairing_pipeline.md)。完整清单与原始回执保留在本地/服务器的忽略输出目录，GitHub仅发布汇总。

## 事件可辨识性研究（Shadow）

新增自动事件、来源可追溯弱标签、严格部分路径、合成插入/删除/合法变速及独立重放。固定预算与指标边界见 [研究协议](docs/事件可辨识性研究协议.md)。历史 A/B/C、Native31、默认生成路径保持冻结。真实事件评价仍需要独立审核。

本轮服务器六组固定预算实验及独立重放已完成；结果与边界见 [事件可辨识性实验报告](reports/event_identifiability_report.md)。合成身份/NULL恢复改善，真实时钟内容判别未超过时间先验，弱对应未显示独立收益，继续保持 Shadow。

## 局部表征可辨识性 R1 / 条件内容 R2（Shadow）

在01c0515上冻结编码器、旧头与默认生成，新增统一网格的信息可访问性探针、Token谱/Hubness、固定路径内容重评分及五类损失梯度审计。严格控制候选人口足够时才执行等参数R2；不足不放宽。见 [预注册协议](docs/局部表征可辨识性研究协议.md)。

R1服务器审计与独立重放已完成，严格条件候选Train/Val均为0，R2按门限未启动。详见 [局部表征可辨识性报告](reports/local_identifiability_report.md) 与脱敏回执；没有新检索模型或生成质量提升声明。


## Dense A1 / G1 research audit (2026-11-12)

See [experiment report](reports/dense_condition_20261112.md) and [registered protocol](docs/DENSE_CONDITION_PROTOCOL.md). Real source coverage: 936 recordings / 8,510 windows; strict Train 18 queries / 10 recordings, Val 0; training was not started. Frozen EDGE audit: 14 real generations plus 14 exact recomputations, Native 0/14 and source-follow gate failed. Explicit research visibility passes the six matched unknown-storage and six bypass invariants; legacy remains the default. No bilateral generation or SQP qualification is claimed.


## G2 / A2 source-identifiability study (2026-11-12)

[Report](reports/source_identifiability_20261112.md) · [receipt](reports/source_id_03_receipt.json) · [code boundaries](research/README.md) · [registered protocol](docs/G2_A2_PROTOCOL.md).

New studies are isolated in research/source_identifiability; historical generation trainers moved to research/legacy, and four duplicate script wrappers were removed. Three matched 357,424-parameter Adapters completed 400 updates each. All model arms failed source-following qualification (0/4); Native 0/52. All44 diffusion outputs and8 nonlearned controls were independently recomputed exactly. New A2 population: Train69 queries/63 recordings/6 groups, Val0; no A2 training. Final regression164 passed +1 dependency skip; supplemental EDGE13 passed. Default generation and Native remain unchanged, and observed-trained checkpoints require a matching research runtime.


## A2-V3 candidate support / G2-M mechanism study (2026-11-13)

See [report](reports/A2V3候选支持域与G2M机制对照报告.md), [registered protocol](docs/source_necessity_v1.md), and [sealed receipt](reports/source_necessity_receipt.json). New separately registered A2 tasks have 1,023/885 Val queries; the historical strict task remains unchanged. Five matched Adapters completed 400 updates each. All source-following arms remain 0/4; Native 11/52 overall (bypass 1/4, learned arms 2/8 each). All52 generations and156 motion/FK arrays were independently replayed exactly with zero optimizer calls. Production retrieval/generation and G3 remain unqualified.


## Real source CF and fixed-candidate content tasks (2026-11-13)

[Report](reports/来源效应与G2CF及A2内容对照报告.md) · [protocol](docs/source_tasks_v1.md) · [receipt](reports/source_tasks_receipt.json). Saved-source response was independently measured; real paired high-noise source qualification and four matched CF arms completed. All CF source-follow gates remain 0/4 (Native9/44). Four equal-parameter15/30/dual/shuffle content models were trained on the same frozen offline A2 candidates and independently replayed. Default retrieval/generation, Native and G3 remain unqualified. The pipeline now rejects legacy anchor selection which a whole-song Planner would overwrite.


## Frozen capacity / local path audit (2026-11-13)

[Report](reports/冻结来源控制容量与局部路径审计报告.md) · [protocol](docs/source_capacity_v1.md) · [receipt](reports/source_capacity_receipt.json). Zero optimizer updates. All72 derivative probes passed numeric checks; no site passed directional control qualification.32 frozen controls and96 motion/FK arrays were exactly regenerated in an independent process. All4 frozen content encoders were compared on fixed candidates using same-frame, banded DTW, partial NULL and content controls, then independently replayed. Group intervals do not support a reliable local-path gain. No production or G3/R4 promotion.

## Reference-only guidance / event information audit (2026-11-13)

[Report](reports/参考动作可用引导与事件信息审计报告.md) · [protocol](docs/reference_guidance_v1.md) · [sealed receipt](reports/reference_guidance_receipt.json).
All96 reference-only direction derivatives passed numeric checks; protected all-case qualification failed, so no full-chain guidance or mapping training started.44 single-step Native readouts and132 arrays were independently replayed exactly. Four frozen encoders received60 Train-only diagnostic linear readouts; all scores/heads were replayed without refitting. Source registry exposes60 shared choreography identities despite valid known-song isolation. Proxy decodability does not qualify cross-modal event semantics, production generation or a new blind benchmark.

## Protected source and event/planner audits (2026-11-13)

[Report](reports/受保护来源控制与事件信息及规划集成报告.md) · [protocol](docs/protected_source_v1.md) · [receipt](reports/protected_source_receipt.json). Fixed small-subspace source projections pass all measured relative-motion protection checks and improve joint Train readouts beyond matched shrink controls, but full source qualification fails and Native acceptance is unchanged. Interval/pose audits fit no model weights and do not support new encoder training. E1-W is consumed only by a Shadow research path. All three stages have fresh-process replay; production remains unchanged.


## R6 source expression and independent review preparation (2026-11-13)

[Report](reports/来源动态控制与独立盲审准备报告.md) · [protocol](docs/source_expression_r6_v1.md) · [receipt](reports/source_expression_receipt.json). Five fixed protected objectives completed 240 local projections and a full fresh replay; no arm passed all-case source qualification, so no new training or full generation started. Three distinct real audio assets yielded nine verified Shadow plans. Thirty-five blind-review items are PREPARATION_ONLY with zero human records. Media-only continuation preserves the original failed WAV-byte receipt and SHA-sealed scientific stages. Production remains unchanged.


## Blind-viewer playback correction V2
[Report](reports/盲审骨架播放修正说明.md) · [receipt](reports/review_viewer_v2_receipt.json). Preserves original pose samples, restores a separate unverified fitted-Root view, uses elapsed/audio clocks and stops at the final frame. Viewer corrections do not establish motion-quality or contact qualification; original review artifacts remain frozen.


### R33 来源路由与相对动态监督

已完成单一 Lodge 主干上的冻结通用参考路由、ChangE 参考600步等预算对照及独立重放。Dyn改善训练琵琶的Core动态误差，额外来源构形和全窗物理仍未合格，整曲0。详见[实验报告](reports/R33来源路由与相对动态实验报告.md)及[公开统计](reports/r33_release/r33_summary.json)。

## R38：观测桥任务学习与有序检索（研究）

R37规划及默认EDGE路径保持冻结。R38用真实连续录制训练Context完成观测外端几何桥上的补间，并实际执行有序SDMR多正例训练。先交付可观看比较，再完成独立复验；构形有部分收益，动态与Root质量仍未通过，SDMR顺序置乱仍更高，因此不升级选路或整曲。

- 实际观看：`outputs/bridge_r38_01/viewer_final/index.html`（本机交付，不托管私有媒体）。
- [实验报告](reports/R38观测桥任务学习与有序检索实验报告.md)，[公开统计](reports/r38_release/r38_summary.json)。
- 隔离实现：`src/dunhuang/research/bridge_r38`；完整回归554 passed、8 skipped。

## R39：连续Root与端点约束RTS（研究）

固定R38权重，先交付四栏真实观看：Geometry／Raw／RootContinuous／ResidualRTS。连续米制Root残差修复边缘极值，约束RTS降低身体高频并保留多数构形收益；接触、完整物理和整曲资格仍未建立。原始相对Root导数已恢复，冻结SDMR的位置捷径诊断继续保持Shadow。

- 本机观看：`outputs/stabilize_r39_01/viewer_final02/index.html`（不在Git托管私有媒体）。
- [实验报告](reports/R39连续Root写回与端点约束RTS实验报告.md)、[公开统计](reports/r39_release/r39_summary.json)。
- 隔离实现：`src/dunhuang/research/stabilize_r39`；562 passed、8 skipped；没有新增神经训练，默认EDGE和R37计划不变。
