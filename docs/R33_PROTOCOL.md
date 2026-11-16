# R33 单主干条件接口、域隔离与来源配对动力学

基线7a207915daf33d6fc9839fc088055d27b3a39e86，保留正式EDGE/Planner/Native和R31/R32全部冻结代码、权重与失败记录。新代码control_r33独立，只加载一个Lodge与固定R31 CanonicalNormalizer/ActionIO。通用参考使用冻结R31 CanonicalIO参考，ChangE参考从均衡R32末态开始。按来源登记域为整个窗口选择，不学域分类器、不逐帧切换；这种结构性保留不是单一共享适配器跨域泛化。

条件接口显式区分source_valid、context_valid、target_valid和editable；未知上下文可用where(mask,value,0)编码，但未知目标不形成完整x0。Context为B×Nc×139规范模型值，逐通道Mask、窗口时间、左右、observed/generated/planned标签，必须共享窗口坐标。小MLP285→128→64及1/4/7层独立QKV/零初始化输出；单个hook中两条件都读原始hidden，开关clear_source/clear_context/clear_all互不混淆。只完成默认关闭、零残差与Mask/参数绑定资格，不将上下文训练混入R33动力学对照。音乐CFG两支读取相同source/context，unc只是无音乐。

预检实际重采样通用AIST四Train/四历史持出、两种子、正确/交换/错时钟/无参考，严格等于R31；ChangE三种参考严格等于R32。再验证真实边界和缺失Root/C4的planned-pose零上下文不改变结果。合计22调用/96轨迹，新进程全部重放，未训练。ActionIO forward显式绑定t，支持位置和关键词参数，不再靠args索引。通用参考、I/O、上下文与主干必须保持哈希和梯度缺席。

额外适配持出按既有目录身份锁定female_lotus_sequence和female_meditation_sequence。排除训练飞天/琵琶原始SHA及录制UID，分别选静态时长≥96的最小event_uid；不看任何生成结果、不自动换源。只做合法映射求解、SO(3)重采样与同一时间轴FK目标；源特征与动态目标都用重采样后的96帧，30Hz求导。若固定候选数学不合法则保留失败，不能挑效果好的替代。它们是R33未参与适配的已曝光目录来源，不称全新盲测、舞者独立或文化真值。Root/C4仍UNKNOWN。

动力学仅作用于Core内Root-relative身体误差E=(P-P_root)-P_source_relative，保留骨盆朝向贡献；按五部位和连续有效模板平衡，Huber(D1E/sv)+Huber(D2E/sa)，30Hz。D3/Jerk只评价。尺度在两条Train ChangE、两音乐、实际R32完整访问状态的999/519/119/19读出：各部位误差RMS跨四时刻取中位数，分别以.05m/s和.5m/s²作数值底线。lambda=.1×min_g median_t(||grad base||g)/median_t(||grad dyn||g)，g为encoder与injection，只读Train，所有量必须有限正值。源构形、坐标差分、局部几何、同状态first7/外侧NoRef原项保留；不引入未知Root/C4真值，不惩罚绝对运动趋近零。

校准除任务梯度的范数/夹角/饱和外，还从同一起点在四时刻各做一项原目标与一项动态目标的实际单步AdamW，共8步独立重置；保存pre/post。候选的平均身体及局部几何损失≤初始1.1倍，活动/幅度相对同状态初始.8–1.25，且更新/梯度有限，才准入600步主比较；不依靠标量总损失下降代替内容保留。所有探针权重随后丢弃，主训练回到同一R32末态；不通过则停止候选，不调持出。PCGrad不自动加入，也不追加新RootNet/频域/阶段网络。

主比较冻结R32 ChangE作基准；Extra和Dyn两臂从同一R32末态各600步、同AdamW .0002、同300配对/300部分观察、均衡16来源每条75次配对呈现、同50步时刻/噪声。AIST回放以显式change_replay训练候选分支，通用推理仍走冻结general；不能把推理路由保留称候选本身不遗忘。辅助状态用预检实际轨迹，300步时各自完整采样刷新一次，不是每步on-policy。只有Dyn在ChangE部分任务增加一个校准后的身体动态包，原Root/C4未知不构造q。

四代表时刻仅用于诊断与校准，主辅助任务覆盖全部50个采样时刻；可信AIST配对q保持原时间支持。实际更新pre/post与零优化器重放、全部最终完整采样独立复验。琵琶配对D2/D3与额外Jerk改善、内容不收缩、飞天收益保留、通用AIST和NoRef精确恢复共同报告；Core外/边界/Root单列，旧物理门不改，Native24仍NOT_EVALUABLE。

R33不跑音乐重定时训练、上下文衔接训练、生成历史短链或整曲；相应独立接口和Mask不是已学会能力。模型不自动晋升。代码先GitHub后tmux；保留所有源SHA、模型SHA、账本、梯度/更新、失败/恢复回执，临时SSH仅清理本轮实际文件。R33媒体另按本批载荷授权。
