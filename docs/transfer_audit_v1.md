# Data V3 A2 资格与冻结 G2 归因：独立研究协议

本轮固定4adee59为默认基线，不修改Planner、E1-W、采样、权重、Native门限。研究编排隔离于research/transfer_audit。先推源码至GitHub，再服务器精确拉取，在新目录启动tmux。不得看结果后放松候选门限或选择训练权重。

## A2：新角色和真实连续支持域

消费Data V3已封存的AIST_strict_unseen_song。先按登记SHA筛选新Train/Val，再读取原始动作/真实音频/时钟；新Test仅允许读取身份元数据，禁止特征和张量。历史Test被分入新Train/Val时明确记录historical_role，不宣称新盲测。

每个新Train/Val来源重新解码完整SMPL、先计算全录制SO(3)导数、独立提取动作事件；从完整原音频提取冻结50D观测及音频事件，使用哈希绑定的原librosa0.11.0/soxr环境，不补齐缺失缓存。不是新的Jukebox提取或151D生成目标。

窗口150帧、中心120帧；预注册stride15，并以同源同角色stride60子集作密度反事实。维持原A2五项严格匹配阈值、查询/录制/组数准入要求。候选负例同实际音频时刻、不同音乐/编舞/关联组，只表示真实来源关联的非目标，不表示艺术不适合。伪事件仍为未审核代理。

修正旧A2调用接口以允许注入任务校验器；旧默认validate_roles保留。新任务隔离音乐/音频SHA/录制/原始SHA及曲目组，允许同一编舞跨歌曲角色；不能套用旧“所有身份同时隔离”的闭包。

独立重算Train-only音乐50D与动作74D统计：每个完整资产只计一次，Val不拟合，未知通道均值0/标准差1。保存registry/population/statistics绑定；不替换旧Encoder或EDGE统计。新模型训练还要求作品别名、预训练曝光、任务授权及实际新checkpoint绑定，候选数量充分并不替代这些条件。当前V3这些资格未完成，登记审查/统计预算而不启动投影或Encoder训练；未来训练需另行固定任务及预算。

## G2：冻结模型的信息链归因

固定已封存source_id_03中g2_0/1/2/3权重、两个音乐案例、两种来源和种子20261112/20261113。全部为已曝光历史诊断案例，不能称新泛化验证。所有优化器构造器在进程中禁用。

1. 教师加噪t=100/500/900/999：同音乐、同x_t、A/B/NULL来源交换，测153D输入、128D Token、512D残差、八层Decoder、条件/无条件、CFG及裁剪后响应。
2. g2_2按原去噪MSE分量与source监督计算分组梯度范数与夹角，不执行优化器。检查任务监督是否能到达来源通道，不将非零梯度当作可生成证据。
3. 32次固定50步真实采样：每案例/种子运行bypass、g2_0 A/B、g2_2 A/B、硬关键帧A/B、NULL；同噪声，运行原Native，NULL必须等于bypass。已有28次对应生成与48次教师预测须与冻结旧张量精确相等。
4. 在bypass轨迹step0/24/49捕获真实x_t，再固定同状态交换来源并比较四权重，分开报告教师输入和自由轨迹的信息响应。
5. 报告旧训练t范围50–949与真实采样时刻的覆盖差异、残差/音乐范数、条件响应衰减、裁剪饱和、隐藏帧误差、双向偏好、活动度与原门槛。覆盖差异是机制假设，不提前宣称唯一原因。

不调增残差系数、不改CFG、不放宽旧0/4来源跟随合同。新增方向性诊断不授予G3资格。若未形成合格新训练人口和可辨识任务，不开展新Adapter训练。

## 执行与独立复验

    tmux new-session -d -s ddg-transfer-audit 'bash scripts/run_transfer_audit.sh outputs/transfer_audit_01 outputs/transfer_audit_01_receipt > outputs/transfer_audit_01.log 2>&1'

主进程与独立新进程分别从原音频/SMPL重建A2、重算Train统计、候选、旧模型响应/梯度及全部32次生成/Native；逐值比较科学数组、指标和原始输出。重复拟合只限独立重算Train统计，优化器调用必须为0。任务全部完成后再核查exit_status=0并执行：

    python scripts/export_transfer_audit.py --out outputs/transfer_audit_01 --receipt outputs/transfer_audit_01_receipt --seal

原始身份/资产/梯度与层张量留服务器，公开仅聚合统计、哈希与回执。失败目录不可覆盖；只清理本轮创建的临时SSH辅助文件。FineDance、E2人工边界、G3与整曲/SQP不被本轮审查自动准入。
