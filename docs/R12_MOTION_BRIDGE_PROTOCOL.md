# R12：显式动作状态 Bridge（执行前登记）

基准 75868e812e31adf2173e4f62c564c56c4c13d219。R10 C2 与 R11 证据冻结。
正式生成路径、Native 阈值、EDGE 权重不变。研究代码仅在
src/dunhuang/research/motion_bridge，通过显式研究入口运行。

参考：[CondMDI](https://setarehc.github.io/CondMDI/) 的观测动作/掩码输入，
[BADM](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Bidirectional_Autoregessive_Diffusion_Model_for_Dance_Generation_CVPR_2024_paper.pdf)
的双向上下文思想。这里是适配 EDGE 的新实验实现，不是两篇论文的等价复现；
不加载 HumanML3D 权重，不宣称已学会敦煌舞文化语义。

## 人口与顺序

使用已核验的 AIST++ 1833 Train 窗、48 音乐组。按登记种子和组身份 SHA 排序：
4 组通路构造、4 组通路持出、4 组训练监测、剩余 36 组拟合。
所有特征 RMS 仅由最后 36 组拟合。旧 Val、历史 R8 案例均已曝光，
只能在结构、损失、预算和最终 checkpoint 冻结之后做读出，不能称新盲测。
FineDance、ChangE 不构造不存在的配对 Root/contact 真值。

## P0：固定幅度局部可控性

三个输出均为 150×512：音乐 cond_encoder、动作 input_projection、
首个 Decoder 的 cross-attention 输出（FiLM 前）。只向 CFG 条件支注入；
无条件支、CFG=2、clamp、x_t、目标、支持域相同。

每例四个预登记 t，方向由真实 8+8 已观测 Root/SO3 构造：
正确观测梯度、同等幅度错配观测梯度、随机方向。全张量 RMS 固定为 0.1。
不使用未知区真值构造方向。未知区误差仅作读出。
这是 feature-space 一步诊断，不是模型训练或生成质量证明。

4 个持出 Train 组中至少 3 例平均目标下降、至少 3 例正确方向同时优于错配与随机，
可作为新机制最小训练准入。优先 motion，其次 decoder；music 是旧入口对照。
不从曝光 Val 选择入口；若均不通过，本轮停止新模型训练并保留失败。

## 新模型与最小对照

B3：当前噪声、冻结音乐特征、扩散步、时间位置作为 Query；
真实 Root/24关节 SO3、观测掩码及有效连续帧速度，作为左右已知状态。
双侧交叉注意力产生 Motion Token 残差。
B2：参数量相同，替换为四层逐位置模块，保留有序左右池化。
两个模型从头初始化、同样最终 400 更新，不延续 C2。
共有参数同种子同初值，采样行/时间/噪声调度相同，无 Val 选择最优 checkpoint。

内容响应扣除同掩码零内容响应，消除纯音乐/掩码捷径；空上下文严格零。
残差 RMS 上限 0.1，与 P0 幅度一致。没有输出动作写入器或后处理 Root 修复。

## 训练分布与目标

新登记 B0 真实 inpainting 轨迹缓存：8 个拟合窗、4 个监测窗，
每窗 50 DDIM 步，保存第 0/24/44/49 步完整 x_t、raw x0、噪声摘要。
缓存独立重跑并逐数组比对。训练 50% 实际轨迹状态、50% 真实配对 q_sample。
缓存绝不称作当前训练模型的 on-policy 分布；覆盖率只限登记窗口和步数。
软上下文单侧 dropout 不改缓存硬约束，训练支持清楚记录。

损失直接作用硬覆盖前的 raw x0：
未知区 147D 重建、已观测 Root/SO3、左右接缝 Root D1/D2 和相对旋转速度。
不把已知真值混入预测后再计算损失，不增加伪造 contact 标签。
SI 单位和各项权重写在 config；保存分项梯度和有限性检查。

## 生成与发布

B0 无 Adapter、B1 冻结 C2 是历史基线；B2/B3 完成新训练后再对照。
先看单窗两侧状态响应、来源姿态、Root/FK、两端连续性与原 Native。
正确/移除/左右错配只改变软条件，硬约束和噪声保持一致。
未通过来源与绝对物理验收，不运行三首整曲；human_reviewed=false，
operational_qualified=false。人工审核不阻断客观算法实验。

代码先发布 GitHub，再以精确 SHA 在 tmux 运行；新进程重放零 optimizer；
完整数组留 outputs，公开精简报告/哈希/回执。仅清理本轮临时 SSH 辅助物。

## 训练开始后的预冻结读出细则（尚未读取本轮 DEV）

生成固定 8 个旧案例，B0/B3/B2 各 8 次，匹配原种子；B1 为已冻结 C2。
B0 的 normalized 数组必须与历史 zero 逐位相等。完整采样与 Native 新进程重放。
4 个 Val 的 soft 正确/移除/单侧移除/左右错配在同一 correct-driven x_t 上比较；
它们不是独立反事实完整采样。报告误差、Root 已知区误差两套胜率。

单窗准入：左右错配对照中两套胜率分别至少 3/4；8 例原绝对 Native 与
150 帧 ROI Native 全过；每例姿态误差不超过 C2+0.01rad，历史 R8 来源姿态
误差也不超过 C2+0.01rad；左右接缝各自 Jerk maximum 不超过对应 C2+1e-6。
这些是研究资格而非新 Native 阈值。未全部通过时整曲仍为 0。
