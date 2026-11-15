# R21 当前代码调用与实验分工

## 正式路径保持原样

pipeline.py → MusicTimeline/Event Catalog/Planner → 全曲Event Plan →
edge_backend.py/原Temporal V0.5 → 冻结EDGE → 原Native。
SDMR仍Shadow；R21权重没有被加入默认生成入口。

## R21研究路径

scripts/run_r21.sh 的每一阶段是独立进程，不是多个生成网络串联：

1. context_r21.data：全部候选来源/时钟/完整录制审计，必要编辑域筛查；
   输出24 Train +4曝光DEV及源几何代理、独立Native parity。
2. neutrality：相同R19 N2 warm模型、旧R20条件和种子，必须逐值复现两条冻结轨迹。
   这是新仪表回归，不是再次研究双端可见性。
3. train B0/B1/B2/B3：四个独立同起点模型，各1200更新。
   B0/B2使用两个150视图，B1/B3单225；目标混合或边界/内部独立均值。
4. calibrate：B3固定末态的Train-only支撑代理梯度预算，无优化器。
5. train Q/M/F：B3末态的三个独立分支，各600更新、同一4来源机制人口。
   Q是共用继续训练对照，M仅替换半数输入x_t，F仅新增支撑机制。
6. rollout：全部模型固定后才读取DEV模型成绩；两种观察布局及固定上下文干预。
7. evaluate：独立Native、实际硬Mask分区、绝对q/实际采样误差、支撑和运动范围。
8. verify_r21.py：读取所有新进程重建结果/快照，不构建优化器。
9. render_r21.py：世界Root固定相机、共同尺度、同步可选双视频、真实配对音频。
10. collect_r21.py：只向全新发布目录写入公开汇总与回执。

数据/权重加载调用旧模块是为了保持资产与权重身份，不意味着R13至R21网络逐个运行。

## 单次模型计算

同一225帧x_t、观察值/Mask、音乐、代理metadata →
split两视图或joint全局225 →
动作主投影与观察投影 + 同参数状态头 →
同一EDGE的有音乐/空音乐两次前向 →
CFG=2 → clip[-1,1] →
split均值融合/joint恒等 →
真实硬观察保留 → 原50步全局逆扩散更新。

训练在guided输出上计算目标；两音乐分支仅另做读出。
音乐编码器及除首Decoder/输入输出之外的参数保持冻结，激活可反传。
没有独立Root网络，没有新的状态头参数，没有附加后处理平滑器。
未来Root与首尾来自真实源录制；这仍不是整曲FREE任务或ChangE风格资格。

## 归因区域

hard_domains使用实际Mask，完全固定、跨硬观察边界、完全可编辑三类模板互斥完整。
legacy_r20_measurement只保留旧位置定义供比较，不能称纯编辑域。
cfg_preclip是CFG之后、裁剪之前；C4始终不作为真实接触监督。
