# 真实配对生成资产：两阶段预检

原始数据只在被忽略的 assets/datasets 保存，不推送 Git。复用旧来源 population 的完整哈希和历史角色，不把已曝光 Val/Test 重新称为盲测，不随机重分配。

第一阶段先复制真实动作字节，再完成 CPU 转换与 Normalizer 支持域检查：

    python scripts/stage_paired_assets.py --source D:/danceRAG --population /path/to/historical/population.json --out assets/datasets
    python scripts/paired_assets_experiment.py --assets outputs/event_plan_01/prepare/assets.json --raw-root assets/datasets --out outputs/paired_assets_new

AIST++ 按 smpl_scaling 还原米制 Root，按固定 EDGE 的 ::2 从 60Hz 取到 30Hz；Root/姿态由 Y-up 转为 EDGE Z-up/row Rot6D。保留原高度，不为支持域另拟合地面。分别审计官方全局 Root 和当前训练的窗口首帧 XZ 原点。训练审计固定 150/75 网格，不能外推为所有切片均可/不可表示。Contact4 使用固定 EDGE 的脚步三维前向位移 <0.01m，窗口最后一帧为1；它是派生监督而非接触传感真值。

FineDance 的 315D 保留 Root 与 body22 旋转，末端手部 22/23 以 identity 占位且监督为 false；不能把占位当成完整 SMPL24 真值。mask 分 observed、derived 和 supervision，不允许未知通道进入 loss。

只有第一阶段存在表示合格的 AIST Train 候选，才复制两组预注册候选所需音频，执行真实 Jukebox Layer66 提取烟测：

    python scripts/stage_paired_audio.py --source D:/danceRAG --raw-root assets/datasets --audit outputs/paired_assets_new/audit
    python scripts/paired_assets_experiment.py --mode features --audit outputs/paired_assets_new/audit --assets outputs/event_plan_01/prepare/assets.json --raw-root assets/datasets --out outputs/paired_features_new

跨机器先按清单哈希同步动作及被选音频。音频切片不 padding、不循环、不拉伸，绑定原音频 SHA、真实 sample_start/end、offset、窗口 UID、固定提取器及特征 SHA。50D/35D/803D 特征均不能替代 150x4800 Jukebox。默认最多提取来自两个不同旧 Train 关联组的首个表示合格窗，未运行大规模提取或模型优选。

paired_generation_manifest_v2 是严格训练接口。除完整支持域外，还必须具备：
- 原始音频/动作、转换后目标、三种 mask、Jukebox 与提取收据的哈希；
- 与模型/Scaler 绑定的 150 帧/30Hz 真实时钟；
- 每窗有方法记录且绑定源 SHA/offset 的 alignment_attestation；
- 关联来源跨角色隔离，以及绑定原始源 SHA、录制/编舞/音乐身份和角色的 split_attestation；
- 独立 Train/Val；Test 先做身份检查再排除拟合。

发布包声明零 offset 不自动获得已校准配对资格。历史“known components”也不是已证实独立录制。缺少证据时 paired 预检明确 BLOCKED，零优化器步；完整训练完成也默认保持 music/full_motion_qualified=false，另行验证生成质量。本阶段不修改 Native 或冻结 Normalizer，不裁剪越域目标，也不拿 ChangE 零补全作为真值。

已有单次生成、V0.8 闭环及上身探针入口保持原行为；旧的无 mask/时钟/来源收据的 paired 清单现在拒绝使用。
