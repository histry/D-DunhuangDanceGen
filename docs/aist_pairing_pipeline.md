# AIST++ 可核验配对与有界训练

本入口把源身份、时钟、表示支持、Jukebox 和模型训练分开设门。代码先发布到 GitHub，再在独立输出目录执行；原始 WAV/视频、拟合动作、特征和权重仍为忽略资产，不随源码发布。

## 来源和时钟

使用官方精修视频 CSV、基础舞 choreo.xlsx、AIST++ 相机 mapping/all/ignore 记录及原始动作/音频 SHA。录制 UID 按官方 cAll 对应的多机位捕获命名，基础编舞 UID 为官方 (genre, ch编号) 联合键，音乐按源文件身份。复核所有关联键跨历史角色是否互斥；不重新随机划分，不声称旧开发人口是新盲测。

高级舞有共享编舞，ch01..ch21 不是已获证的独立编舞 ID，当前完整来源台账记录它们但不准入首轮训练。通用动作名称如 walk 跨 genre 出现不被隐藏，也不被宣称为语义独立。

每条准入来源下载官方明确列出的 2 Mbps/c01 精修视频，核对完整 CSV 的身份字段和文件大小。精修音轨和原 WAV 在8kHz上做三个3秒锚点的归一化波形相关；最小相关系数0.98、残差/锚间差不超过1/60秒、视频音轨与60FPS注释时长差不超过34ms。禁止用动作节拍最优相位拟合音频，禁止时间拉伸。循环相似峰明确记录，零点结合出版方精修起点约定固定，不把重复小节叫作唯一峰证明。

## 顺序执行

以下命令在项目根目录、匹配的 EDGE Python 下执行，输出目录必须全新。使用 tmux 及持久日志，记录每步退出码。

    python scripts/aist_pairing.py --stage calibrate --workers 8 --assets outputs/event_plan_01/prepare/assets.json --raw-root assets/datasets --authority assets/datasets/aistpp/authority_20261112_v2 --source-audio /path/to/hash-verified/original/aist/wavs --out outputs/aist_calibration_new
    python scripts/aist_pairing.py --stage index --assets outputs/event_plan_01/prepare/assets.json --raw-root assets/datasets --calibration outputs/aist_calibration_new --audit outputs/paired_assets_01/audit --out outputs/aist_index_new
    python scripts/aist_pairing.py --stage features --assets outputs/event_plan_01/prepare/assets.json --raw-root assets/datasets --index outputs/aist_index_new --out outputs/aist_features_new
    python scripts/aist_pairing.py --stage train --steps 300 --assets outputs/aist_features_new/assets.json --raw-root assets/datasets --out outputs/aist_train_new

calibrate 的 --smoke 仅用于每角色一条的工程核验，不能编译完整训练人口。存在技术失败时 index 明确拒绝；不能默默靠下载成功的人口选模型。异常保留错误和旧目录，新目录重试时可复用已校验的视频，不能覆盖已测回执。

index 以原150/75冻结网格和模型 Normalizer 为基础，重新检查真实目标、观测/派生mask、Source/Role和实测音频范围。无 Jukebox 的预检必须显式 require_features=False，回执标明只验证目标/时钟/身份；训练入口始终使用完整特征预检，不能以该模式跳过音乐资产。

features 对相同原 WAV SHA、精确采样范围、固定提取器、Layer66/30Hz计算一次Jukebox，多来源共享同一真实音频窗时复用数组。每个目标仍有独立、绑定窗口 UID 的提取收据。不 padding、不循环、不改速率；全量清单含 Test 资产，但模型拟合和选权不使用 Test。

## 训练和资格

首轮只执行300步全151D配对补全，固定种子20261111、batch2、AdamW学习率1e-4、梯度裁剪1。冻结 EDGE，训练 PrimitiveAdapter；每50步和结束按固定Val隐藏通道MSE选择权重。保留zero_adapter、observed_source、condition_removed对照。这里的输入是部分已观测参考和真实配对Jukebox，训练成功不能自动证明音乐演绎、文化语义或Native生成质量。

## 独立复核

    python scripts/verify_aist_pairs.py --assets outputs/event_plan_01/prepare/assets.json --raw-root assets/datasets --authority assets/datasets/aistpp/authority_20261112_v2 --calibration outputs/aist_calibration_new --index outputs/aist_index_new --features outputs/aist_features_new --training outputs/aist_train_new --out outputs/aist_verify_new.json
    python scripts/replay_paired_training.py --assets outputs/aist_features_new/assets.json --training outputs/aist_train_new --out outputs/aist_validation_replay_new.json

第一条重核验来源与视频字节、身份隔离、三角色代表时钟重算、特征清单和训练收据；第二条在新进程只读重放选定权重的固定Val补全指标，零优化器调用。物理Native阈值和V0.8默认生成入口不因本轮变化。
