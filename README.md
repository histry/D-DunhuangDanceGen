# DunhuangDanceGen

独立的音乐驱动敦煌舞生成实验工程：Music Timeline → Retrieval / Anchor Conditioning → 冻结 EDGE → 原 Native 物理验收。

唯一生成网络是 EDGE。没有 Refiner、第二个 Diffusion、Graph-SB/WFC 或 SQP。旧项目只通过固定版本的隔离进程提供动作解码和完整物理审核；没有复制旧工程。模型条件仍为真实 Jukebox layer 66、4800D，30 FPS，150 帧窗口、75 帧步长。

## 运行

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/generate.py --config configs/generation.json
# 已有服务器环境；先拉取代码，再在 tmux 内执行：
bash scripts/setup_server.sh
tmux new-session -d -s dunhuang-v1 'bash scripts/run_server.sh'
```

`configs/generation.json` 给出服务器上的已存在资产；换音乐时，先在 EDGE 环境执行 `scripts/extract_music.py --edge-root ... --audio ... --output ...`，把新特征、WAV 和提取 receipt.json 的 SHA256 填入配置。所有输出使用新目录，拒绝覆盖已有实验。新 WAV 的时长必须与其特征窗口对应。环境按官方 EDGE 安装；Native 使用旧项目 dance-rag 环境。外部依赖 commit 见 `configs/dependencies.json`。

## 动作与控制合同

- `MotionAdapter` 明确转换 Native Y-up/列式 Rot6D 与 EDGE Z-up/行式 Rot6D，米制 SMPL24；只有 Root 的局部旋转左乘坐标变换。接触预测保持原值，真实支撑由 Native FK 推导。
- `baseline` 使用官方长序列 DDIM 和原 Root / quaternion SLERP 拼接。
- `anchored` 在每个 DDIM 步联合施加重叠共识与稀疏姿态条件，最终步也属于约束采样。它是新的采样器，不能冒充原 EDGE inpaint；不在生成后粘贴姿态。记录原始去噪器误差，精确满足约束不代表模型自然地生成这些姿态。
- 初版仅控制上身关节 12–21，不约束 Root、接触和未观测手部。冻结归一化不支持的姿态直接拒绝，不裁剪掩盖。
- 检索支持带显式描述符的最近邻和无查询时固定顺序选择。当前实验采用预先固定三种来源，不宣称已实现 Router 语义检索或 Planner 学习。
- Native 原绝对阈值从固定旧版本 baseline 环境加载；不继承旧父动作相对保护或 50 Core / 49 Bridge 验收。每例保存阈值、整曲审计和真实 FK 交叉检查。

## 实验与证据

`scripts/run_experiments.py`：同音乐、同 seed，依次运行 5 秒、15 秒、整曲纯 EDGE 与稀疏锚点对照。选用真实 catalog 三个来源的首个事件中间姿态，锚点映射到真实音频检测节拍；不会按生成结果选择姿态。保存匹配姿态误差、音乐速度极小值对齐、原始窗口重叠误差、Native 接触/Root/jerk 等指标。

输出在 `outputs/experiment_v1`，包括输入哈希、动作 NPY、FK、日志和 summary。模型参数前后哈希相等；没有训练。代码运行成功、物理通过、敦煌风格成立分别判断。独立风格与人工自然性评价尚未实现，结果明确为 NOT_EVALUATED。

现有权重 SHA256 已冻结，但发布者下载来源未独立核实，不能宣称严格复现官方论文效果。音乐特征复用旧项目已提取的真实缓存，不重新声称本轮做过特征提取。运行前检查 Git 版本与受跟踪文件，环境依赖仍需使用配置指定的已安装环境。

## 代码

`src/dunhuang/pipeline.py` 是唯一主入口；`contracts.py`、`music.py`、`retrieval.py`、`conditioning.py`、`sampling.py`、`edge_backend.py`、`physical.py`、`evaluation.py` 分别承担清晰职责。测试覆盖表示往返、FK 坐标等价、音乐边界、跨窗口锚点和逐步采样。测试里的微型去噪器只用于算法检查，不用于实验生成。

首轮真实结果与复核方法：[实验报告](reports/实验报告.md) · [机器可读指标](reports/experiment_v1.json)。

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
