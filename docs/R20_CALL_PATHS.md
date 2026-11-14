# 默认调用与R20研究调用

本图对应本仓库实现。复用加载器/数据/坐标/FK，不代表串联二十个生成网络。

## 默认生成

```mermaid
flowchart LR
  A[scripts/generate.py] --> B[pipeline.run_once]
  B --> C[planning.worker]
  C --> D[Event Catalog / Timeline / Planner]
  D --> E[Event Plan / anchors]
  E --> F[edge_backend]
  F --> G[冻结EDGE / Temporal V0.5默认]
  G --> H[physical / 原Native]
```

来源：src/dunhuang/pipeline.py、planning/worker.py、edge_backend.py、physical.py。
研究训练权重不因本轮结果自动替换configs/default或运行资产。

## R20实验组织

```mermaid
flowchart TD
  W[R19 N2末态] --> S0[S0 双150 / N2目标]
  W --> S1[S1 真225 / N2目标]
  S0 --> G[仅Train的结构继续条件]
  S1 --> G
  G --> P[冻结共同起点 / 重置各优化器]
  P --> A0[A0 四时刻额外训练]
  P --> A1[A1 扩展时刻支持]
  P --> A2[A2 配对全身时序]
  A2 --> K[仅Train的采样落差条件]
  K --> A3[A3若准入 / 自身x_t缓存]
  A3 --> V[全部组固定后的Train与曝光DEV评价]
  A0 --> V
  A1 --> V
  A2 --> V
```

各分支是独立训练对照，不是一次前向串联的模型。
temporal_r20/common.py复用R19/R18/R17加载器和资产血缘；实际主干仍是一份EDGE。

## 每个去噪步骤

```mermaid
flowchart LR
  X[全局225 x_t / 观察 / Mask / 真实音乐] --> L{布局}
  L --> D[split:两个150视图]
  L --> J[joint:一次225视野]
  D --> N[同一EDGE参数 / CFG两支]
  J --> N
  N --> C[CFG=2 / clamp]
  C --> M[split均值 或 joint恒等]
  M --> H[明确的硬保留]
  H --> R[下一逆扩散步]
  R --> X
  H --> F[逆归一化 / FK / Native]
```

Joint仅对null_cond_embed作固定长度适配，已有音乐编码器权重保持。
两个布局都是INBETWEEN可靠双端研究；整曲右端动作Root并非天然真值。
R20不运行生成历史训练，也不把缓存x_t叫已生成舞蹈历史。
