# Paper 1 支撑接口与来源容量先导

统一交换对象在 `contracts.py`，冻结算法仍由原研究包提供。正式 EDGE 入口没有依赖此包。

本轮只处理经过 SHA 核验的14条原始轨道，区分11个录制组、8条 Solo、6条待人工审查多人轨道。未核验“72条独立录制”这一说法。多人轨道不自动加入编排。

来源与预算先登记，再计算 GTW 或 G1。完整低活动区必须位于原 R47 单元内部，两侧各有30个非低活动时间单元；找不到机会的来源保留缺失。240/360/450帧预算选择最小合法值，不截断原语凑长度。语义完整性仍为 UNKNOWN。

主候选先展示同来源 Uniform 与冻结 GTWLocal 选择。观看验收回执通过后才启动对照：固定原语的时间对照，以及同一录制、来源长度相差不超过10%的更低活动单元替换对照。来源替换会改变姿态内容、来源自然时长和可用时钟池，不解释成纯 Warp 收益。MinHoldDiagnostic 只是可达性端点，不是正式音乐方法。

49格有限时钟域复用 R53 DAG 枚举，固定首末各4帧实际旋转及来源时钟；不是原 E1-W 离散域，也不是连续全域最优。每次选择重新构造48帧 G1，不继承资格。GTW 是现有论文机制改编基线，非原论文完整动态规划。

运行（仓库根目录、已有历史封存输入）：
```
PYTHONPATH=src python -m dunhuang.research.paper1_support.study main
PYTHONPATH=src python -m dunhuang.research.paper1_support.view main
# 主候选浏览器验收之后：
PYTHONPATH=src python -m dunhuang.research.paper1_support.study controls
PYTHONPATH=src python -m dunhuang.research.paper1_support.view controls
PYTHONPATH=src python -m dunhuang.research.paper1_support.verify
```

当前没有独立人类偏好；音乐质量、定势艺术意义、世界Root／接触和Native24保持不可评。无网络训练、R74、SDMR路由或整曲准入升级。历史源代码与回执属于复验依赖，不作无依据删除。

归因、匿名评价和统一入口分别由 `postprocess attribution`、`postprocess reviews`、`postprocess index` 生成（均为 `python -m dunhuang.research.paper1_support.postprocess ...`）。匿名顺序只在首次生成时创建，管理员映射保留在私有 `controls/admin`，不重抽顺序挑结果。

真实记录导入复用 `python -m dunhuang.research.gtw_kinematic_baseline.reviews --admin outputs/paper1_support_01/controls/admin/map.json --roster <独立核验名册> --reviews <本人实际记录JSON> --out <全新统计路径>`。未提供记录不填充同分、不生成偏好；至少每项两位独立合格评价者后才汇总来源级描述。
