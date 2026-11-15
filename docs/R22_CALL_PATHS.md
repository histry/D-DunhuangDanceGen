# R22调用路径

正式 pipeline→旧Planner→Temporal V0.5→冻结EDGE→原Native 不修改。
历史R21包与其所有封存产物保持原样。

研究入口 scripts/run_r22.sh：
1. branch_r22.data → 原R21独立数据重建＋源编辑域／前瞻机制清单；
2. branch_r22.sanity → R21 B3加载＋G0数值／真实梯度核对；
3. branch_r22.lodge → 忽略inputs中的固定官方Decoder和原生采样，只有共同22几何诊断；
4. branch_r22.train → 新objective分支重建＋复用R21 guided物理算子；
5. branch_r22.rollout → 复用R21原采样器（CFG/clip/hard完全不改）；
6. context_r21.evaluate → 同一原Native和真实Mask分区；
7. verify_r22.py → 独立重建／真实更新读出／完整采样逐值核验；
8. render_r22.py → 复用固定世界相机渲染，新研究标签；
9. collect_r22.py → 全部结果与独立回执，拒绝覆盖。

训练分支直接由 final_layer 的保留梯度hook取得；Telemetry仅用于记录。
G1/G2仅改变重建目标，未修改网络、两条前向或物理路径。
Lodge不通过补齐缺失关节进入正式Native；原生soft条件不等同R22硬端点。
未来ChangE参考阶段、生成历史或未知未来Root均尚未由该调用图接入。
