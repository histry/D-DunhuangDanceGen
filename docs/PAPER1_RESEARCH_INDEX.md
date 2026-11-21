# Paper 1 当前研究入口

正式生成、研究先导和冻结对照分别维护。研究模块没有自动取得正式路由资格。

| 类别 | 入口 | 证据边界 |
|---|---|---|
| 本轮薄集成与来源支持 | `src/dunhuang/research/paper1_support` | 统一5种交换对象；14轨道资产审查；固定来源时钟与来源替换分别归因；独立质量未知 |
| 当前强基线 | R52 Heuristic + E1-W + G1；R63 Fixed48 | 保持来源内容和合法执行合同；数值通过不等于音乐或Native24合格 |
| 自然时间改编基线 | `gtw_kinematic_baseline` | GTWLocal/Extended/Context，非原论文完整DP复现 |
| 来源容量诊断 | `hold_timing` | 既有12位置的Hold代价不区分候选；全段低活动不能用固定总时长Warp消除 |
| 音乐独立证据 | `music_events`、R65评价工具 | 自动Beat/Downbeat/结构候选不是人工真值；缺少独立音乐质量记录 |
| 冻结评分研究 | R41/SDMR、R67共享合法候选 | 保留评分与共同域对照；没有正式路由资格 |
| 历史机制与执行依赖 | R36/R47/R53/R59–R73 | 保留来源身份、候选、阈值、失败和复验；不作为最新算法逐项堆叠 |
| 独立生成研究 | EDGE正式链、Lodge参考引导研究 | 属于不同合同；本轮不启动Diffusion、R74或新训练 |

推进顺序：核验真实来源 → 先观看实际主候选 → 同来源时间对照／来源替换对照 → 独立自然性与音乐偏好 → 条件成立后再考虑SDMR融合和整曲。没有人工记录时，不用机器代价代替独立艺术评价。

旧README各轮条目作为历史交付索引保留；本页按当前研究职责导航，不重写冻结结果。

自动评价入口：`src/dunhuang/research/paper1_auto_eval`。改编BAS、只读音乐证据、真实配对非循环错位与受控运动干预各自报告；不改变旧Uniform／GTW／G1，未准入MotionCritic、BAP、PFC或Native24。

Q02：[完整候选音乐时钟及运动可观测性资格](../reports/paper1_auto_eval_q02_release/report.md)。冻结自动评价仍未取得跨域准入；历史人口不作为新音乐泛化，来源×时间机会保持登记，联合决策尚未执行。

SourceTime-Q01：[有限共同池决策实验](../reports/paper1_source_time_release/report.md)。固定来源×时间与强顺序对照；BAS只读，不参与选路，未宣称艺术质量或未见泛化。

Reachability-Q01：[精确可达证书及完整139单位结构审查](../reports/paper1_reachability_release/report.md)。单锚点和联合窗口分开，连续可达不升级G1；有限对应假设无合法解，不扩搜或宣称艺术收益。

RoleAlignment-Q01：[有限事件对应资格与来源执行](../reports/paper1_role_alignment_release/report.md)。复用1–3节点可达证书、共同候选和原G1；新配对诊断仍未确认角色信息超越密度／位置控制，不进入正式路由。
