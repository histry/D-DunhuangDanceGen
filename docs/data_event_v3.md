# Data Split V3 与 ChangE-E1：运行协议

本阶段只做来源审计、任务化角色清单和冻结 Core 的 E0/E1 对照。配置固定于 configs/data_event_v3.json；不训练、不拟合统计、不调用 EDGE 或 Native，不改变正式规划、权重和验收门限。旧 Source-ID G2 未通过，不能据此启动 G3、E2 或 SQP。

## 数据协议

- 私有 source_registry_v3.json 同时保留 historical_role/exposure 与独立 roles；单人准入必须有外部来源证据，单轨数组不能证明单人表演。
- AIST++ 取官方 crossmodal 清单与已通过来源时钟、完整目标/Jukebox 核验来源的交集，分别报告标称列表数与实际人口。官方 Val/Test 允许共享音乐；录制/编舞不跨角色。另建 strict_unseen_song / unseen_choreography / unseen_performer 协议，分别按对应身份、原始 SHA 与录制闭包；不以共同舞者合并所有音乐。作品范围限于 publisher canonical track ID，跨 ID 同作品别名仍未知。
- FineDance 用哈希绑定的官方代码 AST 读取 cross_genre/cross_dancer Test 与 ignore，绝不执行供应商代码。原始20 Test 与 ignore 相交后应按实际可用人口计数。没有官方 Val；缺少可靠作品身份、单人来源与表示/时钟准入时不构造内层 Val，也不把缺失手部补零当完整151D真值。
- ChangE 保留8个单人来源的共享 Catalog，与源留出8折分开。每折6 Train / 1 Val / 1 Test，只登记重建索引及 Train-only 统计义务；本阶段不拟合或授予任何旧权重资格。6个双人录制轨道只保留盘点，排除单人资源。
- 新角色不是新的盲测：历史数据角色/开发曝光和预训练暴露 UNKNOWN 均保留。所有协议 ready_for_new_training=false。下次训练须单独绑定清单、Train-only 统计和 checkpoint_binding，禁止继承旧模型资格。

## E0 / E1

保留原499个 UID、Core边界与全录制发现依赖。E0从原始SMPL独立解码，精确复现旧12–21十关节峰值、35%–65%中心和冻结Scaler支持域。E1在完整录制上先计算SO(3)导数与可信FK，再切Core，按实际受控12–19八关节取峰；original_peak、controlled_peak、safe_anchor分别记录。安全锚点不移动真实峰，不按Scaler筛选；真峰没有25帧上下文时只可选另一个实际观测阶段，否则NO_SAFE_ANCHOR。

启动、速度峰、方向转折和定格是运动学代理；Root转向不成为上身锚点，Root/contact不新增控制或真值。多关键姿态最多3个，可不足；边界上下文只来自原来源，不能宣称为生成邻窗一致性。

注册脚本在观测E1结果前，按真实音乐timeline内容SHA与路径稳定选定150/225/450/3601帧各一条。E0/E1共用slot、beat、全局及局部0.75–1.33时间映射约束，拒绝零长/非单调映射；保持原time_map数值算子不变。分别记录 Temporal 冻结Scaler支持和 Adapter 原生观测模式，合法候选数不是生成质量。

## 执行与复验

先 push 本提交，再让服务器精确 fast-forward 到提交。使用：

    tmux new-session -d -s ddg-data-v3 'bash scripts/run_data_event_v3.sh outputs/data_event_v3_01 outputs/data_event_v3_01_receipt > outputs/data_event_v3_01.log 2>&1'

输出与回执目录必须不存在。launcher运行完整回归、EDGE补充测试、主研究和全新进程复验，并保留UTC、执行提交和退出码。复验重新读取原始SMPL、复制哈希一致的原始资产、重解码全部8来源、重建499条事件特征并逐值比较，禁止复用主运行的解码数组或统计。主运行和复验均无优化器、统计拟合或模型前向。

私有源身份、原始资产和全账本留在服务器；export_data_event_v3.py只导出白名单聚合数与证据SHA。完成后另核实 exit_status=0、public_summary、fresh replay和Git同步。清理范围只含本轮SSH临时文件，不删除实验输出/失败回执。

官方定义：[AIST++ Facts and Figures](https://google.github.io/aistplusplus_dataset/factsfigures.html)，[FineDance官方划分代码](https://github.com/li-ronghui/FineDance/blob/main/dataset/FineDance_dataset.py)。实际执行以缓存源码SHA和本轮来源账本为准。
