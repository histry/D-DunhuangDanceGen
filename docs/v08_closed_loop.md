# V0.8 Native 反馈闭环

本阶段增加可审计控制回路，冻结 EDGE、V0.5 默认单次路径及固定 Native 阈值保持不变。无训练；完整 151D 目标、真实音乐配对、双侧上下文和 SQP 是后续有条件阶段。

planning.feedback 重新执行完整 Native audit 并与保存结果逐字段相等，保存指标、数值、阈值、四帧 Jerk 支持域、真实 150/75 窗归属及事件时间关联。局部诊断使用与完整 audit 相同的推导接触状态，不改写保存数组。靠近窗边界不证明接缝因果，时间关联不证明事件不可行。

planning.replanner 只做有限预算与进程编排。外循环复用原 beam planner 的完整候选计划；随后仅替换反馈关联最高的一个事件槽，固定其余槽，记录父计划及反馈哈希。排除仅对本次父计划该槽生效，无全局黑名单。内循环按预注册顺序改 Temporal 强度（默认 .55 → .35）或种子。Adapter 内循环仅换种子。

默认至少半数帧出现局部违反即停止模型诊断；纯边界故障停止等待上下文方法。50% 是路由阈值，不是 Native 验收阈值。显式诊断模式最多允许一次大范围失败后的重规划，仅验证接口。相同种子/条件下替换事件后，normalized motion RMS 不超过预注册 1e-3 时，报告该对照响应不足并停止。没有训练过的风险概率。

候选初始 UNKNOWN；生成成功且精确 Native 回验后才记录轨迹 PASS/FAIL。执行错误与超时保持 UNKNOWN，计划的 native_feasibility 保持 UNKNOWN。150/225 帧 PASS 不代表整曲，物理 PASS 不代表音乐演绎或文化语义合格。

缓存绑定所有请求资产与参数、完整计划哈希、模型/Adapter、种子、特征与提取收据、采样时钟、Native 合约、运行时及项目源码哈希；命中前校验保存数组和报告。缓存仅在本次目录内复用。每个候选有独立进程组，超时终止该组。

输出目录首次必须不存在。完成后 --resume 校验缓存，不再生成；中断半成品不自动重试，标记 UNKNOWN 并停止，需新目录显式重跑。程序不接管存活的旧进程，强制杀死控制器后恢复前应确认旧进程退出。

运行：

    python scripts/generate.py --config configs/closed_loop.example.json --assets outputs/event_plan_01/prepare/assets.json --out outputs/loop_new
    python scripts/generate.py --config configs/closed_loop.example.json --assets outputs/event_plan_01/prepare/assets.json --out outputs/loop_new --resume

有界诊断协议：

    python scripts/v08_experiment.py --assets outputs/event_plan_01/prepare/assets.json --out outputs/v08_new

state.json 保存预算、启动次数、决策、资产绑定、停止原因。feedback.json 保存精确 Native 回验及定位。planning_N/event_plans.json 保存尚未测试的 UNKNOWN 候选；selected.json 保存执行计划与 lineage。质量指标分别报告，不把违反较小的失败样本写成验收成功。
