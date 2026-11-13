# 盲审骨架播放修正 V2

依据用户对抖动和滑步的反馈，先检查冻结盲审包及来源身份。基准4df4f68，不更改原 source_expression_04 包或模型。

原显示有两个混淆因素：逐帧减掉 Root 位移；无音频时用 setInterval 按帧计步并自动回到首帧。16个 EVENT 片段的循环接缝平均关节位移为片段内单帧位移P95的3.7—35.6倍。部分片段原拟合位移超过1米。上述是呈现诊断，不是独立动作质量真值。

新查看器保留两个模式：默认包含已封存 Native 的拟合 Root 位移，减掉的只是首帧常量平移；另一模式固定 Root，只看姿态。世界位置未校准、真实地面与接触未知，不添加假地面或足锁，不从画面推断真实滑步。

原姿态坐标逐帧保持完全一致；只按同一30FPS时钟添加已核对SHA的Root路径，世界模式序列化允许1.01e-6米误差。requestAnimationFrame按实际经过时间计算帧号；音频任务使用音频时钟。停止在末帧，需手动重新播放。没有姿态平滑、插值、动作修补或生成模型更新。若片段内部仍抖动，继续作为原始拟合动作的待审问题，不能声称已修复动作本身。

35个盲编号、对应来源及片段区间与原包逐项核对，原媒体保留。新媒体SHA不同，审核导入时必须使用新protocol；旧记录不得套用到新媒体。当前记录数仍为0，human_reviewed=false，operational_qualified=false。

独立研究入口：python -m dunhuang.research.source_expression.viewer_v2 --source outputs/source_expression_04/review/reviewer --catalog outputs/event_plan_01/prepare/catalog/catalog.json --out NEW_DIR。服务器通过 scripts/run_review_viewer.sh NEW_STUDY NEW_RECEIPT 在tmux中进行回归、重建、字节重放及独立坐标核验。仅修正呈现，不重新运行R6控制对照或训练。
