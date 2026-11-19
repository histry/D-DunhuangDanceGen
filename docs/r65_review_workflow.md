# R65 实际审核操作

此目录是研究工具说明，不包含任何人工标签。未收到真实记录时，系统正常运行并返回NOT_EVALUABLE，不会补造评价者或评分。

## 观看与填写

已看过R64的人可以继续比较动作效果，但须勾选历史曝光；这类意见作为开发反馈保留，不能重新称为独立盲评。

管理员给两名不同评价者分配F01／F02表单。先只发`reviewer/music.html`及其三首完整WAV，分别标记普通拍点、强拍、重音、乐句或UNSURE；不要提供自动检测结果、动作或其他人的标签。每类选择完成、部分、无明确事件或无法判断，未标注不代表无事件。导出后由管理员封存原始文件及摘要。

管理员记录首次提供动作页面的时间和证据，再发`reviewer/index.html`与短音乐片段。相同表单固定全部29例及A—D位置。只记录实际可感知的音乐协调、转场、连续性和丰富度；可以同分或无法判断。不根据已知方法身份追求预期答案。页面会本地保存草稿，也可下载；正式导出的ID和声明仍需管理员核验。

两名评价者的原始意见分别保存，分歧和UNSURE不得覆盖、删除或自动转换。此工具不判断敦煌文化语义、世界Root或足接触。

## 管理员绑定

`outputs/quality_r65_01/packet/admin/mapping.json`和`private_seed.json`不能发给盲评者，也不能放进临时HTTP服务。身份表、认证证据、真实意见都不提交Git。

私有名册`entries`逐人包含：`reviewer_id`、`reviewer_roster_ref`、稳定的真实人员映射`person_uid`、`assigned_form_id`、`verified`、`independence_verified`，以及`authentication_evidence_path`和实际SHA256。布尔值是管理员完成核验后的签证，不是程序自动证明身份。

为了核验“先听音乐、后看动作”，同一名册行还需`audio_before_motion_verified`、`music_review_id`、`music_review_digest`、`music_sealed_at`、`motion_first_access_at`、`order_evidence_path`及SHA256。`music_review_digest`为`common.digest(records.validate_document(音乐原记录, admin映射))`，绑定规范化合同；原文件仍按字节封存。两个管理员时间必须带时区并严格先后，证据应记录独立任务封存与首次动作交付。客户端导出时间单独不足以放行。

没有管理员名册、证据或顺序封存时，记录可格式导入和保留，但不自动取得独立资格。部分填写和UNSURE不删除其余已审核数据；统计按共同完成范围和有效评分进行，单人记录仅作描述。

## 导入与计算

先固定代码、音频、动作及私有映射；在新输出目录运行：

```text
PYTHONPATH=src python scripts/quality_r65.py evaluate --reviews /private/actual_records --roster /private/verified_roster.json --out outputs/quality_r65_review_01
```

`--reviews`只放实际导出的音乐与动作原记录；不要混入草稿、UI测试或身份表。同一个任务的修订不能冒充额外独立评价者；修订需保留原件和管理员版本登记，再明确本轮主记录。

程序校验pack、协议、媒体、表单、时间、人员与覆盖，逐字节复制原件到私有分析目录。事件误差按原音频秒算；UNSURE区域不当作负例。双方无事件、全并列或常数评分的相关性保持未知。自动结构项只是代理，双人一致性也不等于艺术真值。当前三首音乐已曝光，不支持新音乐泛化主张。
