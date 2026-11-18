# R45 preregistration: evidence-calibrated review and source-feasible phase clocks

Baseline main 8524471220e6c50e9a63f9688da3289d8f87ba30. Formal EDGE/Planner/Native and R38–R44 scientific roots stay frozen. New code is isolated in research/phase_r45 and research/target_review_r45. No new SDMR weights or generator samples in A/B. No Test tensors. No Val/held-based parameter selection. Same historical population; no claim of a new blind music benchmark.

## Effect first

Same 22 R43/R44 cases, raw Geometry, intervals and two saved seeds. Geometry duplicates do not count as independent samples. Primary Phase candidate, Geometry, saved R44C2 and R43Core failure are exported before new factor/negative controls. First deliver all22 interactive trajectories and three fixed source clips: female_FeiTian, female_meditation, male_pipa_2. Inspect actual browser audio/frame/case/clock interaction before running controls. Source original hidden gap is evaluation-only.

## B: strict interpolation factor and music-controlled selection

B1 MatchedLinear and B2 MatchedC2 use EXACTLY the same actual feasible source/target knots stored in R44 G2, not the unconstrained old G1 knots. Both use the identical R44 spatial SO(3) RotationSpline and Root CubicSpline interpolants, support, masks, audio and population. B2 must reconstruct saved R44 C2 arrays exactly. This isolates time interpolant shape. Compare both per-knot ±3-frame stencils and interior 30Hz FK D1/D2/D3.

B3 Phase extracts kinematic speed/onset/braking/turn/hold proxies from the visible source chunks BEFORE cropping Core; no hidden target or neural prediction enters detection. At most4 separated stages per Core. Same source-only fixed patterns (at most25) are used for every music condition, with <=6-frame shifts, original [0.75,1.33] music/source rate and analytical continuous rate certificates. Target knots are actual integer frames; d1=1,d2=0. Source-only candidate constraints: activity ratio [.85,1.15], D2<=1.05 baseline, D3<=1.05 baseline. Identity is always available; low activity gets no forced phase. These constraints are an ENGINEERING FILTER, not independent physics evidence. No pipa2 exception or post-result adjustment.

Music selection uses audio-only spectral flux and beat proximity, fixed type-compatibility rules and identity/strain penalties (.12,.02). WrongMusic/NoMusic use the identical source phases, clock pool and eligible set; only acoustic scoring changes. FixedMapMusicB must preserve motion bit-exact. NoMusic removes all acoustic scores and chooses identity. Waveform RMS accents and waveform first-difference energy peaks are separate evaluation proxies; the planning objective is not a success metric. Two evaluation families: old R44 upper-speed/RMS timing and wrist-FK-acceleration/waveform-transient timing. Empty/quiet cases remain non-evaluable and denominators are reported.

Gate for further time-learning preparation: at least6 of8 paired Train music groups evaluable for each proxy/control; music-group bootstrap95% lower bound >0 for BOTH proxies vs WrongMusic AND NoMusic; at least3/4 ChangE Train sources preserve activity/D3 and protected exactness. Held cases are diagnostics, not selection. Full Native24/contact/world Root remain unqualified where unsupported. B3 source-feasibility success by construction is not evidence of musical learning or independent physical qualification.

Detailed pipa2 attribution is registered alongside FeiTian, pipa1 and quiet female_meditation. Decompose the continuous chain rule into source-speed, clock-acceleration and clock-jerk terms, integrate with fixed tensor Gauss orders3/5 to compare with ACTUAL30Hz finite third differences, and retain numerical/quadrature residual. Source derivatives use 7-point stencils with step0.025 source frames. Report signed covariance/energy contributions, residual and quadrature sensitivity; do not claim exact causality when residual is large. PWL knot singularities are diagnosed via direct discrete stencils, not the smooth chain formula.

## A: target-domain evidence, no training

Preserve the552 frozen Native pairs. Preselect a source-balanced subset by choosing one candidate per available raw source/query via SHA256(blind ID), before viewing scores (23queries x8 sources if all available). Reviewer media hides model/source/role/scores; no semantic labels are inferred. EVENT/POSE/MUSIC_MATCH separate, UNSURE and query-level NO_SUITABLE_CANDIDATE supported. Browser export is an UNVERIFIED client record, never automatically human truth. Formal import requires auditable roster, independent reviewer records, source/media hashes and disagreement handling. At least two independent reviews; third-person adjudication preserves both originals. Missing real records => NOT_EVALUABLE; does not block B algorithms.

Frozen Global/Diagonal/Partial scores come from R43 Native. Native Debias uses reconstructed frozen R41 global vectors, checked against saved Global scores, and the SAME24 R44 Train calibration groups. Fixed Train query bank comprises the already registered query at each90/150/240 length; aggregate candidate affinity per Train group then top10 groups. Score=2*Global-mean(top10 group affinity). No Native label fitting, no full BeatDance reproduction. Report full and balanced source availability/tie-aware concentration; never call changed source distribution artistic accuracy.

nDCG and model preference agreement require verified complete MUSIC_MATCH judgments in the frozen pool. Refusal is evaluated only with an explicit independent query-level none-suitable judgment; missing judgments/UNSURE are not negatives. Report coverage, disagreement and source/music-group denominators, with song-group bootstrap (only3 historical songs). No real labels => no numeric human metrics or routing qualification. New independent music population is absent unless separately supplied and frozen.

## C: conditional learning only

No new time/space branch unless B supplies credible net benefit and real paired reference/target lineage is admitted. Audit same-song independent raw recordings, clock support, choreography duplication, actual representation/joint/Root/contact support and stage correspondence. Same physical music clock is weak correspondence, not exact cross-performance pose truth. Do not synthesize a correct ChangE music target by swapping audio. A lack of manual labels is not itself a prohibition on objective experiments. Any allowed training would need its own frozen initialization/budget/targets before execution; otherwise stop without Lodge retraining or whole songs.

All stages: publish code before server execution; tmux; fresh-process replay with zero optimizer calls; exact protected frames, baseline and parent hashes; retain success/failure receipts. Private media/arrays/weights are not pushed to Git. Clean only this run's temporary SSH helpers.

References checked: MotionBeat official https://motionbeat2025.github.io/ and https://github.com/wangxuanchen88/MotionBeat ; GTCC CVPR2024 https://openaccess.thecvf.com/content/CVPR2024/html/Donahue_Learning_to_Predict_Activity_Progress_by_Self-Supervised_Video_Alignment_CVPR_2024_paper.html ; no claim of full reproduction or learned semantic phases.

补充固定读出定义：导数归因将连续SO(3)路径采样与实际139D编码后再FK的差异单列为representation round-trip increment，和积分/源导数残差分开；这不是一个新精度候选，也不据此更改任何门槛。检索额外保留固定EnergyHeuristic（Train归一化音乐通道0与动作通道264均值差），它不是完整Planner代价，更不是真值。

媒体边界修正（仅修正审核包导出）：两个已登记查询的末帧音频bin不满1/30秒，但最后姿态帧中心仍有真实音频。导出保留真实EOF与实际采样区间，记录partial_final_audio_bin，不补零、不移动槽位或姿态时钟。若末帧中心本身已越过EOF则拒绝导出。首次失败包与非零回执保留。

种子合同澄清：全部22案例的身体6D轨迹相同，但old_failed/new_plan压力案例继承的模型Root和接触通道随种子不同。时间图筛查只使用共同身体FK；世界Root/接触仍不合格。保留逐通道差异，不把它们强制改成相同，也不把两个种子当独立来源。首次评价断言过强的失败日志保留；仅修正验证范围与页面说明，候选、音乐、权重和门槛不改。
