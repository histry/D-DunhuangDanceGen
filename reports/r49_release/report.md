# R49 事件级音乐对齐实验

先验收三首敦煌音乐的代表槽位，再运行消融、历史 Val 和独立复算。观看页播放 ChangE 原动作经合法时间映射后的 22 关节骨架，左右栏同步使用同一真实音乐片段。

本轮为 95 个时段的 3861/4177 个变体取得分数。其余 316 个变体所在时段超出真实音乐特征支持，均按 MUSIC_SUPPORT 排除。每个事件在同一目标时段只试一个 R48 变体。
冻结 R41 在历史 Val 有内容分辨信号，但相对换音乐的配对区间跨零，事件排序未达到接管 Planner 的证据要求。

## 历史 Val

Train 与 Val 的作品组、音频 SHA、原始动作 SHA 和录制身份不相交。历史 Val 参与过开发，因此只用于诊断。现有 Data V3 的 119 条 Test 登记仍属已曝光或曝光未核验。新盲测等待具有物理时钟、身份和曝光证据的输入。置信区间以录制内均值再按音乐重采样，动作窗口或相位数量不构成独立歌曲数。

| 任务 | 方法 | 曲目 | R@1 | 音乐组 95% CI |
|---|---|---:|---:|---|
| phase_offsets（118项） | Content | 6 | 25.46% | [+18.80%, +32.13%] |
| phase_offsets（118项） | Duration | 6 | 20.00% | [+20.00%, +20.00%] |
| phase_offsets（118项） | Energy | 6 | 18.98% | [+9.17%, +29.44%] |
| phase_offsets（118项） | Rule | 6 | 19.35% | [+10.37%, +29.17%] |
| phase_offsets（118项） | MusicSwap | 6 | 21.20% | [+17.04%, +25.37%] |
| phase_offsets（118项） | BothZero | 6 | 20.00% | [+20.00%, +20.00%] |
| equal_duration_near_motion_energy（128项） | Content | 6 | 30.00% | [+10.00%, +50.00%] |
| equal_duration_near_motion_energy（128项） | MusicSwap | 6 | 24.44% | [+6.11%, +46.67%] |
| equal_duration_near_motion_energy（128项） | Energy | 6 | 8.29% | [+2.50%, +14.86%] |
| equal_duration_near_motion_energy（128项） | Duration | 6 | 20.00% | [+20.00%, +20.00%] |
| equal_duration_near_motion_energy（128项） | BothZero | 6 | 20.00% | [+20.00%, +20.00%] |
| same_song_wrong_time（20项） | Content | 1 | 20.00% | 不可估计（仅 1 个音乐组） |
| same_song_wrong_time（20项） | MusicSwap | 1 | 25.00% | 不可估计（仅 1 个音乐组） |
| same_song_wrong_time（20项） | Energy | 1 | 5.00% | 不可估计（仅 1 个音乐组） |
| same_song_wrong_time（20项） | Duration | 1 | 20.00% | 不可估计（仅 1 个音乐组） |
| same_song_wrong_time（20项） | BothZero | 1 | 20.00% | 不可估计（仅 1 个音乐组） |

| 任务 | 内容相对对照 | R@1 差值 | 配对 95% CI |
|---|---|---:|---|
| phase_offsets | Duration | +5.46% | [-1.20%, +12.13%] |
| phase_offsets | Energy | +6.48% | [-3.06%, +15.83%] |
| phase_offsets | Rule | +6.11% | [-6.39%, +16.11%] |
| phase_offsets | MusicSwap | +4.26% | [-4.07%, +12.59%] |
| phase_offsets | BothZero | +5.46% | [-1.20%, +12.13%] |
| equal_duration_near_motion_energy | MusicSwap | +5.56% | [-19.72%, +34.72%] |
| equal_duration_near_motion_energy | Energy | +21.71% | [-3.61%, +46.68%] |
| equal_duration_near_motion_energy | Duration | +10.00% | [-10.00%, +30.00%] |
| equal_duration_near_motion_energy | BothZero | +10.00% | [-10.00%, +30.00%] |
| same_song_wrong_time | MusicSwap | -5.00% | 不可估计（仅 1 个音乐组） |
| same_song_wrong_time | Energy | +15.00% | 不可估计（仅 1 个音乐组） |
| same_song_wrong_time | Duration | +0.00% | 不可估计（仅 1 个音乐组） |
| same_song_wrong_time | BothZero | +0.00% | 不可估计（仅 1 个音乐组） |

五偏移任务的零标签来自已核验源时钟。错音乐与原音乐共用实际支持。固定长度的事件任务以本录制的对应时段为正例，并挑四个来自不同录制的负例。近能量难度使用离线目标动作统计。同歌错时只有一个音乐组可以组成固定五选池，组间区间不可估计。上述任务评价物理关联和时段分辨，艺术适配的独立标签仍缺失。

## 敦煌音乐同池消融

| 评分臂 | 共同槽位 | 首选变化 | 候选数 |
|---|---:|---:|---:|
| B0_activity_proxy | 95 | 70 | 3861 |
| B1_linear_local | 95 | 10 | 3861 |
| B2_linear_local_multi | 95 | 4 | 3861 |
| B4_coverage | 95 | 0 | 3861 |
| swapped_local | 95 | 2 | 3861 |

表中一般与主评分的首选比较，换音乐与正确音乐的 Local 首选比较。活动度臂在相同 R48 候选池重算，以排除候选库变化。主评分相对同池活动度代理改变 70/95 个首选，单独以局部内容换音乐只改变 2/95 个首选。现有目标域观察不足以把主评分差异归因于音乐内容，作品适配也缺少独立标签。

## 阶段、接法与训练资格

在 48 组同编舞异录制候选上，从 6927 个可观测源阶段中，多近邻、互为第一、余量与顺序检查保留 4 个，其中 2 个落在 15 帧时钟带内。筛查只使用姿态与顺序，时钟差留作结果。真实阶段语义继续记为 UNKNOWN。
冻结几何策略审计 12 条候选接法，10 条达到 R48 数值筛查，0 条目标时段实际留出 48 帧。几何证据及时间空间可行性分别保存，Root 与接触真值有待补齐。
单事件神经重新演绎保留训练和干预合同，尚欠可核验的参考与目标监督。现有 Lodge 可变长路径对参考和查询阶段输入置零，当前轮次未执行神经训练或实际采样。

## 复验

服务器回归 639 passed, 10 skipped, 5 warnings in 28.69s。主评分在两次对照任务中精确复算，Val 与资格审计逐哈希一致。复核 41 个冻结 R48 文件，17 个新文件逐项封存。首次启动的历史音频资产清单错误已有修正，原失败回执保留。浏览器验收使用本地只读 HTTP Range 服务支持音频跳转，服务已关闭。
