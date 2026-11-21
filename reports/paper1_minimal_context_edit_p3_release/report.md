# Paper1-MinimalContextEdit-P3：完整验收与Source-only对照

本轮完成授权的3个观看文件回传、实际浏览器验收、冻结Source-only对照及独立复验。科学生产提交为`f4e4ea9b9c22125c3076bd965341ec46d19c597c`；限定标量导出工具为`25ef1197c765627a59715abb721ae7c14245eef6`，科学包及旧P0／P1／P2／R53／G1均未修改。

## 真实完成的结果

| 条件 | 实际整曲／登记任务 | 保留缺失 | 不同SO(3)／FK数组 | 已测G1物化 |
|---|---:|---:|---:|---:|
| Source＋Warp字典序主方案 | 10/14 | 4 | 10／10 | 56 |
| Source-only＋UID平局对照 | 10/14 | 4 | 10／10 | 58 |

两阶段跨方案共20条不同旋转轨迹；所有新轨迹均实际执行为13-Core整曲，不是仅节点路径。合并有67种不同边界键，两个原实验共114次已测边物化，连同两次独立重放共228次真实G1构造。新增UNKNOWN边评价为0。

## 最少修改及Source-only对照

来源原语身份由source_uid、event_uid、source_interval共同确定；同原语的时钟变化按实际q数组SHA判定。以下来源变化均相对历史B0，包含指定中心，不能混淆为仅改变邻居数。

| 音乐／中心槽 | 指定来源 | P2强顺序换源 | 两方案共同最少换源 | 主方案保留原语的额外时钟变化 | Source-only额外时钟变化 |
|---|---|---:|---:|---:|---:|
| dunhuangwu2／1 | male_pipa_2 | 6 | 2 | 0 | 3 |
| dunhuangwu3／11 | male_pipa_2 | 7 | 2 | 0 | 2 |
| dunhuangwu4／0 | male_pipa_2 | 6 | 2 | 0 | 2 |
| dunhuangwu4／4 | male_ribbon_FenHe | 8 | 3 | 0 | 2 |
| dunhuangwu4／4 | male_meditation | 8 | 2 | 0 | 3 |
| dunhuangwu4／4 | female_meditation | 8 | 2 | 0 | 3 |
| dunhuangwu4／8 | male_ribbon_FenHe | 7 | 2 | 0 | 3 |
| dunhuangwu4／8 | male_meditation | 7 | 2 | 0 | 3 |
| dunhuangwu4／8 | female_meditation | 7 | 3 | 0 | 2 |
| dunhuangwu4／8 | male_pipa_2 | 7 | 2 | 0 | 3 |

8个任务图内最少换2个原语，2个任务最少换3个；指定中心占1个，因此其余上下文只需改1–2个。原强顺序的6–8次换源并非连接约束全部强制要求。

Source-only的最少来源数与主方案相同，但在保留B0原语上额外改变2–3个时钟。这说明Warp次目标在当前有限图内可保留更多原执行时钟，是目标性质和实际选择结果的证据，不是艺术自然性或音乐协调增益。

两方案有7组完整来源身份相同；另外3组——dunhuangwu4槽4的male_ribbon_FenHe、male_meditation、female_meditation——还改变了零基槽3（Core 4）的邻居原语。因此“同样少换源”不等于“换了完全相同的原语”；不能将全部动态差异解释为纯Warp效应。所有逐槽来源和时钟身份均保留。

4个无完整路径任务仍为：dunhuangwu2槽9、dunhuangwu3槽6与槽8、dunhuangwu4槽4的male_pipa_2。dunhuangwu3槽6明确保留局部有见证而整曲无见证的状态。它们仅在当前测量图中没有路径，不能宣称连续候选域无解。

## 实际动作与证书

DP保存全部层、全部节点的Bellman标签、最优前缀、前驱及终点最小值；完整检查所有等式，另用随机小图穷举交叉验证。最优性严格限于相同P1已测、策略可准入且通过原G1数值门的有限图，不读取或补齐UNKNOWN边。

每个新Core重新从原录制和实际q重建，旋转及q哈希精确一致。Bridge均按原G1方法重做，哈希及数值读数精确一致。三首仍覆盖3601／3598／3601帧，所有帧恰好写入一次。完整FK D1／D2／D3、SO(3)角速度与运输加速度、逐Core／Bridge及跨区差分模板重新计算，不按指标调整路径。

历史B0原数组及每首9G0／3G1混合桥仍为独立历史参照，不把它当成全G1图中的已知可行原路径。P2强顺序同样保留其原先选出的来源和时钟。三首均为历史开发音乐，14个任务不能作为14首独立音乐样本。

## 浏览器验收

首批3文件共13,396,308字节，已逐一核对事前SHA；复用本机已授权P2数据和三首音频，没有新增音频或原始旋转回传。

实际检查了14个任务、3个同步画面和4个缺失任务。缺失任务的强顺序与Min-Edit栏均留空。共同播放、暂停、0.5倍速、拖动、前后单帧和原生音频进度条均可联动：1470帧为49秒，1471帧为49.033333秒；三首末帧分别为3600／120秒、3597／119.9秒、3600／120秒。

原速来源具有独立30 FPS时钟，播放和拖动原速来源时主音乐停留在49秒；原速没有被变速伪装成同步。页面无脚本错误。三栏为Root-relative身体22关节FK，不能代替完整世界Root、接触或Native24。截图及本机观看文件保留在私有输出；浏览器工程验收与独立重放记录也包含于本批公开的independent.json。

## 版本、复验与运行限制

36项原合同及回归测试通过。主实验与Source-only各由新的tmux进程重做DP、所有证书、来源读取、G1物化、SO(3)／FK数组及指标，全部文件精确一致。主清单SHA为`85ed56a85f350268f26c67c642393c7429f3c81abdb68c886f13f03523c4c747`，控制清单SHA为`5977cc575a38baa781d5fff8fb285c818471ae23b5f8ba90081f4743d2bbc9c5`。优化器调用为0。

本轮服务器直连GitHub发生GnuTLS连接中断。通过本机实时ls-remote、服务器HEAD及干净源码确认科学代码同为f4e4ea9后，使用此前已拉取的完全相同提交执行Source-only及复验。新增发布脚本先推GitHub，再用2472字节的单提交Git bundle经用户SSH中继，核对公开SHA后Git快进；bundle SHA为`23055e003e75c3190c8132ce8933c3bb84507daeb628e07688d1b0ec1939974b`。没有关闭TLS校验、修改全局Git设置或改动科学包。网络与版本回执另存。

限定9个标量JSON已由两个进程精确生成，清单SHA为`f02d0e487f3ff71f7c654f780be78734aad448ba45ae524a9840b57a1dcf6184`。本批9个JSON共6,258,097字节，已取得本批回传与公开授权，按冻结SHA回传并随本报告发布。原始动作坐标、旋转数组、音频、输入张量与权重不在公开批次中；旧P1／P2待授权结果没有因本轮自动公开。

当前成立的是有限已测图内的最少修改与实际执行结果。独立音乐质量、艺术自然性、新音乐泛化及世界Root／Contact／Native24继续不可评价。SDMR仍为Shadow，`can_route=false`。本轮临时只读服务已关闭，临时SSH与Git中继文件按限定清单清理；最终运行及清理回执保留于本机与服务器私有输出。

## 公开证据入口

[登记与路径](minimal_edit_paths.json) · [完整Bellman标签与图内证书](exact_measured_graph_optima.json) · [来源／时钟归因与次目标对照](source_warp_change_attribution.json) · [完整及分区运动指标](full_motion_metrics.json) · [实际执行身份与Bridge哈希](execution.json) · [输入与版本绑定](provenance.json) · [浏览器及独立复验](independent.json) · [资格边界](qualification.json) · [本批清单](manifest.json)。

最优性核验需要清单绑定的原P0／P1有限图输入；本批证书不构成对未公开父数据或连续候选域的替代证明。
