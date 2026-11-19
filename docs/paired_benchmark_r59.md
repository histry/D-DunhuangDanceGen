# R59：真实配对固定网格资格与R58目标审计

冻结基线 d2558c6。先推GitHub，在服务器tmux构建完整原始录制固定网格，交付六个按元数据固定的真实来源／Heuristic／Frozen回放及视频，浏览器验收后才运行其余冻结评分对照。没有新增优化器、SDMR权重或编舞权重。

## 数据及可证范围

AIST++沿用Data V3的919 Train、118历史曝光Val来源（实际数量再由原始清单核验）；每个来源复核原动作、派生Native、R41旋转、真实音频和官方时钟证据SHA，再在完整共享支持上生成150帧、75步长和末端窗口。没有受正例/H分箱/历史特征落盘范围限制。动作求导在完整录制上完成，R37 Train统计冻结，50D音乐＋274D姿态/几何/动态通道接口保持。

FineDance从用户既有danceRAG原资产按官方@Genre Train编号固定取001–011与013，逐文件核对已存下载清单SHA并复制到本项目私有inputs；从完整真实音频用同一librosa0.11.0/50D提取器重提，使用原始315D前22身体旋转。009等动作/音频尾部不等长时只保留共同前缀并记录未配对尾部，不使用旧四样本的2帧长度差作为普遍条件。保留原始共同零点，Lodge中004等的双侧等量裁剪不当作音频偏移。@Dancer角色另外报告，不把两种官方划分混合；不称12条@Genre Train诊断为完整FineDance官方测试。该源库没有多录制配对证明，时间检索候选含对应录制的真实目标窗；只能报告同录制Recorded Correspondence，不能冒充跨来源泛化。原始平移保存，显示局部身体；世界坐标/物理单位和接触不因本轮获得Native资格。

官方依据：https://google.github.io/aistplusplus_dataset/factsfigures.html 明确音乐/编舞重复及Val/Test音乐共享；https://github.com/li-ronghui/FineDance 的固定官方加载器0476cd42619e57fb6ec4840e81acaf7743b1507d区分cross_genre/cross_dancer。当前Data V3还可能跨角色共享编舞，必须如实报告，不宣称未见编舞。旧Val、旧开发观看和完整预训练谱系未知都不更名为新盲测。

## 固定评价

每音乐组按动作SHA选前两录制，每录制在完整网格均匀取最多三个查询（仅元数据）。开放池同数据集/角色/长度，AIST排除同原动作SHA；同歌曲池仅再限制音频身份；困难池按查询音乐活动度与候选动作活动度距离取64，不查看目标动作的活动度、不插入正例。目标缺失也保留查询，单独报告覆盖与缺失，不能只公布有正例人口的条件R@K。UNKNOWN继续占据真实排序池，但不称为艺术负例。

六种评分：Heuristic、Frozen R41、R55 Conditional/NoClock原残差与两者TeacherRescore代数重评分。后两者不是重新训练Teacher-Preserved模型。主模型完全冻结，音乐置零、等活动不同音乐及动作Token时间反转分别输出。固定1e-7分箱下的同分按随机排列的解析期望计R@1/R@5/MRR，禁止UID排位制造命中。按录制内、音乐组间均衡，配对差值CI以音乐组重采样，不把窗/帧当独立音乐作品。

偏移任务使用[-15,-8,0,8,15]帧，在150帧真实连续观察内做非循环共同裁剪，目标为注入偏移的相反数；报告准确率与MAE，能量时序相关为Heuristic时钟基线。这证明受控时钟敏感性，不是事件语义或唯一艺术对应。活动强度/音频SHA并非内容泛化证明。

## 执行与复验

    python -m dunhuang.research.paired_benchmark_r59 data --out outputs/paired_benchmark_r59_01/data
    python -m dunhuang.research.paired_benchmark_r59 main --out outputs/paired_benchmark_r59_01/main
    python -m dunhuang.research.paired_benchmark_r59 render --source outputs/paired_benchmark_r59_01/main --out outputs/paired_benchmark_r59_01/main_video
    # 实际回传并验收后，要求 receipts/main_browser.json passed=true
    python -m dunhuang.research.paired_benchmark_r59 controls --out outputs/paired_benchmark_r59_01/controls
    python -m dunhuang.research.paired_benchmark_r59 verify_data --source outputs/paired_benchmark_r59_01/data --out outputs/paired_benchmark_r59_01/independent_data
    python -m dunhuang.research.paired_benchmark_r59 verify --source outputs/paired_benchmark_r59_01/controls --out outputs/paired_benchmark_r59_01/independent_controls
    python -m dunhuang.research.paired_benchmark_r59 diagnostics --out outputs/paired_benchmark_r59_01/r58_diagnostics

所有目录须全新。源码/冻结依赖、数据/特征/权重SHA绑定；新进程从原始录制重建网格及数组，重新编码全部候选并复验分数NPZ字节。R58各C1/C2/C3路径完整重建内部得分与逐槽各项差值；所有动态比未知的节点单列UNKNOWN，旧零代价行为只审计，不改写封存结果。

训练与正式选路保持关闭。资格报告必须同时给出数据独立性、目标覆盖、正确音乐内容依赖与相对Frozen/Heuristic的音乐组配对区间。没有可核验的新独立评价人口时，不以历史Val优势授权新主模型；这不妨碍本轮客观实验。

## 独立质量评价准备

R58真实乐句及艺术偏好仍不可评。交付只听音频的边界/重音/重复/UNSURE记录入口和B0/B1/C1/C2同音乐盲评协议，至少两独立评价者、保留分歧；没有实际审核记录时human_reviewed=false。评价记录不用于本轮改权重。真实配对动作也不是唯一正确的完整编舞。
