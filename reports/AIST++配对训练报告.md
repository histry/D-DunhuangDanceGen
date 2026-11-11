# AIST++ 官方记录配对与首轮训练汇总

本轮先发布实现，再在服务器tmux中完成校准、配对预检、批量Jukebox、训练和独立重放。主实验固定提交为86fa32bd35015db82f01ec4ce02f815070280771；独立复核与最终完整测试为a910910f31682d92edfa0511e165305aa0035f99。主流水线、复核及测试退出码均为0。

## 来源与时钟

完整来源台账覆盖1,363条AIST++动作，其中1,166条基础舞通过，197条高级舞因实际共享编舞身份尚未明确仍排除。没有技术/时钟失败，也没有静默按下载成功人口选取数据。

使用官方精修视频CSV、基础编舞工作簿、AIST++视角映射与原始动作/音频SHA。官方精修起点约定见[精修规则](https://aistdancedb.ongaaccel.jp/how_to_refine_raw_videos/)；[AIST++下载说明](https://google.github.io/aistplusplus_dataset/download.html)给出60FPS注释及对应关系。逐条核对[官方完整CSV](https://aistdancedb.ongaaccel.jp/database_download/)的文件大小及身份字段。

每条准入视频音轨与原始音乐在8kHz上做三个3秒锚点相关；固定NCC≥0.98、残差/锚间差≤1/60秒、时长差≤34ms。1,166条offset均为0秒，最小锚点NCC为0.9975809849363375，最大锚点残差0，最大时长差约13.333ms。428条周期性候选结合官方起点和前缀匹配固定零点，不宣称相关峰唯一。没有按动作节拍调相位或时间拉伸。

基础编舞键使用[官方工作簿](https://aistdancedb.ongaaccel.jp/data/choreo.xlsx)的(genre,ch)联合键。高级舞存在跨舞者共享编舞，编号不足以确定独立性，见[官方数据库结构](https://aistdancedb.ongaaccel.jp/database_structure/)。

## 清单、特征与训练

完整配对清单为2,418个150帧窗口：Train1,480、Val481、Test457。保持历史角色和150/75网格；原始录制、音乐字节身份、官方基础编舞键、数据集舞者ID跨角色互斥。通用动作名称仍跨角色共享，不能推出语义或统计独立性；历史曝光和EDGE预训练曝光不构成新盲测。

目标147D来自拟合pose/root，4D Contact为派生量。目标/时钟/身份预检通过后才提取真实Layer66 Jukebox；按音频SHA、精确采样范围和提取器去重得到127份150×4800 FP32数组。每个目标保留独立收据。完整特征预检通过后才训练，不补零、不循环、不改Scaler。Test只做资产准备，不参与Adapter拟合或权重选择。

冻结EDGE，仅训练PrimitiveAdapter。固定300步、seed20261111、batch2、AdamW学习率1e-4、梯度裁剪1；每50步评价固定8个Val窗，预定规则选出第300步。模型参数前后哈希相同。

| 固定Val隐藏通道补全对照 | MSE |
|---|---:|
| zero_adapter | 0.01162724118330516 |
| observed_source | 0.011398248922584267 |
| condition_removed | 0.011400366929592565 |

约1.97%的补全MSE下降不证明音乐演绎；移除来源参考几乎不变，来源参考利用仍弱。训练状态为PAIRED_COMPLETION_TRAINED_UNQUALIFIED，music_qualified/full_motion_qualified均false。未执行新Native生成，默认生成路径和Native阈值不变。

## 复核与保留证据

服务器完整测试96 passed、2 warnings。三角色代表时钟重算与保存结果一致，两份独立小批次Jukebox与全量新提取逐元素一致。新进程加载固定权重重放三个验证对照，MSE最大差0、优化器调用0、模型及Adapter参数不变。

完整来源台账、配对清单、目标、特征、权重及逐条回执保留在本地和服务器。8,849个交付输出文件、738,994,767字节完成逐文件SHA核验。完整报告位于本地忽略目录outputs/aist_delivery_review_01；原始运行产物保留在outputs/aist_*_01。

GitHub仅发布实现、汇总和[六次训练指标](aist_training_metrics.csv)。含原始服务器路径的完整台账/回执未发布；原始数据、特征与权重继续作为忽略资产。临时传输ZIP和bundle已删除，SSH使用内存连接，没有创建临时SSH辅助文件。

阶段入口、严格预检和独立重放命令见[流水线说明](../docs/aist_pairing_pipeline.md)。尚待解决高级舞真实共享编舞映射、弱参考响应和自由生成Native质量，不能将本轮补全训练升级为整曲质量结论。
