# Paper 1 只读自动评价

冻结输入为Paper1-Support-Q01的30个显示位置。此分支不重选动作、不改变来源／时钟／G1，也不训练新网络。

复用Music Events固定Beat This!、All-In-One和R64目录，保留原秒数、浮点帧、取整支持和SHA。BAS参照MACE官方固定版本的3D数值函数，明确改编为30 FPS、身体相对22关节；速度低谷平滑σ=5帧，音乐到动作的高斯距离σ=0.1秒。空事件返回不可评。它不是原论文2D/24关节排行榜指标，也不等于速度峰代理。

先运行并回传观看包，浏览器验收之后才运行受控时钟／动作验证：
```
PYTHONPATH=src python -m unittest dunhuang.research.paper1_auto_eval.test_contracts
PYTHONPATH=src python -m dunhuang.research.paper1_auto_eval.experiment
PYTHONPATH=src python -m dunhuang.research.paper1_auto_eval.view preview
# 实际浏览器验收回执绑定preview/manifest后：
PYTHONPATH=src python -m dunhuang.research.paper1_auto_eval.controls
PYTHONPATH=src python -m dunhuang.research.paper1_auto_eval.view controls
PYTHONPATH=src python -m dunhuang.research.paper1_auto_eval.verify
```

官方BAS参考快照应放在私有 `outputs/paper1_auto_eval_01/reference/mace_metric.py`，SHA已冻结。只抽取BA/get_db两函数在合成24关节数据上检查数值一致性；不导入完整第三方项目或模型。

18个旧比较保留并分类：VALID_DISTINCT才进入方法偏好统计，VALID_IDENTICAL只作平局控制，UNAVAILABLE不能评分。匿名次序复用旧私有映射，使用新pack ID；静音自然性与音乐协调分开。现有GTW/R69独立名册合同仍有效。导入：
```
PYTHONPATH=src python -m dunhuang.research.paper1_auto_eval.reviews --admin outputs/paper1_auto_eval_01/preview/admin/map.json --roster <独立名册> --reviews <实际本人记录> --out <全新统计路径>
```

音乐时移为±0.25/±0.5秒，不循环；正确音乐的查询事件身份／数量在所有偏移下相同。ChangE的正确条件只指登记音乐，不是原始配对真值。另行重算已有六组历史Train真实配对的时钟诊断，不称未见泛化。

抖动、非法局部跳时、关节偏移、重复姿态和合法时间调整分开记录；合法保持延长没有预设艺术错误标签。源外写入、非法时钟和合成动作永不进入正式候选池。指标不合成艺术总分。

MotionCritic只做表示资格审查：原始55关节到SMPL24手部映射、坐标变换，以及生成G1世界Root尚未验证。未补零、未下载权重、未推理。BAP缺少登记音乐目标锚点，PFC/足滑/Native24缺少真实世界合同，FID/DIV缺少足够独立同域人口，均保持不可评。非退化来源×时钟仅作前瞻登记；没有评价有效性与独立质量证据时不启动联合决策、整曲或R74。
