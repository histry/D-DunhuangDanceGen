# R53 合法时间演法—连接耦合研究

本轮是独立 CPU 机制实验，不训练 SDMR、Lodge 或 EDGE。正式规划和 R52 的18条强基线输出保持不变。实现位于 `src/dunhuang/research/warp_coupling_r53/`，注册配置为同目录 `protocol.json`。

## 冻结人口与边界

- 基线：`984fd0b3826b64e6acbf9e977a29cef5eeaa02a4`；主实现：`f6dbdaf2a5b608baf2e71545cccbb3f82a003272`。
- 原R47目录139单元、注册预算16、每槽10个合法事件、Core240帧，事件数2/3、Gap24/48/72，3首已曝光敦煌音乐，共18任务。
- 沿用原E1-W离散节点和0.75–1.33局部斜率，K=4；Top-K与随机K均包含冻结W0，并匹配每事件的实际候选数量。线性对照同样属于原图。
- 不改原G0/G1或R48阈值，不添加来源/槽位。共同22关节的Root-relative FK不是完整世界Root、接触或Native24真值。
- 三首歌都曾用于开发，未见音乐与G4未测试；本轮无人工审核记录。

## 先观看，再对照

各输出目录必须不存在；不得覆盖失败产物。以下命令在仓库根目录、`PYTHONPATH=src`、单线程BLAS环境执行，服务器以tmux持久运行。

```bash
python -m pytest -q
python -m dunhuang.research.warp_coupling_r53 preview --out outputs/warp_coupling_r53_01/preview
# 回传viewer/protocol.json列出的原字节文件并验SHA；浏览器完成播放/逐帧/条件切换检查。
# 只将播放验收写入 receipts/preview_browser.json，不能记为人工舞蹈质量通过。
python -m dunhuang.research.warp_coupling_r53 main --out outputs/warp_coupling_r53_01/main
python -m dunhuang.research.warp_coupling_r53 examples --out outputs/warp_coupling_r53_01/examples
# 完成main观看验收，记录 receipts/main_browser.json；并检查P2_investment_gate。
python -m dunhuang.research.warp_coupling_r53 controls --out outputs/warp_coupling_r53_01/controls
```

观看页使用已保存的真实FK，原始采样无额外平滑。局部矩阵可以查看所有变体；拒绝桥留空。局部恢复案例按案例/事件ID排序，取每音乐/Gap首个恢复边，不按收益大小挑选。它们不代表原18任务由失败恢复。

## 独立重算

```bash
python -m dunhuang.research.warp_coupling_r53 preview --out outputs/warp_coupling_r53_01/replay/preview
python -m dunhuang.research.warp_coupling_r53 main --out outputs/warp_coupling_r53_01/replay/main
python -m dunhuang.research.warp_coupling_r53 controls --out outputs/warp_coupling_r53_01/replay/controls
python -m dunhuang.research.warp_coupling_r53 verify --out outputs/warp_coupling_r53_01/receipts/independent.json
python -m dunhuang.research.warp_coupling_r53 release --out outputs/warp_coupling_r53_01/publication
python -m dunhuang.research.warp_coupling_r53 recheck --out outputs/warp_coupling_r53_01/receipts/final_recheck.json
```

每阶段重新核验R51/R52封存、原始来源哈希、原E1-W、SO(3)重采样及18条R52输出逐位不变。重算链使用独立重建的父图，不读取原运行的桥结果作为重算答案。JSON全失败图及确定性NPZ字节均须完全一致，训练/优化器调用为0。

缓存的完整身份含来源、区间、目标槽位、音乐/结构、完整Warp、算法/参考/骨架及Gap。桥构造读各侧3帧，但原D3筛查读到第4帧，因此二级缓存必须绑定全部8帧精确字节；内部变速负对照固定各侧4帧并绕过缓存直接重算。

## 如何解释指标

G1要求真实采样动作变化；G2要求真实边界速度/加速度变化且桥数值或资格变化。投资门另要求资格翻转或超过max(1 m/s³,5%)的Gap D3变化。Gap D3用于搜索评分，不作为独立效果证据。

B0固定W0；B1先按局部活动匹配独立定Warp；B2随机K联合；B3 Top-K联合。另保留线性W3及K=1/2/4。相同事件与候选数不意味着同样墙钟开销：共享候选图、实际桥构造、读集复用、所需边数、搜索访问和耗时分别保存。

G3要求至少2首歌：B3相对B0/B1/B2均改善未用于搜索的边界邻域D3至少5%，Core D3不退化超过5%，节拍代理不下降超过0.02。该门只是开发代理资格，不能证明感知自然性或跨作品泛化。原18任务B0均通过，因此全局失败恢复分母固定18，恢复数为0；局部候选边恢复另报。

公开仅发布代码、协议、统计及回执；动作、完整候选图、音频和观看数据保留于本地与服务器输出目录。临时SSH传输脚本/Bundle清理，tmux实验脚本与科学日志作为回执保留。
