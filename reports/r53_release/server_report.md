# R53 合法时间演法与连接机制实验

零神经训练；原R47目录139单元、预算16、每槽10合法事件，3首历史曝光音乐，18任务。局部斜率0.75–1.33、G0/G1、R48筛查不变。

{'G1': True, 'G2': True, 'P2_investment_gate': True, 'G3': False, 'G4': False}

同一Warp保留完整来源、实际端点帧中心与源阶段锚点，仅变目标时间锚点。K-best从原离散DAG求解，随机K从同图的合法路径秩均匀抽样并包含W0，不扩展整数域。

原固定Warp在18例的失败数：0，恢复率分母固定18：0.000。原基线全部通过时不能宣称恢复了失败任务。

## 分组结果

[
  {
    "song": 0,
    "comparisons": {
      "B0": {
        "cases": 6,
        "boundary_ratio": 1.011325470509141,
        "core_ratio": 0.9670912771693368,
        "beat_delta": -0.07159394228083059,
        "qualified": false
      },
      "B1": {
        "cases": 6,
        "boundary_ratio": 1.0,
        "core_ratio": 1.0,
        "beat_delta": 0.0,
        "qualified": false
      },
      "B2": {
        "cases": 6,
        "boundary_ratio": 0.9264194500135018,
        "core_ratio": 1.0035141055212524,
        "beat_delta": -0.015290783747833306,
        "qualified": true
      }
    },
    "registered_proxy_pass": false
  },
  {
    "song": 1,
    "comparisons": {
      "B0": {
        "cases": 6,
        "boundary_ratio": 1.0831303576012914,
        "core_ratio": 1.032966450843219,
        "beat_delta": 0.08289222838603215,
        "qualified": false
      },
      "B1": {
        "cases": 6,
        "boundary_ratio": 1.0,
        "core_ratio": 1.0,
        "beat_delta": 0.0,
        "qualified": false
      },
      "B2": {
        "cases": 6,
        "boundary_ratio": 0.964152957456101,
        "core_ratio": 1.0434539951807624,
        "beat_delta": 0.05811262383729598,
        "qualified": false
      }
    },
    "registered_proxy_pass": false
  },
  {
    "song": 2,
    "comparisons": {
      "B0": {
        "cases": 6,
        "boundary_ratio": 0.9306300788722729,
        "core_ratio": 0.9725422226223154,
        "beat_delta": -0.03142634931430255,
        "qualified": false
      },
      "B1": {
        "cases": 6,
        "boundary_ratio": 0.9435255291949156,
        "core_ratio": 0.9775939218305587,
        "beat_delta": -0.003330339402786949,
        "qualified": true
      },
      "B2": {
        "cases": 6,
        "boundary_ratio": 0.9970782093036289,
        "core_ratio": 1.0037119590760966,
        "beat_delta": -0.017818218150554833,
        "qualified": false
      }
    },
    "registered_proxy_pass": false
  }
]

G3是预登记开发代理门，要求至少2首歌同时优于B0/B1/B2的未优化边界D3读数5%，Core D3不退化5%以上、节拍代理不下降0.02以上。不是独立音乐或人类自然性结论。只有3首曝光音乐，bootstrap仅作分组描述；G4未测试。

构造器只读各三帧，但原D3筛查还读各第四帧，因此内部变速负对照固定各四帧，并绕过缓存直接复算。负对照不可构造时记录，不增加时钟节点。缓存主身份绑定完整Warp/源SHA/槽位/音乐/方法/Gap/骨架/参考；二级复用只在所有四帧读集逐字节相同时发生。完整失败证据均保留。

B1/B2/B3有同量K候选及共同预计算图；各方法所需边数、独特边界读集、搜索访问数及时间单列。共享预计算不等同实际墙钟预算严格相等。Gap D3是优化项，不能作为独立收益证据；另报边界邻域、Core和节拍代理。没有World Root/Contact或Native24资格。

## 独立复验

{
  "exact_source_rebuild_replays": [
    "preview",
    "main",
    "controls"
  ],
  "deterministic_archive_bytes": true,
  "full_failed_graphs_compared": true,
  "original_heuristic_trajectories_unchanged": 18,
  "neural_training_steps": 0,
  "optimizer_calls": 0,
  "can_route": false
}
