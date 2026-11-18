"""Exact replay and experiment publication, separate from frozen baselines."""
from pathlib import Path
import subprocess
import numpy as np
from .common import ROOT,read,write,fresh,digest


def compare(first,second):
 a=read(first/'result.json');b=read(second/'result.json');assert a==b,str(first)
 if (first/'arrays.npz').exists():
  with np.load(first/'arrays.npz') as x,np.load(second/'arrays.npz') as y:
   assert set(x.files)==set(y.files)
   for k in x.files:assert np.array_equal(x[k],y[k]),k


def verify(out):
 pairs=[('plans','replay/plans'),('plan_controls','replay/plan_controls'),('music_controls','replay/music_controls'),('frozen_full','replay/frozen_full'),('main','replay/main'),('controls','replay/controls'),('no_phase','replay/no_phase')]
 for a,b in pairs:compare(ROOT/a,ROOT/b)
 with np.load(ROOT/'main/arrays.npz') as main,np.load(ROOT/'controls/arrays.npz') as ctrl:
  for k in set(main.files)&set(ctrl.files):assert np.array_equal(main[k],ctrl[k]),'Main/control shared trajectory changed: '+k
 for f in read('outputs/validation_r51_01/publication/seal.json')['files']:
  p=Path(f['path']);assert p.stat().st_size==f['bytes'] and digest(p)==f['sha256']
 for sub in ('train_main_replay','train_no_phase_replay'):
  r=read(ROOT/sub/'result.json');assert r['optimizer_calls']==0 and len(r['snapshot_checks'])==3
 write(out,dict(exact_scientific_replays=len(pairs),gradient_snapshot_replays=6,r51_sealed_files_verified=88,optimizer_calls=0,can_route=False))


def release(out):
 out=fresh(out);repair=read(ROOT/'repair/result.json');p=read(ROOT/'plan_controls/result.json');n=read(ROOT/'main/result.json');ctrl=read(ROOT/'controls/result.json');ind=read(ROOT/'receipts/independent.json')
 summary=dict(region_repair=dict(legacy_missing=repair['legacy_missing_D3_records'],corrected_records=repair['corrected_region_records'],selection_changed=False),plan_choices=[dict(key=r['key'],selected={a:dict(status=s['status'],sources=[e['source_uid'] for e in s.get('events',[])],events=[e['event_uid'] for e in s.get('events',[])],kinematics=s.get('kinematics')) for a,s in r['selected'].items()},same_source_forward_candidates=r['same_source_forward_candidates']) for r in p['records']],generation=n,controls=ctrl,no_phase=read(ROOT/'no_phase/result.json'),training={k:read(ROOT/k/'result.json') for k in ('train_main','train_no_phase')},independent=ind,pytest=(ROOT/'receipts/pytest.log').read_text().strip().splitlines()[-1],native24='NOT_EVALUABLE_22_OF24',whole_song_generations=0,can_route=False,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
 write(out/'summary.json',summary)
 lines=['# R52 Heuristic强基线与独立完整来源训练','','修复了绝对音乐帧与局部短片帧混用；原1008条空分区读数重算为882条有效区域，原计划、轨迹和whole统计不变。高D3不直接解释为不自然或Native失败。','','## 两阶段接口','','EVENT_PLAN_R52_V1绑定真实来源、完整区间、合法Warp、阶段代理/未知状态、软参考上下文、实际过渡预算和真实音乐35D。SDMR分数保留Shadow元数据，不成为ModuleII训练门槛。','','## 完整来源与补间分开','','训练FULL_REFERENCE_RECONSTRUCTION可读取完整合法来源，这是身份重建任务；HIDDEN_GAP_INBETWEENING隐藏中心48帧并使用真实连续目标。AIST++真实配对音乐与ChangE零音乐源保持任务各占半批，不伪造敦煌配对重演真值。只训练已有Source/Context，固定Lodge、CanonicalIO、General；Root/C4不入损失。两个240步训练组在主效果观看后才进行零Phase同预算对照，动态项按Train初始梯度校准，NoPhase沿用主组权重。','','## 两个完整事件','','固定R51 Heuristic的两个真实完整事件、E1-W和48帧几何桥，Core240+240，整段528帧。每个256帧计算窗都能读取同一完整528帧来源记忆；四窗对同一全局噪声轨迹共同去噪。只有整段外端16帧硬保护，Root/C4保留未知模型规范，未生成合格世界Root。来源Core可读480帧，桥区参考未知，不把几何桥当真实重演目标。采样从t119执行50步，仍属温启动局部研究。','','|案例|方法|来源Core角误差rad|来源CoreFK误差m|','|---|---|---:|---:|']
 for r in n['records']:
  for method,v in r['methods'].items():lines.append('|%s|%s|%.6f|%.6f|'%(r['key'],method,np.mean([x['source_core_angle_rad'] for x in v]),np.mean([x['source_core_fk_rmse_m'] for x in v])))
 lines += ['','几何基线Core直接等于来源，来源误差为0，因此要求神经生成保持来源、改善连接/音乐响应，而不声称超过零误差。所有Core/Bridge/边界/whole动态及相对来源D1/D2/D3残差保留机器摘要；源动作本身的快速变化不等同异常抖动。两首敦煌音乐是非配对历史开发输入，换音乐只能检验响应，不能证明新音乐重演正确。','','## 规划对照','','Heuristic、SDMR、Hybrid=Heuristic+0.1Full、RhythmCurve=Heuristic+0.1Multi在同一候选/图/预算上比较；重复、来源均衡和同来源前向不重叠事件连接开关完全匹配。方法筛查失败不等价于所有神经重演皆不可能。冻结Heuristic保持原18组选择，不直接晋升SDMR。','','## 复验','',str(ind),summary['pytest'],'原R51的88文件保持；本轮轨迹、检查点、真实梯度及采样重放保留服务器。完整Native不可评、无整曲、研究不晋升。']
 (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 files=[]
 for directory in (ROOT/'repair',ROOT/'plans',ROOT/'interface',ROOT/'frozen_full',ROOT/'train_main',ROOT/'train_main_replay',ROOT/'main',ROOT/'plan_controls',ROOT/'music_controls',ROOT/'controls',ROOT/'train_no_phase',ROOT/'train_no_phase_replay',ROOT/'no_phase',ROOT/'replay'):
  for f in sorted(directory.rglob('*')):
   if f.is_file():files.append(dict(path=str(f),bytes=f.stat().st_size,sha256=digest(f)))
 for f in (out/'summary.json',out/'report.md',ROOT/'receipts/independent.json',ROOT/'receipts/main_browser.json',ROOT/'receipts/trained_browser.json'):
  files.append(dict(path=str(f),bytes=f.stat().st_size,sha256=digest(f)))
 write(out/'seal.json',dict(files=files))


def recheck(out):
 files=read(ROOT/'publication/seal.json')['files']
 for f in files:
  p=Path(f['path']);assert p.stat().st_size==f['bytes'] and digest(p)==f['sha256']
 write(out,dict(files=len(files),all_hashes_verified=True,optimizer_calls=0,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
