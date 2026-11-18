"""Replay compares exact science, including full failed graphs, not wall-clock timings."""
import subprocess
from .common import *

def compare(a,b):
 for name in ('baseline.json','result.json','cache.json'):
  assert read(a/name)==read(b/name),str(a/name)
 for name in ('arrays.npz','cache_arrays.npz'):
  assert sha(a/name)==sha(b/name),str(a/name)
  with np.load(a/name) as x,np.load(b/name) as y:
   assert x.files==y.files
   for key in x.files:assert x[key].dtype==y[key].dtype and np.array_equal(x[key],y[key]),key

def run(out,controls=True):
 modes=['preview','main']+(['controls'] if controls else [])
 for mode in modes:compare(ROOT/mode,ROOT/'replay'/mode)
 for baseline in ('preview','main')+ (('controls',) if controls else ()):
  assert len(read(ROOT/baseline/'baseline.json')['records'])==18
 write(out,dict(exact_source_rebuild_replays=modes,deterministic_archive_bytes=True,full_failed_graphs_compared=True,original_heuristic_trajectories_unchanged=18,neural_training_steps=0,optimizer_calls=0,can_route=False))

def release(out):
 out=fresh(out);main=read(ROOT/'main/result.json');has_controls=(ROOT/'controls/result.json').exists();where=ROOT/('controls' if has_controls else 'main')
 from .evaluate import summarize
 summary=summarize(where);summary['gates']=dict(main['gates'],G3=summary['G3'],G4=False);summary['independent']=read(ROOT/'receipts/independent.json');summary['resources']={k:read(ROOT/k/'resources.json') for k in ('preview','main')+ (('controls',) if has_controls else ())};summary['code_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();write(out/'summary.json',summary)
 rows=['# R53 合法时间演法与连接机制实验','','零神经训练；原R47目录139单元、预算16、每槽10合法事件，3首历史曝光音乐，18任务。局部斜率0.75–1.33、G0/G1、R48筛查不变。','',str(summary['gates']),'','同一Warp保留完整来源、实际端点帧中心与源阶段锚点，仅变目标时间锚点。K-best从原离散DAG求解，随机K从同图的合法路径秩均匀抽样并包含W0，不扩展整数域。','',f"原固定Warp在18例的失败数：{summary['baseline_failed_tasks']}，恢复率分母固定18：{summary['recovery_rate_all_registered']:.3f}。原基线全部通过时不能宣称恢复了失败任务。",'','## 分组结果','',json.dumps(summary['music_groups'],ensure_ascii=False,indent=2),'','G3是预登记开发代理门，要求至少2首歌同时优于B0/B1/B2的未优化边界D3读数5%，Core D3不退化5%以上、节拍代理不下降0.02以上。不是独立音乐或人类自然性结论。只有3首曝光音乐，bootstrap仅作分组描述；G4未测试。','','构造器只读各三帧，但原D3筛查还读各第四帧，因此内部变速负对照固定各四帧，并绕过缓存直接复算。负对照不可构造时记录，不增加时钟节点。缓存主身份绑定完整Warp/源SHA/槽位/音乐/方法/Gap/骨架/参考；二级复用只在所有四帧读集逐字节相同时发生。完整失败证据均保留。','','B1/B2/B3有同量K候选及共同预计算图；各方法所需边数、独特边界读集、搜索访问数及时间单列。共享预计算不等同实际墙钟预算严格相等。Gap D3是优化项，不能作为独立收益证据；另报边界邻域、Core和节拍代理。没有World Root/Contact或Native24资格。','','## 独立复验','',json.dumps(summary['independent'],ensure_ascii=False,indent=2)]
 (out/'report.md').write_text('\n'.join(rows)+'\n',encoding='utf8');(ROOT/'index.html').write_text('<meta charset="utf-8"><h1>R53 合法演法与连接机制</h1><p>无神经生成，Root/Contact未知。</p><p><a href="preview/viewer/index.html">先看：同事件多Warp与连接矩阵</a></p><p><a href="main/viewer/index.html">固定Warp与联合Warp</a></p>'+('<p><a href="controls/viewer/index.html">等K对照与K=1/2/4</a></p>' if has_controls else '<p>P2未进入对照：相关门未通过。</p>'),encoding='utf8')
 files=[]
 for mode in modes_for_seal(has_controls):
  for p in sorted((ROOT/mode).rglob('*')):
   if p.is_file():files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 for p in (ROOT/'index.html',out/'summary.json',out/'report.md',ROOT/'receipts/independent.json'):
  files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 write(out/'seal.json',dict(files=files))

def modes_for_seal(controls):return ['preview','main','replay']+(['controls'] if controls else [])
def recheck(out):
 files=read(ROOT/'publication/seal.json')['files']
 for f in files:assert Path(f['path']).stat().st_size==f['bytes'] and sha(f['path'])==f['sha256']
 write(out,dict(files=len(files),all_hashes_verified=True,optimizer_calls=0))
