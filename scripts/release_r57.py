"""Export completed R57 attribution, keeping diagnostics separate from qualification."""
import argparse,csv,io,json,hashlib,subprocess,runpy
from pathlib import Path
import numpy as np
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):Path(p).write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def csvput(p,rows):
 f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);Path(p).write_bytes(f.getvalue().encode('utf8'))
def codecheck(result):
 b=result['binding'];commit=b['commit']
 for name,digest in b['code'].items():assert hashlib.sha256(subprocess.check_output(['git','show',commit+':src/dunhuang/research/tempo_attribution_r57/'+name])).hexdigest()==digest
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('outputs/tempo_attribution_r57_01'));parser.add_argument('--out',type=Path,default=Path('reports/r57_release'));args=parser.parse_args();root,out=args.root,args.out;assert not out.exists();out.mkdir(parents=True)
 main=read(root/'main/result.json');controls=read(root/'controls/result.json');sdmr=read(root/'sdmr/result.json');repair=read(root/'repair/result.json')
 for result in (main,controls,sdmr):codecheck(result)
 for stage in ('main','controls','sdmr','repair'):
  replay=read(root/('independent_'+stage)/'result.json');assert replay['result_exact'] and replay['arrays_bytes_exact'] and replay['optimizer_calls']==0;put(out/('independent_'+stage+'.json'),replay)
 for stage in ('main','controls','repair'):assert 'exit_status=0' in (root/'receipts'/(stage+'_receipt.txt')).read_text()
 assert '705 passed, 1 skipped' in (root/'receipts/pytest_final_repair.log').read_text();assert read(root/'receipts/main_browser.json')['passed'];assert all(r['passes_original_screen'] for r in controls['G1_diagnostic']);assert repair['newly_retained_edges']==len(controls['G1_diagnostic'])
 # Preserve the predecessor seals without mixing different path conventions.
 parents={}
 for name in ('strong_baseline_r52_01','warp_coupling_r53_01','residual_sdmr_r54_01','residual_sdmr_r54_02'):
  rows=read(Path('outputs')/name/'publication/seal.json')['files']
  for row in rows:assert sha(row['path'])==row['sha256'],row['path']
  parents[name]=len(rows)
 validate=runpy.run_path('scripts/release_r55_and_song.py')['validate_seal']
 for name in ('clock_residual_r55_01','clock_residual_r55_02','whole_song_geometry_01','whole_song_geometry_02','music_structure_r56_02'):parents[name]=validate(Path('outputs')/name)
 put(root/'receipts/parent_seals.json',dict(unchanged=True,files=parents))
 A1=[]
 for row in controls['paired_events']:
  v={k:row[k] for k in ('song','slot','event_uid','source_uid')};v.update(source_start=row['source_interval'][0],source_end=row['source_interval'][1]);v.update({'Native_D%d_Q95'%k:row['native']['D%d_Q95'%k] for k in (1,2,3)})
  for arm in ('B0','B1'):
   for kind in ('linear','warp'):
    q=row[arm+'_'+kind];v.update({arm+'_'+kind+'_D%d_Q95'%k:q['metrics']['D%d_Q95'%k] for k in (1,2,3)});v[arm+'_'+kind+'_A3']=q['amplification']['3']['ratio'];v[arm+'_'+kind+'_knot_mass']=q['metrics']['knot_D3']['squared_mass_fraction']
  q=row['transport_B0_shape_to_B1'];v['transport_constructible']=q is not None;v['transport_D3']=q['metrics']['D3_Q95'] if q else None;v['uniform_continuous_cubic_prediction']=row['continuous_uniform_speed_cubic_prediction'];A1.append(v)
 csvput(out/'event_dynamic_profiles.csv',A1);put(out/'factorial.json',controls['factorial']);put(out/'not_common_legal.json',controls['not_common_legal']);put(out/'G1_diagnostic.json',controls['G1_diagnostic']);put(out/'G1_repair.json',repair);put(out/'gradient_records.json',sdmr['gradient_records']);put(out/'open_pool.json',sdmr['open_pool']);put(out/'rescore_group_metrics.json',sdmr['rescore_group_metrics'])
 dynamics={}
 for arm in ('B0','B1'):
  rows=controls['paired_events'];dynamics[arm]=dict(source_normalized_A3_median=float(np.median([r[arm+'_warp']['amplification']['3']['ratio'] for r in rows if r[arm+'_warp']['amplification']['3']['ratio'] is not None])),warp_over_uniform_D3_median=float(np.median([r[arm+'_warp']['metrics']['D3_Q95']/r[arm+'_linear']['metrics']['D3_Q95'] for r in rows if r[arm+'_linear']['metrics']['D3_Q95']>1e-8])),knot_squared_mass_median=float(np.median([r[arm+'_warp']['metrics']['knot_D3']['squared_mass_fraction'] for r in rows if r[arm+'_warp']['metrics']['knot_D3']['squared_mass_fraction'] is not None])))
 rescored={};differences={};val=sdmr['rescore_group_metrics']['val_cross_music']
 for name,groups in val.items():rescored[name]={metric:float(np.mean([m[metric] for m in groups.values()])) for metric in ('recall_at1','recall_at5','mrr')}
 for name in ('Conditional_TeacherRescore','NoClock_TeacherRescore'):
  differences[name]={}
  for baseline in ('Frozen','Heuristic'):
   groups=sorted(val[name].keys()&val[baseline].keys());v=np.array([val[name][g]['recall_at1']-val[baseline][g]['recall_at1'] for g in groups]);rng=np.random.default_rng(55055);draw=v[rng.integers(len(v),size=(5000,len(v)))].mean(1);differences[name][baseline]=dict(mean=float(v.mean()),ci95=np.quantile(draw,[.025,.975]).tolist(),music_groups=len(v),development_exposed=True)
 openpool={}
 for name in ('Frozen','Heuristic'):
  rows=[r for r in sdmr['open_pool'] if r['method']==name];by={g:[r for r in rows if r['group']==g and r['source_clock_target_metrics'] is not None] for g in sorted({r['group'] for r in rows})};openpool[name]=dict(queries=len(rows),music_groups=len(by),pool_size_range=[min(r['pool_size'] for r in rows),max(r['pool_size'] for r in rows)],metrics={key:float(np.mean([np.mean([r['source_clock_target_metrics'][key] for r in group]) for group in by.values() if group])) for key in ('recall_at1','recall_at5','mrr')},scope='HISTORICALLY_MATERIALIZED_FEATURE_POOL_NOT_FULL_RAW_DATABASE; PHYSICAL_CLOCK_TARGET_NOT_ARTISTIC_TRUTH')
 summary=dict(source_bindings={k:v['binding'] for k,v in [('main',main),('controls',controls),('sdmr',sdmr)]},A0=controls['frozen_A0'],shared_events=len(A1),unshared_event_slots=len(controls['not_common_legal']),shared_factorial_slots=sum(r['status']=='SHARED_SUPPORT_FACTORIAL_DECOMPOSITION' for r in controls['factorial']),total_slots=39,factorial_summary=controls['summary'],dynamic_summary=dynamics,transport_constructible=sum(r['transport_constructible'] for r in A1),G1_recovered=repair['newly_retained_edges'],gradients=sdmr['gradient_summary'],rescore_val=rescored,rescore_paired_differences=differences,open_pool=openpool,main_cases=len(main['records']),parents_unchanged=parents,training_steps=0,optimizer_calls=0,new_independent_music=0,human_reviewed=False,phrase_truth='NOT_EVALUABLE',native24='NOT_EVALUABLE',production_changed=False,operational_qualified=False,P1_training='NOT_STARTED; SCORE_AUDIT_NOT_NEW_TRAINING',A3_joint_planning='NOT_STARTED; SHARED_SUPPORT_ATTRIBUTION_IS_PARTIAL');put(out/'summary.json',summary)
 root.joinpath('index.html').write_bytes(('<!doctype html><meta charset="utf-8"><style>body{max-width:900px;margin:50px auto;padding:20px;font:19px system-ui;background:#f4f1e8;color:#173e36}a{display:block;padding:20px;margin:15px;background:white}</style><h1>R57 · 同源时间演法与归因</h1><a href="main/viewer/index.html">八个同来源四栏交互 · 逐帧／倍速</a><a href="main_video/index.html">八段实际轨迹 MP4</a><p>原速来源无配对音乐真值；其他栏为合法重定时，不是Diffusion。结束后隐藏实际支持，不循环。</p><p>音乐结构收益与动作自然性分开；身体22关节Root-relative，Root／接触未知，Native24不可评。</p>').encode())
 for p in (root/'receipts').iterdir():
  if p.is_file() and p.suffix in ('.json','.txt','.log','.sh'):(out/p.name).write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
 seal=runpy.run_path('scripts/release_r56.py')['seal'];rows=seal(root);put(out/'seal.json',dict(files=len(rows),manifest_sha256=sha(root/'artifacts_manifest.json')))
 lines=['# R57：时间演法、来源选择与 G1 方法恢复','','先交付八个共同合法首槽事件的真实原速／B0／结构槽匀速／结构槽E1-W轨迹、八段MP4和浏览器验收，再运行三首历史音乐39槽位审计。没有新独立音乐、人工乐句标注或艺术偏好；主视频不是神经生成。','','## 同源与同槽位归因','','全部16个冻结单位中，共取得298个事件×槽位共同合法实例，79个不共同合法实例明确排除；同一事件比较原速、匀速、原E1-W及B0时间形状整体缩放到B1。只有145个缩放时间形状仍满足每段原[.75,1.33]斜率。选中B0/B1事件×两种槽位的2×2有25/39个槽位完整，其余14个保留缺失，不制造反事实。','','|历史音乐|完整共享槽位|来源选择贡献|时间安排贡献|交互项|Core Q95总差|','|---|---:|---:|---:|---:|---:|']
 for song,r in controls['summary'].items():
  m=r['mean_factorial'];lines.append('|dunhuangwu'+str(int(song)+2)+'|'+str(r['shared_slots'])+'/13|'+ '|'.join(format(m[k],'.3f') for k in ('selection','timing','interaction','total'))+'|')
 lines+=['','来源与时间使用对称因子分解，各自包含一半交互，总和精确等于共享槽位Core D3 Q95差。单位m/s³，按完整共享槽位的描述均值；不是全片Q95的因果分解，前两首时间贡献较大、第三首来源贡献较大不能外推到缺失14槽。共同合法人口298实例也不是298个独立来源。','','|同源指标中位数|B0|B1|','|---|---:|---:|']
 for key in ('source_normalized_A3_median','warp_over_uniform_D3_median','knot_squared_mass_median'):lines.append('|'+key+'|'+format(dynamics['B0'][key],'.6f')+'|'+format(dynamics['B1'][key],'.6f')+'|')
 lines+=['','源归一化A3是Core Q95除以相同来源原速Q95，近静态分母保持未定义。分段线性时间图的锚点附近差分与其他Core区域分别记录；匀速三次方关系仅为连续机制预测，离散SO(3)重采样与FK差分不保证精确缩放。不能把高D3直接叫不自然，也不能据中位数断言所有局部片段均无抖动。详细活动量、D2/D3和结点差分见event_dynamic_profiles.csv；完整时间图留封存原始结果。','','## 根据真实证据修正 G1 调用顺序','','R56的11条G0 Ambiguous SO3 chart异常边逐条独立运行原G1，11/11通过原图域和D2/D3/肩腕数值筛查。新增geometry_recovery_r57：只捕获这一指定G0错误，仍继续原G1，其他错误照常抛出。所有11条修复轨迹与前置只读诊断逐值一致，新进程复验一致；原G0/G1、阈值、R56冻结图和正式Planner不改。这里恢复的是当前方法的合法候选，不是物理合格或整体舞蹈质量提升。','','## SDMR 教师评分与梯度审计','','40个优化音乐组各选一个固定UID任务；Frozen初始、Conditional和NoClock的0/120/239预更新快照共280条读数，初始状态重复不能算独立样本。记录Rank／Clock和教师保留Rank／Clock的余弦、模长、0.5权重后的模长比及梯度哈希。余弦在零附近、不同阶段正负均存在；这不是Clock导致退化的因果证明，不据此引入PCGrad或调权重。','','|历史曝光Val跨音乐|R@1|R@5|MRR|','|---|---:|---:|---:|']
 for name,r in rescored.items():lines.append('|'+name+'|'+ '|'.join(format(r[k],'.4f') for k in ('recall_at1','recall_at5','mrr'))+'|')
 lines+=['','TeacherRescore仅为已有R55权重的F+(旧分数−H)代数重评分，没有重新训练教师保留模型。初始评分等于Frozen而非H的性质已有梯度测试；NoClock重评分56.67%与Frozen52.22%的差值仍需看6个音乐组配对区间，见summary.json，不得当作新方法泛化收益。训练、校准、历史Val仍保留原身份与录制内／音乐组间均衡。','','开放池不依赖已知正例或H分箱：使用同角色、同帧数、不同原动作SHA的全部已落盘特征，UNKNOWN保留在排序池中。18查询、6个历史Val音乐组、70–243候选；Frozen和Heuristic源时钟目标R@1均为11.11%。这不是艺术适配准确率；原特征落盘人口此前受任务构建影响，因此尚不等于完整原始数据库检索，也不是独立盲测。候选范围、R@5、MRR与UNKNOWN数量见open_pool.json及summary.json。FineDance不在本轮冻结人口内，不声称跨数据集迁移。','','## 复验、交付与范围','','主效果、298实例归因、11边诊断、SDMR全部梯度哈希／分数NPZ和G1修复均新进程精确重建，优化器调用0；最终705回归通过、1跳过（含历史5警告）。九个父代共857个封存文件复核未变，本轮科学产物另行封存。主效果使用5981066，完整审计使用3e50307，G1修复使用28d707d；版本均先推GitHub再服务器tmux。两项静态修正发生于相应审计开始前：录制均衡统计与绝对音乐时钟缓存身份；未按结果调权重。','','本轮按方案优先完成P0与直接可证的方法调用修正，没有启动A3联合时长规划或SDMR P1新训练；本轮Val重评分不用于挑选新训练配置。继续保留B0强基线，B1/B3与SDMR都不升级正式资格；世界Root／接触未知、Native24不可评。','','本机统一入口：`D:/DunhuangDanceGen/outputs/tempo_attribution_r57_01/index.html`；四栏交互：`main/viewer/index.html`；八段视频：`main_video/index.html`（均相对于该入口）。']
 (out/'report.md').write_bytes(('\n'.join(lines)+'\n').encode());put(out/'public_manifest.json',dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]));print('R57_EXPORT',len(rows),sum(parents.values()))
if __name__=='__main__':main()
