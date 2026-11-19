"""Publish completed R58 measurements without upgrading scientific qualification."""
import json,csv,io,hashlib,subprocess,runpy
from pathlib import Path
from collections import Counter
import numpy as np
from dunhuang.research.execution_planning_r58.common import ROOT,PARENT,PREVIOUS,CFG,read,write,sha
from dunhuang.research.execution_planning_r58.profiles import Assets
def put(p,v):Path(p).write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def csvput(p,rows):
 b=io.StringIO(newline='');w=csv.DictWriter(b,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);Path(p).write_bytes(b.getvalue().encode('utf8'))
def verify_code(result):
 b=result['binding'];commit=b['commit']
 for name,digest in b['science'].items():assert hashlib.sha256(subprocess.check_output(['git','show',commit+':src/dunhuang/research/execution_planning_r58/'+name])).hexdigest()==digest
 for name,digest in b['dependencies'].items():assert hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name])).hexdigest()==digest
def main():
 root=ROOT;out=Path('reports/r58_release');assert not out.exists();data=read(root/'controls/result.json');main=read(root/'main/result.json');recovery=read(root/'recovery/result.json');support=read(root/'support/result.json');uniform=read(root/'main/uniform_profiles.json');verify_code(main);verify_code(data);verify_code(recovery)
 for name in ('main','controls'):
  assert 'exit_status=0' in (root/'receipts'/(name+'_receipt.txt')).read_text()
  replay=read(root/('independent_'+name)/'result.json');assert replay['result_exact'] and replay['arrays_bytes_exact'] and replay['all_graphs_rebuilt_from_original_sources']
 for name in ('recovery','support'):assert read(root/('independent_'+name)/'result.json')['result_exact']
 assert '710 passed, 1 skipped' in (root/'receipts/pytest_final.log').read_text();assert read(root/'receipts/main_browser.json')['passed'];assert read(root/'review/protocol.json')['independent_records']==0
 assets=Assets();z=np.load(root/'controls/arrays.npz');byunit={u['unit_uid']:u for u in assets.bank};selected=[];records=[];metrics={}
 for r in data['records']:
  song,arm=r['song'],r['arm'];assert r['status']=='NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW';R=z[f'song{song}_{arm}_R'];P=z[f'song{song}_{arm}_P'];cost=[];activity=[]
  for slot,e in enumerate(r['events']):
   n,Q=assets.node(byunit[e['event_uid']],e['target_interval'],assets.pop.tracks[song]);assert n['warp']==e['time_warp'];a,b=e['target_interval'];error=float(np.max(abs(Q-R[a:b])));assert error==0.;cost.append(n['execution_cost']);activity.append(n['activity']);row=dict(song=song,arm=arm,slot=slot,event_uid=e['event_uid'],source_uid=e['source_uid'],source_start=e['source_interval'][0],source_end=e['source_interval'][1],target_start=a,target_end=b,duration=b-a,activity=n['activity'],execution_cost=n['execution_cost'],core_matrix_error=error,native_sha256=n['native_sha256'])
   for k in (1,2,3):row[f'D{k}_Q95']=n['metrics'][f'D{k}_Q95'];row[f'A{k}']=n['source_normalized'][str(k)]['ratio']
   selected.append(row)
  if arm in ('B0','B1'):
   assert np.array_equal(R,assets.z[f'song{song}_{arm}_R']) and np.array_equal(P,assets.z[f'song{song}_{arm}_P'])
  m=dict(whole=r['metrics']['kinematics']['whole'],boundary_proxy_F1=r['metrics']['phrase_boundary_proxy']['F1'],mean_execution_distortion=float(np.mean(cost)),mean_Core_activity=float(np.mean(activity)),whole_mean_activity=float(np.linalg.norm(np.diff(P.astype(float),axis=0,prepend=P[:1])*30,axis=-1).mean()),source_count=len(r['source_histogram']),unique_events=len(r['unit_histogram']),rotation_sha256=r['rotation_sha256'],source_core_error=r['metrics']['source_core_max_matrix_error']);metrics[song,arm]=m;records.append(dict(song=song,arm=arm,metrics=m,durations=[e['target_interval'][1]-e['target_interval'][0] for e in r['events']],event_uids=[e['event_uid'] for e in r['events']],search=r.get('search'),score=r['score']))
 comparisons={}
 for song in CFG['songs']:
  rr={r['arm']:r for r in records if r['song']==song};m={a:r['metrics'] for a,r in rr.items()};base=m['B0']['whole']['D1_Q95'];ratio=m['C2']['whole']['D1_Q95']/base if base>1e-8 else None
  comparisons[str(song)]=dict(C2_minus_C1_execution_distortion=m['C2']['mean_execution_distortion']-m['C1']['mean_execution_distortion'],C2_minus_C1_boundary_proxy=m['C2']['boundary_proxy_F1']-m['C1']['boundary_proxy_F1'],C2_minus_C1_whole_D3=m['C2']['whole']['D3_Q95']-m['C1']['whole']['D3_Q95'],C2_C1_same_rotations=rr['C2']['metrics']['rotation_sha256']==rr['C1']['metrics']['rotation_sha256'],C2_C1_changed_event_slots=sum(a!=b for a,b in zip(rr['C2']['event_uids'],rr['C1']['event_uids'])),C2_C1_changed_durations=sum(a!=b for a,b in zip(rr['C2']['durations'],rr['C1']['durations'])),C2_music_proxy_noninferior_to_B1=m['C2']['boundary_proxy_F1']>=m['B1']['boundary_proxy_F1']-CFG['proxy_boundary_noninferiority'],C2_B0_D1_ratio=ratio,C2_activity_diagnostic_in_registered_band=ratio is not None and CFG['activity_ratio_scope'][0]<=ratio<=CFG['activity_ratio_scope'][1],C2_minus_C3_boundary_proxy=m['C2']['boundary_proxy_F1']-m['C3']['boundary_proxy_F1'])
 graphs={};graphdata={}
 for song in CFG['songs']:
  p=root/('main' if song==CFG['main_song'] else 'controls')/'graphs'/str(song);g=read(p/'result.json');assert sha(p/'result.json')==data['graphs'][str(song)]['result_sha256'];graphdata[song]=g;lookup={n['id']:(k,n) for k,layer in enumerate(g['layers']) for n in layer};fixed=[e for e in g['edges'] if all(lookup[key][1]['target']==g['base_intervals'][lookup[key][0]] for key in (e['left'],e['right']))];graphs[str(song)]=dict(**data['graphs'][str(song)],fixed_subgraph_trials=len(fixed),fixed_subgraph_eligible=sum(e['eligible'] for e in fixed),G0_errors=sum(e['evidence']['G0_error'] is not None for e in g['edges']),same_graph_C1_C2_C3=True,compute_scope='C1_FIXED_SUBGRAPH; C2/C3_SHARE_ALL_GRAPH_BUILD_CALLS; NO_EQUAL_ACTUAL_COST_CLAIM_FOR_C1')
 missing=[]
 for r in support['records']:
  g=graphdata[r['song']];cells=[]
  for cell in r['cells']:
   matching=[n for n in g['layers'][r['slot']] if n['event_uid']==cell['event_uid']];cells.append(dict(**cell,R58_legal_nodes_in_same_layer=len(matching),R58_legal_targets_in_same_layer=[n['target'] for n in matching]))
  missing.append(dict(song=r['song'],slot=r['slot'],cells=cells))
 recovery_rows=[];seen=set();unique_added=0
 for r in recovery['records']:
  graphkey=assets.record(r['song'],r['arm'])['graph_key']
  if graphkey not in seen:unique_added+=len(r['added_edges']);seen.add(graphkey)
  recovery_rows.append({k:r[k] for k in ('song','arm','old_eligible','new_eligible','added_edges','old_path_capacity','new_path_capacity','selected_path_changed','old_events','new_events','old_score','new_score')})
 parents={}
 for name in ('strong_baseline_r52_01','warp_coupling_r53_01','residual_sdmr_r54_01','residual_sdmr_r54_02'):
  entries=read(Path('outputs')/name/'publication/seal.json')['files']
  for e in entries:assert sha(e['path'])==e['sha256']
  parents[name]=len(entries)
 validate=runpy.run_path('scripts/release_r55_and_song.py')['validate_seal']
 for name in ('clock_residual_r55_01','clock_residual_r55_02','whole_song_geometry_01','whole_song_geometry_02','music_structure_r56_02','tempo_attribution_r57_01'):parents[name]=validate(Path('outputs')/name)
 put(root/'receipts/parent_seals.json',dict(unchanged=True,files=parents))
 out.mkdir(parents=True);csvput(out/'selected_execution_profiles.csv',selected);duration=[]
 for r in uniform:
  row=dict(event_uid=r['event_uid'],source_uid=r['source_uid'],duration=r['duration'],legal=r['legal'],reason=r.get('reason'),activity=r.get('activity'),execution_cost=r.get('execution_cost'))
  for k in (1,2,3):row[f'D{k}_Q95']=r['metrics'][f'D{k}_Q95'] if r['legal'] else None;row[f'A{k}']=r['source_normalized'][str(k)]['ratio'] if r['legal'] else None
  duration.append(row)
 csvput(out/'uniform_duration_profiles.csv',duration);put(out/'missing_counterfactual_support.json',missing);put(out/'geometry_graph_comparison.json',recovery_rows);put(out/'comparison_differences.json',comparisons)
 for stage in ('main','controls','support','recovery'):put(out/('independent_'+stage+'.json'),read(root/('independent_'+stage)/'result.json'))
 for p in (root/'receipts').iterdir():
  if p.is_file() and p.suffix in ('.json','.txt','.sh','.log'):(out/p.name).write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
 hub='<!doctype html><meta charset="utf-8"><style>body{max-width:950px;margin:40px auto;padding:20px;font:19px system-ui;background:#f4f1e8;color:#173e36}a{display:block;padding:18px;background:white;margin:14px 0}</style><h1>R58 · 来源执行档案与整曲对照</h1><a href="main_video/B0_B1_C2_full.mp4">完整音乐三栏视频 · B0/B1/C2</a><a href="main/viewer/index.html">首曲主效果 · 共同音乐时钟与事件跳转</a><a href="controls/viewer/index.html">三首整曲五组固定对照</a><a href="profiles/viewer/index.html">来源执行档案 · 匀速曲线与实际E1-W测量</a><a href="review/viewer/index.html">A/B 盲评准备 · 尚无人类记录</a><p>22关节Root-relative几何研究预览；世界Root/接触未知、Native24不可评。自动音乐边界与执行代价不是人工乐句或自然性真值。</p>';root.joinpath('index.html').write_bytes(hub.encode())
 seal=runpy.run_path('scripts/release_r56.py')['seal'];sealed=seal(root);put(out/'seal.json',dict(files=len(sealed),manifest_sha256=sha(root/'artifacts_manifest.json')))
 summary=dict(records=records,comparisons=comparisons,graphs=graphs,uniform_entries=len(uniform),uniform_legal=sum(r['legal'] for r in uniform),source_units=16,missing_counterfactual_slots=len(missing),G1_unique_graphs=len(seen),G1_unique_added_edges=unique_added,G1_selected_plan_changes=sum(r['selected_path_changed'] for r in recovery_rows),timing_controls=data['timing_controls'],source_binding=data['binding'],main_binding=main['binding'],review_protocol=read(root/'review/protocol.json'),profiles_protocol=read(root/'profiles/viewer/protocol.json'),main_video=read(root/'main_video/result.json'),parents_unchanged=parents,neural_training_steps=0,optimizer_calls=0,sdmr_used=False,new_independent_music=0,human_reviewed=False,true_phrase_labels='NOT_EVALUABLE',native24='NOT_EVALUABLE',production_changed=False,operational_qualified=False);put(out/'summary.json',summary)
 lines=['# R58：来源执行档案与有限联合时长规划','','先交付并浏览器验收首曲120秒B0/B1/C2真实轨迹和视频，再执行三首历史音乐五组固定对照、原R56图G1恢复与全部独立重建。没有新神经训练、SDMR选路、独立未见音乐或人工评价结论。','','## 方法与预登记限制','','16个冻结单位分别测180–300帧的1936个匀速时长项；非法项不补齐。节点档案再实测当前真实音乐原E1-W的活动量、D1/D2/D3、相对原生来源的动态比、局部边界角速度/加速度、来源SHA和精确源帧映射。优化代价为三个阶次abs(log(Ak))/k均值，增加和降低均计价，近静态分母保持UNKNOWN；这不是自然性评价器。','','B0/B1为R56封存轨迹，逐数组精确保留。C1固定B1时长，只加入0.1执行变形代价。C2在六个交替边界[-18,0,18]帧格点联合选来源和时长，其他六个边界固定，仍为13个Core、12×48帧过渡、原[.75,1.33] E1-W与G0/G1数值筛查，使用R57指定G0异常后继续G1的修正。音乐边界权重1、长度0.1、Bridge0.01、原重复惩罚0.1/0.02固定；Beam128加显式C1合法路径保底，不宣称全局最优或任意联合时长。','','C3与C2共用完全相同候选图及计算结果，仅循环移动边界奖励。B1格点先验、真实音乐能量和E1-W拍点都仍保留；因此只检验共同图条件下增量边界奖励，不是完全移除音乐结构。C1固定子图与C2全图的实际计算量不同，详细数目见summary.json。全部权重先于主视频冻结，未按观看或对照结果调整。','','## 全部固定结果','','|音乐|方法|边界代理F1|执行变形代价|整曲D1 Q95|整曲D3 Q95|来源/事件|','|---|---|---:|---:|---:|---:|']
 for r in records:
  m=r['metrics'];lines.append('|dunhuangwu'+str(r['song']+2)+'|'+r['arm']+'|'+ '|'.join(format(v,'.5f') for v in (m['boundary_proxy_F1'],m['mean_execution_distortion'],m['whole']['D1_Q95'],m['whole']['D3_Q95']))+'|'+str(m['source_count'])+'/'+str(m['unique_events'])+'|')
 lines+=['','执行代价是优化目标读数，不是独立自然性；D1/D3是相对身体差分，不能直接叫Native物理判定。逐事件D1/D2/D3、A1/A2/A3、活动量、来源身份与实际目标区间见selected_execution_profiles.csv。所有选中Core均与原始来源在实际映射处逐矩阵一致，这是几何构造保持，不是学习式重演成功。','','## C2是否真的超过仅换来源？','','|音乐|C2/C1相同轨迹|来源改变槽数|时长改变槽数|执行代价差|边界F1差|D3差|','|---|---|---:|---:|---:|---:|---:|']
 for song,r in comparisons.items():lines.append('|dunhuangwu'+str(int(song)+2)+'|'+str(r['C2_C1_same_rotations'])+'|'+str(r['C2_C1_changed_event_slots'])+'|'+str(r['C2_C1_changed_durations'])+'|'+ '|'.join(format(r[k],'.6f') for k in ('C2_minus_C1_execution_distortion','C2_minus_C1_boundary_proxy','C2_minus_C1_whole_D3'))+'|')
 lines+=['','同轨迹或改来源已有相同效果时，不能声称联合时长必不可少。另按预登记报告C2相对B1边界代理下降不超过0.05、相对B0全片D1在[.8,1.25]的描述标志；这些不是艺术或物理资格门，不用于改权重或选择最优音乐。三首都已开发曝光；不把1936格点、节点、连接或13事件对当作独立音乐样本计算伪置信区间。','','## 缺失支持与G1恢复的独立影响','','R57缺失14槽逐一保留四个来源×时长单元，报告原比例、不可构造原因、登记替代时长，以及本轮有限格点中同槽位同来源的合法节点数；不是把缺失反事实补成观测。对R56九张原图、十二个旧方案，在完全相同旧槽位、候选、评分和Beam下重现原选路分数与身份，再恢复11条G1边，记录合法边数、精确分层图路径数、最终选路是否改变和真实整曲轨迹。路径容量是数值图容量，非物理可行性或艺术质量。细节见geometry_graph_comparison.json；相同选路时新旧整段R/P数组也要求精确一致。','','## 独立复验与交付','','首曲与三首对照均在新进程从原始动作重新构建全部图、路径、统计和NPZ，结果及字节精确一致；原图G1恢复与14槽支持另行精确重放，优化器/神经模型调用0。710回归通过、1跳过（原5个警告）；全部913个父代封存文件逐SHA复核未变，本轮科学产物另行封存。服务器初次GitHub直连TLS EOF后，用已发布同一SHA的验证Git包同步，未关闭TLS验证。','','本机入口：`D:/DunhuangDanceGen/outputs/execution_planning_r58_01/index.html`；完整视频`main_video/B0_B1_C2_full.mp4`；三首五组`controls/viewer/index.html`；执行档案`profiles/viewer/index.html`；盲评准备`review/viewer/index.html`。C1/C2盲页隐藏方法和来源身份，管理员映射不进入公开报告。这里只准备媒体与协议，人类记录0；音乐边界应由独立音频任务采集，不用动作结果反向当成乐句真值。','','本轮不升级正式Planner。World Root／Contact保持UNKNOWN，Native24、真实乐句及艺术偏好不可评；不能仅凭优化代价或相关边界代理声称整体敦煌舞质量提升。']
 (out/'report.md').write_bytes(('\n'.join(lines)+'\n').encode());put(out/'public_manifest.json',dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]));print('R58_RELEASED',len(sealed),sum(parents.values()))
if __name__=='__main__':main()
