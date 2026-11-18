"""Publish R56 descriptive evidence, keeping proxy results separate from truth."""
import json,hashlib,argparse
from pathlib import Path

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):Path(p).write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
def seal(root):
 rows=[]
 for p in sorted(root.rglob('*')):
  rel=p.relative_to(root).as_posix()
  if not p.is_file() or rel.startswith('publication/') or rel=='artifacts_manifest.json' or rel in ('receipts/final_recheck.json','receipts/cleanup.json'):continue
  rows.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p)))
 put(root/'artifacts_manifest.json',dict(files=rows));return rows
def verify(root):
 rows=read(root/'artifacts_manifest.json')['files']
 for r in rows:assert sha(root/r['path'])==r['sha256']
 return len(rows)
def fmt(x):return 'UNKNOWN' if x is None else format(x,'.4f')

def main():
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path('outputs/music_structure_r56_02'));a.add_argument('--out',type=Path,default=Path('reports/r56_release'));args=a.parse_args();root,out=args.root,args.out;main=read(root/'main/result.json');controls=read(root/'controls/result.json');im=read(root/'independent_main/result.json');ic=read(root/'independent_controls/result.json');assert im['exact_arrays_bytes'] and ic['exact_arrays_bytes'] and im['optimizer_calls']==ic['optimizer_calls']==0;assert read(root/'receipts/main_browser.json')['passed'];assert 'exit_status=0' in (root/'receipts/controls_receipt.txt').read_text();assert '692 passed, 1 skipped' in (root/'receipts/pytest_final.log').read_text()
 records=controls['records'];graphs=controls['graphs'];comparisons={}
 for song in range(3):
  rr={r['arm']:r for r in records if r['song']==song};assert set(rr)=={'B0','B1','B2','B3'};assert rr['B1']['graph_key']==rr['B3']['graph_key'];g=graphs[rr['B1']['graph_key']];assert g['geometry_trials']<=3072
  fields={}
  for new,old in (('B1','B0'),('B1','B2'),('B3','B1'),('B3','B0')):
   if any(rr[k]['status']!='NUMERICAL_WHOLE_SONG_RESEARCH_PREVIEW' for k in (new,old)):fields[new+'-'+old]=dict(status='NOT_COMPARABLE_FAILED_PATH');continue
   def delta(key):
    aa,bb=rr[new]['metrics'],rr[old]['metrics']
    for part in key:aa,bb=aa[part],bb[part]
    return float(aa-bb) if aa is not None and bb is not None else None
   fields[new+'-'+old]=dict(boundary_proxy_F1=delta(['phrase_boundary_proxy','F1']),repetition_proxy_rho=delta(['repetition_structure_proxy','spearman']),whole_D1_Q95=delta(['kinematics','whole','D1_Q95']),whole_D2_Q95=delta(['kinematics','whole','D2_Q95']),whole_D3_Q95=delta(['kinematics','whole','D3_Q95']))
  comparisons[str(song)]=fields
 compact=[]
 for r in records:
  g=graphs[r['graph_key']];compact.append({k:r[k] for k in ('song','arm','frames','regions','boundary_centers','status','graph_key','shuffle')}|dict(events=r.get('events',[]),metrics=r.get('metrics'),geometry_trials=g['geometry_trials'],eligible_edges=g['eligible'],candidate_counts=g['candidate_counts'],chart_error_rejections=sum('rejection' in e for e in g['edges']),edge_ledger_sha256=g['edge_ledger_sha256'],trajectory_sha256=r.get('rotation_sha256'),source_count=len(r.get('source_histogram',{})),unique_events=len(r.get('unit_histogram',{})),core_lengths=[q['interval'][1]-q['interval'][0] for q in r['regions'] if q['kind']=='core']))
 rows=seal(root);verify(root);out.mkdir(parents=True,exist_ok=False)
 for p in sorted((root/'receipts').iterdir()):
  if p.is_file() and p.suffix in ('.json','.txt','.sh','.log'):(out/p.name).write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
 previous=Path('outputs/music_structure_r56_01');failure=dict(status='ABORTED_B3_GRAPH_UNHANDLED_G0_CHART_ERROR',exit_status=1,completed_stage_arrays=False,source_commit='07bd4242eaa7e114779f609648e72311fd0442d3',repair='R56_OUTER_GUARD_REJECTS_EXACT_KNOWN_ERROR; FROZEN_GEOMETRY_AND_THRESHOLDS_UNCHANGED')
 for p in sorted((previous/'receipts').iterdir()):
  if p.is_file() and p.suffix in ('.json','.txt','.sh','.log'):(out/('attempt01_'+p.name)).write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
 put(out/'failed_attempt.json',failure);put(out/'independent_main.json',im);put(out/'independent_controls.json',ic);put(out/'comparison_differences.json',comparisons);put(out/'seal.json',dict(files=len(rows),manifest_sha256=sha(root/'artifacts_manifest.json')))
 summary=dict(records=compact,descriptive_differences=comparisons,source_binding=controls['binding'],music=controls['music'],whole_music_inputs=3,independent_new_music_works=0,phrase_truth='NOT_EVALUABLE',human_preference='NOT_EVALUATED',blind_preparation=read(root/'blind/review_protocol.json'),main_video=read(root/'main_video/result.json'),paper_claim_qualified=False,production_changed=False,SDMR_training_steps=0,neural_training_steps=0,optimizer_calls=0,native24='NOT_EVALUABLE',first_effect_before_controls=True,failed_attempt=failure,seal_files=len(rows));put(out/'summary.json',summary)
 lines=['# R56 整曲音乐结构归因：固定基线与受控对照','','完成完整音乐B0/B3主观看与验收后，再执行3个历史音频×4组固定实验；不是新音乐独立盲测，不将自动代理解释为真实乐句或艺术偏好。','','## 冻结与改变','','B0精确重建原dunhuangwu2：旋转、FK、事件UID一致。13个Core、12×48帧过渡，16个封存单位，原E1-W速度域[.75,1.33]、G0/G1和R48阈值不改。B1用音频谱变化＋停顿代理分配180–300帧Core；B2置乱B1时长排列，保留时长多重集；B3与B1共享完全相同槽位、候选和连接图，额外约束音乐/动作重复结构相似度。Beam128，每布局同一3072次几何上限，实际合法候选/试算数可不同，不能称计算量精确匹配。','','B3固定0.25×平均重复核不一致代价；音乐核用Chroma Core均值/标准差，动作核用16单位32相位局部SO3距离。所谓主题只指自动几何相似性，不是文化语义真值。原Heuristic、单位/来源重复代价(.1/.02)与过渡D3权重(.01)不变。未训练SDMR或Diffusion。','','边界指标用另一路Chroma novelty代理；重复指标用log-frequency与来源相位FK核；动作重音对比音频spectral flux。它们同源且相关，不能称独立人工评价。真正Phrase Boundary Alignment保持NOT_EVALUABLE；没有人类偏好评分。','','## 全部固定结果','','|输入|方法|状态|来源/事件|边界代理F1|重复代理rho|整体D3 Q95|','|---|---|---|---:|---:|---:|']
 for r in compact:
  m=r['metrics'];lines.append('|dunhuangwu'+str(r['song']+2)+'|'+r['arm']+'|'+r['status']+'|'+str(r['source_count'])+'/'+str(r['unique_events'])+'|'+fmt(m['phrase_boundary_proxy']['F1'] if m else None)+'|'+fmt(m['repetition_structure_proxy']['spearman'] if m else None)+'|'+fmt(m['kinematics']['whole']['D3_Q95'] if m else None)+'|')
 lines+=['','逐音乐的B1−B0、B1−B2、B3−B1和B3−B0差值见comparison_differences.json，不根据这轮结果调权重或选最优音乐。13个事件的78个事件对不是78个独立音乐样本；此处只做描述，不报告伪独立置信区间。高D3可能同时受活动量和高频误差影响，不能单独认定动作更差或更好。全片/Core/Bridge/边界D1–D3、来源保持误差、每个事件的实际时间图均保存。','','## 可观看交付','','- 主候选完整音乐视频：`D:/DunhuangDanceGen/outputs/music_structure_r56_02/main_video/B0_vs_B3_full.mp4`。','- 主交互：`D:/DunhuangDanceGen/outputs/music_structure_r56_02/main/viewer/index.html`。','- 全部对照：`D:/DunhuangDanceGen/outputs/music_structure_r56_02/controls/viewer/index.html`，三首完整音乐，共同播放/暂停/倍速/逐帧，音频结构曲线与每方法事件条可点击联动。','- 盲评准备：`outputs/music_structure_r56_02/blind/blind_pair.mp4`，仅显示A/B、同相机/尺度/音乐，隐藏方法名、条件颜色和分段条。admin_mapping.json仅供管理员，未作为公开报告或人类评分。当前PREPARATION_ONLY、人类记录0。','','全部显示与输出是Root-relative身体22关节。世界Root/接触UNKNOWN，Native24不可评，不能声称物理合格、敦煌文化正确或感知收益已经确立。','','## 失败与复验','','01在变长B3连接图遭遇原G0 Ambiguous SO3 chart而退出1，完整失败回执保留。02仅在R56外层拒绝这种已知方法错误，其他ValueError仍抛出；G1未尝试也明确记录，不把该拒绝称为全局物理不可行。音乐算法、时长范围、评分权重、随机种子和几何阈值未根据质量结果重调。','','main和controls均由新进程重新读取原音频/动作，重建全部候选图、路径、评分与轨迹，结果及确定性NPZ字节精确一致。运行提交单独记录，唯一单列的非科学字段为binding.code_commit；原提交的每份数值源码SHA均独立核对。模型/神经优化器调用0。完整回归692通过、1跳过；当前没有新的独立音乐作品或独立人工乐句真值，因此不升级正式规划资格。']
 (out/'report.md').write_bytes(('\n'.join(lines)+'\n').encode('utf8'));put(out/'public_manifest.json',dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]));print('R56_RELEASED',len(rows),len(compact))
if __name__=='__main__':main()
