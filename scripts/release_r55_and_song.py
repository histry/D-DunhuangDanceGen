"""Export completed R55 and whole-song evidence; never infer unexecuted branches."""
import argparse,json,hashlib
from pathlib import Path

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):Path(p).write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
def ended(root,name):
 text=(root/'receipts'/name).read_text();assert 'exit_status=0' in text;return text
def seal(root):
 files=[]
 for p in sorted(root.rglob('*')):
  rel=p.relative_to(root).as_posix()
  if not p.is_file() or rel.startswith('publication/') or rel=='artifacts_manifest.json' or rel in ('receipts/final_recheck.json','receipts/cleanup.json'):continue
  files.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p)))
 put(root/'artifacts_manifest.json',dict(files=files));return dict(files=len(files),bytes=sum(r['bytes'] for r in files),manifest_sha256=sha(root/'artifacts_manifest.json'))
def validate_seal(root):
 obj=read(root/'artifacts_manifest.json');entries=obj['files'];entries=[dict(path=k,sha256=v if isinstance(v,str) else v['sha256']) for k,v in entries.items()] if isinstance(entries,dict) else entries
 for r in entries:assert sha(root/r['path'])==r['sha256'],str(root/r['path'])
 return len(entries)
def export_receipts(root,out):
 for p in sorted((root/'receipts').iterdir()):
  if p.is_file() and p.suffix in ('.txt','.log','.json','.sh'):(out/p.name).write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
def public_manifest(out):put(out/'public_manifest.json',dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]))

def r55(out):
 root=Path('outputs/clock_residual_r55_02');ended(root,'main_receipt.txt');ended(root,'comparison_receipt.txt');q=read(root/'qualification/gate.json');assert q['admitted'];ev=read(root/'evaluation/result.json');ind=read(root/'independent_final/result.json');assert ind['optimizer_calls']==0 and ind['checks']['evaluation']=='ALL_ARRAYS_AND_SUMMARY_EXACT'
 arms={a:read(root/a/'result.json') for a in ('Conditional','NoClock','Unconditional','Shuffled')};assert len({r['initial_digest'] for r in arms.values()})==1 and len({r['schedule_sha256'] for r in arms.values()})==1
 objective_keys=('common.py','study.py','qualify.py','population.py','protocol.json')
 for key in objective_keys:assert len({r['code'][key] for r in arms.values()})==1
 seal_info=seal(root);assert validate_seal(root)==seal_info['files'];out.mkdir(parents=True,exist_ok=False);export_receipts(root,out)
 previous=Path('outputs/clock_residual_r55_01');old_arms={a:read(previous/a/'result.json') for a in arms};previous_info=dict(population_updates=sum(r['updates'] for r in old_arms.values()),qualification_updates=440,evaluation_started=(previous/'evaluation').exists(),evaluation_completed=(previous/'evaluation/result.json').exists(),final_independent_completed=(previous/'independent_final/result.json').exists(),calibration_independent=False,disposition='TRAIN_METADATA_CALIBRATION_CANDIDATE_EXPOSURE; INTERRUPTED_EVALUATION; NOT_USED_FOR_MODEL_OR_THRESHOLD_SELECTION',launcher_exit_status_alone_not_completion=True,models={a:r['model_sha256'] for a,r in old_arms.items()})
 summary=dict(qualification=q,arms=arms,evaluation=ev,independent=ind,seal=seal_info,qualification_updates=440,population_updates=960,all_attempts_qualification_updates=880,all_attempts_population_updates=1920,previous_attempt=previous_info,whole_song_generations=0,learned_ChangE_routing=False,human_reviewed=False)
 put(out/'population_audit.json',read(root/'input/binding.json')['population_audit']);put(out/'previous_attempt.json',previous_info);put(out/'summary.json',summary);put(out/'calibration.json',read(root/'evaluation/calibration.json'));(out/'independent.json').write_bytes((root/'independent_final/result.json').read_bytes());put(out/'seal.json',seal_info)
 lines=['# R55 任务一致的局部残差与可验证时钟恢复','','R54历史23/32及75%资格门保持封存。本轮完成新的任务一致资格、真实主候选、先观看后固定预算对照、按音乐组评价与新进程精确复验。','','## 资格实验','','|项|Held 8候选|Held 32候选|作用|','|---|---:|---:|---|']
 for k,m in q['synthetic'].items():lines.append('|'+k+'|'+(format(m['8_actual'],'.2%') if '8_actual' in m else '—')+'|'+(format(m['32_actual'],'.2%') if '32_actual' in m else '—')+'|'+('池化候选规模诊断' if k in ('S1','S2') else '与真实任务共用局部残差损失')+'|')
 lines+=['','S1/S2在共同无偏移新随机数据上比较池化候选规模，并非只修改R54候选数的逐项复现（原R54随机裁剪起点不同）。S3按6候选训练、Held8/32各4种子；S4从同一连续随机信号非循环裁剪出已知偏移，所有偏移共用支持。零音乐和反序对照保存在summary.json。真实8任务拟合R@1='+format(q['real_fit_R1'],'.2%')+'。合成320更新＋真实拟合120更新；不将此结果视为艺术语义成功。','','## 真实人口与固定对照','','严格R54配对人口48 Train音乐组，预定哈希留出8组只做校准，40组优化；6个Val音乐组已有历史开发曝光。522632总参数／261312可训练参数；各组从原R41初始化，240步同一查询日程，未继承8任务拟合权重。Conditional加入局部时钟辅助；NoClock移除此项；Unconditional移除困难候选限制；Shuffled打乱关联与辅助时钟标签。候选池依赖已知正例分箱，只是离线诊断，不是在线ChangE召回。','','主候选先回传并实际浏览器验收，随后才执行另三组。训练提交逐组保存在summary.json；objective代码、初始化与日程逐项一致。V1仍有留组动作被用作负候选，保留960次主模型更新及评价中断证据；启动器退出码即使为0，也不能代替缺失的最终评价和复验产物。V2同时隔离查询与两种候选池的音乐组、原动作SHA和音频SHA；543个有效训练任务覆盖40个优化组，删除651次候选曝光，排除2个无负例任务，隔离交集为空。8个留组只对本次适配隔离，R41预训练已曝光。','','|历史Val任务|方法|组均衡R@1|组bootstrap95%区间|','|---|---|---:|---|']
 for family in ('cross_music','same_music'):
  for name,m in ev['content']['val_'+family]['methods'].items():lines.append('|'+family+'|'+name+'|'+format(m['mean'],'.2%')+'|'+str(m['ci95'])+'|')
 lines+=['','## 已知注入偏移恢复','','|模型／控制|准确率|MAE帧|','|---|---:|---:|']
 for arm,data in ev['clock'].items():
  for control,m in data.items():lines.append('|'+arm+'/'+control+'|'+format(m['accuracy']['mean'],'.2%')+'|'+format(m['absolute_error_frames']['mean'],'.3f')+'|')
 lines+=['','时钟标签是人为控制的注入偏移；恢复成功并不能证明音乐重音与动作发力的语义正确。ClockOnly、EnergyCurve、Frozen及零音乐/双零/反序在同一任务中独立报告。Train留组决定置信阈值；Val只按该阈值报告接受覆盖和错误率。阈值='+str(ev['threshold_frozen_on_train'])+'，Val接受'+str(ev['accepted_val'])+'/'+str(ev['total_val'])+'。没有资格时拒答，不向ChangE宣称置信概率。','','新增内容增益登记门：'+str(ev['A3_increment_gate'])+'。不升级默认Planner，不宣称敦煌迁移或文化语义成功。完整评价含配对音乐组差值区间及各组原值。','','## 复验和交付','','资格全部评分精确重现；真实拟合0/60/119及四组0/120/239步损失/梯度精确复验、更新后状态改变；全部最终评分NPZ字节和统计重建一致，优化器调用0。观看页是实际来源回放和检索/偏移读数，不是神经动作生成。服务器保留权重和原始输入，本机回传观看页、统计和回执。','','本机观看：`D:/DunhuangDanceGen/outputs/clock_residual_r55_02/main_view/viewer/index.html`。独立整曲实验另见whole_song_geometry_release，未使用本轮SDMR。']
 (out/'report.md').write_bytes(('\n'.join(lines)+'\n').encode('utf8'));public_manifest(out)

def song(out):
 root=Path('outputs/whole_song_geometry_02');ended(root,'run_receipt.txt');d=read(root/'main/result.json');ind=read(root/'independent/result.json');assert ind['optimizer_calls']==0 and ind['model_calls']==0;assert d['P0_time_coverage'] and d['P1_all_connections'];assert sha(root/'main/arrays.npz')==sha(Path('outputs/whole_song_geometry_01/main/arrays.npz'))
 import numpy as np
 with np.load(root/'main/arrays.npz') as z:assert np.all(z['coverage']==1) and np.isnan(z['world_root_translation']).all() and not z['world_root_observed'].any()
 view=read(root/'main/viewer/protocol.json');assert view['audio_embedded_sha256']==d['audio_sha256'];seal_info=seal(root);validate_seal(root);out.mkdir(parents=True,exist_ok=False);export_receipts(root,out)
 summary={k:d[k] for k in ('status','audio_sha256','frames','fps','duration_seconds','regions','selected','connections','source_histogram','unit_histogram','kinematics','beat_proxy','source_core_max_SO3_rad','P0_time_coverage','P1_all_connections','P2_complete_readout','P3_native24','physical_qualified','model_calls','optimizer_calls','binding')};summary.update(edge_trials=len(d['edges']),eligible_edges=sum(e['selected'] is not None for e in d['edges']),independent=ind,seal=seal_info,identical_01_02_trajectory=True)
 movie=read(root/'video/result.json');assert movie['arrays_sha256']==sha(root/'main/arrays.npz') and movie['video_sha256']==sha(root/'video/dunhuangwu2_geometry_full.mp4');summary['video']={k:movie[k] for k in ('frames','fps','video_duration','audio_duration','video_sha256','renderer_sha256','audio_source_sha256','arrays_sha256')};put(out/'summary.json',summary);put(out/'independent.json',ind);put(out/'seal.json',seal_info)
 lines=['# dunhuangwu2 完整音乐纯几何整曲基线','','已形成完整连续的Root-relative身体22关节轨迹，并回传实际观看页；不是Diffusion重新演绎，也未获得Native24物理资格。','','|项目|实测|','|---|---|','|音乐|dunhuangwu2，'+format(d['duration_seconds'],'.8f')+'秒|','|时间覆盖|'+str(d['frames'])+'帧／30FPS；每帧恰好一次|','|Core／Transition|'+str(len(d['selected']))+'／'+str(len(d['connections']))+'|','|不同来源录制／不同事件|'+str(len(d['source_histogram']))+'／'+str(len(d['unit_histogram']))+'|','|候选连接／原筛查可用|'+str(len(d['edges']))+'／'+str(sum(e['selected'] is not None for e in d['edges']))+'|','|选中连接|全部通过原R50 G0/G1与R48阈值|','|对齐来源Core SO3最大误差|'+str(max(d['source_core_max_SO3_rad']))+' rad（重采样来源按构造保持）|','|World Root／Contact／Native24|UNKNOWN／UNKNOWN／NOT_EVALUABLE|','','## 实际算法','','使用封存R51的16个来源单位，不把旧499个短片段全部冒充完整原语。音频采样数决定3601帧，按接近240帧Core选13事件；Core为232/233帧，12段48帧过渡，精确覆盖整首音乐。原E1-W合法速度范围[.75,1.33]，不补造丢失来源帧；短槽不是SDMR或神经乐句检测结果。','','每层8个合法候选，异来源邻接全部进行原G0/G1构造与筛查；128宽Beam按Heuristic、.01相对过渡D3代价及预登记单位/来源重复惩罚(.1/.02)选路。有限搜索若失败会保留失败层，不能证明全局不可能。本次624条边中597条可用。未放宽阈值，无神经推理、优化器或训练。','','## P0–P3','','P0覆盖与P1选中路径数值筛查通过。P2表示全片、Core、过渡、左右边界和来源记录均有读数，不等于全部物理质量合格。全片相对身体D1/D2/D3：']
 lines.append('')
 for key,value in d['kinematics']['whole'].items():lines.append('- '+key+': '+str(value))
 lines+=['','过渡最大gap_D3='+str(max(c['methods'][c['selected']]['gap_D3'] for c in d['connections']))+'，原参考D3='+str(d['binding']['parent']['reference']['D3'])+'。参考只用于同定义的连接筛查，不拿整片D3或快速动作直接判Native失败。全片动力学高峰仍需结合来源与阶段观察，不宣称无抖动、无足滑或艺术自然性。拍点结果是运动学代理，非人工事件对应真值。','','P3尚不可评：数组中世界Root平移显式NaN，观测掩码false；不把Root=0当真值。22关节不足Native24完整合同。观看有3D方位，依然属于Root-relative局部姿态视图，不能掩盖世界移动未知。','','## 完整输出与观看','','`D:/DunhuangDanceGen/outputs/whole_song_geometry_02/main/arrays.npz`保存SO3旋转、3D关节、覆盖、逐帧事件/来源帧及未知Root掩码。`main/result.json`包含全部候选、时间图、连接证据和逐段指标。','','本机观看：`D:/DunhuangDanceGen/outputs/whole_song_geometry_02/main/viewer/index.html`。完整原音乐字节内嵌，SHA与封存原音轨一致；支持播放、暂停、逐帧、事件跳转、倍速、3D观察方位和D3点击诊断。','','01先完成轨迹并观看。02仅修正阶段风险字段名称与内嵌音频显示，布局、评分和阈值未改；两版本轨迹NPZ逐字节相同。02新进程重新读取全部来源、重建624条候选边、选路和轨迹，再次与主结果逐字节精确一致，优化器与模型调用均0。源代码、tmux命令、退出状态和封存清单均保留。本首属于历史曝光开发音乐，不是新作品独立盲测。仅本首整曲，未自动扩至另外两首。完整双视角音乐MP4：`D:/DunhuangDanceGen/outputs/whole_song_geometry_02/video/dunhuangwu2_geometry_full.mp4`，30FPS/3601帧；独立核对视频时长、原音频SHA与输入轨迹SHA。']
 (out/'report.md').write_bytes(('\n'.join(lines)+'\n').encode('utf8'));public_manifest(out)

def main():
 p=argparse.ArgumentParser();p.add_argument('kind',choices=['r55','song']);p.add_argument('--out',type=Path,required=True);a=p.parse_args();globals()[a.kind](a.out);print('RELEASED',a.kind,a.out)
if __name__=='__main__':main()
