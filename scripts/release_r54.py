"""Export evidence-bounded R54 results, including stopped and corrected attempts."""
import argparse,hashlib,json,subprocess
from pathlib import Path

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def put(p,x):p.write_bytes((json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
def source_commit(root):
 name='checks_receipt.txt' if (root/'receipts/checks_receipt.txt').exists() else 'main_receipt.txt'
 return next(s.split('=',1)[1] for s in (root/'receipts'/name).read_text().splitlines() if s.startswith('commit='))
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('reports/r54_release'));a=p.parse_args();out=a.out
 attempts={};audits={};inputs=[]
 for num in ('01','02'):
  root=Path('outputs/residual_sdmr_r54_'+num);meta=read(root/'data/manifest.json');gate=read(root/'checks/gate.json');independent=read(root/'independent/result.json');training=read(root/'checks/result.json');assert independent['source_data_and_deterministic_npz_bytes_exact'] and independent['optimizer_calls']==0
  revision=source_commit(root)
  for name,digest in meta['binding']['code'].items():assert hashlib.sha256(subprocess.check_output(['git','show',revision+':src/dunhuang/research/residual_sdmr_r54/'+name])).hexdigest()==digest
  rows=meta['rows'];tasks=[t for t in meta['tasks'] if t['role']=='train'];spans=[max(rows[i]['body_activity'] for i in t['candidates'])-min(rows[i]['body_activity'] for i in t['candidates']) for t in tasks];donors=[abs(rows[t['query']]['music_activity']-rows[t['donor']]['music_activity']) for t in tasks]
  audits[num]=dict(scope='TRAIN_METADATA_ONLY_NO_VAL_SCORES',tasks=len(tasks),motion_spread_over_005=sum(s>.05+1e-9 for s in spans),maximum_motion_spread=max(spans),maximum_wrong_music_activity_delta=max(donors))
  attempts[num]=dict(source_commit=revision,population=meta['statistics'],music_groups={k:len(v) for k,v in meta['groups'].items()},queries=len(meta['queries']),unavailable=len(meta['missing']),gate=gate,fit_parameters={k:training[k] for k in ('total_parameters','trainable_parameters','updates','model_sha256')},independent=independent,source_manifest_sha256=hashlib.sha256((root/'data/manifest.json').read_bytes()).hexdigest(),qualification_updates=200,population_model_updates=0 if not gate['admitted'] else None,Val_neural_evaluation_executed=(root/'evaluation/result.json').exists())
  inputs.extend([root/'data/manifest.json',root/'checks/gate.json',root/'independent/result.json'])
 assert not attempts['01']['gate']['admitted'] and not attempts['02']['gate']['admitted'];assert audits['02']['motion_spread_over_005']==0 and audits['02']['maximum_wrong_music_activity_delta']<=.05
 root=Path('outputs/residual_sdmr_r54_02');check=read(root/'receipts/final_recheck.json');assert check['all_hashes_verified'];fit=read(root/'independent/fit_final.json');assert fit['recomputed_R1']==1 and fit['optimizer_calls']==0
 out.mkdir(parents=True,exist_ok=False)
 for num in ('01','02'):
  r=Path('outputs/residual_sdmr_r54_'+num)
  for rel in ('publication/summary.json','publication/seal.json','independent/result.json'):
   dest=out/(num+'_'+Path(rel).name if rel!='independent/result.json' else num+'_independent.json');dest.write_bytes((r/rel).read_bytes().replace(b'\r\n',b'\n'))
  for source in sorted((r/'receipts').iterdir()):
   if source.is_file() and source.suffix in ('.json','.txt','.log','.sh') and source.stat().st_size: (out/(num+'_'+source.name)).write_bytes(source.read_bytes().replace(b'\r\n',b'\n'))
 put(out/'fit_final.json',fit);put(out/'design_audit.json',audits)
 summary=dict(attempts=attempts,fit_final_replay=fit,total_qualification_updates=400,total_population_model_updates=0,Val_neural_evaluation_executed=False,A3_population_comparison='NOT_RUN_CHECKS_GATE_FAILED',A4='NOT_RUN',A5='NOT_RUN',whole_song_generations=0,human_reviewed=False,production_planning_unchanged=True,source_inputs=[dict(path=p.as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs]);put(out/'summary.json',summary)
 lines=['# R54 条件困难负例与残差SDMR：工程修正和资格实验','','**完成了候选混杂修正、可观看拟合诊断与独立复验；未启动主模型或消融，不宣称新的泛化收益。**','','## 代码与人口','','同一R41架构522632参数，其中261312参数用于音乐/动作输入及local分支，其余冻结。Residual=当前local−冻结local；Total=Heuristic+0.25 Residual；温度0.1，残差约束0.01。Conditional/Unconditional/Shuffled的240步同预算代码已实现，但本轮全部未启动。R52/R53及生产路径未修改。','','数据来自已封存R41 AIST++录制和真实音频，保留既有完整事件边界；90/150/240帧是检索诊断子窗，不是重新把ChangE完整动作切碎。FineDance未加入本轮。48个Train组、6个历史曝光Val组；没有新盲测或未见编舞结论。','','|版本|Train跨音乐/同歌|Val跨音乐/同歌（仅构造，未评分）|动作活动量最大跨度|错音乐活动量最大差|','|---|---:|---:|---:|---:|']
 for num in ('01','02'):
  d=attempts[num]['population'];z=audits[num];lines.append('|'+num+'|'+str(d['train_cross_music'])+'/'+str(d['train_same_music'])+'|'+str(d['val_cross_music'])+'/'+str(d['val_same_music'])+'|'+format(z['maximum_motion_spread'],'.6f')+'|'+format(z['maximum_wrong_music_activity_delta'],'.6f')+'|')
 lines+=['','V1同绝对Heuristic分箱仍允许活动量在平均音乐强度两侧对称而不同。Train714任务中239项活动量跨度超过0.05，最大0.98098。V2新增独立动作活动量0.05分箱，错音乐活动量差≤0.05，并新增EnergyCurve评价基线；没有根据Val评分修改方法。候选为正例定位的离线条件诊断，不是在线召回器。','','同音频跨录制、时钟IoU≥0.6的关系为多正例；模糊重叠与共享编舞跨音乐为UNKNOWN。同歌错时仅为物理时钟任务非目标，不自动认定艺术不合理；近重复指纹也不构成语义真值。','','## 实际训练与停止点','','|版本|合成正对照（32候选）|预登记门|8个Train拟合R@1|主模型更新|','|---|---:|---:|---:|']
 for num in ('01','02'):
  g=attempts[num]['gate'];lines.append('|'+num+'|'+format(g['synthetic_held_realizations_accuracy'],'.5%')+'|75%|'+format(g['real_train_fit_R1'],'.2%')+'|0|')
 lines+=['','每版本仅80次合成更新和120次固定Train拟合，两版本合计400次资格更新。真实拟合达到100%，合成23/32=71.875%，低于原门75%；没有增加步数、重选种子或降低阈值。32个合成例很少，该失败不能推出编码器不能学习；它只说明这项已登记资格未通过。训练是8候选辅助任务，评价32候选，任务差异也须在下一份独立协议中审查。','','因此没有执行Conditional主训练、Unconditional/Shuffled对照、Val神经评分、A4合法Warp训练、A5编舞接入或整曲生成。本轮无法判断残差SDMR是否优于原模型，也不能用Train拟合代替内容泛化。','','## 观看效果','','本机统一入口：`D:/DunhuangDanceGen/outputs/residual_sdmr_r54_02/index.html`。严格候选预览6例，固定Train拟合诊断8例，支持音频共同时间轴、暂停、逐帧、倍速及来源详情。四栏是实际来源回放与模型选择，不是生成新动作。页面明确标为TrainFitDiagnostic。','','初版观看清单自引用旧哈希的问题已修复，失败目录保留，原动作与评分data.js逐字节不变。公开Git仅统计和回执，不包含私有媒体、模型或训练数组。','','## 独立复验','','两个版本均重新从封存R41动作/特征与真实音频构造R54输入；数据NPZ字节、合成评分、Train拟合0/60/119步损失及梯度精确复验，确认更新后的权重变化。没有重跑SMPL转换，也没有重训全部更新。新进程另从最终权重重算八个拟合任务，R@1仍为100%，优化器调用0。未执行分支的评分未被冒称重放。','','最终封存：V1共43文件、V2共47文件；全体90文件哈希复核通过。完整数组、快照和权重在服务器；本机回传观看文件、元数据和回执。R53原248文件另行核验保持不变。最终完整回归见02_pytest_final.log。','','复现当前严格版本使用本轮代码；复现V1需使用其固定e1eeefa提交。源代码指纹与提交已逐项核对。协议见docs/residual_sdmr_r54.md。此报告可由scripts/release_r54.py从封存产物重新导出。']
 (out/'report.md').write_bytes(('\n'.join(lines)+'\n').encode('utf8'));put(out/'public_manifest.json',dict(scope='PUBLIC_UTF8_LF_TEXT_COPIES; ORIGINAL_SCIENCE_SEALS_UNCHANGED',files=[dict(name=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.iterdir()) if p.is_file()]));print('R54_PUBLIC_EXPORTED',len(list(out.iterdir())))
if __name__=='__main__':main()
