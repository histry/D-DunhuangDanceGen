"""Source, actual gradient and complete frozen-score replay, without retraining."""
import subprocess
import torch
from torch.nn import functional as F
from .common import *
from .learn import Study,fit,schedule,synthetic

def verify(out):
 out=fresh(out)
 from .data import build
 build(out/'data');assert read(out/'data/manifest.json')==read(ROOT/'data/manifest.json');assert sha(out/'data/arrays.npz')==sha(ROOT/'data/arrays.npz')
 s=Study();g=read(ROOT/'checks/gate.json');net=s.model();net.load_state_dict(torch.load(ROOT/'checks/synthetic.pt',map_location=s.device,weights_only=True)['state']);net.eval()
 with torch.no_grad():
  m,d,dm,mv,_=synthetic(65054,32);mt,dt,*_=net.content(m,d,dm,mv);scores=(F.normalize(mt.mean(1),dim=-1)@F.normalize(dt.mean(1),dim=-1).T).cpu().numpy()
 assert np.array_equal(scores,np.load(ROOT/'checks/synthetic_eval.npz')['scores']);fit_out=out/'fit';fit_out.mkdir();fit(s,s.model(),[g['fit_task_indices'] for _ in range(CFG['fit_steps'])],fit_out,'Conditional',ROOT/'checks');arms=[]
 if g['admitted']:
  for arm in ('Conditional','Unconditional','Shuffled'):
   target=out/arm;target.mkdir();fit(s,s.model(),schedule(s.meta['tasks'],CFG['steps'],CFG['seed']),target,arm,ROOT/arm);arms.append(arm)
  from .evaluate import run
  run(out/'evaluation');assert read(out/'evaluation/result.json')==read(ROOT/'evaluation/result.json');assert sha(out/'evaluation/scores.npz')==sha(ROOT/'evaluation/scores.npz')
 write(out/'result.json',dict(source_data_and_deterministic_npz_bytes_exact=True,synthetic_scores_exact=True,train_fit_loss_gradients_exact=True,trained_arms=arms,full_actual_and_counterfactual_scores_exact=bool(arms),optimizer_calls=0,test_tensor_reads=0,can_route=False))

def release(out):
 out=fresh(out);gate=read(ROOT/'checks/gate.json');data=read(ROOT/'data/manifest.json');independent=read(ROOT/'independent/result.json');evaluation=read(ROOT/'evaluation/result.json') if gate['admitted'] else None
 result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source=data['binding'],population=data['statistics'],groups={k:len(v) for k,v in data['groups'].items()},queries=len(data['queries']),unavailable=len(data['missing']),checks=gate,evaluation=evaluation,independent=independent,network_updates=CFG['synthetic_steps']+CFG['fit_steps']+(3*CFG['steps'] if gate['admitted'] else 0),human_reviewed=False,finedance_admitted=False,unseen_music_generalization=False,can_route=False,production_planning_unchanged=True)
 write(out/'summary.json',result);lines=['# R54 条件困难负例与残差SDMR实验','', '冻结R52/R53与R41架构，先回传真实动作检索预览，再做固定训练对照。没有生成新舞蹈。','', '人口：'+str(result['population'])+'；音乐组：'+str(result['groups']), '', '全部任务时长相同，在固定0.05宽度Heuristic分箱及动作活动量分箱内竞争，错音乐活动量差不超过0.05。音乐为整录制真实RMS百分位，动作为完整录制先FK求导后的相对身体速度/2截断。复用R36/R53定义，但不含E1-W或Geometry，不冒称完整冻结规划器。', '', '候选构造使用正例所在Heuristic箱，是离线配对诊断人口，不是在线召回器。不同录制同音频时钟重叠IoU≥0.6为多正例；共享编舞跨音乐及重叠模糊时段UNKNOWN；同歌错时为物理时钟非目标，不认定艺术错误。自动重复指纹只排除显然近重复，不保证所有合理替代都已发现。', '', 'Residual=当前局部分数−冻结R41局部分数；Total=Heuristic+0.25×Residual，温度0.1，仅pairwise软间隔排序与0.01残差约束。没有扩大网络或增加几何预测头。总参数522632，音乐/动作输入与local路径可训练，其余分支冻结。各训练臂相同初始化、240更新、同查询顺序；Unconditional仅改变负例，Shuffled仅置乱同池标签，匹配候选数量。', '', '合成随机信号与真实Train拟合资格：'+str(gate),'','六组Val历史曝光，全部训练预算与权重固定后统一评价；未读取Test，不证明敦煌语义、未见音乐或未见编舞泛化。FineDance未进入本轮准入人口。']
 if evaluation:
  lines+=['','## 留出诊断（组内先录制均衡，再音乐宏平均）','','|任务|方法|R@1|95%音乐组区间|','|---|---|---:|---|']
  for family in ('cross_music','same_music'):
   for method,v in evaluation['summary']['val_'+family]['methods'].items():
    q=v['recall_at1'];lines.append('|'+family+'|'+method+'|'+format(q['mean'],'.4f')+'|'+str(q['ci95'])+'|')
  lines+=['','A3增量门：'+str(evaluation['A3_increment_gate'])+'；A4准入：'+str(evaluation['A4_admitted'])+'；A5未启动。', '', 'A3要求至少4个Val音乐组，且相对Heuristic、Frozen、HybridFrozen、EnergyCurve、Unconditional、Shuffled和活动量匹配错音乐的配对R@1差值区间下界均大于0。A4还要求同歌错时相对Heuristic的区间下界大于0；不足则不追加Warp训练或规划接入。完整MRR/R@5、组均值、配对差值与覆盖人口见summary.json。']
 lines+=['','## 独立复验','',str(independent),'', '新进程从封存R41动作/特征数组和真实音频重建R54活动量、候选及归一化输入，不重跑SMPL转换或声学特征提取；数据NPZ及完整正确/错音乐评分NPZ字节一致。每个训练臂0/120/239及Train拟合0/60/119步的真实损失/梯度精确复验，核实更新后权重变化；复验无优化器调用。未声称重新训练一遍全部更新。', '', '数组、模型、快照、训练日志和观看资产在服务器封存；公开代码、统计与回执。生产选路、G0/G1和Native不改。']
 (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');files=[]
 for mode in ('data','checks','Conditional','Unconditional','Shuffled','evaluation','independent','preview','preview_v2','main_view','fit_view'):
  if not (ROOT/mode).exists():continue
  for p in sorted((ROOT/mode).rglob('*')):
   if p.is_file():files.append(dict(path=p.as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
 write(out/'seal.json',dict(files=files));write(ROOT/'receipts/seal_check.json',dict(files=len(files),all_hashes_verified=all(sha(f['path'])==f['sha256'] for f in files),optimizer_calls=0));print('R54_RELEASE',len(files),flush=True)
