"""Evidence-only release and scientific seals after all fresh-process checks finish."""
import argparse,collections,subprocess
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.sdmr_event_rank.evaluate import group_ci

def run(generation,ranking,out):
    G=Path(generation);R=Path(ranking);out=Path(out)
    if out.exists():raise ValueError('Fresh public release')
    assert (R/'receipts/finish_all.exit').read_text().strip()=='0'
    audit=read(R/'independent.json');assert audit['total_training_updates']==720 and audit['optimizer_steps']==0
    equivalence=read(R/'scoring_equivalence.json');assert equivalence['global_and_diagonal_exact'] and equivalence['exact_score_matrices']==1732
    out.mkdir(parents=True);q=read(G/'qualification.json');guard=read(G/'main_guarded/guard.json');b=read(R/'benchmark/manifest.json');frozen=read(R/'evaluation/result.json');diagnostic=read(R/'diagnostics/result.json');native=read(R/'change_shadow_v3/result.json');p2={arm:read(R/('evaluation_'+arm+'_v2/result.json')) for arm in ('NoRank','Rank','ShuffledRank')};comparisons={}
    for role in ('train','val'):
        for pool in ('same_music','energy_matched','cross_music'):
            key='/'.join((role,pool,'actual'));models={a:p2[a]['summary'][key]['Diagonal'] for a in p2};models['Frozen']=frozen['summary'][role][pool]['actual']['Diagonal'];table={a:v['group_macro_recall_at1'] for a,v in models.items()}
            diffs={}
            for baseline in ('Frozen','NoRank','ShuffledRank'):
                x=models['Rank']['group_means'];y=models[baseline]['group_means'];assert x.keys()==y.keys();values=[x[g]-y[g] for g in sorted(x)];diffs[baseline]=dict(mean=float(np.mean(values)),music_groups=len(values),cluster_ci95=group_ci(values,43043,5000))
            comparisons[role+'/'+pool]=dict(diagonal_group_macro_recall_at1=table,rank_minus=diffs)
    native_summary={}
    for col,name in enumerate(('Global','Diagonal','Offset','Partial','Hybrid')):
        winners=[];zero_changes=0;swap_changes=0;coverage=[]
        for row in native['records']:
            values=np.array(row['variants']['actual']);a=int(values[:,col].argmax());winners.append(row['candidates'][a]['source_uid']);coverage.append(float(values[a,5]));zero_changes+=a!=int(np.array(row['variants']['music_zero'])[:,col].argmax());swap_changes+=a!=int(np.array(row['variants']['music_swap'])[:,col].argmax())
        native_summary[name]=dict(top_source_counts=dict(collections.Counter(winners)),top1_changes_music_zero=zero_changes,top1_changes_music_swap=swap_changes,mean_top1_partial_coverage=float(np.mean(coverage)),semantic_accuracy=None)
    receipts={}
    for name,root in [('generation',G),('ranking',R)]:
        exits={p.name:p.read_text().strip() for p in (root/'receipts').glob('*.exit')};receipts[name]=dict(exits=exits,preserved_nonzero={k:v for k,v in exits.items() if v!='0'})
    summary=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),baseline='5183d2078c034ed0f7cb16e590120ecdd3119dbe',generation=dict(cases=22,seeds=2,main_trajectories=88,control_trajectories=132,independently_replayed_trajectories=220,generator_optimizer_steps=0,train_music_groups_passed=sum(v['passed'] for v in q['train_music_gate'].values()),train_music_groups=len(q['train_music_gate']),train_change_passed=sum(v['passed'] for v in q['train_change_gate'].values()),train_change_sources=len(q['train_change_gate']),training_admitted=q['training_admitted'],guard=guard,official_fk_executed=q['official_fk_executed']),ranking=dict(subwindows=len(b['rows']),queries=len(b['queries']),valid_tasks=len(b['tasks']),unavailable=len(b['unavailable']),p2_gate=frozen['p2_gate'],training_updates=720,training_arms=['NoRank','Rank','ShuffledRank'],equal_initialization_and_schedule=True,comparisons=comparisons,native_queries=len(native['records']),native_unavailable=native['unavailable'],native_shadow=native_summary),token_clock={a:diagnostic['models'][a]['clock'] for a in diagnostic['models']},independent=audit,scoring_equivalence=equivalence,receipts=receipts,human_reviewed=False,semantic_event_truth='UNKNOWN',native24='NOT_EVALUABLE_22_OF24',contact_truth='UNKNOWN',whole_song_generations=0,can_route=False,operational_qualified=False)
    write(out/'summary.json',summary)
    lines=['# R43 音乐条件 Core 试验与 SDMR-E2 事件排序报告','','本轮先交付实际生成和来源候选观看，再执行固定对照。正式 EDGE、旧 Planner、R41/R42 及历史资产未修改。所有结论限于登记人口。','','## 生成侧','','22 个案例、两种子：主候选 88 条、三个条件控制 132 条；另有 220 条独立完整采样重放。Lodge、Source 和 Context 全部冻结，新生成器更新 0。保护域和权限合同通过，完整物理资格不等于通过。',f"Train 正确音乐同时优于置零/错音乐 {summary['generation']['train_music_groups_passed']}/8 组；ChangE 动态保持 {summary['generation']['train_change_passed']}/4。新生成训练未准入。事后工程风险门接受 Core {guard['accepted']}/44，回退 Geometry {guard['fallback']}/44；回退不是新的预登记模型胜利，也不改变原始失败。",'','主观看：`outputs/music_reperformance_r43_01/viewer_guarded/index.html`。来源、Frozen、Core、另一音乐、NoMusic、WrongSource 与 Guarded 可切换。两个固定 MP4 不是整曲；阶段代理用于权限，未宣称新神经阶段学习。','','## 检索侧','','10,107 个 90/150/240 帧子窗口、178 查询、433 个有效任务、101 个不可用查询池；子窗口继承 Parent Event，不重写完整动作边界。同长度、不同录制，时钟标签不是艺术匹配。六 Train 与六历史曝光 Val 音乐组；Test 不读。',f"P1 Train 准入 {frozen['p2_gate']['admitted']}。NoRank/Rank/ShuffledRank 各 240 步，R41 同初始化 522632 参数、相同 Train 顺序及优化器设置；不按 Val 选择权重或预算。",'','下表为音乐组宏平均 Diagonal R@1（%），不同候选池可能大量重合，不能当成独立复验。','','|角色/候选池|Frozen|NoRank|Rank|ShuffledRank|','|---|---:|---:|---:|---:|']
    for key,v in comparisons.items():lines.append('|'+key+'|'+'|'.join(f"{100*v['diagonal_group_macro_recall_at1'][a]:.2f}" for a in ('Frozen','NoRank','Rank','ShuffledRank'))+'|')
    lines+=['','Rank 相对各基线的音乐组配对 bootstrap 区间、R@5/MRR、内容置零与交换、六目标梯度及已知 Token 时钟扰动结果均保留在机器摘要和原始回执；不将这项辅助时钟任务称为语义事件真值。',f"三首历史敦煌音乐选择 24 个完整事件位置，其中 {len(native['records'])} 个有真实支持；其余未替补。每个最多 24 个合法 E1-W 候选，从原生 SO(3) 实际重定时并重新求导。全部适配标签 UNKNOWN，不把旧规划选择标为正例，SDMR 不接管选路。",'','## 验证与限制','','独立进程逐数组重放采样、特征、评分与完整事件；三组训练在 0/120/239 步复验全部六项目标的实际共享梯度和损失，并确认更新后权重改变，复验优化器调用 0。五个冻结父代逐文件 SHA 校验见 independent。','', '保留首次 viewer 命名冲突、完整库字段错误、FK/Token 显示键冲突及评价中断回执。显示修复增加真实 XYZ，不改变此前评分；缓存优化对 1,732 份冻结评分矩阵逐元素完全一致。页面可播放验收与模型质量判断分开。','','完整 Native24、手部和敦煌语义资格仍未取得，ChangE 世界 Root/Contact 真值保持未知；新整曲 0。AIST 生成例来自历史适配缓存，不能称新的 Data V3 Test；检索 Val 为历史曝光组，不是新盲测。','','参考机制：[BADM, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Bidirectional_Autoregessive_Diffusion_Model_for_Dance_Generation_CVPR_2024_paper.html)、[Drop-DTW, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/729c68884bd359ade15d5f163166738a-Abstract.html)。本轮复用仓库单主干/严格部分路径机制，不声称完整复现这些论文。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for root in (G,R):
        manifest=root/'artifacts_manifest.json'
        if manifest.exists():raise ValueError('Never replace an existing scientific seal')
        files=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('artifacts_manifest.json','publication_receipt.json')]
        write(manifest,dict(source_commit=summary['source_commit'],files=files,optimizer_steps=sum(read(R/('p2_'+a+'/result.json'))['optimizer_steps'] for a in ('NoRank','Rank','ShuffledRank')) if root==R else 0,operational_qualified=False));print('SEALED',root,len(files),sum(f['bytes'] for f in files),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generation',required=True);p.add_argument('--ranking',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.generation,a.ranking,a.out)
