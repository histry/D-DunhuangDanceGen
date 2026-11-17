"""Release observed results only; immutable scientific archives and replay receipts."""
import argparse,subprocess,collections,re
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write
from dunhuang.research.sdmr_event_rank.evaluate import group_ci
from .replay import digest

def run(out):
    G=Path('outputs/reperformance_r44_01');A=Path('outputs/retrieval_audit_r44_01');out=Path(out)
    if out.exists():raise ValueError('Fresh release')
    receipts={}
    for p in (G/'primary_receipt.txt',G/'controls_receipt.txt',A/'receipt.txt',G/'readout_receipt.txt'):
        value=p.read_text();assert 'exit_status=0' in value,str(p);receipts[str(p)]=value
    for root,name in ((G,'main'),(G,'controls'),(A,'native'),(A,'evaluation'),(A,'native_calibration')):
        r=read(root/(name+'_replay_receipt.json'));assert r['exact_arrays'] and r['exact_scientific_json'] and r['optimizer_calls']==0
    acceptance=read(G/'acceptance.json');quality=read(G/'quality.json');assert quality==read(G/'quality_replay.json');assert quality['official_fk_executed'];population=read(A/'benchmark/population_audit.json');evaluation=read(A/'evaluation/result.json');native=read(A/'native/result.json');cal=read(A/'native_calibration/result.json');parents=read(G/'parent_verification.json');assert parents['all_parent_hashes_unchanged'];counterfactual={}
    from dunhuang.research.paired_event_r41.events import validate_roles
    benchmark=read(A/'benchmark/manifest.json');isolation=validate_roles(benchmark['rows']);parent=read('outputs/sdmr_event_rank_01/benchmark/manifest.json');assert [t for t in benchmark['tasks'] if t['role']=='val']==[t for t in parent['tasks'] if t['role']=='val'];assert len(set(benchmark['calibration_groups']))==len(set(benchmark['probe_groups']))==24;assert not set(benchmark['calibration_groups'])&set(benchmark['probe_groups']);assert population['train_groups']==48
    assert read(G/'acceptance.json')['neural_training_admitted'] is False
    for role in ('train','val'):
        for pool in ('same_music','energy_matched','cross_music'):
            for method in ('Global','Diagonal','Offset','Partial','CalibratedPartial','TrainBankDebias'):
                actual=evaluation['summary'][role][pool]['actual'][method]['group_means']
                for control in ('both_zero','music_swap','motion_swap'):
                    other=evaluation['summary'][role][pool][control][method]['group_means'];assert actual.keys()==other.keys();v=[actual[g]-other[g] for g in sorted(actual)];counterfactual['/'.join((role,pool,method,control))]=dict(mean=float(np.mean(v)) if v else None,ci95=group_ci(v,44041,5000),groups=len(v))
    regression=re.search(r'\d+ passed[^\n]*', (G/'full_pytest_final.log').read_text()).group(0)
    summary=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),baseline='5e0506c2f29ca6fb8600620c60affed641cab34f',regression=regression,generation=acceptance,population=population,retrieval_summary=evaluation['summary'],calibration=evaluation['calibration'],retrieval_counterfactual_differences=counterfactual,native_hubness=native['hubness'],native_availability=native['availability'],native_calibrated=cal,official_fk=dict(trajectories=quality['official_fk_trajectories'],maximum_error_m=quality['maximum_fk_error_m']),parents=parents,receipts=receipts,optimizer_steps=0,new_neural_samples=0,whole_song_generations=0,can_route=False,human_reviewed=False,operational_qualified=False)
    out.mkdir(parents=True);write(out/'summary.json',summary)
    lines=['# R44：来源姿态保持的音乐时钟与冻结检索审计','','本轮先交付并在浏览器验收22案例主候选，再运行后续对照。新增研究与正式EDGE/Planner/Native分开；R38–R43父代均保持封存。新训练0、新神经采样0、新整曲0。','','## 观看','','- `outputs/reperformance_r44_01/videos/index.html`：飞天、女冥想、琵琶2三个预登记来源，五栏实际256帧视频。','- `outputs/reperformance_r44_01/viewer/index.html`：先交付的主候选，22案例，共同音乐时钟、暂停/倍速/逐帧、Root-relative/世界视角、诊断及时间图。','- `outputs/reperformance_r44_01/viewer_controls/index.html`：加入错音乐时间图、固定图换音乐与无音乐时间图。','- 原时钟真实来源的隐藏缺口只作评价；音乐重定时参考与它不同步的地方不应误读为来源丢失。几何双种子输出相同，不是独立样本。','','## 合法时钟与实际动力学','','G0是冻结Geometry，G1是旧E1-W分段线性时钟，G2是有解析速度界的C²五次时钟，G4是R43已保存Core失败样本。两种重定时使用同一SO(3) RotationSpline和Root CubicSpline，不新增姿态残差；Bridge及保护域精确保持。G2同时限制相位位移和速度变化，因此G1/G2不是只改变C²单一因素的消融。','', '|来源|角色|C²活动度/Geometry|C²Core D3/Geometry|动力学筛查|','|---|---|---:|---:|---|']
    for r in acceptance['records'][:8]:lines.append(f"|{r['key']}|{r['role']}|{r['activity_ratio']:.4f}|{r['core_D3_ratio']:.4f}|{r['dynamics_screen']}|")
    screen=acceptance['train_only_screen'];lines+=['',f"ChangE Train筛查 {screen['change_passed']}/{screen['change_total']}；正确音乐的RMS重音时间代理相对WrongMapB和NoMusicTime的音乐组bootstrap区间分别为 {screen['music_proxy']['WrongMapB']['ci95']}、{screen['music_proxy']['NoMusicTime']['ci95']} 秒。总准入 {screen['passed']}，决定：{acceptance['training_decision']}。",'固定时间图换播放音乐时，动作逐元素相同；改变音乐并重新求图才会改变动作时钟。这证明的是显式时间映射，而非神经网络新学到音乐表达。RMS重音和上身速度峰均为代理，不能当作真实逐事件语义。','',f"官方FK核对 {quality['official_fk_trajectories']} 条轨迹，最大误差 {quality['maximum_fk_error_m']:.3g} m。另保存模型Root D1/D2/D3、世界FK和足部速度代理；未知接触不能计算足滑真值，22关节不能通过完整Native24。Root是继承模型先验，非ChangE世界轨迹。",'','## 冻结SDMR人口与代价','','全48 Train音乐组，两个原始录制、每长度90/150/240一个元数据中位窗；原Val任务不变。同一录制候选排除，UNKNOWN不硬标。正例是官方物理时钟关联，不是艺术适配真值。共享编舞与缺失支持完整报告，不宣称未见编舞泛化。',f"查询 Train={population['train_queries']}，Val={population['val_queries']}，有效任务={population['tasks']}；不可用查询池={len(population['unavailable'])}，缺失查询长度支持={len(population['missing'])}。",'','|角色/候选池|Global|Diagonal|Offset|Partial|CalibratedPartial|TrainBankDebias|','|---|---:|---:|---:|---:|---:|---:|']
    for role in ('train','val'):
        for pool in ('same_music','energy_matched','cross_music'):
            methods=evaluation['summary'][role][pool]['actual'];lines.append('|'+role+'/'+pool+'|'+'|'.join(f"{100*methods[m]['group_macro_recall_at1']:.2f}" if methods[m]['group_macro_recall_at1'] is not None else 'N/A' for m in ('Global','Diagonal','Offset','Partial','CalibratedPartial','TrainBankDebias'))+'|')
    c=evaluation['calibration'];lines+=['',f"固定路径match=1-sim，双侧drop总计0.6，局部无差异相似度约0.4；这不是全局DP的充分必要条件。原始23×24候选全部与冻结路径评分核对。Train24组负例q95中位数校准阈值={c['threshold']:.6f}，drop={c['drop_cost']:.6f}；另外24个Train probe音乐组宏FPR={c['probe_macro_fpr']:.4f}，TPR={c['probe_macro_tpr']:.4f}，信息性筛查={c['informative']}。这些probe仍是编码器已训练人口，不是新泛化集。",'TrainBankDebias为固定Train查询库K=10的候选亲和度校正，未完整复现BeatDance。仅在配对候选任务评价；没有把该校正未经验证地接到Native选路。候选池含离线构造正例，不能当部署检索总体准确率。','', '## Native来源集中与拒配','','|方法|分数并列均分Top1来源计数|所选候选平均路径覆盖|','|---|---|---:|']
    for m,v in native['hubness'].items():lines.append('|'+m+'|'+str(v['fractional_top1'])+'|'+f"{100*v['mean_selected_coverage']:.3f}%"+'|')
    lines+=['',f"Train校准移到Native后，平均并列Top1覆盖 {100*np.mean([r['mean_tied_top1_coverage'] for r in cal['records']]):.3f}%；增加覆盖不等于拒配/语义正确。来源可用频次和Top1/可用候选比例保留在摘要；不能把飞天数量多的候选池与音乐适配能力混为一谈。",'`native/blind_items.json`、`admin_map.json`仅为552项独立盲审准备，未产生人工结论；E/P/M任务、弃权和争议处理保持分开。无人工标签不阻断本轮客观实验，但不能声称艺术适配通过。','','## 可复核边界','','原始/校准路径、完整候选评分、音乐置换/动作置换/双置零及音乐组置信区间均封存。特征、候选人口、Native、主候选和全部后续对照均由新进程复算；复验不优化权重。完整回归605 passed、10 skipped；7个父代逐文件SHA复核；发布回执及失败记录保留。', '保持正式选路关闭。下一步应先解决音乐代理收益不稳和留出动态风险，不因合同通过增加训练或强跑整曲。','','参考机制：[SciPy RotationSpline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RotationSpline.html)、[BPoly.from_derivatives](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BPoly.from_derivatives.html)、[BeatDance ICMR2024官方实现](https://github.com/XulongT/BeatDance)、[Drop-DTW官方实现](https://github.com/SamsungLabs/Drop-DTW)。本轮路径为严格一对一match/drop，并非完整Drop-DTW；C²不保证Jerk降低。']
    lines=[line.replace('完整回归605 passed、10 skipped','最终完整回归 '+regression) for line in lines]
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for root in (G,A):
        manifest=root/'artifacts_manifest.json'
        if manifest.exists():raise ValueError('Never replace a seal')
        files=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('artifacts_manifest.json','publication_receipt.json')];write(manifest,dict(source_commit=summary['source_commit'],files=files,optimizer_steps=0,operational_qualified=False));print('R44_SEALED',root,len(files),sum(f['bytes'] for f in files),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
