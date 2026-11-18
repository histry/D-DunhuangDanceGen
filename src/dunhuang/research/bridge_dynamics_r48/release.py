"""Measured evidence release; distinguish absolute source proxies and physics truth."""
import argparse,collections,re,subprocess
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write
from dunhuang.research.reperformance_r44.replay import digest

ROOTS=('event_feasibility_r48_01','bridge_dynamics_r48_01','phase_discrimination_r48_01','event_feasibility_r48_replay','phase_discrimination_r48_replay')

def run(out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh release')
    pub=Path('outputs/r48_publication');G=Path('outputs/bridge_dynamics_r48_01');C=Path('outputs/event_feasibility_r48_01');P=Path('outputs/phase_discrimination_r48_01');g=read(G/'controls/result.json');c=read(C/'result.json');p=read(P/'result.json');ind=read(pub/'independent.json');root=read(pub/'root_contact_audit.json');assert ind['all_scientific_values_exact'];receipts={}
    for name in ('primary','controls','verify'):
        value=(pub/(name+'_receipt.txt')).read_text();assert 'exit_status=0' in value;receipts[name]=value
    for folder,name in ((G/'viewer','main'),(G/'viewer_controls','controls'),(C/'viewer','event')):
        receipt=read(G/'receipts'/(name+'_browser_acceptance.json'));assert receipt['passed'] and receipt['protocol_sha256']==digest(folder/'protocol.json')
    counts=collections.Counter(r['selected'] or 'REJECT' for r in g['records']);summary=[]
    for r in g['records']:
        summary.append({k:r[k] for k in ('key','kind','source_window','methods','certificates','screens','selected','rejection')}|dict(left_event=r['left']['unit_uid'],right_event=r['right']['unit_uid'],metrics={name:{k:v for k,v in m.items() if k!='jerk_curve'} for name,m in r['metrics'].items()}))
    regression=re.search(r'\d+ passed[^\n]*',(pub/'full_pytest_final.log').read_text()).group(0)
    catalog=dict(candidates=len(c['events']),supported_frames=c['source_frames_covered'],unknown_frames_excluded=c['unknown_source_frames_excluded'],actual_music_slot_candidates=c['actual_music_slots'],events_with_legal_clocks=sum(bool(e['clock_variants']) for e in c['events']),legal_clock_variants=sum(len(e['clock_variants']) for e in c['events']),maximum_source_frames=max(e['source_interval'][1]-e['source_interval'][0] for e in c['events']),longer_than450=sum(e['source_interval'][1]-e['source_interval'][0]>450 for e in c['events']),legacy499_replaced=False)
    result=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),baseline=g['config']['baseline'],regression=regression,catalog=catalog,bridge=summary,selection_counts=dict(counts),absolute_source_reference=g['reference'],phase={k:v for k,v in p.items() if k not in ('records','recording_balanced_group_scores')},root_contact=root,independent=ind,receipts=receipts,neural_optimizer_steps=0,new_neural_samples=0,whole_song_generations=0,human_reviewed=False,can_route=False,operational_qualified=False)
    out.mkdir(parents=True);write(out/'summary.json',result)
    lines=['# R48：事件可行性、固定锚点动态桥与冻结内容对照','','本轮先回传实际主候选、验收逐帧与4条真实视频，再执行G2及音乐内容对照。保留正式Planner/EDGE/Native、旧499及R41–R47所有权重/封存。新代码分别位于event_feasibility_r48、bridge_dynamics_r48、phase_discrimination_r48、root_contact_audit_r48。','','## 观看入口','','- `outputs/bridge_dynamics_r48_01/viewer/index.html`：32个新身份任务，来源/G0/G3主效果；G1可切换。','- `outputs/bridge_dynamics_r48_01/videos/index.html`：飞天、女冥想、琵琶2、异源cross_00，Source/G0/G3四条真实256帧视频。','- `outputs/bridge_dynamics_r48_01/viewer_controls/index.html`：加入G2的实际数组。','- `outputs/event_feasibility_r48_01/viewer/index.html`：139候选的真实音乐槽位、E1-W、完整记忆和局部窗口；可打开R47完整来源骨架。','本轮没有新增神经动作，没有将新异源连接配上不存在的真实音乐。逐帧117=3.9s、118=3.933333s，播放到255不自动循环；异源Source缺口留空。局部姿态不显示伪世界Root或接触真值。','','## P0：候选与实际执行合同','',f"139事件保持{catalog['supported_frames']}个已接受来源帧，{catalog['unknown_frames_excluded']}个质量未知帧全部排除。5个超过450帧，最长{catalog['maximum_source_frames']}；不因模型256帧接口重新切开长保持。",f"对三首历史音乐的原R36槽位及相邻2/3槽集合，共{catalog['actual_music_slot_candidates']}个实际目标时段。{catalog['events_with_legal_clocks']}/139事件有至少一个合法时间图，共{catalog['legal_clock_variants']}个合法变体。没有合法时段的事件保留不可用状态，不拉伸范围。",'EventCandidateV2保存原始/Native SHA、连续区间、有效组件、阶段证据、前后上下文、真实合法时长、具体时钟与256/128窗口。重叠窗口读取完全相同源帧位置，完整来源记忆不截断；短尾只记录真实帧数，不伪造padding观测。阶段名显式适配E1-W上身峰值/转折优先级。','这些只是研究候选/时间图，未改三首正式计划、未生成完整歌舞。can_route=false。','','## P1：新人口与六维内部动态','', '每来源按元数据三分位选两个候选，在接受组件内取连续256帧，16个同录制缺口；另按来源顺序形成16个异源出口/入口。这是区别于R47旧22案例的新固定人口。G4对应更换事件端点的连接审计，不与同端点G0–G3混成一个误差任务。','G3保持4段、5个测地锚点和两侧三帧身体角速度/角加速度。仅优化3个内部速度比例[0,2]及3个沿测地轴加速度比例[-2,2]；一个初始点，最大36次求解评估，额外有限差分调用全部计数。D2/D3/相对肩腕形状/偏离权重0.1/1/0.05/0.05。没有扩大图域、没有读隐藏动作、没有优化Root/contact。','所有方法自身π−0.05图域先检查，G3解析身体jet残差<1e−7；真实30Hz边界率、加速度和D3另外评价。G2内部角速度/角加速度连续，但未钳制同一观察jet；不能以低D3证明满足G1/G3边界合同。',f"固定Train来源参考：D2 Q95={g['reference']['D2']:.6g}，D3 Q95={g['reference']['D3']:.6g}，肩腕距离Q01={g['reference']['shoulder_wrist_lower_m']:.6g}米。只用Train四录制已接受完整支持建立，不读取持出结果校准。这是来源相对身体数值分布，不是普适物理阈值。",f"研究性选择计数：{dict(counts)}；G0/G1/G3仅在自身图域、绝对D2/D3参考及肩腕参考全部通过时按Gap D3选，否则拒绝。没有用隐藏来源误差选桥。",'','|任务|类别|G0图域|G1 Gap D3|G3 Gap D3|G3/G1|选择|','|---|---|---|---:|---:|---:|---|']
    for r in g['records']:
        m=r['metrics'];a=m.get('G1',{}).get('gap_D3');b=m.get('G3',{}).get('gap_D3');ratio=b/max(a,1e-12) if a is not None and b is not None else None;lines.append(f"|{r['key']}|{r['kind']}|{r['certificates']['G0']['chart_eligible']}|{a}|{b}|{ratio}|{r['selected'] or 'REJECT'}|")
    lines+=['','G3测试的是这六个共享标量的受限动态协调，未覆盖任意关节的所有可能内部状态；局部解不证明全局最优或不可行。源隐藏误差仅在16个同录制任务评价，异源不存在唯一答案。保护区保持精确原数组。数值候选也不能证明文化构形、碰撞、自然性或接触合格。','','## P2：同任务冻结R41内容评分','','沿用R47的476条跨录制记录/444来源对和48个Train音乐组、相同五偏移。各主要方法只读取共同13帧有效卷积与所有偏移支持，保留原始/共同阶段数；音频置换也用相同保留人口。整录制求导、原Train归一化和最终R41权重均逐SHA验证；输出为冻结96D局部token。此处48组属于既有训练人口，不是新的独立泛化评价。','音频配置22050Hz/735hop恰好30Hz；显式审计n*hop/sr、source+audio_offset、13帧有效掩码及正偏移读取更晚音乐。没有凭低于机会值反推一个新偏移或调权重。','', '|方法|音乐组|R@1|组bootstrap95%区间|','|---|---:|---:|---|']
    for name,v in p['summary'].items():lines.append(f"|{name}|{v['groups']}|{v['accuracy']:.6f}|{v['ci95']}|")
    lines+=['','|B2Content 相对对照|改善|95%区间|必要检验|','|---|---:|---|---|']
    for name,v in p['comparisons'].items():lines.append(f"|{name}|{v['mean_improvement']}|{v['ci95']}|{v['passed']}|")
    lines+=['',f"内容必要检验={p['necessary_content_screen']}，训练准入=False。",'B2TypeShuffled与B2Content相同是因为阶段类型未输入该内容头；它检验接口，不作为要求模型胜过自身的晋级门。B0/B1按同共同支持重算，不能直接对比R41旧30.16%或R47旧13.13%–18.62%区间。','没有已验证的阶段真值，B3Verified=NOT_EVALUABLE。B3UncalibratedProxyOnly仅在R47代理支持子集上把内容分数与0.1阶段项相加，人口不同，不能混入主公平比较或叫阶段收益证明。代理对应和Train重用均不授权新音乐阶段监督训练。','','## P3/P4与科学边界','','R45 184项匿名审查包保留，真实审核0，艺术/文化语义NOT_EVALUABLE；未因缺少人工审核停止本轮客观实验。8条ChangE Native数组的形状、有限值、Root范围、接触通道逐项审计；数字存在不等于来源观察真值。ChangE世界Root和接触仍UNKNOWN，AIST为核验时钟的拟合平移而非接触真值。完整Native24/真实足滑继续不可评，FineDance未重新准入。','','## 独立重放、发布与清理','',f"完整回归：{regression}。两个桥数组包、候选/时钟合同、全部真实音频内容评分、Root/contact审计均新进程重算，所有科学数值精确一致。冻结父代16根目录共{ind['parent_files']}文件逐SHA不变。"]
    for v in ind['official_fk']:lines.append(f"- 官方FK实际执行{v['trajectories']}条桥轨迹，最大差{v['maximum_error_m']:.9g}米；零Root只用于相对身体实现一致性，不作物理通过。")
    lines+=['','代码先推GitHub，再服务器tmux。成功/失败、launcher和SHA回执保留，观看包逐哈希回传并实际浏览器验收；最终发布/临时SSH清理见publication_receipt与cleanup_receipt。仍被研究依赖及封存引用的旧代码保留，没有为清理外观破坏基线。','本轮神经训练0、新神经采样0、整曲0。不得将几何优化叫学习式音乐演绎；下一步应以真实时钟内容检验、合法来源支持和实际动态证据决定。','','## 实现依据','','采用[SciPy bounded least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)的受限局部最小二乘机制，未声称二阶SQP或全局解；音频时间按[librosa0.11 onset中心定义](https://librosa.org/doc/0.11.0/generated/librosa.onset.onset_strength.html)审计，未通过经验偏移调参重贴时钟标签。']
    ratio=np.median([r['metrics']['G3']['gap_D3']/max(r['metrics']['G1']['gap_D3'],1e-12) for r in g['records'] if 'G3' in r['metrics'] and 'G1' in r['metrics']])
    lead=['## 实际结论','',f"冻结R41内容在同合同Train时钟任务R@1={p['summary']['B2Content']['accuracy']:.4%}，优于能量、固定规则、置零和换音乐，必要内容检验通过。48组既有训练音乐重用，不能提升为独立泛化或语义阶段真值。G3/G1的Gap D3中位比={ratio:.8f}，仅极小整体变化，不支持统一升级。研究候选{dict(counts)}，所有正式路由、物理资格仍关闭。",'没有经过验证的阶段监督，B3Verified不可评；此处缺失的是可核验阶段关系，不要求必须由人工提供。已有物理时钟足以开展本轮客观测试，但不能冒充语义阶段监督。没有新增神经训练或整曲。','']
    lines[4:4]=lead
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for name in ROOTS:
        root=Path('outputs')/name;m=root/'artifacts_manifest.json'
        if m.exists():raise ValueError('Do not replace a seal')
        files=[dict(path=f.relative_to(root).as_posix(),bytes=f.stat().st_size,sha256=digest(f)) for f in sorted(root.rglob('*')) if f.is_file() and f.name not in ('artifacts_manifest.json','publication_receipt.json')];write(m,dict(source_commit=result['source_commit'],files=files,neural_optimizer_steps=0,operational_qualified=False));print('R48_SEALED',name,len(files),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
