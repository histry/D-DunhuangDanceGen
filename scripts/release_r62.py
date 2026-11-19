"""Publish R62 scalar evidence and receipts, never private motion or viewer data."""
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from dunhuang.research.duration_r62.common import ROOT,CFG,read,sha,fresh

EXECUTION=Path('outputs/r62_execution_01')


def write_json(path,value):
    Path(path).write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))


def csvfile(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf8',newline='') as f:
        if keys:
            w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)


def columns(prefix,interval):
    return {prefix+'_start':None if interval is None else interval[0],
            prefix+'_end_exclusive':None if interval is None else interval[1]}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def scalars(value,prefix=''):
    """Do not export coordinate arrays, clocks, curves, mask lists or full results."""
    out={}
    for k,v in value.items():
        if k in ('arrays_key','jerk_curve'):continue
        if isinstance(v,dict):out.update(scalars(v,prefix+k+'_'))
        elif v is None or isinstance(v,(str,bool,int,float)):out[prefix+k]=v
    return out


def mask_counts(case,kind,audit):
    row=dict(key=case['key'],case=case['case'],source_uid=case['source_uid'],dataset=case['dataset'],
        gap_frames=case['gap'],audit_group=kind,edit_target_frames=audit['edit_target_frames'],
        edited_center_times=len(audit['edited_centers']),center_scope='UNIQUE_SOURCE_FRAME_TIMES_NOT_EVENTS',
        verified_status=audit['verified_status'],verified_intrusion_fraction=audit['verified_intrusion_fraction'],
        task_status=audit['task_status'],
        denominator_scope='SELECTED_PHASE_SOURCE_SUPPORT' if kind=='TARGET_PHASE' else 'WHOLE_RECORD_ALL_EVIDENCE_SOURCE_UNION_NOT_LOCAL_WINDOW')
    for name in ('raw','band'):
        x=audit[name]
        row.update({name+'_target_evidence_frames':sum(x['target_mask']),
            name+'_edit_overlap_target_frames':x['edit_overlap_target_frames'],
            name+'_edited_source_frame_count':x['edited_source_frame_count'],
            name+'_source_support_frame_count':x['source_support_frame_count'],
            name+'_source_intrusion_fraction':x['source_intrusion_fraction']})
    return row


def p0_summary_rows(summary,stratum='ALL_DIFFICULTY'):
    rows=[]
    for gap,datasets in summary.items():
        for dataset,s in datasets.items():
            row=dict(gap_frames=int(gap),dataset=dataset,difficulty=stratum,
                registered=s['registered'],eligible=s['eligible'],not_eligible=s['not_eligible'],sources=s['sources'])
            for name,value in s['methods'].items():
                row.update({name+'_'+k:v for k,v in value.items()})
            if not s['paired']:rows.append(row)
            for comparison,stats in s['paired'].items():
                rows.append(dict(row,comparison=comparison,paired_sources=stats['sources'],paired_tasks=stats['tasks'],
                    difference_mean=stats['mean'],ci95_lower=None if stats['ci95'] is None else stats['ci95'][0],
                    ci95_upper=None if stats['ci95'] is None else stats['ci95'][1],scope=stats.get('scope')))
    return rows


def method_failures(task,method,value,reference,stage):
    base=dict(stage=stage,key=task['key'],source_uid=task['source_uid'],
        gap_frames=task.get('gap_frames',task.get('gap')),method=method)
    if not value['constructible']:
        return [dict(base,failure='CONSTRUCTION_REJECTION',reason=value.get('reason','FROZEN_METHOD_REJECTION'))]
    if value['passes_original_screen']:return []
    m=value['metrics'];reasons=[]
    if m['gap_D2']>reference['D2']:reasons.append('D2_REFERENCE')
    if m['gap_D3']>reference['D3']:reasons.append('D3_REFERENCE')
    if m['shoulder_wrist_min_m']<reference['shoulder_wrist_lower_m']:reasons.append('SHOULDER_WRIST_REFERENCE')
    return [dict(base,failure='ORIGINAL_NUMERICAL_SCREEN',reason='|'.join(reasons))]


def fmt(value):
    return '不可计算' if value is None else '%.6f'%value


def release():
    main=read(ROOT/'main/result.json');controls=read(ROOT/'controls/result.json')
    cmain=read(ROOT/'main/continuous/result.json');xmain=read(ROOT/'main/cross_source/result.json')
    continuous=read(ROOT/'controls/continuous/result.json');cross=read(ROOT/'controls/cross_source/result.json')
    independent=read(ROOT/'independent/result.json')
    browser=read(ROOT/'receipts/main_browser.json')
    assert browser['passed'] and browser['main_result_sha256']==sha(ROOT/'main/result.json')
    for stage,root in ((main,ROOT/'main'),(controls,ROOT/'controls')):
        assert stage['continuous_result_sha256']==sha(root/'continuous/result.json')
        assert stage['cross_result_sha256']==sha(root/'cross_source/result.json')
        assert stage['training_steps']==stage['optimizer_calls']==stage['planner_runs']==stage['joint_budget_planning_runs']==0
    assert cross['main_G1_exact'] and cross['main_result_sha256']==sha(ROOT/'main/cross_source/result.json')
    assert independent['fresh_source_decoding_and_curve_solves'] and independent['optimizer_calls']==0
    assert {r['mode'] for r in independent['records']}=={'main','controls'}
    for row in independent['records']:
        assert row['scientific_values_exact'] and row['arrays_bytes_exact']
        assert row['source_result_sha256']==sha(ROOT/row['mode']/'result.json')
    receipts={}
    for action in ('main','controls','verify','tests'):
        receipts[action]=read(EXECUTION/(action+'_receipt.json'))
        assert receipts[action]['exit_status']==0
    log=(EXECUTION/'tests.log').read_text(encoding='utf8')
    match=re.search(r'(\d+) passed',log);assert match
    skip=re.search(r'(\d+) skipped',log)
    tests=dict(passed=int(match.group(1)),skipped=int(skip.group(1)) if skip else 0,
        log_sha256=sha(EXECUTION/'tests.log'))
    assert tests['passed']>0
    assert continuous['baseline48_parity']==cmain['baseline48_parity']
    for row in continuous['baseline48_parity']:
        assert all(row[k] for k in ('truth_exact','G1_curve_exact','scalar_metrics_exact','target_and_all_masks_exact'))
    assert continuous['difficulty_stratification']==cmain['difficulty_stratification']
    assert main['parent_seal']==controls['parent_seal']

    out=fresh('reports/r62_release')
    for action,value in receipts.items():write_json(out/(action+'_receipt.json'),value)
    prior_receipts=[]
    for name,expected in (('main_failed_serialization',1),('tests_initial',0)):
        p=EXECUTION/(name+'_receipt.json')
        if p.exists():
            value=read(p);assert value['exit_status']==expected
            write_json(out/(name+'_receipt.json'),value);prior_receipts.append(name)
    failed_archive=ROOT.parent/'duration_r62_failed_serialization_01'
    if failed_archive.exists():
        write_json(out/'failed_serialization_archive.json',dict(
            scope='RETAINED_FAILED_PREVIEW_WITHOUT_NEW_SCORE_OR_THRESHOLD_SELECTION',
            files=[dict(path=p.relative_to(failed_archive).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
                   for p in sorted(failed_archive.rglob('*')) if p.is_file()],
            original_data_bytes_published=False))
    write_json(out/'independent.json',independent);write_json(out/'main_browser.json',browser)
    controls_viewer=ROOT/'receipts/controls_browser.json'
    if controls_viewer.exists():
        value=read(controls_viewer);assert value['passed']
        if 'controls_result_sha256' in value:assert value['controls_result_sha256']==sha(ROOT/'controls/result.json')
        write_json(out/'controls_browser.json',value)

    tasks=[];metric_rows=[];audits=[];failures=[];guard_rows=[]
    for row in continuous['records']:
        q=dict(key=row['key'],case=row['case'],source_uid=row['source_uid'],dataset=row['dataset'],role=row['role'],
            source_R_sha256=row['source_R_sha256'],gap_frames=row['gap'],status=row['status'],
            allocated_gap_seconds=row['gap']/30,endpoint_separation_seconds=(row['gap']+1)/30,
            motion_difficulty=row['motion_difficulty'],baseline48_endpoint_mean_angle_rad=row['baseline48_endpoint_mean_angle_rad'],
            human_reviewed=False,verified_phase='UNKNOWN')
        q.update(columns('source',row.get('source_interval')));q.update(columns('source_edit',row.get('edit_source_interval')))
        if row['status']!='CONTINUOUS_GAP_EVALUATED':
            failures.append(dict(stage='P0',key=row['key'],source_uid=row['source_uid'],gap_frames=row['gap'],
                failure='SOURCE_SUPPORT_OR_ROSTER',reason=row.get('reason',row['status'])))
        else:
            q.update(columns('target_edit',row['edit_target_interval']))
            q.update(center=row['center'],phase_type=row['target_phase']['type'],target_frames=row['frames'])
            q.update({k:v for k,v in row.items() if k.startswith('G1_minus_')})
            for name,value in row['methods'].items():
                metric_rows.append(dict(key=row['key'],case=row['case'],source_uid=row['source_uid'],dataset=row['dataset'],
                    gap_frames=row['gap'],difficulty=row['motion_difficulty'],method=name,**scalars(value)))
                failures+=method_failures(row,name,value,continuous['reference'],'P0')
                guard=row['solver']['guards'].get(name,{})
                trials=guard.get('trials') or [{}]
                for i,trial in enumerate(trials):
                    guard_rows.append(dict(stage='P0',key=row['key'],method=name,trial_index=i,gap_frames=row['gap'],
                        selected_segments=guard.get('selected_segments'),
                        **scalars(guard,'guard_'),**scalars(trial,'trial_')))
            for name,key in [('TARGET_PHASE','target_phase_audit'),('ALL_EVIDENCE','all_evidence_audit')]:
                audits.append(mask_counts(row,name,row[key]))
        tasks.append(q)
    csvfile(out/'continuous_tasks.csv',tasks);csvfile(out/'continuous_metrics.csv',metric_rows)
    csvfile(out/'edit_support_audit.csv',audits)
    summaries=p0_summary_rows(continuous['summary'])
    for label,summary in continuous['difficulty_summary'].items():summaries+=p0_summary_rows(summary,label)
    csvfile(out/'continuous_paired_summary.csv',summaries)
    write_json(out/'difficulty_stratification.json',continuous['difficulty_stratification'])
    write_json(out/'baseline48_R61_parity.json',dict(records=continuous['baseline48_parity'],
        R61_fixed_result_sha256=continuous['R61_fixed_result_sha256'],R61_fixed_arrays_sha256=continuous['R61_fixed_arrays_sha256']))

    pair_rows=[]
    for pair in cross['roster']:
        q=dict(pair_case=pair['case'],pair_uid=pair['pair_uid'],frozen_plan_key=pair['frozen_plan_key'],
            frozen_edge_index=pair['frozen_edge_index'],frozen_gap_frames=pair['frozen_gap_frames'])
        for side in ('left','right'):
            e=pair[side]
            q.update({side+'_'+k:e[k] for k in ('event_uid','source_uid','source_R_sha256','full_core_R_sha256','full_core_frames','independent_source_recompile_exact')})
            q.update(columns(side+'_source',e['source_interval']));q.update(columns(side+'_original_target',e['target_interval']))
            q[side+'_warp_sha256']=digest(e['time_warp'])
            q[side+'_warp_knots']=len(e['time_warp'])
            q[side+'_native_sha256']=e['native_reference'].get('sha256')
        pair_rows.append(q)
    csvfile(out/'cross_source_pairs.csv',pair_rows)
    xmetrics=[];xtasks=[]
    for row in cross['records']:
        q=dict(key=row['key'],pair_case=row['pair_case'],pair_uid=row['pair_uid'],source_uid=row['source_uid'],
            gap_frames=row['gap_frames'],status=row['status'],local_total_frames=row['local_total_frames'],
            allocated_gap_seconds=row['gap_frames']/30,endpoint_separation_seconds=(row['gap_frames']+1)/30,
            fixed_full_pair_total_frames=row['fixed_full_pair_total_frames'],fixed_music_budget=False,
            retained_core_edit_overlap_frames=row['retained_core_edit_overlap_frames'],
            full_core_hashes_unchanged=row['full_core_hashes_unchanged'],
            boundary_poses_and_jets_identical_across_durations=row['boundary_poses_and_jets_identical_across_durations'],
            available_boundary_readset_sha256=row['available_boundary_readset_sha256'],observed_jets_sha256=row['observed_jets_sha256'],
            hidden_truth_available=False,transition_source_support='UNKNOWN_DIFFERENT_SOURCE_GAP')
        q.update(columns('target_edit',row['edit_target_interval']));xtasks.append(q)
        for name,value in row['methods'].items():
            xmetrics.append(dict(key=row['key'],pair_case=row['pair_case'],pair_uid=row['pair_uid'],
                source_uid=row['source_uid'],gap_frames=row['gap_frames'],method=name,**scalars(value)))
            failures+=method_failures(row,name,value,cross['reference'],'P1')
            guard=row['solver']['guards'].get(name,{})
            trials=guard.get('trials') or [{}]
            for i,trial in enumerate(trials):
                guard_rows.append(dict(stage='P1',key=row['key'],method=name,trial_index=i,gap_frames=row['gap_frames'],
                    selected_segments=guard.get('selected_segments'),
                    **scalars(guard,'guard_'),**scalars(trial,'trial_')))
    csvfile(out/'cross_source_tasks.csv',xtasks);csvfile(out/'cross_source_metrics.csv',xmetrics)
    csvfile(out/'failures.csv',failures)
    csvfile(out/'solver_admission.csv',guard_rows)
    write_json(out/'cross_duration_response.json',cross['summary'])

    source_ids=sorted({p[s]['source_uid'] for p in cross['roster'] for s in ('left','right')})
    source_incidence=Counter(p[s]['source_uid'] for p in cross['roster'] for s in ('left','right'))
    write_json(out/'source_dependence.json',dict(cross_unique_sources=source_ids,cross_pair_source_incidence=dict(source_incidence),
        cross_pair_count=len(cross['roster']),cross_pairs_share_sources=True,
        independent_edge_bootstrap_performed=False,independence='DEVELOPMENT_FROZEN_HEURISTIC_PAIRS; NO_UNSEEN_SOURCE_OR_PRETRAINING_INDEPENDENCE_CLAIM'))
    versions=dict(main_commit=main['binding']['commit'],controls_commit=controls['binding']['commit'],
        main_package_hashes=main['binding']['code'],controls_package_hashes=controls['binding']['code'],
        frozen_P0_geometry_sha256=continuous['frozen_geometry_sha256'],frozen_P1_algorithms=cross['algorithms'],
        R61_main_result_sha256=continuous['R61_main_result_sha256'],
        R61_fixed_result_sha256=continuous['R61_fixed_result_sha256'],R61_fixed_arrays_sha256=continuous['R61_fixed_arrays_sha256'],
        source_selection=cross['population'],frozen_numerical_reference=scalars(continuous['reference']))
    write_json(out/'versions.json',versions)
    summary=dict(schema='R62_DURATION_LOCAL_DIAGNOSTIC_PUBLIC_V1',main_commit=versions['main_commit'],
        controls_commit=versions['controls_commit'],durations=CFG['durations'],continuous_registered=len(continuous['records']),
        continuous_source_centers=len({r['case'] for r in continuous['records']}),
        continuous_eligible=sum(r['status']=='CONTINUOUS_GAP_EVALUATED' for r in continuous['records']),
        continuous_summary=continuous['summary'],continuous_difficulty_summary=continuous['difficulty_summary'],
        cross_pairs=len(cross['roster']),cross_source_count=len(source_ids),cross_records=len(cross['records']),
        cross_summary=cross['summary'],R61_parity_cases=len(continuous['baseline48_parity']),
        direct_parent_files_unchanged=controls['parent_files_unchanged'],
        inherited_frozen_parent_files_unchanged=independent['frozen_parent_files_unchanged'],tests=tests,
        training_steps=0,optimizer_calls=0,planner_runs=0,joint_budget_planning_runs=0,can_route=False)
    write_json(out/'summary.json',summary)
    write_json(out/'qualification.json',dict(
        P0='WITHIN_DURATION_SAME_TARGET_PAIRED_RECONSTRUCTION; NO_ERROR_RANKING_ACROSS_HIDDEN_LENGTHS',
        P1='FIXED_CORE_FIXED_WARP_FIXED_BOUNDARY_DURATION_RESPONSE; NO_SINGLE_HIDDEN_TRUTH',
        G0='ALL_REGISTERED_SOURCE_SUPPORT_AND_CONSTRUCTION_FAILURES_RETAINED',
        G1='DESCRIPTIVE_LOCAL_DURATION_RESPONSE_NOT_FIXED_BUDGET_QUALITY_GAIN',
        G2='ADAPTIVE_SELECTOR_NOT_QUALIFIED; NO_FROZEN_OUT_OF_SAMPLE_DECISION_NET_GAIN',
        G3='CORE_TRANSITION_JOINT_TOTAL_BUDGET_PLANNING_NOT_RUN',
        independent_edge_bootstrap_performed=False,cross_source_time_policy='EXTRA_GAP_TIME_NOT_BORROWED_FROM_CORES',
        human_reviewed=False,semantic_completeness='UNKNOWN',verified_phase='UNKNOWN',world_root_contact='UNKNOWN',
        Native24='NOT_EVALUABLE',training_steps=0,optimizer_calls=0,planner_runs=0,can_route=False))

    lines=['# R62 过渡时长与时间预算局部诊断', '',
        '主效果代码：`'+versions['main_commit']+'`；固定目标方法对照代码：`'+versions['controls_commit']+'`。先推GitHub、服务器拉取后tmux执行，主效果页面实际验收后才运行方法对照；本轮不训练神经模型、不扩展Planner。', '',
        '## 先看实际效果', '',
        '本机统一入口：`D:/DunhuangDanceGen/outputs/duration_r62_01/index.html`。真实连续补间与跨来源连接分开展示，24/36/48/60/72帧均按30FPS原速播放。不同长度不拉伸到同一时长，较短结果结束后不得当作继续生成；构造拒绝保留空缺。观看数据、骨架坐标、旋转NPZ和私有完整result不进入Git。', '',
        '## P0：真实连续动作的变长隐藏缺口', '',
        f"沿用R61的{summary['continuous_source_centers']}个来源中心，五时长共登记{summary['continuous_registered']}个任务，实际具备真实上下文{summary['continuous_eligible']}个；缺少支持时不平移中心或另选来源。每个时长内部，三方法使用相同隐藏目标与同一组4+4可用边缘观测。", '',
        'SLERP仅读两个邻接端点；SingleChartHermite与冻结G1实际使用相同的最近3+3姿态，估计相同身体角速度／角加速度及SO(3)导数约束。二者可用的第4帧保持不动但不进入冻结导数估计器。Hermite为单个Lie代数五次图，G1使用4/8/16重锚定分段及内部测地线状态；因此该比较同时涉及内部锚点／分段组织，不能称为只改变一项抽象几何性质。两者沿用同半径、同导数容差，拒绝不回退。', '',
        '|隐藏长度|数据|合格来源任务|G1−SLERP角误差rad，95%来源重采样区间|G1−单图Hermite角误差rad，95%来源重采样区间|',
        '|---:|---|---:|---|---|']
    for gap,datasets in continuous['summary'].items():
        for dataset,s in datasets.items():
            if dataset=='AIST++' and not s['registered']:continue
            values=[]
            for name in ('SLERP','SingleChartHermite'):
                q=s['paired']['G1_minus_'+name+'_error_rad'];value=fmt(q['mean'])
                if q['ci95'] is not None:value+=' ['+fmt(q['ci95'][0])+', '+fmt(q['ci95'][1])+']'
                values.append(value)
            lines.append('|%s|%s|%d/%d|%s|%s|'%(gap,dataset,s['eligible'],s['registered'],*values))
    lines += ['',
        '负值只表示同一个长度、同一个隐藏目标上G1平均重建误差更低。长度变化会改变真实隐藏内容和边界状态，不能把不同长度平均误差直接排序成“更长更好”。配对均值以双方均可构造为条件；全部拒绝、原数值筛查失败和人口缺失在failures.csv另列，不能把失败填零。', '',
        '纯观测难度分层先用固定48帧任务的可见端点平均SO(3)距离取全来源中位数，再把相同来源标签用于全部长度。分层不读取重建成绩、不用于选最佳长度；逐来源、数据域和固定难度分层的同长度差值完整公开。自动阶段侵入仍是代理重叠，不是语义完整性真值；目标阶段分母与整条来源证据分母明确区分。', '',
        f"48帧基线的{summary['R61_parity_cases']}个可评案例与R61原真值、G1轨迹、共享标量读数及目标／全来源掩码逐项精确一致。", '',
        '## P1：固定两个真实Core，仅改变连接帧数', '',
        f"按冻结R51案例编号和边顺序，最多请求{cross['population']['requested_maximum_pairs']}组，实际得到{summary['cross_pairs']}组不重复事件／Warp／边缘读集组合，涉及{summary['cross_source_count']}条来源。该短人口不足12组时保留实际短缺，不能补选更容易的案例。全部完整Core与合法Warp、端点姿态、最近4+4读集和估计边界jet在五时长间保持精确一致。", '',
        '这些边共享来源和历史启发式选择，不是独立未见事件样本；不对边做独立性假设的bootstrap。本轮没有新的独立音乐或人工偏好评分。跨来源间隙没有唯一真实隐藏动作，SO(3)/FK重建误差保持null；只报告构造、边界动态、活动幅度、数值筛查和真实观看结果。', '',
        '|过渡帧数|连接组数|G0可构造／原数值门|G1可构造／原数值门|', '|---:|---:|---:|---:|']
    for gap,s in cross['summary']['by_duration'].items():
        cells=[]
        for name in ('G0','G1'):
            q=s['methods'].get(name,dict(constructed=0,passed_original_screen=0))
            cells.append('%d/%d · %d/%d'%(q['constructed'],s['tasks'],q['passed_original_screen'],s['tasks']))
        lines.append('|%s|%d|%s|%s|'%(gap,s['tasks'],*cells))
    patterns=Counter('|'.join(map(str,r['passing_G1_durations'])) or 'NONE' for r in cross['summary']['pair_responses'])
    lines += ['', '固定连接组的G1通过长度集合：'+ '；'.join(k+'：'+str(v)+'组' for k,v in sorted(patterns.items()))+'。', '',
        '通过集合和最短通过长度均是观察到的事后Oracle描述，不是已经训练或验证的自适应决策器。D2/D3下降可能仅来自给过渡分配更多时间，需同时查看路径长度、关节活动、姿态范围和边界变化，不能自动解释为自然性提高。', '',
        '**本轮长过渡直接增加总时长，没有从固定Core合法时长中偿还帧数。** 因此不是固定音乐总帧数下的时间预算收益，也没有完成Core×Transition联合规划；新P2与整曲对照保持未运行。只有冻结选择规则在独立评价上超过固定48帧和简单规则、且没有静止化退化，才有依据启动联合时间预算研究。', '',
        '## 复验、封存与资格', '',
        f"完整回归{tests['passed']} passed、{tests['skipped']} skipped。独立新进程重新解码来源并重构main/controls两阶段的P0/P1全部曲线；科学值和确定性NPZ逐字节精确一致。直接R61父代封存{summary['direct_parent_files_unchanged']}个文件、继承历史父代{summary['inherited_frozen_parent_files_unchanged']}个文件保持不变。", '',
        '真实人工记录0，阶段语义与感知质量尚未建立独立验证；世界Root和接触UNKNOWN，身体22关节相对Root诊断不能升格为Native24。生产Catalog、冻结G0/G1与历史实验没有被覆盖。训练／优化器／Planner／固定总预算联合规划均0，正式路由关闭。所有图表数值都来自本轮保存的统计，未追加来源挑选、阈值放宽或结果驱动调参。']
    if 'main_failed_serialization' in prior_receipts:
        lines += ['', '## 保留的工程失败与唯一修正', '',
            '首轮主效果在跨来源边缘支持的NumPy整数JSON序列化处退出，失败回执与部分科学档案完整保留。修正仅将边缘支持索引转成Python原生整数，并补充序列化回归；未改变来源人口、几何求解、筛查阈值或评分。之后从同一预登记合同完整重跑主效果，再进行观看与对照。首轮测试回执与最终回归分别记账，不以成功运行覆盖失败谱系。']
    (out/'report.md').write_bytes(('\n'.join(lines)+'\n').encode('utf8'))
    write_json(out/'scientific_files.json',dict(files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(ROOT.rglob('*')) if p.is_file()],scope='HASH_DIRECTORY_ONLY; NO_RAW_ARRAY_OR_MEDIA_BYTES'))
    write_json(out/'public_manifest.json',dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(out.iterdir()) if p.is_file()],media_published=False,body_coordinates_published=False,
        raw_arrays_published=False,private_complete_result_published=False))
    print('R62_RELEASE',len(list(out.iterdir())),flush=True)


if __name__=='__main__':release()
