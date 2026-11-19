"""Public R61 numerical release: no source tensors, coordinates, or media."""
import csv
import json
import re
from collections import Counter
from pathlib import Path

from dunhuang.research.edit_support_r61.common import ROOT, read, sha, fresh

EXECUTION = Path('outputs/r61_execution_01')


def write_json(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf8'))


def csvfile(path, rows):
    if not rows:
        Path(path).write_bytes(b'')
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def mask_row(case, group, audit):
    """Export counts, never full positions/masks or motion coordinates."""
    row = dict(case=case['case'], source_uid=case['source_uid'], dataset=case['dataset'],
               audit_group=group, edit_target_frames=audit['edit_target_frames'],
               edited_centers_unique_source_frame_count=len(audit['edited_centers']),
               verified_status=audit['verified_status'],
               verified_intrusion_fraction=audit['verified_intrusion_fraction'],
               task_status=audit['task_status'],
               clock='ORIGINAL_SOURCE_30FPS_IDENTITY_CORRESPONDENCE',
               center_count_scope='UNIQUE_SOURCE_FRAME_TIMES_NOT_INDEPENDENT_EVENTS',
               denominator_scope=('TARGET_PHASE_SOURCE_SUPPORT' if group=='P2_TARGET_PHASE' else
                                  'UNION_OF_ALL_PROVIDED_WHOLE_RECORD_SOURCE_EVIDENCE_NOT_56_FRAME_WINDOW'))
    for kind in ('raw', 'band'):
        x = audit[kind]
        row.update({kind+'_target_evidence_frames': sum(x['target_mask']),
                    kind+'_edit_overlap_target_frames': x['edit_overlap_target_frames'],
                    kind+'_edited_source_frame_count': x['edited_source_frame_count'],
                    kind+'_source_support_frame_count': x['source_support_frame_count'],
                    kind+'_source_intrusion_fraction': x['source_intrusion_fraction'],
                    kind+'_evidence_status': x['evidence_status']})
    return row


def interval_columns(prefix, interval):
    if interval is None:
        return {prefix+'_start': None, prefix+'_end_exclusive': None}
    return {prefix+'_start': interval[0], prefix+'_end_exclusive': interval[1]}


def fmt(value, digits=6):
    return '不可计算' if value is None else ('%.*f' % (digits, value))


def release():
    main = read(ROOT/'main/result.json')
    fixed = read(ROOT/'fixed_gap/result.json')
    replay_main = read(ROOT/'independent_main/result.json')
    replay_fixed = read(ROOT/'independent_fixed/result.json')
    browser = read(ROOT/'receipts/main_browser.json')
    assert browser['passed'] and browser['main_result_sha256']==sha(ROOT/'main/result.json')
    assert replay_main['population_exact'] and replay_main['crop_R_and_removed_masks_exact']
    assert replay_fixed['result_exact'] and replay_fixed['arrays_bytes_exact']
    assert fixed['main_result_sha256']==sha(ROOT/'main/result.json')
    assert fixed['phase_labels_used_by_solvers'] is False
    assert fixed['optimizer_calls']==fixed['training_steps']==fixed['planner_runs']==0
    receipts = {}
    for action in ('main', 'fixed', 'verify_main', 'verify', 'tests'):
        p = EXECUTION/(action+'_receipt.json')
        value = read(p)
        assert value['exit_status']==0, str(p)
        receipts[action] = value
    test_log = (EXECUTION/'tests.log').read_text(encoding='utf8')
    passed_match = re.search(r'(\d+) passed', test_log)
    skipped_match = re.search(r'(\d+) skipped', test_log)
    assert passed_match and skipped_match
    tests = dict(passed=int(passed_match.group(1)), skipped=int(skipped_match.group(1)),
                 log_sha256=sha(EXECUTION/'tests.log'))
    assert tests['passed']==780 and tests['skipped']==1

    main_cases = main['cases']
    matched = [c for c in main_cases if c['selection']['status']=='MATCHED']
    fixed_cases = fixed['records']
    for c in matched:
        q = c['selection']; a,b = q['A4']; A,B = q['A5']; x,y = q['evidence']['evidence_span']
        assert a==A and a<B<b and a<=x<y<=b
        assert q['added_frames']==0 and q['removed_raw_frames']>0
        assert c['source_only_edit_audit']['edit_target_frames']==0
        assert c['source_only_edit_audit']['verified_status']=='UNKNOWN'
    for row in fixed_cases:
        if row['status']!='FIXED_GAP_EVALUATED':
            continue
        assert row['edit_target_interval']==[4,52] and row['frames']==56
        assert row['solver']['solver_hidden_read'] is False
        for key in ('target_phase_audit','all_evidence_audit'):
            assert row[key]['edit_target_frames']==48
            assert row[key]['verified_status']=='UNKNOWN'
            assert row[key]['verified_intrusion_fraction'] is None

    out = fresh('reports/r61_release')
    for action, value in receipts.items():
        write_json(out/(action+'_receipt.json'), value)
    write_json(out/'independent_main.json', replay_main)
    write_json(out/'independent_fixed.json', replay_fixed)
    write_json(out/'main_browser.json', browser)
    if (ROOT/'receipts/fixed_browser.json').exists():
        fixed_browser = read(ROOT/'receipts/fixed_browser.json')
        assert fixed_browser['passed'] and fixed_browser['fixed_result_sha256']==sha(ROOT/'fixed_gap/result.json')
        write_json(out/'fixed_browser.json', fixed_browser)

    population = []
    masks = []
    for c in main_cases:
        q = c['selection']
        row = dict(case=c['case'], source_uid=c['source_uid'], dataset=c['dataset'], role=c['role'],
                   phase_type=c['phase_type'], status=q['status'], source_R_sha256=c['source_R_sha256'],
                   human_reviewed=False, verified_phase='UNKNOWN', semantic_completeness='UNKNOWN')
        if q['status']=='MATCHED':
            p = q['evidence']; x,y = p['evidence_span']
            row.update(interval_columns('A4', q['A4']))
            row.update(interval_columns('A5', q['A5']))
            row.update(interval_columns('raw_evidence', p['evidence_span']))
            row.update(interval_columns('crop_removed', q['removed_interval']))
            row.update(phase_center=p['frame'], phase_body_region=p['body_region'],
                       source_length_A4=q['source_length_A4'], source_length_A5=q['source_length_A5'],
                       raw_evidence_frames=y-x, removed_raw_frames=q['removed_raw_frames'],
                       removed_raw_fraction=q['removed_raw_frames']/(y-x),
                       existing_center_removed=q['existing_center_removed'], added_frames=q['added_frames'],
                       raw_previously_fully_in_A4=q['raw_previously_fully_in_A4'],
                       intervention_type=q['intervention_type'],
                       P0_task='CROP_VALIDITY_NOT_LENGTH_CONTROLLED_QUALITY_EFFECT',
                       reconstruction_edit_target_frames=c['source_only_edit_audit']['edit_target_frames'])
            masks.append(mask_row(c, 'P0_SOURCE_ONLY', c['source_only_edit_audit']))
        else:
            row['rejection_counts'] = json.dumps(q.get('rejection_counts', {}), sort_keys=True)
        population.append(row)
    csvfile(out/'intervention_validity.csv', population)

    tasks = []
    method_rows = []
    for c in fixed_cases:
        task = dict(case=c['case'], source_uid=c['source_uid'], dataset=c['dataset'], status=c['status'],
                    task_label=c['task_label'], source_R_sha256=c['source_R_sha256'],
                    human_reviewed=False, verified_phase='UNKNOWN')
        task.update(interval_columns('source_context', c.get('source_interval')))
        if c['status']=='FIXED_GAP_EVALUATED':
            task.update(interval_columns('target_context', c['target_interval']))
            task.update(interval_columns('source_edit', c['edit_source_interval']))
            task.update(interval_columns('target_edit', c['edit_target_interval']))
            task.update(phase_type=c['target_phase']['type'], phase_center=c['target_phase']['frame'],
                        G1_minus_SLERP_error_rad=c['G1_minus_SLERP_error_rad'],
                        G1_minus_SLERP_fk_error_m=c['G1_minus_SLERP_fk_error_m'],
                        G1_constructible=c['methods']['G1']['constructible'],
                        SLERP_constructible=c['methods']['SLERP']['constructible'],
                        solver_hidden_read=c['solver']['solver_hidden_read'],
                        G1_rejection_fallback=c['solver']['G1_rejection_fallback'],
                        clock=c['source_to_target_clock'],
                        task_scope='SAME_TARGET_METHOD_EFFECT_NOT_A4_VS_A5_EFFECT')
            for name, m in c['methods'].items():
                metric = dict(case=c['case'], source_uid=c['source_uid'], dataset=c['dataset'], method=name,
                              source_start=c['source_interval'][0], source_end_exclusive=c['source_interval'][1],
                              edit_start=c['edit_source_interval'][0], edit_end_exclusive=c['edit_source_interval'][1])
                metric.update({k:v for k,v in m.items() if k not in ('metrics','arrays_key') and not isinstance(v, (list,dict))})
                metric.update({'original_'+k:v for k,v in m.get('metrics',{}).items() if not isinstance(v,(list,dict))})
                method_rows.append(metric)
            masks.append(mask_row(c, 'P2_TARGET_PHASE', c['target_phase_audit']))
            masks.append(mask_row(c, 'P2_ALL_EVIDENCE', c['all_evidence_audit']))
        tasks.append(task)
    csvfile(out/'fixed_gap_tasks.csv', tasks)
    csvfile(out/'fixed_gap_metrics.csv', method_rows)
    csvfile(out/'edit_support_audit.csv', masks)

    versions = dict(main_commit=main['binding']['commit'], fixed_commit=fixed['binding']['commit'],
                    main_code_sha256=main['binding']['code'], fixed_code_sha256=fixed['binding']['code'],
                    frozen_G1_sha256=fixed['frozen_geometry_sha256'],
                    R60_population_sha256=main['parent_population_sha256'],
                    R60_controls_arrays_sha256=main['parent_controls_arrays_sha256'])
    write_json(out/'versions.json', versions)
    summary = dict(schema='R61_INTERVENTION_EDIT_SUPPORT_FIXED_GAP_PUBLIC_V1',
                   registered_cases=len(main_cases), matched_source_crops=len(matched),
                   source_recordings=len({c['source_uid'] for c in matched}),
                   source_counts=dict(Counter(c['dataset'] for c in matched)),
                   phase_proxy_counts=dict(Counter(c['phase_type'] for c in matched)),
                   nested_crops_with_zero_added_frames=sum(c['selection']['added_frames']==0 for c in matched),
                   total_existing_raw_evidence_frames_removed=sum(c['selection']['removed_raw_frames'] for c in matched),
                   centers_removed=sum(c['selection']['existing_center_removed'] for c in matched),
                   fixed_gap=fixed['summary'], tests=tests,
                   parent_files_unchanged=replay_main['parent_files_unchanged'],
                   main_commit=versions['main_commit'], fixed_commit=versions['fixed_commit'],
                   human_records=0, AIST_registered_cases=0, can_route=False)
    write_json(out/'summary.json', summary)
    write_json(out/'qualification.json', dict(
        INTERVENTION_VALIDITY='OBSERVED_KINEMATIC_EVIDENCE_REMOVAL_VERIFIED; SEMANTIC_PHASE_VALIDITY_UNKNOWN',
        EDIT_SUPPORT_AUDIT='COMPLETE_EDIT_OVERLAP_REPORTED; INTRUSION_IS_NOT_AVOIDED_BY_THIS_AUDIT',
        INBETWEENING_QUALITY='PAIRED_SAME_48_FRAME_TARGET_SLERP_VS_G1',
        phase_semantic_effect_established=False, independent_perceptual_quality_established=False,
        verified_phase='UNKNOWN', human_reviewed=False, semantic_completeness='UNKNOWN',
        source_independence='RECORDING_GROUPED_DEVELOPMENT_DIAGNOSTIC; NOT_NEW_UNSEEN_OR_PRETRAINING_INDEPENDENT_TEST',
        world_root_contact='UNKNOWN', Native24='NOT_EVALUABLE',
        training_steps=0, optimizer_calls=0, planner_runs=0, duration_grid_runs=0, can_route=False,
        negative_delta_meaning='G1_HAS_LOWER_MEAN_HIDDEN_RECONSTRUCTION_ERROR_THAN_SLERP_ON_THE_SAME_TARGET',
        missing_construction_policy='FAILURES_RETAINED; PAIRED_MEAN_CONDITIONAL_ON_BOTH_METHODS_CONSTRUCTIBLE',
        protected_source_claim_authorized=False))

    lines = ['# R61 阶段干预有效性与编辑支持一致性', '',
        '主来源裁剪代码：`'+versions['main_commit']+'`；固定缺口代码：`'+versions['fixed_commit']+'`。代码先推送GitHub，服务器拉取后以tmux执行。主观看页面及其SHA验收通过后才启动固定缺口对照。', '',
        '## 可观看产物', '',
        '本机统一入口：`D:/DunhuangDanceGen/outputs/edit_support_r61_01/index.html`。来源页展示原速完整上下文、A4和嵌套裁剪A5；固定缺口页展示同一个真实隐藏动作与SLERP／冻结G1。媒体、源动作张量、身体坐标及私有完整result不进入Git。', '',
        '## INTERVENTION_VALIDITY：先确认已有内容确实被裁掉', '',
        f"预登记{len(main_cases)}例，实际可构造{len(matched)}例，来自{summary['source_recordings']}条来源录制；类型及来源按元数据预选，不按补间成绩选例。已有六个事后诊断区间排除，不据此宣称来源未曝光。", '',
        f"{summary['nested_crops_with_zero_added_frames']}/{len(matched)}个裁剪满足A5为A4严格子区间、左边界相同、新增来源帧为0；累计删除{summary['total_existing_raw_evidence_frames_removed']}帧原先存在于A4的原始运动学证据，删除检测中心的案例数为{summary['centers_removed']}。保持／根角运动的半段删除与峰值证据删除分别记账。", '',
        '这是精确来源内容删除，不是独立确认的舞蹈阶段或艺术意图。A5更短是干预有效性试验的设计，不能据此声称排除了长度因素或证明阶段完整性收益。未修改R47分段器、阈值或12帧扩张半径。', '',
        '## EDIT_SUPPORT_AUDIT：整个编辑范围必须记账', '',
        'P0裁剪删除范围M_crop_removed与P2重建覆盖M_edit分开。P0来源回放没有生成覆盖；P2每例实际编辑目标帧[4,52)，对应真实来源中心前后各24帧。未知的P_verified保持UNKNOWN，侵入率为null，不转换为零。', '',
        '目标阶段审计的分母是所选原始证据／保护带的来源帧并集；all_evidence审计的分母是整条录制中所有传入证据的来源帧并集，不是56帧窗口内的证据数量。二者不能混比。edited_centers统计去重来源帧时刻，不代表独立阶段事件个数。', '',
        '目标帧经source_positions映射，以floor/ceil两点支持审计插值对应范围。本轮为真实30FPS恒等时钟；该对应掩码并不表示G1读取了隐藏来源。固定缺口围绕所选证据中心，侵入被完整登记只能证明合同清楚，不能称阶段已被避开或原样保留。', '',
        '## INBETWEENING_QUALITY：固定同一个48帧目标', '',
        '两方法具有相同的两侧4+4帧可用观测。SLERP实际使用两个邻接端点，G1还使用边缘姿态的速度／加速度信息；这是不同补间方法的比较。隐藏真值只用于评价，不参与求解；G1拒绝时保留失败，不以SLERP或真实来源替代。', '',
        '|数据|任务|两方法均可构造|来源录制数|G1−SLERP角误差rad，95%来源重采样区间|',
        '|---|---:|---:|---:|---|']
    for dataset, s in fixed['summary'].items():
        p = s['primary']; value = fmt(p['mean'])
        if p['ci95'] is not None:
            value += ' ['+fmt(p['ci95'][0])+', '+fmt(p['ci95'][1])+']'
        lines.append('|%s|%d|%d|%d|%s|'%(dataset,s['tasks'],s['paired_constructible'],p['sources'],value))
    lines += ['',
        '负值表示G1在相同隐藏目标上平均角重建误差较低。先对同录制任务求均值，再按录制均衡和bootstrap；构造失败数与原数值筛查分别保留，配对均值不静默填补失败。12条开发来源不等于12条谱系独立、未见音乐或文化语义样本。AIST本轮未登记，0任务表示未评价。', '',
        '同时发布FK误差、相对真实运动的边界角速度／加速度误差、目标代理中心姿态误差、FK活动范围及路径长、近静态时保持未定义的动态比。原R48相对身体数值筛查不是Native24；D3变低也不能直接代表艺术自然性。', '',
        'P2不是A4−A5阶段效应，不证明阶段标签、完整原语、音乐语义或新生成器有效。现有G1不读取阶段标签，不能将补间收益归因到阶段理解。', '',
        '## 独立复验与结论边界', '',
        f"来源裁剪人口、原旋转数组和删除掩码独立重放一致；固定缺口科学JSON和确定性NPZ新进程逐字节一致；封存父文件{replay_main['parent_files_unchanged']}个未变。完整回归{tests['passed']} passed、{tests['skipped']} skipped。主结果与复验SHA分别见versions.json及运行回执。", '',
        '独立人工记录0，verified_phase=UNKNOWN，semantic_completeness=UNKNOWN；世界Root／接触和Native24没有取得资格。训练、优化器、Planner与Core×Transition时长网格执行均0。R60、生产Catalog和原几何阈值保持封存。', '',
        '本轮完成干预与编辑合同验证及同目标补间比较；不将其升级为感知完整性、阶段核心创新、整曲编舞质量或跨数据集泛化结论。']
    evaluated=[c for c in fixed_cases if c['status']=='FIXED_GAP_EVALUATED']
    bands=[c['all_evidence_audit']['band']['edit_overlap_target_frames'] for c in evaluated]
    lines += ['', '## 实测编辑侵入与原数值门', '',
        f"全部{len(evaluated)}例的实际编辑区均侵入所选阶段证据中心；整段编辑区与所有工程保护带重叠{min(bands)}–{max(bands)}帧（每例共48个编辑帧）。这是预设中心缺口任务的事实，不是检测准确率。", '',
        '|数据|SLERP原数值门通过|G1原数值门通过|', '|---|---:|---:|']
    for dataset,s in fixed['summary'].items():
        lines.append('|%s|%d/%d|%d/%d|'%(dataset,s['methods']['SLERP']['passed_original_screen'],s['tasks'],s['methods']['G1']['passed_original_screen'],s['tasks']))
    lines += ['', '两方法平均重建误差有差异，并未带来本轮原数值门通过数量的提升。FineDance仅4例；所有区间均为固定开发人口的来源重采样描述，不能用于新增独立泛化主张。']
    (out/'report.md').write_bytes(('\n'.join(lines)+'\n').encode('utf8'))
    write_json(out/'scientific_files.json', dict(files=[dict(path=x.relative_to(ROOT).as_posix(), bytes=x.stat().st_size, sha256=sha(x))
        for x in sorted(ROOT.rglob('*')) if x.is_file()],
        scope='HASH_DIRECTORY_ONLY; SCIENTIFIC_MEDIA_ARRAY_AND_PRIVATE_RESULT_BYTES_NOT_PUBLISHED'))
    write_json(out/'public_manifest.json', dict(files=[dict(path=x.name, bytes=x.stat().st_size, sha256=sha(x))
        for x in sorted(out.iterdir()) if x.is_file()],
        media_published=False, raw_arrays_published=False, body_coordinates_published=False,
        private_complete_result_published=False))
    print('R61_RELEASE', len(list(out.iterdir())), flush=True)


if __name__=='__main__':
    release()
