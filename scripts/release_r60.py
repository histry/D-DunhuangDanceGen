"""Publish text statistics and receipts only; keep source arrays/media in outputs."""
import csv,shutil,subprocess,math
from collections import Counter
from dunhuang.research.phase_integrity_r60.common import ROOT,read,write,sha,fresh

def csvfile(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def release():
    out=fresh('reports/r60_release');p=read(ROOT/'population/result.json');r=read(ROOT/'controls/result.json');v=read(ROOT/'independent/result.json')
    assert v['result_exact'] and v['arrays_bytes_exact'];assert read(ROOT/'receipts/main_browser.json')['passed']
    for name in ('main','controls','verify','tests'):
        receipt=read('outputs/r60_execution_02/'+name+'_receipt.json');assert receipt['exit_status']==0
        write(out/(name+'_receipt.json'),receipt)
    write(out/'independent.json',v);write(out/'main_browser.json',read(ROOT/'receipts/main_browser.json'))
    write(out/'summary.json',dict(schema='R60_PHASE_INTEGRITY_ATTRIBUTION',scientific_commit=r['binding']['commit'],
                                sources=len(p['sources']),source_counts=dict(Counter(s['dataset'] for s in p['sources'])),
                                units=len(p['units']),summary=r['summary'],accepted_change_frames=p['legacy_accepted_frames'],
                                legacy_events=len(p['population_views']['legacy499']),r47_units=len(p['population_views']['R47_139']),r58_bank=len(p['population_views']['R58_16']),
                                parent_files_unchanged=v['parent_files_unchanged'],main=read(ROOT/'main/result.json'),
                                legal_domain_note=r['legal_domain_note']))
    write(out/'qualification.json',dict(G1_phase_truth='NOT_ESTABLISHED',G2_numerical_results=r['summary'],
                                       G3_joint_planning='NOT_ADMITTED',human_reviewed=False,semantic_completeness='UNKNOWN',
                                       can_route=False,training_updates=0,Native24='NOT_EVALUABLE',
                                       pre_score_correction='V1_RETAINED; V2_REQUIRES_EQUAL_REAL_CONTEXT_AND_INTERVENED_SEAM',
                                       no_V1_controls=True,no_production_or_frozen_algorithm_changes=True))
    pop=[]
    for u in p['units']:
        q=u['pair'];a,b=u['source_interval'];row=dict(unit_uid=u['unit_uid'],source_uid=u['source_uid'],dataset=u['dataset'],source_start=a,source_end=b,length=b-a,
                                                  protected_crossings=len(u['protected_crossings']),status=q['status'],reason=q.get('reason',''),semantic_completeness='UNKNOWN')
        if q['status']=='MATCHED':row.update(shift=q['shift'],A5_start=q['A5'][0],A5_end=q['A5'][1],relative_activity_difference=q['relative_activity_difference'],
                                          common_hidden_seams='|'.join(q['common_hidden_seams']),context_frames_each_side=q['context_frames_each_side'])
        pop.append(row)
    csvfile(out/'population.csv',pop)
    csvfile(out/'matched_results.csv',[dict(unit_uid=u['unit_uid'],source_uid=u['source_uid'],dataset=u['dataset'],
            status=u['pair']['status'],seams=u.get('common_constructible_seams'),primary_error_A4_minus_A5=u.get('primary_error_A4_minus_A5'),
            error_A4=u.get('error_rad',{}).get('A4'),error_A5=u.get('error_rad',{}).get('A5'),pass_fraction_A4=u.get('G1_pass_fraction',{}).get('A4'),
            pass_fraction_A5=u.get('G1_pass_fraction',{}).get('A5')) for u in r['units']])
    csvfile(out/'fixed_lengths.csv',r['fixed_lengths'])
    profiles=[]
    for q in r['legal_profiles']:
        row={k:q[k] for k in ('unit_uid','source_uid','dataset','arm','source_length','target_length','theoretical_ratio','legal','protected_crossings','phase_evidence_truncated')}
        if q['legal']:
            for k in (1,2,3):row['D%d_Q95'%k]=q['metrics']['D%d_Q95'%k];row['D%d_amplification'%k]=q['amplification'][str(k)]['ratio']
            row['warp_knots']=len(q['warp'])-2;row['knot_D3_squared_mass_fraction']=q['metrics']['knot_D3']['squared_mass_fraction']
        profiles.append(row)
    csvfile(out/'duration_profiles.csv',profiles)
    csvfile(out/'hidden_bridges.csv',[dict(key=q['key'],unit=q['unit'],arm=q['arm'],side=q['side'],method=method,constructible=m['constructible'],
            passes_original_screen=m['passes_original_screen'],hidden_SO3_error_rad=m['error_rad'],hidden_FK_error_m=m.get('fk_error_m'),
            gap_D2=m.get('metrics',{}).get('gap_D2'),gap_D3=m.get('metrics',{}).get('gap_D3'),virtual_split_array_exact=q['virtual_split_array_exact'],solver_hidden_read=q['solver_hidden_read'])
            for q in r['hidden_bridges'] for method,m in q['methods'].items()])
    write(out/'source_audit.json',dict(sources=[dict(source_uid=s['source_uid'],dataset=s['dataset'],role=s['role'],frames=s['frames'],
            R_sha256=s['R_sha256'],accepted_components=s['accepted_components'],protected_spans=len(s['protected_spans']),phase_proxies=len(s['phase_evidence'])) for s in p['sources']],
            source_selection=p['source_selection'],no_new_phase_truth=True,legacy_views=p['population_views'],
            unknown_quality_frames_excluded=p['unknown_quality_frames_excluded']))
    write(out/'parent_seal.json',dict(files=p['parent_seal'],verified_files=v['parent_files_unchanged']))
    write(out/'scientific_files.json',dict(files=[dict(path=str(x.relative_to(ROOT)),sha256=sha(x),bytes=x.stat().st_size) for x in sorted(ROOT.rglob('*')) if x.is_file()]))
    lines=['# R60 阶段支持完整性与原语时长归因','',
           '研究代码：`'+r['binding']['commit']+'`。依据所给方案，R59 编号已占用，故独立使用 R60。先发布代码，在服务器 tmux 生成并回传观看页、完成浏览器验收，再运行对照。未训练 SDMR/Diffusion，未修改生产 Catalog、原 E1-W、G0/G1 或历史基线。','',
           '## 可观看交付','',
           '本地 `outputs/phase_integrity_r60_02/main/viewer/index.html`：原速连续来源、A4 与 A5，旧499切口和阶段保护带，统一播放/暂停/拖动/倍速/逐帧。`main/video/index.html` 为实际编码视频。灰色只表示区间外上下文，不是补全输出；本轮无音乐、非新舞蹈生成。','',
           '## 预评分修正','',
           '初版来源页暴露支持域混杂：录制起点没有等长左上下文时，负对照可能仅在左侧切入保护区，误差却在未干预右侧计算。首轮输出保留于 `phase_integrity_r60_01`，没有计算对照分数。唯一预评分修正要求两组左右均有30帧真实上下文，且只评价真正被截断、具有相同48帧隐藏真值支持的一侧。不改匹配容差、质量阈值或优化成绩。','',
           '## 人口与合同','',
           f"{len(p['sources'])}条来源、{len(p['units'])}个候选。保留旧499、R47全部139、R58冻结16单位及41,400已接受帧；全部来源质量空隙继续排除。AIST++ 12个Train音乐组，FineDance 12条官方@Genre Train诊断录制；外部每来源最多3个按时间元数据选出的R47候选。两者都没有天然完整舞蹈阶段真值。",'',
           'A4/A5同录制、同支持分量、相同长度、至少70%重叠；边界移动不超过30帧，上身平均角活动量差不超过15%。先按活动差、位移和方向确定唯一配对，之后才计算误差。NOT_MATCHABLE保留，不用其他录制补齐。','',
           '## 主要结果','',
           '|数据|候选|合格配对|独立来源数|A4−A5 G1隐藏动作误差，rad（95%来源重采样区间）|','|---|---:|---:|---:|---|']
    for d,s in r['summary'].items():
        t=s['primary'];value='不可计算' if t['mean'] is None else '%.6f [%.6f, %.6f]'%(t['mean'],*t['ci95'])
        lines.append('|%s|%d|%d|%d|%s|'%(d,s['units'],s['matched'],t['sources'],value))
    lines += ['','负值只表示在本轮同录制、同长度、真实上下文条件下，A4对隐藏48帧的固定G1重建误差更低；不能等价为舞蹈自然性、文化语义或阶段检测准确率。所有方法的构造失败、原数值门失败与缺失人口均保留。主误差使用构造成功的共同接缝；原数值门资格另外报告，不能只挑合格连接来计算误差。','',
              'Virtual Split逐数组精确回拼，误差为0；Actual Split+Bridge替换相同48帧，不增加总时长，隐藏真值从不进入求解器。原ChangE Train参照阈值用于外域时仅为冻结迁移诊断，不是已验证的通用物理门。','',
              '同长度A4/A5合法时长覆盖相同：原E1-W的线性回退保证端点速率域，阶段拍点为可选吸引项。因此当前接口本身不能证明阶段完整性改变合法时长集合。D1/D2/D3、结点放大和边界状态另存完整私有档案；高D3不直接等于动作不自然，近静态来源归一化保持未定义。','',
              '96/180/240/300帧控制仅描述相同中心附近的保护风险、来源支持与长度关系；不是随机化时长试验，也不把每种长度都标为完整动作。','',
              '## 资格与下一步','',
              'G1：独立人工/跨录制阶段真值未建立。G2：只报告本轮运动学边界与连续补间关联。G3：不启动联合编舞扩搜。`human_reviewed=false`，`semantic_completeness=UNKNOWN`，`can_route=false`，`Native24=NOT_EVALUABLE`。未因缺少人工标签中止本轮可客观执行的研究。','',
              '13项本轮测试通过；服务器完整回归734 passed、1 skipped（5条历史warning）。独立新进程重建全部人口、重算所有时间映射和桥，科学JSON与确定性NPZ逐字节一致；验证封存父文件 '+str(v['parent_files_unchanged'])+' 个不变。训练和优化器调用0。','',
              '结论边界：不以自动保护规则满足率证明语义准确性，不以一轮代理实验决定更换论文主线；保留R47工程方案、R52强基线与R58历史结论。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    write(out/'public_manifest.json',dict(files=[dict(path=x.name,bytes=x.stat().st_size,sha256=sha(x)) for x in sorted(out.iterdir()) if x.is_file()],media_and_raw_arrays_published=False))
    print('R60_RELEASE',len(list(out.iterdir())),flush=True)
if __name__=='__main__':release()
