"""Derive compact public R53 tables only after byte-exact source replay succeeds."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from statistics import mean


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('outputs/warp_coupling_r53_01'))
    parser.add_argument('--out', type=Path, default=Path('reports/r53_release'))
    a = parser.parse_args()
    source, out = a.source, a.out
    independent = read(source/'receipts/independent.json')
    assert independent['exact_source_rebuild_replays'] == ['preview', 'main', 'controls']
    assert independent['optimizer_calls'] == 0
    assert read(source/'receipts/final_recheck.json')['all_hashes_verified']
    data = read(source/'controls/result.json')
    preview = read(source/'preview/result.json')
    main_data = read(source/'main/result.json')
    summary = read(source/'publication/summary.json')
    entries = [e for r in data['records'] for layer in r['banks'] for e in layer]
    population = dict(event_slot_instances=len(entries), distinct_sampled_motion=sum(e['distinct_warp_count'] > 1 for e in entries), changed_boundary_jets=sum(e['max_jet_delta'] > 1e-7 for e in entries), original_selected_adjacencies=len(preview['pairs']), selected_numeric_effects=sum(p['has_numeric_effect'] for p in preview['pairs']), selected_relevant_effects=sum(p['has_effect'] for p in preview['pairs']), direct_negative_control_recalculations=preview['negative_control_direct_calls'], full_fixed_event_pairs=len(main_data['full_pair_mechanisms']), full_numeric_effects=sum(p['has_numeric_effect'] for p in main_data['full_pair_mechanisms']), full_relevant_effects=sum(p['has_effect'] for p in main_data['full_pair_mechanisms']), recovered_pairs=summary['fixed_event_pairs_with_recovered_edges'], lost_pairs=summary['fixed_event_pairs_with_lost_edges'], full_global_task_recoveries=summary['recovered_tasks'], fixed_global_task_denominator=summary['registered_tasks'])
    tables = []
    for song in range(3):
        for arm in data['records'][0]['plans']:
            plans = [r['plans'][arm] for r in data['records'] if r['song'] == song]
            admitted = [p for p in plans if p['status'] == 'NUMERICAL_SHORT_PLAN_NOT_NATIVE']
            tables.append(dict(song=song, arm=arm, admitted=len(admitted), registered=len(plans), boundary_D3=mean(p['metrics']['independent_boundary_D3_mean'] for p in admitted) if admitted else None, core_D3=mean(p['metrics']['core_D3_Q95_mean'] for p in admitted) if admitted else None, beat_proxy=mean(p['metrics']['beat_proxy'] for p in admitted) if admitted else None, requested_edges=sum(p['required_edges'] for p in plans), per_case_readset_count_sum=sum(p['required_unique_bridge_readsets'] for p in plans), paths_visited=sum(p['paths_visited'] for p in plans)))
    pair_lookup = {(p['case'], p['slot'], p['left_event'], p['right_event']): p for p in data['full_pair_mechanisms']}
    recovered_selected = []
    for r in data['records']:
        for slot, (left, right) in enumerate(zip(r['plans']['B3']['nodes'], r['plans']['B3']['nodes'][1:])):
            key = (r['key'], slot, data['nodes'][left]['event_uid'], data['nodes'][right]['event_uid'])
            if not pair_lookup[key]['cells'][0]['admitted']:
                recovered_selected.append(dict(case=r['key'], slot=slot, left_event=key[2], right_event=key[3]))
    resources = {mode: read(source/mode/'resources.json') for mode in ('preview', 'main', 'controls')}
    inputs = [source/'controls/result.json', source/'preview/result.json', source/'main/result.json', source/'publication/summary.json', source/'publication/seal.json', source/'receipts/independent.json', source/'receipts/final_recheck.json']
    binding = [dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs]
    out.mkdir(parents=True, exist_ok=False)
    write(out/'tables.json', dict(population=population, music_method_tables=tables, resources=resources, source_bindings=binding, selected_recovered_edges=recovered_selected, all_methods_numerically_admitted=sum(p['status']=='NUMERICAL_SHORT_PLAN_NOT_NATIVE' for r in data['records'] for p in r['plans'].values()), B3_B1_identical_paths=sum(r['plans']['B3']['nodes']==r['plans']['B1']['nodes'] for r in data['records']), scope='EXPOSED_DEVELOPMENT_BODY22_RELATIVE_NOT_NATIVE24'))
    for p in (source/'publication').iterdir():
        if p.suffix in ('.json', '.md'): shutil.copyfile(p, out/('server_'+p.name if p.name=='report.md' else p.name))
    for p in (source/'receipts').iterdir():
        if p.suffix in ('.json', '.txt', '.log'): shutil.copyfile(p, out/p.name)
    rows = ['# R53 合法时间演法与连接机制实验报告', '', '**结论：局部机制得到支持，联合规划的预登记开发收益门未通过。保留 R52 强基线；不新增神经训练，不进入整曲。**', '', '先推送代码并在tmux执行，先回传/验收实际观看，再跑固定对照。完整原始动作、音频、失败图和数组留于本地/服务器输出；Git发布统计、协议和回执。', '', '## 冻结协议与真实人口', '', 'R52基线984fd0b；物理算法提交f6dbdaf；c96f970只补局部恢复展示与公开统计。R47目录139、注册预算16、每槽10合法事件；3首曝光音乐，2/3事件×24/48/72帧过渡，共18任务；Core240帧，斜率0.75–1.33，K≤4。原G0/G1和R48筛查不变。', '', json.dumps(population, ensure_ascii=False, indent=2), '', '406是按1e-10量化后不同的实际重采样轨迹实例数，不是独立来源或歌曲数。38组恢复与20组丢失是候选事件对上的变化；原18全局任务全部可行，故失败任务恢复为0/18。', '', '## 未用于搜索评分的边界读数', '', '边界D3为相对身体FK三阶差分指标，单位m/s³；Core为D3 Q95；Beat为自动峰—拍点代理。Gap D3被搜索使用，不作为独立效果证据。', '', '|音乐|方法|通过任务|边界D3|Core D3|Beat代理|', '|---|---|---:|---:|---:|---:|']
    for r in tables:
        if r['arm'] in ('B0','B1','B2','B3'):
            rows.append(f"|{r['song']+2}|{r['arm']}|{r['admitted']}/{r['registered']}|{r['boundary_D3']:.3f}|{r['core_D3']:.3f}|{r['beat_proxy']:.4f}|")
    rows += ['', 'B0固定W0；B1先按局部活动匹配独立选择Warp；B2随机K联合搜索；B3 Top-K联合搜索。B3与B1在15/18任务的选择完全相同。B3实际用了2条W0被拒绝、变速后通过的边，但这两例B1也选到了相同结果，不能据此声称联合选择有额外收益。全部9种方法×18任务均通过原数值筛查；通过不意味着自然性、世界Root/接触或完整Native资格。', '', f"G3={summary['G3']}；至少两首歌相对B0/B1/B2同时改善边界5%、Core退化≤5%、Beat下降≤0.02的门未通过。三首开发音乐不支持跨作品泛化断言；G4未测。不调阈值、种子或损失争取过门。", '', '## 计算与K趋势', '', '|方法|K|请求边数（跨任务和）|搜索访问数|边界D3（三曲均衡）|', '|---|---:|---:|---:|---:|']
    for arm, k in [('B0',1),('B1',4),('B2_K1',1),('B2_K2',2),('B2',4),('B3_K1',1),('B3_K2',2),('B3',4),('W3',1)]:
        rr=[r for r in tables if r['arm']==arm]
        rows.append(f"|{arm}|{k}|{sum(r['requested_edges'] for r in rr)}|{sum(r['paths_visited'] for r in rr)}|{mean(r['boundary_D3'] for r in rr):.3f}|")
    rows += ['', 'K=1的Top-K和随机K完整动作逐位等于R52。增加K未获得独立指标的单调收益。B1在每事件K个Warp中选一条后才搜索，因此请求桥边与搜索数更少；B2/B3候选数及搜索访问数相同，独特边界读集数不同。共享缓存不能称为严格相同墙钟预算。', '', '|阶段|候选图新桥构造|父进程峰值KiB|单个子进程最大KiB|科学阶段秒数|', '|---|---:|---:|---:|---:|']
    for mode, r in resources.items():
        rows.append(f"|{mode}|{r['bridge_new_evaluations']}|{r['parent_peak_rss_kib']}|{r['child_max_rss_kib']}|{r['seconds']:.2f}|")
    rows += ['', '候选图计数不含每阶段27次冻结基线重建与preview的52次直接负对照；科学阶段时间包含这些复算、搜索与数组保存，但不含随后观看文件导出。内存读数不是所有并发进程RSS之和；时间不参与确定性重放比较。完整节点、桥ID、失败原因、实际读集命中数及逐方法耗时存于原产物。', '', '## 反事实与独立复验', '', '52次内部变速负对照直接重算（不用缓存），固定两侧各4帧，实际内部轨迹不同而桥与筛查完全相同。各3帧足够构造桥，但D3筛查需要第四帧，因此不能只固定三个端帧。', '', 'preview/main/controls均用新进程从SHA核验的真实来源重建；JSON全失败图、原始旋转/FK及确定性NPZ字节完全一致。没有优化器或神经训练调用；R52原18轨迹不变。详见independent.json、final_recheck.json与阶段exit_status=0回执。', '', '服务器GitHub直拉两次TLS EOF后，使用已发布、核对SHA的Git bundle精确快进；未关闭TLS校验。观看验收只说明播放与身份/时钟正确，不是独立人工舞蹈评价。', '', '## 观看与复现', '', '本机统一入口：`D:/DunhuangDanceGen/outputs/warp_coupling_r53_01/index.html`。含27原选连接的完整Warp矩阵、18条B0/B3主比较、18条等K对照、3个按登记顺序取出的局部恢复案例。拒绝桥留空；无渲染平滑。', '', '运行与边界合同见`docs/warp_coupling_r53.md`。本报告表格可用`python scripts/release_r53.py --source outputs/warp_coupling_r53_01 --out /一个新的目录`从封存产物重新生成。`tables.json`记录所有输入SHA和完整表格。']
    (out/'report.md').write_text('\n'.join(rows)+'\n', encoding='utf8')
    print(json.dumps(dict(public_files=len(list(out.iterdir())), G3=summary['G3'], population=population), ensure_ascii=False))


if __name__ == '__main__':
    main()
