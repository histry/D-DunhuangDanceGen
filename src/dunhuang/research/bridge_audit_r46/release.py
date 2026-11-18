"""Evidence-bound R46 publication with immutable scientific artifacts."""
import argparse,subprocess,re
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write
from dunhuang.research.reperformance_r44.replay import digest

def run(out):
    root=Path('outputs/bridge_audit_r46_01');out=Path(out)
    if out.exists():raise ValueError('Fresh release')
    receipts={}
    for name in ('primary','controls'):
        p=root/(name+'_receipt.txt');value=p.read_text();assert 'exit_status=0' in value;receipts[name]=value
    audit=read(root/'main/result.json');oracle=read(root/'oracle/result.json');ind=read(root/'independent.json');guard=read(root/'admission.json');assert guard==read(root/'admission_replay.json');assert ind['all_scientific_values_exact'] and not oracle['neural_training_admitted']
    coverage=read(root/'coverage.json');assert coverage==read(root/'coverage_replay.json')
    browser={}
    for name,folder in (('main','viewer'),('oracle','viewer_oracle')):
        r=read(root/'receipts'/(name+'_browser_acceptance.json'));assert r['passed'] and r['viewer_protocol_sha256']==digest(root/folder/'protocol.json');browser[name]=r
    regression=re.search(r'\d+ passed[^\n]*',(root/'full_pytest_final.log').read_text()).group(0)
    source_records=[dict(key=r['key'],source={k:v for k,v in r['source'].items() if k!='audit'},audit=r['source']['audit']) for r in audit['records']]
    result=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),baseline=audit['config']['baseline'],regression=regression,bridge=source_records,chart_guard=guard,full_source_coverage=coverage,phase_oracle_summary=oracle['summary'],oracle_capacity_screen=oracle['oracle_capacity_screen'],fixed_score_screen=oracle['fixed_score_screen'],oracle_conditions=[dict(key=r['key'],donors=r['donors'],available=r['available'],metrics=r['metrics']['methods']) for r in oracle['records']],independent=ind,receipts=receipts,browser=browser,optimizer_steps=0,new_neural_samples=0,whole_song_generations=0,human_reviewed=False,can_route=False,operational_qualified=False)
    out.mkdir(parents=True);write(out/'summary.json',result)
    lines=['# R46：几何桥责任、完整来源覆盖与Train音乐阶段上限','','本轮按附件R46方案，先回传并验收22案例实际轨迹与3条视频，再运行固定时钟集合的Oracle和条件对照。没有新增Lodge/SDMR训练，没有新神经采样或整曲生成。正式EDGE/Planner/Native及11个父代冻结；研究代码独立在bridge_audit_r46。未删除仍被基线复验依赖的历史代码。','','## 观看','','- `outputs/bridge_audit_r46_01/videos/index.html`：飞天、女冥想、琵琶2四栏实际256帧视频。','- `outputs/bridge_audit_r46_01/viewer/index.html`：22案例，来源／冻结Geometry／仅观察端点重建／冻结Phase，支持117帧跳转、共同音频时钟、逐帧和两种坐标视角。','- `outputs/bridge_audit_r46_01/viewer_oracle/index.html`：8个AIST++ Train音乐组，正确／错音乐／同能量异起音／已知错位／无音乐与事后Oracle。该页是诊断上限，不是新生成模型。','','## P0：构形差异的实际责任','','仅用101/102/103/152/153/154帧的观测旋转重建原R38五次桥，隐藏48帧NaN与干扰值产生同一结果。所有22例最大旋转重建差如下；原Root/contact先验保持。Geometry与Phase的Bridge一致；不是修改Core时钟可以修复的部位。','', '|案例|重建最大角差rad|117帧平均角差rad|连续图最大范数rad|局部图域检查|','|---|---:|---:|---:|---|']
    guards={r['key']:r['certificate'] for r in guard['records']}
    for r in source_records:
        a=r['audit'];v=a['frame117_so3_mean_rad'];g=guards[r['key']];lines.append(f"|{r['key']}|{a['rebuild_vs_geometry_max_angle_rad']:.3g}|{'N/A' if v is None else f'{v:.4f}'}|{g['continuous_chart_maximum_rad']:.4f}|{g['decision']}|")
    rejected=sum(not r['certificate']['chart_eligible'] for r in guard['records']);lines+=['',f"原端点检查只限制观测相对旋转，没有检查中间五次轨迹。新增独立sidecar将既有π−0.05局部图半径检查延伸到整条曲线；{rejected}/{len(guard['records'])}例超域。方法对范数平方多项式的导数求根并检查端点及实根极值；这是解析极值候选加数值求根，不是区间算术严格证明。",'这项检查限制当前局部坐标插值的适用域，不能证明对应动作全局不可连接，也不代表图域内的桥自然或物理合格。原桥未覆盖或裁剪，正式路由未升级；不将拒绝样本伪装成修复成功。', '同录制缺口20例可评价真实隐藏动作；2个异源压力连接没有唯一真实过渡，117帧源误差N/A。全部保存源SHA、边界log速度/加速度、肩腕/双腕/脚踝/躯干几何、D1/D2/D3、Root范围及骨长误差，避免仅用低Jerk推断自然性。官方FK用于排查坐标与实现差异，不评价艺术质量。','','## P3：飞天和冥想为何保持原时间','','|案例|左Core原因|右Core原因|左/右可行非恒等图数量|','|---|---|---|---|']
    for r in source_records[:8]:
        c=r['audit']['capacity'];lines.append(f"|{r['key']}|{c[0]['reason']}|{c[1]['reason']}|{c[0]['nonidentity_eligible']}/{c[1]['nonidentity_eligible']}|")
    lines+=['','NO_DETECTED_STAGE包含低活动和没有通过检测/保护范围的阶段，不能全部解读成动作库没有运动。还独立扫描完整8条Native来源，避免把首256帧当成整条录制的能力。最大活动窗只用于诊断覆盖，不当作完整语义动作、不用于模型选择或新训练。','', '|来源|总帧|当前256帧覆盖|当前上身平均角速rad/s|完整来源95分位角速rad/s|最高均值256帧区间|最高窗均值rad/s|','|---|---:|---:|---:|---:|---|---:|']
    for r in coverage['records']:lines.append(f"|{r['source_uid']}|{r['source_frames']}|{100*r['current_coverage_fraction']:.2f}%|{r['current_mean_upper_speed_radps']:.4f}|{r['whole_recording_p95_upper_speed_radps']:.4f}|{r['maximum_mean_activity_256frame_interval']}|{r['maximum_mean_activity_256frame_upper_speed_radps']:.4f}|")
    lines+=['','这些字段构成Planner可读取的研究sidecar：来源、当前窗口、阶段证据、可行时间图、幅度/速率范围及接法图域风险。没有让Planner凭速度认定文化表达正确，也不强迫冥想定格产生新运动。','','## P1：不扩大时钟集合的Train Oracle','','只读取8个已登记AIST++ Train组，重建同一<=25图银行；Correct与冻结Phase逐元素一致，NoMusic与Geometry逐元素一致。OracleCorrect按每Core两个已定义代理误差之和事后选图，OracleWrong在错误音乐代理下选图，最终全部对正确配对音乐评价。代理缺失时不填0，支持与分母完整保留。', 'EnergyMatched从真实其他Train音乐中按RMS接近选择，并要求起音余弦<0.9且拍点列表不同；若无候选则记不可用。该名称只表示控制构造，不表示BPM/曲式/语义完全相同。已知音频偏移为−15/−8/+8/+15帧，无循环补齐；未观测尾部带mask，不作为负例。这里只测试评分/时钟响应，没有训练偏移分类器或声称恢复准确率。','','|比较（正值表示代理误差改善）|音乐组|均值秒|95%组bootstrap区间秒|登记筛查|','|---|---:|---:|---|---|']
    for k,v in oracle['summary'].items():lines.append(f"|{k}|{v['groups']}|{v['mean']}|{v['ci95']}|{v['passed']}|")
    lines+=['',f"Oracle必要筛查={oracle['oracle_capacity_screen']}；固定评分筛查={oracle['fixed_score_screen']}。",'Oracle是读取评价代理后的同样本可达上限，不能当部署收益、独立验证或真实语义对应。即使部分区间正，也不能因此直接训练生成网络。所有方法的完整曲线与候选逐项误差保留，不按曝光Val挑参数或权重。','','## P2与训练决定','','沿用R45的184项匿名准备包与导入合同，未收到新的真实独立审核文件。真实审核数0，艺术适配nDCG/偏好/拒配正确性保持NOT_EVALUABLE。人工记录缺失不阻断本轮客观P0/P1/P3。三首敦煌音乐仍为历史曝光开发资产。',f"跨录制候选物理时钟对{ind['cross_performance_audit']['physical_clock_pairs']}个，经同编舞和实际>=90帧支持筛选剩{ind['cross_performance_audit']['same_choreography_supported_pairs']}个；逐事件阶段关系仍未验证。共享时钟不是精确演绎目标，不通过音频互换制造ChangE真值。",'新训练与整曲维持0：没有将Oracle输出当新监督，也没有自动晋升SDMR或Geometry。下一阶段优先处理当前桥局部图域适用性和真实阶段信息；在单事件来源保持、音乐净收益及物理支持共同成立前不扩展整曲。','','## 验证与恢复','',f"完整回归：{regression}。两包科学数组与JSON新进程精确重放；覆盖侧写、图域检查及官方FK分别重算。11个父代共{sum(p['files'] for p in ind['parents'])}文件逐SHA保持。"]
    for q in ind['official_fk']:lines.append(f"- {q['name']}官方FK实算{q['count']}轨迹，最大差{q['maximum_error_m']:.9g}米。")
    lines+=['','完整Native24与真实接触足滑仍不可评。局部视角移除每帧Root；世界视角不伪造来源Root，生成Root仍为保存先验。', '首次导出误将Phase显示时钟标为恒等，现已绑定冻结真实时钟；原首次页面/数组/回执保留。独立检查确认除Phase clock元数据外所有动作与诊断数组不变。首轮截图为旧帧缓存，最终验收在新一轮读取中确认117帧=3.9秒，随后118帧=3.933秒与半速播放；最终截图保留。', '新增连续图域极值检查的测试暴露了多项式乘积自动裁短导致的形状错误，已用固定阶卷积修正，合成恒等与超域案例通过；未修改真实动作或阈值来通过测试。', '代码先推GitHub，服务器tmux运行；GitHub fetch遇TLS EOF后用同一已发布SHA的Git bundle同步，未关闭证书验证。媒体及公开报告回传本机，完整数组/日志/成功失败回执留服务器。封存与临时SSH/HTTP清理状态以发布回执为准。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    seal=root/'artifacts_manifest.json'
    if seal.exists():raise ValueError('Never replace seal')
    files=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('artifacts_manifest.json','publication_receipt.json')];write(seal,dict(source_commit=result['source_commit'],files=files,optimizer_steps=0,operational_qualified=False));print('R46_SEALED',len(files),sum(f['bytes'] for f in files),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
