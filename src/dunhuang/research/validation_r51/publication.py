"""Fresh-process exact scientific replay and bounded publication."""
from pathlib import Path
from collections import Counter
import subprocess
import numpy as np
from dunhuang.research.complete_r36.core import read,write
from dunhuang.research.reperformance_r44.replay import digest
from .common import ROOT,fresh


def equivalent(a,b):
    a,b=Path(a),Path(b)
    assert read(a/'result.json')==read(b/'result.json'),str(a)
    if (a/'arrays.npz').exists():
        with np.load(a/'arrays.npz') as x,np.load(b/'arrays.npz') as y:
            assert set(x.files)==set(y.files)
            for name in x.files:assert np.array_equal(x[name],y[name]),name


def verify(out):
    pairs=[('main','replay/main'),('plans','replay/plans'),('controls/sample','replay/sample_controls'),('controls/plans','replay/plan_controls'),('controls/paired','replay/paired')]
    for a,b in pairs:equivalent(ROOT/a,ROOT/b)
    # Main results are shown first and duplicated under the control process unchanged.
    with np.load(ROOT/'main/arrays.npz') as a,np.load(ROOT/'controls/sample/arrays.npz') as b:
        for k in a.files:assert np.array_equal(a[k],b[k]),'Main/control common array differs: '+k
    for f in read('outputs/r50_publication/seal.json')['files']:
        p=Path(f['path']);assert p.stat().st_size==f['bytes'] and digest(p)==f['sha256']
    write(out,dict(exact_scientific_replays=len(pairs),main_common_sampling_arrays_exact=True,r50_sealed_files_verified=29,optimizer_calls=0,can_route=False))


def release(out):
    out=fresh(out);ind=read(ROOT/'receipts/independent.json');assert ind['exact_scientific_replays']==5
    acceptance=read(ROOT/'browser_receipt.json');assert acceptance['passed']
    for sub,h in acceptance['protocol_sha256'].items():assert digest(ROOT/sub/'viewer/protocol.json')==h
    assert acceptance['verified_unix'] < (ROOT/'controls/sample/result.json').stat().st_mtime
    p=read(ROOT/'controls/plans/result.json');n=read(ROOT/'controls/sample/result.json');paired=read(ROOT/'controls/paired/result.json')
    planning=[]
    for r in p['records']:
        for arm,s in r['selected'].items():
            events=s.get('events',[]);planning.append(dict(key=r['key'],arm=arm,status=s['status'],events=[e['event_uid'] for e in events],sources=[e['source_uid'] for e in events],repeated_event=len(events)!=len({e['event_uid'] for e in events}),kinematics=s.get('kinematics'),score=s.get('score')))
    stages=[]
    for r in p['records']:
        for slot,pool in enumerate(r['pools']):
            catalog=r['catalog_pools'][slot]
            ranking=sorted(pool,key=lambda c:(-c['scores'][0],c['unit']['unit_uid']))
            top=[];seen=set()
            for c in ranking:
                uid=c['unit']['source_uid']
                if uid not in seen:top.append(c['unit']['unit_uid']);seen.add(uid)
                if len(top)==4:break
            stages.append(dict(key=r['key'],slot=slot,catalog_legal=len(catalog),registered_legal=len(pool),catalog_sources=dict(Counter(c['unit']['source_uid'] for c in catalog)),registered_sources=dict(Counter(c['unit']['source_uid'] for c in pool)),top4=top,raw_top_source=ranking[0]['unit']['source_uid'] if ranking else None,catalog_top_source=max(catalog,key=lambda c:c['scores'][0])['unit']['source_uid'] if catalog else None))
    planning_aggregate={}
    for arm in sorted({r['arm'] for r in planning}):
        rows=[r for r in planning if r['arm']==arm]
        planning_aggregate[arm]=dict(cases=len(rows),admitted=sum(r['status']=='NUMERICAL_SHORT_PLAN' for r in rows),unique_sources=len({s for r in rows for s in r['sources']}),unique_events=len({s for r in rows for s in r['events']}),repeated_event_cases=sum(r['repeated_event'] for r in rows),sequences=[r['events'] for r in rows])
    neural=[]
    for r in n['records']:
        base=r['methods']['C2Phase']
        for arm,v in r['methods'].items():
            neural.append(dict(key=r['key'],domain=r['domain'],arm=arm,gap_angle_mean=float(np.mean([x['gap_angle_rad'] for x in v['final']])),proposal_gap_angle_mean=float(np.mean([x['gap_angle_rad'] for x in v['raw_proposal']])),protected_raw_exact=all(x['protected_raw_exact'] for x in v['final']),gap_activity=float(np.mean([x['gap_activity'] for x in v['final']]))))
    summary=dict(planning=planning,planning_aggregate=planning_aggregate,funnel_stages=stages,neural=neural,weights=n["weights"],context_checkpoint_sha256=digest("outputs/bridge_r38_01/training/context.pt"),paired={k:{t:v for t,v in row.items() if t in ('summary','source_balanced_summary')} for k,row in paired['tasks'].items()},independent=ind,pytest=(ROOT/'receipts/pytest.log').read_text().strip().splitlines()[-1],phase='FIXED_WEIGHTS_NEWLY_ENABLED_KINEMATIC_PROXY_NOT_TRAINED',new_training_steps=0,can_route=False,native24='NOT_EVALUABLE_22_OF24',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write(out/'summary.json',summary)
    lines=['# R51 来源集中归因与真实冻结条件验证','','先交付规划与四个真实冻结Lodge案例，再开展注册对照。全部原始动作、投影前proposal和层残差均保留；保护区由显式投影保留。','','## 规划选择','','完整预算池与原Top4的选择、错误音乐及禁止重复均按固定人口报告。来源增加不等于音乐内容更准确。','','|计划|评分/约束|选中来源|重复事件|','|---|---|---|---|']
    for r in planning:lines.append('|%s|%s|%s|%s|'%(r['key'],r['arm'],' → '.join(r['sources']),r['repeated_event']))
    lines += ['', '各臂汇总：'+str({k:{t:v for t,v in r.items() if t!="sequences"} for k,r in planning_aggregate.items()}), '', '规划保存原139事件合法性与分项，连接图仅使用R50两条/来源的16条元数据预算。FullPool取消Top4截断，NoExactRepeat为工程约束，不能作为音乐选择成功证明。','','## 真实冻结采样','','|来源|条件|Gap角误差rad|投影前Gap角误差|活动度|外侧精确保留|','|---|---|---:|---:|---:|---|']
    for r in neural:lines.append('|%s|%s|%.6f|%.6f|%.6f|%s|'%(r['key'],r['arm'],r['gap_angle_mean'],r['proposal_gap_angle_mean'],r['gap_activity'],r['protected_raw_exact']))
    lines += ['','Source仍沿用R38的1-C2_ENVELOPE注入门控，中心处直接Source残差为零；必须通过完整自注意力与Context间接传播，本轮未擅自修改旧门控。阶段来自可见时钟及运动学代理；隐藏缺口不参与条件。Source的双侧观测可能与Context冗余，去单侧Context不能解释为删除了全部单侧信息。ChangE零音乐不提供跨音乐真值，Root只有模型曲线，没有完整世界Root与接触资格。','','## 同人口配对排序','']
    for task,row in summary['paired'].items():
        lines.append('### '+task)
        for budget in ('summary','source_balanced_summary'):
            lines.append(budget+'：'+str(row[budget]))
    lines += ['','配对排序沿用历史曝光Val，不声称新音乐盲测。90帧候选Identity Warp为零，Full与Local的差异来自Multi。来源均衡子集与完整人口分开报告。','','## 资格与复验','',summary['pytest']+'。五组新进程科学JSON/全部数组精确重算；主效果和对照共同数组完全相同，R50的29个封存文件保持。没有新训练和整曲，Native24仍不可评，SDMR和神经候选保持Shadow。统一EDGE/Lodge公平选型仍需统一帧数、动作表示和监督任务，本轮不把它们混成架构性能结果。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    files=[]
    for directory in [ROOT/'main',ROOT/'plans',ROOT/'controls',ROOT/'replay']:
        for f in sorted(directory.rglob('*')):
            if f.is_file():files.append(dict(path=str(f),bytes=f.stat().st_size,sha256=digest(f)))
    for f in (out/'summary.json',out/'report.md',ROOT/'index.html',ROOT/'main_preview.png',ROOT/'browser_receipt.json',ROOT/'receipts/independent.json'):
        files.append(dict(path=str(f),bytes=f.stat().st_size,sha256=digest(f)))
    write(out/'seal.json',dict(files=files))


def recheck(out):
    manifest=read(ROOT/'publication/seal.json')
    for item in manifest['files']:
        path=Path(item['path'])
        assert path.resolve().is_relative_to(ROOT.resolve())
        assert path.stat().st_size==item['bytes'] and digest(path)==item['sha256'],str(path)
    write(out,dict(files=len(manifest['files']),all_hashes_verified=True,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),optimizer_calls=0))
