"""Frozen R73 domain scoring. No new source search, labels, tuning or training."""
from pathlib import Path
from collections import Counter
import sys,time,json
import numpy as np
from .features import ReferenceCost,METHODS,select
from dunhuang.research.compensation_r73 import warp
from dunhuang.research.local_timing_r68.common import read,write,sha
from dunhuang.research.local_timing_r68.study import pair_metrics,plain
from dunhuang.research.complete_r36.core import rotation_sample
from dunhuang.research.sdmr_warp_rank import connectivity
from dunhuang.research.warp_coupling_r53.common import array_hash,save_arrays
from dunhuang.research.rhythm_r66.metrics import _motion
from dunhuang.research.complete_r36.core import relative_fk

ROOT=Path('outputs/gtw_baseline_01');PARENT=Path('outputs/compensation_r73_01/controls')

def binding():return {n:sha(Path(__file__).with_name(n)) for n in ['study.py','features.py','protocol.json']}

def run(mode='preview',dest=None):
    out=Path(dest) if dest else ROOT/mode;out.mkdir(parents=True,exist_ok=False)
    assert sha(PARENT/'manifest.json')=='05f3cf963c8d92f49e2da3020ac68c484a28da6777781527399ff448fd8f0192'
    if mode=='controls':
        receipt=read(ROOT/'receipts/preview_browser_acceptance.json');assert receipt['status']=='PASS' and receipt['manifest_sha256']==sha(ROOT/'preview/manifest.json')
    cfg=read(Path(__file__).with_name('protocol.json'));ctx=connectivity.prepare();cases=read(PARENT/'registration.json')['cases'];proposals=read(PARENT/'proposals.json');groups=read(PARENT/'groups.json')
    groups=[g for g in groups if mode=='controls' or g['delta']==6]
    write(out/'registration.json',dict(protocol=cfg,binding=binding(),parent_manifest_sha256=sha(PARENT/'manifest.json'),cases=cases,groups=groups))
    rows=[];costs=[];selection=[];arrays={};models={};start=time.perf_counter()
    for g in groups:
        case=cases[g['case_index']];key=case['key'];d=g['delta'];unit=f'{key}_{d:+d}';source=ctx['lib'].rotations[case['source_uid']];a,b=case['source_interval']
        modelkey=(case['source_uid'],a,b)
        if modelkey not in models:models[modelkey]=ReferenceCost(source[max(0,a-30):min(len(source),b+30)],ctx['J'],ctx['parents'])
        model=models[modelkey];base=warp.old.baseline(case['time_warp'],case['target_interval']);q0=base['q'];R0=rotation_sample(source,q0)
        assert array_hash(R0)==case['baseline_core_R_sha256']
        options=[];motions={};own=[p for p in proposals if p['case_key']==key and p['delta']==d]
        for p in own:
            row=dict(id=p['id'],unit=unit,parent_status=p['status'],eligible=False)
            if p['status']=='LEGAL':
                spec={k:p[k] for k in ['width','segments','activity','kind','rank']};m=warp.solve(case,spec,d);assert m['status']=='LEGAL'
                R=rotation_sample(source,m['q']);assert array_hash(R)==p['core_R_sha256'] and array_hash(m['q'])==p['q_sha256']
                scores,coarse=model.score(R);row.update(scores=scores,coarse=coarse,actual_peak=p.get('actual_peak_frame'),core_sha256=array_hash(R))
                motions[p['id']]=(R,m['q'],m['compensation_mask']);options.append(row)
            costs.append(row)
        # Freeze an alignment stratum before any scoring-based selection. Never fill NO_EVENT.
        available=[r for r in options if r['actual_peak'] is not None]
        expected=None if not available else next(p['mapped_anchor_target_frame'] for p in own if p['status']=='LEGAL')
        if available:
            peak=min({r['actual_peak'] for r in available},key=lambda v:(abs(v-expected),v))
            for r in available:r['eligible']=r['actual_peak']==peak
        else:peak=None
        pool=[r for r in options if r['eligible']];poolhash=sha_json(sorted(r['id'] for r in pool))
        boptions=[]
        if mode=='controls':
            for r in options:
                _,coarse=model.score(motions[r['id']][0],coarse_only=True)
                boptions.append(dict(r,coarse=coarse,scores={}))
            for method in METHODS:
                shortlist=sorted([r for r in boptions if r['eligible']],key=lambda r:(r['coarse'][method],r['id']))[:4]
                for r in shortlist:r['scores'].update(model.score(motions[r['id']][0],only=method)[0])
        selections=[]
        for protocol in (['A'] if mode=='preview' else ['A','B']):
            for method in METHODS:
                chosen=select(options if protocol=='A' else boptions,method,None if protocol=='A' else 4)
                record=dict(unit=unit,case_index=g['case_index'],delta=d,protocol=protocol,method=method,selected_id=None if chosen is None else chosen['id'],
                    candidate_pool_sha256=poolhash,eligible_count=len(pool),coarse_queries=len(options) if protocol=='B' else 0,
                    full_score_queries=len(options) if protocol=='A' else min(4,len(pool)),status='NO_ELIGIBLE_CANDIDATE' if chosen is None else 'SELECTED',actual_peak=peak,
                    selected_score=None if chosen is None else chosen['scores'][method],
                    expected_anchor=expected,quality='NOT_EVALUABLE',phase='UNKNOWN',human_records=0)
                selection.append(record)
                if chosen is not None:selections.append((protocol+'_'+method,chosen['id'],*motions[chosen['id']]))
        uq=np.linspace(a,b-1,240);uniform_legal=.75<=(239/(b-a-1))<=1.33
        baselines=[('E1W','FROZEN',R0,q0,np.zeros(240,bool))]
        if uniform_legal:baselines.append(('Uniform','UNIFORM',rotation_sample(source,uq),uq,np.zeros(240,bool)))
        else:rows.append(dict(unit=unit,method='Uniform',status='NOT_FEASIBLE',reason='FROZEN_RATE_LIMIT'))
        for method,cid,R,q,mask in baselines+selections:
            ctx['cache'].clear()  # Independently solve every displayed execution; do not inherit certificates.
            conn,geo=connectivity.check(case,R,ctx)
            row=dict(unit=unit,case_index=g['case_index'],delta=d,method=method,candidate_id=cid,status='EXECUTED',connection=conn,core_sha256=array_hash(R),q_sha256=array_hash(q),
                core_dynamics=_motion(R,relative_fk(R,ctx['J'],ctx['parents']),False),protected_edges_exact=np.array_equal(R[:4],R0[:4]) and np.array_equal(R[-4:],R0[-4:]),
                same_alignment_task_as_interventions=method not in ['E1W','Uniform'],Native24='NOT_EVALUABLE',human_reviewed=False)
            if 'full_R' in geo:row['pair_dynamics']=pair_metrics(geo['full_R'],ctx['J'],ctx['parents'])
            rows.append(row)
            for name,value in dict(R=R,q=q,compensation_mask=mask,**geo).items():arrays[unit+'__'+method+'__'+name]=value
        print('GTW_GROUP',len({r['unit'] for r in selection}),len(groups),flush=True)
    if mode=='controls':
        old=read(ROOT/'preview/records.json');lookup={(r['unit'],r['method']):r for r in rows}
        for r in old:
            if r['status']=='EXECUTED':assert r['core_sha256']==lookup[r['unit'],r['method']]['core_sha256']
    save_arrays(out/'arrays.npz',arrays);write(out/'records.json',plain(rows));write(out/'costs.json',plain(costs));write(out/'selections.json',plain(selection))
    summary=dict(groups=len(groups),proposals=len(costs),legal=sum('scores' in r for r in costs),selection_status=dict(Counter(r['status'] for r in selection)),
        coarse_knn_queries=sum(r['coarse_queries'] for r in selection),full_knn_queries=sum(r['full_score_queries'] for r in selection),
        actual_geometry=ctx['counters'],distinct_source_intervals=len(models),human_records=0,phase_labels=0,preference_prediction='NOT_EVALUABLE',R74='NOT_STARTED',new_training_steps=0)
    write(out/'summary.json',summary);write(out/'runtime.json',dict(seconds=time.perf_counter()-start,scientific_replay_excludes_timing=True))
    write(out/'manifest.json',dict(binding=binding(),files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='runtime.json'}))

def sha_json(v):
    import hashlib
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def replay(mode):
    m=read(ROOT/mode/'manifest.json');assert m['binding']==binding();out=ROOT/(mode+'_replay');run(mode,out)
    differences=[n for n,h in m['files'].items() if sha(out/n)!=h];write(out/'independent.json',dict(fresh_process=True,exact=not differences,differences=differences,source_manifest_sha256=sha(ROOT/mode/'manifest.json')));assert not differences

if __name__=='__main__':
    mode='controls' if '--controls' in sys.argv else 'preview'
    replay(mode) if '--replay' in sys.argv else run(mode)
