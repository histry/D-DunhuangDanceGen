"""Finite, preregistered source compensation experiment. No Planner or training."""
from pathlib import Path
from collections import Counter
import itertools,sys
import numpy as np
from scipy.signal import find_peaks
from . import warp
from dunhuang.research.local_timing_r68.common import read,write,sha
from dunhuang.research.local_timing_r68.study import select_event,plain,pair_metrics
from dunhuang.research.sdmr_warp_rank import connectivity
from dunhuang.research.complete_r36.core import rotation_sample,relative_fk
from dunhuang.research.paired_event_r41.events import profiles
from dunhuang.research.rhythm_r66.metrics import _motion
from dunhuang.research.warp_coupling_r53.common import array_hash,save_arrays

ROOT=Path('outputs/compensation_r73_01');PARENT=Path('outputs/retiming_baselines_r69_01/main')

def binding():
    return {n:sha(Path(__file__).parent/n) for n in ['warp.py','study.py','protocol.json']}

def compact(p):return plain({k:v for k,v in p.items() if k not in ['q','R','offset','edited_mask','compensation_mask']})

def choose(pool):
    high=[p for p in pool if p['kind']=='High' and p.get('actual_peak_frame') is not None]
    low=[p for p in pool if p['kind']=='Low' and p.get('actual_peak_frame') is not None]
    matches=[]
    for h,l in itertools.product(high,low):
        if h['width']!=l['width'] or h['activity']<=1.05*l['activity']:continue
        m=warp.match(h,l)
        matches.append((not m['eligible'],max(m['gaps'].values()),-h['activity']+l['activity'],h['id'],l['id'],h,l,m))
    if matches:
        _,_,_,_,_,h,l,m=min(matches,key=lambda p:p[:5]);chosen=[h,l]
    else:
        h=high[0] if high else None;l=low[0] if low else None
        chosen=[p for p in [h,l] if p];m=dict(eligible=False,reason='NO_COMMON_WIDTH_ACTIVITY_CONTRAST_AND_REDETECTED_PEAK')
    ds=[p for p in pool if p['kind']=='Distributed' and p.get('actual_peak_frame') is not None]
    if ds and chosen:
        d=min(ds,key=lambda d:(max(warp.match(d,chosen[0])['gaps'].values()),d['id']));chosen.append(d)
    return chosen,m

def run(mode='preview',destination=None):
    out=Path(destination) if destination else ROOT/mode
    if out.exists():raise ValueError('Fresh output required')
    out.mkdir(parents=True)
    cfg=read(Path(__file__).with_name('protocol.json'))
    previous=None
    if mode=='controls':
        accepted=read(ROOT/'receipts/preview_browser_acceptance.json')
        assert accepted['status']=='PASS' and accepted['manifest_sha256']==sha(ROOT/'preview/manifest.json')
        previous={(r['case_key'],r['method']):r for r in read(ROOT/'preview/records.json')}
    if sha(PARENT/'manifest.json')!='57ea35f5d7bd1d5d63da065e46e59951698d78b05c844de9aae711f890b08abe':raise ValueError('R69 seal changed')
    ctx=connectivity.prepare();cases=read(PARENT/'population.json')['cases'];assert len(cases)==12
    deltas=[6] if mode=='preview' else cfg['doses']
    write(out/'registration.json',dict(protocol=cfg,binding=binding(),cases=cases,doses=deltas,
        source_interval_count=8,selection_frozen_before_dynamics=True,human_reviewed=False))
    rows=[];proposal_rows=[];arrays={};groups=[]
    for ci,case in enumerate(cases):
        key=case['key'];source=ctx['lib'].rotations[case['source_uid']]
        base=warp.old.baseline(case['time_warp'],case['target_interval']);q0=base['q'];R0=rotation_sample(source,q0)
        assert array_hash(R0)==case['baseline_core_R_sha256']
        bconn,bgeo=connectivity.check(case,R0,ctx)
        baseline=dict(case_key=key,case_index=ci,method='E1W',status='LEGAL',connection=bconn,
                      core_R_sha256=array_hash(R0),pair_dynamics=pair_metrics(bgeo['full_R'],ctx['J'],ctx['parents']))
        rows.append(baseline);arrays[key+'__E1W__R']=R0;arrays[key+'__E1W__full_R']=bgeo['full_R'];arrays[key+'__E1W__q']=q0
        event,event_selection=select_event(case,'upper_speed_peak')
        for delta in deltas:
            group=dict(case_key=key,case_index=ci,delta=delta,event=event,event_selection=event_selection,
                       status='NO_EVENT' if event is None else 'NO_MATCH',methods=[],primary_match=None)
            if event is None:groups.append(group);continue
            w=np.asarray(case['time_warp']);centre=float(np.interp(event['frame'],w[:,0],w[:,1])-case['target_interval'][0]);expected=centre+delta
            activity=profiles(R0)['upper'];pool=[]
            for pi,p in enumerate(warp.proposals(centre,activity)):
                candidate=warp.solve(case,p,delta);candidate.update(id=f'{key}_{delta}_{pi}',case_key=key,case_index=ci)
                if candidate['status']=='LEGAL':
                    R=rotation_sample(source,candidate['q']);assert np.array_equal(R[:4],R0[:4]) and np.array_equal(R[-4:],R0[-4:])
                    v=profiles(R)['upper'];peaks=find_peaks(v)[0];near=[int(i) for i in peaks if abs(i-expected)<=8]
                    peak=min(near,key=lambda i:(abs(i-expected),i)) if near else None
                    candidate.update(R=R,actual_peak_frame=peak,mapped_anchor_target_frame=expected,
                        actual_peak_error_frames=None if peak is None else abs(peak-expected),core_R_sha256=array_hash(R),q_sha256=array_hash(candidate['q']))
                    pool.append(candidate)
                proposal_rows.append(compact(candidate))
            selected,match=choose(pool);group['primary_match']=match;group['status']='MATCHED_HIGH_LOW' if match['eligible'] else 'NO_MATCH'
            for p in selected:
                method=f'{p["kind"]}_{delta:+d}';R=p['R'];conn,geo=connectivity.check(case,R,ctx)
                compare=warp.match(p,selected[0]) if p['kind']=='Distributed' else match
                row=dict(case_key=key,case_index=ci,method=method,delta=delta,status='LEGAL',candidate=compact(p),
                    match=compare,connection=conn,core_R_sha256=array_hash(R),q_sha256=array_hash(p['q']),
                    core_dynamics=_motion(R,relative_fk(R,ctx['J'],ctx['parents']),False),
                    bridge_identical_to_rebuilt_baseline=array_hash(geo['bridge_R'])==array_hash(bgeo['bridge_R']) if 'bridge_R' in geo else None,
                    event_semantics='UNKNOWN',human_reviewed=False,Native24='NOT_EVALUABLE')
                if 'full_R' in geo:row['pair_dynamics']=pair_metrics(geo['full_R'],ctx['J'],ctx['parents'])
                rows.append(row);group['methods'].append(method)
                for n,a in dict(R=R,q=p['q'],offset=p['offset'],edited_mask=p['edited_mask'],compensation_mask=p['compensation_mask'],**geo).items():arrays[key+'__'+method+'__'+n]=a
            groups.append(group)
        print('R73_CASE',ci+1,12,flush=True)
    save_arrays(out/'arrays.npz',arrays);write(out/'records.json',rows);write(out/'proposals.json',proposal_rows);write(out/'groups.json',groups)
    if previous:
        for row in rows:
            if (row['case_key'],row['method']) in previous:
                old=previous[row['case_key'],row['method']]
                assert row['core_R_sha256']==old['core_R_sha256'] and row['connection'].get('full_R_sha256')==old['connection'].get('full_R_sha256')
    write(out/'summary.json',dict(groups=len(groups),status_counts=dict(Counter(g['status'] for g in groups)),
         proposed=len(proposal_rows),legal_proposals=sum(p['status']=='LEGAL' for p in proposal_rows),
         actual_G1_requests=ctx['counters'],human_records=0,human_preference='NOT_EVALUABLE',
         semantic_phase_truth='UNKNOWN',Native24='NOT_EVALUABLE',R74='NOT_STARTED',new_training_steps=0))
    write(out/'manifest.json',dict(binding=binding(),files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print('R73_COMPLETE',mode,len(groups),flush=True)

def replay(mode):
    parent=ROOT/mode;m=read(parent/'manifest.json');assert m['binding']==binding()
    out=ROOT/(mode+'_replay');run(mode,out)
    differences=[n for n,h in m['files'].items() if sha(out/n)!=h]
    write(out/'independent.json',dict(fresh_process=True,source_manifest_sha256=sha(parent/'manifest.json'),differences=differences,exact=not differences))
    assert not differences

if __name__=='__main__':
    mode='controls' if '--controls' in sys.argv else 'preview'
    replay(mode) if '--replay' in sys.argv else run(mode)
