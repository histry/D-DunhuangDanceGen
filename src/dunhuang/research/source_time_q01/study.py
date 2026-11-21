"""Registered source x clock feasibility, actual SO(3) execution and fixed G1."""
import sys
from dataclasses import asdict
import numpy as np
from .common import ROOT,HERE,CFG,read,write,sha,prepare,checked_parent,fresh,seal,verify,binding,parents
from . import baselines
from dunhuang.research.paper1_support import study as old
from dunhuang.research.paper1_support.contracts import MusicEvidence,boundary_unchanged
from dunhuang.research.paper1_auto_eval import music
from dunhuang.research.paper1_auto_eval.experiment import dynamics
from dunhuang.research.complete_r36.core import multi_warp,rotation_sample
from dunhuang.research.warp_coupling_r53.common import array_hash,identity,save_arrays
from dunhuang.research.warp_coupling_r53.warp_lattice import validate

def register(ctx):
    oldreg,_=checked_parent();pre=read('reports/paper1_auto_eval_release/preview_registration.json')['source_time_preflight'];cats=music.catalogs();cases=[];missing=[]
    for p in pre:
        if p['status']!='REGISTERED_NONDEGENERATE_PAIR':missing.append(dict(case_key=p['case_key'],source_uid=p['source_uid'],reason='PREEXISTING_NONDEGENERATE_PAIR_MISSING',replacement_used=False));continue
        c=next(c for c in oldreg['cases'] if c['key']==p['case_key']);sources=[]
        for k,v in enumerate(p['sources']):
            u=next(u for u in ctx['units'] if u['unit_uid']==v['event_uid']);a,b=u['source_interval'];assert [a,b]==v['source_interval'];runs=ctx['natural'][u['source_uid']]['regions']['body']['low_runs'];mask=ctx['natural'][u['source_uid']]['regions']['body']['low_cells']
            h=c['hold'] if k==0 else next(h for h in runs if a+4<h['cells_half_open'][0]<h['cells_half_open'][1]<b-5 and sum(not z for z in mask[a:h['cells_half_open'][0]])>=30 and sum(not z for z in mask[h['cells_half_open'][1]:b-1])>=30)
            sources.append(dict(order=k,event_uid=u['unit_uid'],source_uid=u['source_uid'],source_interval=[a,b],source_sha256=u['source_sha256'],hold=h,expected_preflight_clocks=v['distinct_clocks']))
        T=c['target_frames'];A=c['music_start_frame'];cat=cats[c['music_song']];ev=[e for e in cat['events'] if e['source']=='beat_this' and e['kind']=='beat' and e['seconds_supported'] and A/30<e['seconds']<(A+T-1)/30];targets=[]
        for f in CFG['target_fractions']:
            e=min(ev,key=lambda e:(abs(e['seconds']*30-A-f*(T-1)),e['seconds'],e.get('uid',''))) if ev else None
            targets.append(dict(fraction=f,event_uid=None if e is None else e.get('uid'),target_frame=None if e is None else float(e['seconds']*30-A),target_seconds=None if e is None else float(e['seconds']),status='MISSING_BEAT_TARGET' if e is None else 'REGISTERED_AUTOMATIC_TARGET_NOT_TRUTH',music_semantic_correctness='UNKNOWN'))
        n=c['neighbor'];cases.append(dict(key=c['key'],source_uid=c['unit']['source_uid'],sources=sources,target_frames=T,bridge_frames=48,pair_frames=2*T+48,music_song=c['music_song'],music_start_frame=A,audio_sha256=cat['audio_sha256'],music_catalog_sha256=music.CFG['music_catalog_sha256'][c['music_song']],targets=targets,neighbor=dict(event_uid=n['unit_uid'],source_uid=n['source_uid'],source_interval=n['source_interval'],source_sha256=n['source_sha256'])))
    return dict(protocol=CFG,binding=binding(),parents=parents(),cases=cases,missing=missing,target_registration_precedes_costs=True,exposure='THREE_HISTORICALLY_EXPOSED_RECORDINGS; PAIRED_INTERVALS_NOT_INDEPENDENT_SOURCES',artistic_quality='NOT_EVALUABLE')

def connect(ctx,R,neighbor,T,cache):
    a,b=neighbor['source_interval'];right=rotation_sample(ctx['lib'].rotations[neighbor['source_uid']],np.linspace(a,b-1,T));obs,gap=old.make_observed(R,right,48);edge=np.concatenate([R[-4:],right[:4]]);key=identity([array_hash(edge),48]);hit=key in cache
    if not hit:
        curves,proof=old.solve_boundary(obs,gap);Q=curves['G1'];cache[key]=(None if Q is None else Q[gap[0]:gap[1]].copy(),proof)
    middle,proof=cache[key]
    if middle is None:return None,dict(constructible=False,passes_original_screen=False,boundary_key=key,cache_hit=hit,actual_geometry_calls=int(not hit),inherited=False)
    small=obs.copy();small[gap[0]:gap[1]]=middle;m=old.measure(small,obs,ctx['J'],ctx['parents'],gap,ctx['gate']);full=np.concatenate([R,middle,right]);assert np.array_equal(full[:T],R) and np.array_equal(full[T+48:],right)
    return full,dict(m,boundary_key=key,cache_hit=hit,actual_geometry_calls=int(not hit),bridge_sha256=array_hash(middle),proof=proof,inherited=False,core_exact=True)

def execute(reg,ctx):
    cats=music.catalogs();rows=[];arrays={};geometry_cache={};rejected=[]
    for c in reg['cases']:
        T=c['target_frames'];A=c['music_start_frame'];evidence=MusicEvidence(c['audio_sha256'],(A,A+T));cat=cats[c['music_song']];beats=[e['seconds']*30 for e in cat['events'] if e['source']=='R64' and e['kind']=='beat' and e['seconds_supported']]
        for s in c['sources']:
            u=next(u for u in ctx['units'] if u['unit_uid']==s['event_uid']);source=ctx['lib'].rotations[s['source_uid']];a,b=s['source_interval'];baseq=np.linspace(a,b-1,T);baseR=rotation_sample(source,baseq)
            pool,bank,uniform=old.evaluate_pool(ctx,u,s['hold'],T,evidence);assert len(pool)==s['expected_preflight_clocks'];scorer=old.ReferenceCost(source[max(0,a-30):min(len(source),b+30)],ctx['J'],ctx['parents'])
            for r in pool:r['labels']=['Uniform' if r['key']==uniform else 'HoldGrid']
            phases=[dict(p,type=p['body_region']+'_'+p['type']) for p in u['phases']];ew=multi_warp([a,b],[A,A+T],phases,beats)
            if ew is not None:
                w=np.asarray(ew['time_warp']);w[:,1]-=A;validate(w,[a,b],[0,T]);q=np.interp(np.arange(T),w[:,1],w[:,0]);duplicate=next((k for k,(oldq,_) in bank.items() if np.allclose(q,oldq,atol=1e-9,rtol=0)),None)
                if duplicate is not None:next(r for r in pool if r['key']==duplicate)['labels'].append('E1W')
                else:
                    R=rotation_sample(source,q);holds=old.metrics.occupancy(q,ctx['natural'][s['source_uid']]['regions']['body']['low_runs'],[a,b]);k=array_hash(q);bank[k]=(q,R);pool.append(dict(key=k,labels=['E1W'],GTWLocal=scorer.score(R,only='GTWLocal')[0]['GTWLocal'],mapped_low_seconds=sum(h['target_seconds'] for h in holds),candidate=asdict(old.ExecutionCandidate(old.SourcePrimitive(u['unit_uid'],s['source_uid'],(a,b),u['source_sha256']),evidence,T,tuple(map(tuple,w.tolist()))).validate()),protected_four_frames_exact=boundary_unchanged(q,baseq,R,baseR)))
            else:rejected.append(dict(case_key=c['key'],source_order=s['order'],kind='E1W',reason='FROZEN_SOLVER_NO_LEGAL_PATH',replacement_used=False))
            for r in pool:
                q,R=bank[r['key']];legal=bool(np.all(np.diff(q)>=1/1.33-1e-8) and np.all(np.diff(q)<=1/.75+1e-8));assert legal;uid=identity([c['key'],u['unit_uid'],r['key']]);full,connection=connect(ctx,R,c['neighbor'],T,geometry_cache);arrays[uid+'__q']=q;arrays[uid+'__R']=R
                if full is not None:arrays[uid+'__full_R']=full
                anchor=float(np.interp(s['hold']['cells_half_open'][1],q,np.arange(T)));support=(T-1)/30-r['mapped_low_seconds'];deformation=float(np.sqrt(np.mean(((q-baseq)/(b-a-1))**2)))
                rows.append(dict(uid=uid,case_key=c['key'],source_order=s['order'],source_uid=s['source_uid'],source_interval=[a,b],labels=r['labels'],GTWLocal=r['GTWLocal'],mapped_low_seconds=r['mapped_low_seconds'],moving_support_seconds=support,normalized_clock_RMS=deformation,max_clock_shift_frames=float(np.max(abs(q-baseq))),anchor_source_frame=s['hold']['cells_half_open'][1],anchor_target_frame=anchor,clock_legal=legal,G1_pass=connection['passes_original_screen'],connection=connection,rotation_sha256=array_hash(R),clock_sha256=array_hash(q),full_rotation_sha256=None if full is None else array_hash(full),protected_first_last_four_exact=boundary_unchanged(q,baseq,R,baseR),source_resampling_exact=True,target_frames=T))
            rawid=c['key']+'__S'+str(s['order']);arrays[rawid+'__raw_R']=source[a:b]
        print('SOURCE_TIME_POOL',c['source_uid'],T,sum(r['case_key']==c['key'] for r in rows),flush=True)
    return rows,arrays,rejected,dict(candidate_G1_requests=len(rows),actual_G1_solves=len(geometry_cache),cache_reuses=sum(r['connection']['cache_hit'] for r in rows),common_pool_execution_charged_to_every_arm=True)

def decisions(reg,rows,methods):
    out=[]
    for c in reg['cases']:
        own=[r for r in rows if r['case_key']==c['key']]
        for target in c['targets']:
            for method in methods:
                if target['target_frame'] is None:out.append(dict(case_key=c['key'],fraction=target['fraction'],method=method,success=False,candidate_uid=None,diagnostic_uid=None,missing='TARGET_UNAVAILABLE'));continue
                d=baselines.choose(method,own,target['target_frame']);out.append(dict(d,shared_pool_actual_G1_solves=sum(r['connection']['actual_geometry_calls'] for r in own),shared_pool_candidate_G1_requests=len(own),case_key=c['key'],fraction=target['fraction'],target_frame=target['target_frame'],target_event_uid=target['event_uid'],target_frames=c['target_frames'],pair_frames=c['pair_frames']))
    return out

def preview():
    out=fresh(ROOT/'preview');ctx=prepare();reg=register(ctx);write(out/'registration.json',reg);rows,arrays,rejected,budget=execute(reg,ctx);write(out/'candidates.json',rows);write(out/'rejected.json',rejected);write(out/'geometry_budget.json',budget);save_arrays(out/'arrays.npz',arrays);write(out/'decisions.json',decisions(reg,rows,CFG['preview_methods']));seal(out)
    from .view import build
    build(out,CFG['preview_methods']);print('SOURCE_TIME_PREVIEW_READY',len(rows),flush=True)

def controls(dest=None):
    verify(ROOT/'preview')
    if dest is None:
        receipt=read(ROOT/'receipts/browser_acceptance.json');assert receipt['status']=='PASS' and receipt['preview_manifest_sha256']==sha(ROOT/'preview/manifest.json')
    out=fresh(dest or ROOT/'controls');reg=read(ROOT/'preview/registration.json');rows=read(ROOT/'preview/candidates.json');all_d=decisions(reg,rows,CFG['methods']);write(out/'decisions.json',all_d);ctx=prepare();cats=music.catalogs();metrics=[]
    with np.load(ROOT/'preview/arrays.npz',allow_pickle=False) as z:
        for r in rows:
            R=z[r['uid']+'__R'];d,P=dynamics(R,ctx);c=next(c for c in reg['cases'] if c['key']==r['case_key']);A=c['music_start_frame']/30;end=A+(len(R)-1)/30;events=[e['seconds'] for e in cats[c['music_song']]['events'] if e['source']=='beat_this' and e['kind']=='beat' and e['seconds_supported'] and A+.5<=e['seconds']<end-.5]
            d['BAS_diagnostic_only']=music.gaussian_match(events,music.motion_beats(P)+A);d['source_uid']=r['source_uid'];d['candidate_uid']=r['uid'];metrics.append(d)
    summary=[]
    for method in CFG['methods']:
        own=[d for d in all_d if d['method']==method];ok=[d for d in own if d['success']];summary.append(dict(method=method,registered_tasks=len(own),successful_tasks=len(ok),source_recording_groups=len(reg['cases']),mean_target_error_success_only=None if not ok else float(np.mean([d['target_error_frames'] for d in ok])),failure_count=len(own)-len(ok),artistic_quality='NOT_EVALUABLE'))
    equality=[]
    for d in [d for d in all_d if d['method']=='Joint']:
        other=next(v for v in all_d if v['method']=='ExhaustiveSequential' and (v['case_key'],v['fraction'])==(d['case_key'],d['fraction']));assert d['candidate_uid']==other['candidate_uid'];equality.append(dict(case_key=d['case_key'],fraction=d['fraction'],identical_selection=True,both_fail=d['candidate_uid'] is None))
    write(out/'metrics.json',metrics);write(out/'summary.json',summary);write(out/'exhaustive_equivalence.json',equality);write(out/'qualification.json',dict(mechanism='FINITE_SHARED_POOL_FEASIBILITY_ONLY',joint_vs_exhaustive='IDENTICAL_BY_SAME_OBJECTIVE_AND_FULL_BUDGET',independent_generalization=False,music_artistic_validity='NOT_EVALUABLE',new_training_steps=0,new_G1_calls_after_shared_pool=0,BAS_or_SDMR_used_for_selection=False,can_route=False));seal(out)
    print('SOURCE_TIME_CONTROLS_READY',flush=True)

def replay():
    original=verify(ROOT/'preview');ctx=prepare();reg=register(ctx);assert reg==read(ROOT/'preview/registration.json');rows,arrays,rejected,budget=execute(reg,ctx);out=fresh(ROOT/'replay');save_arrays(out/'arrays.npz',arrays);assert sha(out/'arrays.npz')==sha(ROOT/'preview/arrays.npz');assert rows==read(ROOT/'preview/candidates.json') and rejected==read(ROOT/'preview/rejected.json') and budget==read(ROOT/'preview/geometry_budget.json');controls(ROOT/'controls_replay');m=verify(ROOT/'controls');diff=[n for n,h in m['files'].items() if sha(ROOT/'controls_replay'/n)!=h];assert not diff,diff;write(ROOT/'receipts/independent.json',dict(status='PASS',fresh_process=True,arrays_exact=True,candidate_ledger_exact=True,scalar_differences=diff,parents_unchanged=parents(),new_training_steps=0,can_route=False));print('SOURCE_TIME_REPLAY_PASS',flush=True)

if __name__=='__main__':
    command=sys.argv[1]
    if command=='preview':preview()
    elif command=='controls':controls()
    elif command=='replay':replay()
    else:raise ValueError(command)
