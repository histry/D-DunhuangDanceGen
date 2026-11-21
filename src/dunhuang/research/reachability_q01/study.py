"""Bounded certificates first; complete metadata population and hypotheses later."""
import sys
import numpy as np
from .common import ROOT,OLD,HERE,CFG,read,write,sha,prepare,fresh,seal,verify,binding,parents
from dunhuang.planning.anchor_reachability import single,multiple
from dunhuang.research.paper1_auto_eval import music
from dunhuang.research.paper1_auto_eval.experiment import dynamics
from dunhuang.research.source_time_q01.study import connect
from dunhuang.research.complete_r36.core import rotation_sample
from dunhuang.research.warp_coupling_r53.common import array_hash,save_arrays
from dunhuang.research.warp_coupling_r53.warp_lattice import validate

def certificates():
    parents();reg=read(OLD/'preview/registration.json');rows=read(OLD/'preview/candidates.json');dec=read(OLD/'controls/decisions.json');result=[]
    with np.load(OLD/'preview/arrays.npz',allow_pickle=False) as z:
        for c in reg['cases']:
            for target in c['targets']:
                sources=[];w=next(d for d in dec if d['case_key']==c['key'] and d['fraction']==target['fraction'] and d['method']=='Joint')
                for s in c['sources']:
                    cert=single(s['source_interval'],s['hold']['cells_half_open'][1],c['target_frames'],target['target_frame'],CFG['target_tolerance_frames'],CFG['rates'],['FINITE_CLOCK_FAMILY','G1','MOVING_SUPPORT','FOUR_FRAME_GUARD_WHERE_APPLICABLE']);cert.update(source_uid=s['source_uid'],source_order=s['order'])
                    witness=None
                    if w['candidate_uid'] is not None and w['source_order']==s['order']:
                        r=next(r for r in rows if r['uid']==w['candidate_uid']);q=z[r['uid']+'__q'];t=float(np.interp(cert['anchor_source_frame'],q,np.arange(len(q))));assert abs(t-r['anchor_target_frame'])<1e-10 and cert['rate_status']=='REACHABLE_BY_RATE';assert cert['continuous_reachable_interval'][0]-1e-8<=t<=cert['continuous_reachable_interval'][1]+1e-8
                        witness=dict(candidate_uid=r['uid'],actual_anchor_target_frame=t,actual_clock_sha256=array_hash(q),actual_full_rotation_sha256=r['full_rotation_sha256'],G1_original_screen=r['G1_pass'],source_resampling_exact=r['source_resampling_exact'],protected_first_last_four=r['protected_first_last_four_exact'],target_error_frames=abs(t-target['target_frame']));assert r['G1_pass'] and abs(t-target['target_frame'])<=3+1e-8
                    cert['actual_frozen_execution_witness']=witness;sources.append(cert)
                possible=any(s['rate_status']=='REACHABLE_BY_RATE' for s in sources);assert possible==w['success'];result.append(dict(case_key=c['key'],source_uid=c['source_uid'],fraction=target['fraction'],target_frame=target['target_frame'],target_frames=c['target_frames'],sources=sources,rate_upper_bound_success=possible,actual_success=w['success'],scope='FIXED_TWO_SOURCES_ANCHOR_BUDGET_AND_RATES_ONLY'))
    return result

def preview():
    out=fresh(ROOT/'preview');write(out/'registration.json',dict(protocol=CFG,binding=binding(),parents=parents(),scope='READONLY_EXISTING_EXECUTIONS; NO_NEW_CLOCK_OR_G1_SOLVE'));rows=certificates();write(out/'certificates.json',rows);write(out/'summary.json',dict(tasks=len(rows),rate_permitted_tasks=sum(r['rate_upper_bound_success'] for r in rows),rate_excluded_tasks=sum(not r['rate_upper_bound_success'] for r in rows),actual_witness_tasks=sum(r['actual_success'] for r in rows),upper_bound_attained=True,source_groups=3,new_G1_calls=0,training_steps=0,can_route=False));seal(out)
    from .view import build_A
    build_A(out);print('REACHABILITY_PREVIEW_READY',flush=True)

def population(ctx):
    cats=music.catalogs();rows=[]
    for i,u in enumerate(sorted(ctx['units'],key=lambda u:(u['source_uid'],u['source_interval'],u['unit_uid']))):
        a,b=u['source_interval'];T=next((T for T in CFG['budgets'] if .75<=(T-1)/(b-a-1)<=1.33),None);ph=[p for p in u['phases'] if a<=p['evidence_span'][0]<=p['frame']<p['evidence_span'][1]<=b and a<p['frame']<b-1];on=next((p for p in ph if p['type']=='motion_onset' and p['body_region']=='upper'),None);peak=next((p for p in ph if p['type']=='speed_peak' and p['body_region']=='upper' and on is not None and p['frame']>on['frame']),None);hold=next((h for h in ctx['natural'][u['source_uid']]['regions']['body']['low_runs'] if peak is not None and a<h['cells_half_open'][0]<h['cells_half_open'][1]<b-1 and h['cells_half_open'][1]>peak['frame']),None)
        source_events=None if hold is None else [dict(kind='upper_motion_onset',source_frame=on['frame'],evidence_span=on['evidence_span']),dict(kind='upper_speed_peak',source_frame=peak['frame'],evidence_span=peak['evidence_span']),dict(kind='body_low_run_end',source_frame=hold['cells_half_open'][1],evidence_span=hold['cells_half_open'])];song=i%3;cat=cats[song];targets={};A=900
        if T is not None:
            for name,streams in CFG['hypotheses'].items():
                events=[]
                for f,(model,kind) in zip(CFG['target_fractions'],streams):
                    ev=[e for e in cat['events'] if e['source']==model and e['kind']==kind and e['seconds_supported'] and A/30<e['seconds']<(A+T-1)/30];e=min(ev,key=lambda e:(abs(e['seconds']*30-A-f*(T-1)),e['seconds'],e.get('uid',''))) if ev else None;events.append(None if e is None else dict(event_uid=e.get('uid'),source=model,kind=kind,seconds=e['seconds'],local_target_frame=e['seconds']*30-A,annotation='AUTOMATIC_UNVALIDATED'))
                targets[name]=events
        neighbor=None if T is None else next((dict(event_uid=v['unit_uid'],source_uid=v['source_uid'],source_interval=v['source_interval']) for v in sorted(ctx['units'],key=lambda v:(v['source_uid'],v['source_interval'],v['unit_uid'])) if v['source_uid']!=u['source_uid'] and .75<=(T-1)/(v['duration_frames']-1)<=1.33),None)
        rows.append(dict(unit_uid=u['unit_uid'],source_uid=u['source_uid'],source_interval=[a,b],source_sha256=u['source_sha256'],target_frames=T,source_events=source_events,missing=[] if T is not None and source_events is not None else (['NO_LEGAL_REGISTERED_BUDGET'] if T is None else [])+(['NO_ORDERED_SUPPORTED_TRIPLE'] if source_events is None else []),music_song=song,music_start_frame=A,audio_sha256=cat['audio_sha256'],targets=targets,neighbor=neighbor,semantic_completeness='UNKNOWN',event_correspondence_truth='UNKNOWN'))
    return rows

def controls(dest=None):
    verify(ROOT/'preview')
    if dest is None:
        r=read(ROOT/'receipts/browser_acceptance.json');assert r['status']=='PASS' and r['preview_manifest_sha256']==sha(ROOT/'preview/manifest.json')
    out=fresh(dest or ROOT/'controls');ctx=prepare();pop=population(ctx);display=[]
    for source in sorted({r['source_uid'] for r in pop}):
        first=next((r for r in pop if r['source_uid']==source and not r['missing']),None)
        display.append(dict(source_uid=source,unit_uid=None if first is None else first['unit_uid'],status='MISSING_METADATA_COMPLETE_TRIPLE' if first is None else 'REGISTERED_BEFORE_CERTIFICATES'))
    write(out/'population.json',dict(protocol=CFG,units=pop,display=display,full_R47_units=len(pop),source_groups=len({r['source_uid'] for r in pop}),registered_before_certificates=True));results=[];arrays={};cache={}
    for u in pop:
        for name in CFG['hypotheses']:
            key=u['unit_uid']+'__'+name;r=dict(unit_uid=u['unit_uid'],source_uid=u['source_uid'],hypothesis=name,key=key,status='MISSING',semantic_correspondence='UNKNOWN',artistic_quality='NOT_EVALUABLE')
            if u['missing']:results.append(dict(r,reason=u['missing']));continue
            ev=u['targets'][name]
            if any(e is None for e in ev):results.append(dict(r,reason=['MISSING_MUSIC_EVENT_ID']));continue
            if not np.all(np.diff([e['local_target_frame'] for e in ev])>0):results.append(dict(r,reason=['NONORDERED_AUTOMATIC_TARGET_EVENTS']));continue
            anchors=[dict(source_frame=s['source_frame'],requested=e['local_target_frame'],tolerance=3) for s,e in zip(u['source_events'],ev)];individual=[single(u['source_interval'],s['source_frame'],u['target_frames'],e['local_target_frame'],3,CFG['rates']) for s,e in zip(u['source_events'],ev)];cert=multiple(u['source_interval'],u['target_frames'],anchors,CFG['rates'],additional_constraints=['G1','SOURCE_SUPPORTED_SO3_SAMPLING']);r.update(status=cert['rate_status'],certificate=cert,individual_rate_status=[v['rate_status'] for v in individual],individual_intervals=[v['continuous_reachable_interval'] for v in individual],marginals_all_possible=all(v['rate_status']=='REACHABLE_BY_RATE' for v in individual))
            if cert['witness_clock'] is not None:
                T=u['target_frames'];w=np.asarray(cert['witness_clock']);validate(w,u['source_interval'],[0,T]);q=np.interp(np.arange(T),w[:,1],w[:,0]);source=ctx['lib'].rotations[u['source_uid']];R=rotation_sample(source,q);arrays[key+'__q']=q;arrays[key+'__R']=R;d,P=dynamics(R,ctx);r.update(clock_sha256=array_hash(q),rotation_sha256=array_hash(R),dynamics=d,sampled_inverse_rate_legal=bool(np.all(np.diff(q)>=1/1.33-1e-8) and np.all(np.diff(q)<=1/.75+1e-8)),new_clock_has_four_frame_guard=False)
                assert r['sampled_inverse_rate_legal']
                if u['neighbor'] is not None:
                    full,connection=connect(ctx,R,u['neighbor'],T,cache);r['connection']=connection;r['actual_G1_pass']=connection['passes_original_screen'];r['full_rotation_sha256']=None if full is None else array_hash(full)
                    if full is not None:arrays[key+'__full_R']=full
                else:r.update(actual_G1_pass=None,execution_status='NEIGHBOR_MISSING_NO_G1_CERTIFICATE')
            results.append(r)
        if u['unit_uid'] in {v['unit_uid'] for v in display if v['unit_uid'] is not None}:
            a,b=u['source_interval'];T=u['target_frames'];source=ctx['lib'].rotations[u['source_uid']];arrays[u['unit_uid']+'__raw_R']=source[a:b];arrays[u['unit_uid']+'__Uniform_R']=rotation_sample(source,np.linspace(a,b-1,T))
    save_arrays(out/'arrays.npz',arrays);write(out/'results.json',results);write(out/'summary.json',dict(full_units=len(pop),source_groups=len({r['source_uid'] for r in pop}),metadata_complete_units=sum(not r['missing'] for r in pop),hypotheses=len(results),missing_hypotheses=sum(r['status']=='MISSING' for r in results),rate_feasible=sum(r['status']=='REACHABLE_BY_RATE' for r in results),rate_excluded=sum(r['status']=='EXCLUDED_BY_RATE' for r in results),marginals_possible_joint_excluded=sum(r.get('marginals_all_possible',False) and r['status']=='EXCLUDED_BY_RATE' for r in results),actual_G1_requests=sum('actual_G1_pass' in r and r['actual_G1_pass'] is not None for r in results),actual_G1_passes=sum(r.get('actual_G1_pass') is True for r in results),actual_G1_solves=len(cache),artistic_quality='NOT_EVALUABLE',event_correspondence_accuracy='NOT_EVALUABLE',whole_song='NOT_STARTED',can_route=False,training_steps=0));seal(out);print('MULTI_ANCHOR_CONTROLS_READY',len(results),flush=True)

def replay():
    verify(ROOT/'preview');assert certificates()==read(ROOT/'preview/certificates.json');controls(ROOT/'controls_replay');m=verify(ROOT/'controls');diff=[n for n,h in m['files'].items() if sha(ROOT/'controls_replay'/n)!=h];assert not diff,diff;write(ROOT/'receipts/independent.json',dict(status='PASS',fresh_process=True,exact_control_arrays=True,scalar_differences=diff,A_certificates_exact=True,parents_unchanged=parents(),training_steps=0,can_route=False));print('REACHABILITY_REPLAY_PASS',flush=True)

if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='preview':preview()
    elif stage=='controls':controls()
    elif stage=='replay':replay()
    else:raise ValueError(stage)
