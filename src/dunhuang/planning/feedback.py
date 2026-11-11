"""Read-only Native evidence -> scoped temporal feedback. No search or editing."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
from dunhuang.io import checked,pinned,sha,write
from dunhuang.contracts import MotionAdapter
from .event_plan import semantic_plan_id

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def associated_slots(plan,start,end):
    return [i for i,e in enumerate(plan['events']) if max(start,e['music_interval'][0])<min(end,e['music_interval'][1])]

def jerk_localization(joints,raw_windows,plan,limit,halo=3):
    """Four-frame support, real 150/75 ownership; boundary proximity is not cause."""
    j=np.asarray(joints);values=np.linalg.norm(np.diff(j,n=3,axis=0)*30**3,axis=-1)
    maximum=values.max(1) if len(values) else np.zeros(0)
    raw=[np.linalg.norm(np.diff(w,n=3,axis=0)*30**3,axis=-1).max(1) for w in raw_windows]
    starts=np.flatnonzero(maximum>limit);counts={};rows=[];covered=near=0
    boundaries=sorted({b for i,w in enumerate(raw_windows) for b in (75*i,75*i+len(w)) if 0<b<len(j)})
    for start in starts:
        start=int(start);center=start+2
        owners=[w for w,profile in enumerate(raw) if 0<=start-75*w<len(profile)]
        raw_values=[float(raw[w][start-75*w]) for w in owners]
        exists=any(v>limit for v in raw_values);covered+=int(exists)
        boundary=any(abs(center-b)<=halo for b in boundaries);near+=int(boundary)
        slots=associated_slots(plan,start,start+4)
        for slot in slots:counts[slot]=counts.get(slot,0)+1
        rows.append(dict(stencil=[start,start+4],frame=center,value=float(maximum[start]),
            geometric_max_joint=int(values[start].argmax()),window_owners=owners,raw_window_values=raw_values,
            supporting_raw_window_exceeded=exists,near_window_boundary=boundary,associated_event_slots=slots,
            attribution_status='TEMPORAL_ASSOCIATION_ONLY'))
    return dict(schema='jerk_window_ownership_v1',limit=float(limit),violating_stencils=len(starts),
        supporting_raw_window_exceeded_count=covered,near_boundary_count=near,
        non_boundary_count=len(starts)-near,event_counts={str(k):v for k,v in counts.items()},rows=rows,
        raw_window_scope='saved sampler output windows before final Root/SLERP assembly, not raw per-step denoiser predictions',
        causal_attribution='UNKNOWN',boundary_halo_frames=halo)

def native_context(q):
    root=pinned(q['legacy_root'],q['legacy_commit']);sys.path.insert(0,str(root))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    from contracts.physical_quality import PhysicalQualityLimits
    from motion.config import MotionGenerationConfig
    from dataclasses import asdict
    cfg=MotionGenerationConfig()
    value=dict(schema='native_contract_identity_v1',legacy_commit=q['legacy_commit'],
        limits=asdict(PhysicalQualityLimits.from_environment()),
        motion_config=asdict(cfg),
        runtime=dict(python=sys.version,numpy=np.__version__),
        support_policy='final_fail_closed',metrics_schema='dunhuang_physical_metrics_si_v5_contact_states')
    return dict(value,id=digest(value)),cfg

def build(case,contract):
    case=Path(case);q=json.loads((case/'request.json').read_text())
    actual,cfg=native_context(q)
    if actual!=contract:raise ValueError('Native contract changed')
    physical=json.loads((case/'physical.json').read_text());plan=json.loads(checked(q['event_plan']).read_text())
    motion=np.load(case/'native.npy',allow_pickle=False);fk=np.load(case/'native_fk.npy',allow_pickle=False)
    from motion.physics import audit_motion_np,derive_contacts_np,full_sequence_physical_diagnostics_np
    from motion.geometry import fk_24_np
    from contracts.physical_quality import evaluate_physical_audit
    audit=audit_motion_np(motion,cfg,precomputed_joints=fk)
    clean=lambda value:json.loads(json.dumps(value))
    if clean(audit)!=physical['audit'] or clean(evaluate_physical_audit(audit))!=physical['absolute']:
        raise ValueError('Read-only Native replay differs from saved full audit')
    if physical['limits']!=contract['limits']:raise ValueError('Saved audit limits differ')
    # audit_motion_np already derives these contacts internally. Use precisely
    # that population for the existing read-only localizer, never new labels.
    temporary=motion.copy();temporary[:,:4]=derive_contacts_np(motion,cfg,precomputed_joints=fk)[0]
    diagnostic=full_sequence_physical_diagnostics_np(temporary,cfg,top_k=8)
    raw=np.load(case/'windows.npy',allow_pickle=False);windows=[]
    for w in range(len(raw)):
        valid=max(0,min(150,len(motion)-75*w))
        if valid>=4:windows.append(fk_24_np(MotionAdapter.to_native(raw[w,:valid].copy())))
    jerk=jerk_localization(fk,windows,plan,physical['limits']['joint_jerk_mps3_max'])
    violations=[];slot_counts={int(k):v for k,v in jerk['event_counts'].items()}
    for reason in physical['absolute']['reasons']:
        if reason.endswith('_too_high'):metric=reason[:-9];direction='high'
        elif reason.endswith('_too_low'):metric=reason[:-8];direction='low'
        else:metric=reason;direction='unknown'
        value=physical['audit'].get(metric);limit=physical['limits'].get(metric)
        violations.append(dict(metric=metric,reason=reason,actual=value,limit=limit,direction=direction,
            support_scope='full_requested_motion',attribution_status='UNKNOWN'))
    ranked=[]
    for family,entries in diagnostic['ranked_violations'].items():
        for row in entries:
            a,b=row['frame_span'];slots=associated_slots(plan,a,b)
            ranked.append(dict(row,family=family,associated_event_slots=slots,attribution_status='TEMPORAL_ASSOCIATION_ONLY'))
            for slot in slots:slot_counts[slot]=slot_counts.get(slot,0)+1
    ratio=diagnostic['violating_frames']/max(1,len(motion))
    broad=ratio>=.5
    if physical['absolute']['ok']:route='ACCEPT_NATIVE_REQUESTED_SCOPE'
    elif broad:route='MODEL_DIAGNOSIS'
    elif jerk['non_boundary_count']:route='REGENERATE_CONDITION_OR_SEED'
    elif jerk['near_boundary_count']:route='BOUNDARY_CONTEXT_DIAGNOSIS'
    else:route='DIAGNOSE_THEN_REPLAN'
    result=dict(schema='physical_feedback_v1',generated_motion=sha(case/'native.npy'),
        generated_fk=sha(case/'native_fk.npy'),raw_windows=sha(case/'windows.npy'),
        plan_id=q['event_plan']['sha256'],semantic_plan_id=semantic_plan_id(plan),
        generator_id=dict(checkpoint=q['checkpoint']['sha256'],adapter=q.get('adapter',{}).get('sha256'),mode=q['mode']),
        audio_sha256=q['audio']['sha256'],features_sha256=q['features']['sha256'],seed=q['seed'],
        native_contract_id=contract['id'],native_result='PASS' if physical['absolute']['ok'] else 'FAIL',
        full_song_scope=len(motion)==json.loads((case/'clock.json').read_text())['frames'],
        frames=len(motion),violations=violations,jerk=jerk,ranked_diagnostics=ranked,
        associated_slots=sorted(slot_counts,key=lambda i:(-slot_counts[i],i)),
        slot_temporal_association_counts={str(k):v for k,v in slot_counts.items()},
        violating_frame_fraction=ratio,broad_failure=broad,recommended_route=route,
        native_replay_exact=True,event_globally_infeasible=False,learned_physical_risk=None,
        music_alignment_status='NOT_CALIBRATED',event_following_status='NOT_QUALIFIED',
        attribution_note='Associations are not event/joint causal proof. Aggregate percentile failures are not individual-frame acceptance decisions.')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--contract-only',action='store_true');ap.add_argument('--case');ap.add_argument('--contract-file')
    a=ap.parse_args();q=json.loads(Path(a.request).read_text())
    if a.contract_only:value=native_context(q)[0]
    else:
        contract=json.loads(Path(a.contract_file).read_text());value=build(a.case,contract)
    path=Path(a.out)
    if path.exists():raise FileExistsError(path)
    write(path,value)
if __name__=='__main__':main()
