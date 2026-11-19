"""Nested crops delete previously observed evidence; no new-neighbor confound."""
from .common import *
from dunhuang.research.phase_integrity_r60.population import Assets

def choose(source,kind,exclusions=(),context=45):
    reasons={};eligible=[]
    for p in sorted(source['phase_evidence'],key=lambda p:(p['frame'],p['body_region'],p['type'])):
        if p['type']!=kind:continue
        x,y=map(int,p['evidence_span']);a=x-context;b=y+context
        if kind=='speed_peak' and p['body_region']!=CFG['speed_peak_region']:continue
        if y-x<CFG['minimum_raw_support_frames'][kind]:continue
        component=next((v for v in source['accepted_components'] if v[0]<=a and b+4<=v[1]),None)
        if component is None:reasons['NO_FULL_REAL_CONTEXT']=reasons.get('NO_FULL_REAL_CONTEXT',0)+1;continue
        if any(max(a,lo)<min(b,hi) for lo,hi in exclusions):reasons['OVERLAP_POSTHOC_DIAGNOSTIC']=reasons.get('OVERLAP_POSTHOC_DIAGNOSTIC',0)+1;continue
        cut=x if kind=='speed_peak' else (x+y)//2
        assert a<cut<b and a<=x<y<=b
        eligible.append(dict(evidence=p,A4=[a,b],A5=[a,cut],context_interval=[a,b],component=component,
                             removed_interval=[cut,b],removed_raw_interval=[max(cut,x),y],
                             existing_center_removed=bool(cut<=p['frame']<b),removed_raw_frames=y-max(cut,x),
                             added_frames=0,raw_previously_fully_in_A4=True,
                             intervention_type='EXISTING_PHASE_REMOVAL_CANDIDATE',
                             exact_fact='EXISTING_KINEMATIC_EVIDENCE_DELETED; NO_SEMANTIC_CONFIRMATION',
                             source_length_A4=b-a,source_length_A5=cut-a))
    return dict(status='MATCHED',**eligible[0],eligible_candidates=len(eligible),rejection_counts=reasons) if eligible else dict(status='NOT_MATCHABLE',rejection_counts=reasons)

def population():
    parent=read(PARENT/'population/result.json');a=Assets().build()
    assert list(a.sources.values())==parent['sources'] and a.units==parent['units']
    cases=[]
    for i,(uid,kind) in enumerate(CFG['roster']):
        s=a.sources[uid];q=choose(s,kind,CFG['excluded_diagnostic_intervals'].get(uid,[]),CFG['context'])
        cases.append(dict(case=i,source_uid=uid,dataset=s['dataset'],role=s['role'],phase_type=kind,selection=q,
                          source_R_sha256=s['R_sha256'],human_reviewed=False,verified_phase='UNKNOWN',semantic_completeness='UNKNOWN',can_route=False))
    return a,parent,cases

def enrich_cases(assets,cases):
    from .mask_audit import audit_mask
    for c in cases:
        if c['selection']['status']!='MATCHED':continue
        a,b=c['selection']['A4']
        s=assets.sources[c['source_uid']];c['source_only_edit_audit']=audit_mask(np.arange(a,b),np.zeros(b-a,bool),
            [p['evidence_span'] for p in s['phase_evidence']],s['protected_spans'],sorted(set(p['frame'] for p in s['phase_evidence'])))
        c['crop_removal_scope']='REMOVED_SOURCE_FRAMES_NOT_RECONSTRUCTION_EDIT';c['M_crop_removed']=c['selection']['removed_interval'];c['M_edit']=[]
    return cases

def main(out):
    out=fresh(out);assets,parent,cases=population();cases=enrich_cases(assets,cases);arrays=dict(J=assets.J,parents=assets.parents)
    for c in cases:
        if c['selection']['status']!='MATCHED':continue
        a,b=c['selection']['A4'];arrays['case%d_R'%c['case']]=assets.R[c['source_uid']][a:b]
        arrays['case%d_removed'%c['case']]=np.arange(a,b)>=c['selection']['A5'][1]
    save_arrays(out/'arrays.npz',arrays)
    write(out/'result.json',dict(binding=binding(),protocol=CFG,cases=cases,matched=sum(c['selection']['status']=='MATCHED' for c in cases),
        parent_population_sha256=sha(PARENT/'population/result.json'),parent_controls_arrays_sha256=sha(PARENT/'controls/arrays.npz'),
        reference=assets.reference,roster_frozen_before_quality=True,raw_observed_support_only=True,
        phase_independent_confirmation='UNKNOWN',human_records=0,training_steps=0,can_route=False))
    from .view import crop_view
    crop_view(out,assets,cases)
    print('R61_INTERVENTION',sum(c['selection']['status']=='MATCHED' for c in cases),'/',len(cases),flush=True)

def verify_main(out):
    out=fresh(out);expected=read(ROOT/'main/result.json');assets,parent,cases=population();cases=enrich_cases(assets,cases);assert cases==expected['cases']
    with np.load(ROOT/'main/arrays.npz') as z:
        for c in cases:
            if c['selection']['status']!='MATCHED':continue
            a,b=c['selection']['A4'];assert np.array_equal(z['case%d_R'%c['case']],assets.R[c['source_uid']][a:b])
            assert np.array_equal(z['case%d_removed'%c['case']],np.arange(a,b)>=c['selection']['A5'][1])
    n=verify_parent(parent)
    write(out/'result.json',dict(population_exact=True,crop_R_and_removed_masks_exact=True,matched=expected['matched'],parent_files_unchanged=n,
                                 main_result_sha256=sha(ROOT/'main/result.json'),commit=binding()['commit'],optimizer_calls=0))
