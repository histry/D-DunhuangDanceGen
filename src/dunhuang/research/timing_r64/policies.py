"""Frozen local objectives on one shared real geometry graph."""
import math
import numpy as np
from .common import CFG

POLICIES=('Fixed48','Fixed24','LocalShortest','BalancedEqual','LocalBalanced','BeatOnly','LocalBeat','ShuffledBeat')

def original_pass(record):
    m=record.get('methods',{}).get('G1',{})
    return bool(m.get('constructible') and m.get('passes_original_screen'))

def balanced_cost(record,reference):
    if not original_pass(record):return None
    numerator=denominator=0.
    for side in ('left','right'):
        w=np.asarray(record[side]['time_warp'],float)
        source=np.diff(w[:,0]);slope=np.diff(w[:,1])/source
        numerator+=float(np.sum(source*(np.log(slope)/math.log(1.33))**2))
        denominator+=float(source.sum())
    core=numerator/denominator
    m=record['methods']['G1']['metrics']
    if min(reference['D2'],reference['D3'])<=0:raise ValueError('Positive frozen geometry reference required')
    bridge=.5*(m['gap_D2']/reference['D2']+m['gap_D3']/reference['D3'])
    weights=CFG['balanced_cost']
    return dict(core_map_cost=core,bridge_cost=bridge,
        balanced_cost=weights['core_map_weight']*core+weights['bridge_weight']*bridge,
        scope='PLANNING_SURROGATES_NOT_INDEPENDENT_QUALITY')

def guard(record,baseline):
    if not original_pass(record):return False
    cfg=CFG['retention_guard']
    for side in ('left','right'):
        r=record[side].get('retention_vs_fixed48') or {}
        for name,threshold in (('FK_path_length_mean_m_ratio',cfg['core_FK_path_min_ratio']),
                               ('FK_motion_range_rms_m_ratio',cfg['core_FK_range_min_ratio'])):
            if r.get(name) is None or r[name]<threshold:return False
    return bool(record['methods']['G1']['full_dynamics']['D3_Q95'] <=
                baseline['methods']['G1']['full_dynamics']['D3_Q95']*cfg['whole_D3_Q95_max_ratio'])

def objective(record,name):
    cost=record.get('balanced_cost')
    if cost is None:return None
    value=cost['balanced_cost']
    if name in ('BeatOnly','LocalBeat','ShuffledBeat'):
        music=record['music_reward_shuffled' if name=='ShuffledBeat' else 'music_reward']
        primary=music.get('phase_beat_score') if name=='BeatOnly' else music.get('primary_music_score')
        if primary is None:return None
        value-=CFG['music_reward']['phase_beat_accent_weight']*primary
        section=music.get('section_proxy_score')
        if name!='BeatOnly' and section is not None:value-=CFG['music_reward']['section_proxy_weight']*section
    return float(value)

def choose(records,name):
    if name not in POLICIES:raise ValueError('Unregistered R64 policy')
    if name in ('Fixed48','Fixed24'):
        rows=[r for r in records if r['gap_frames']==(48 if name=='Fixed48' else 24) and r['timing_offset_frames']==0]
        if len(rows)>1:raise ValueError('Duplicate fixed allocation')
        return rows[0] if rows else None
    rows=[r for r in records if original_pass(r)]
    if name in ('LocalShortest','BalancedEqual'):
        rows=[r for r in rows if r['timing_offset_frames']==0]
    if name=='LocalShortest':
        return min(rows,key=lambda r:r['gap_frames']) if rows else None
    rows=[r for r in rows if objective(r,name) is not None]
    return min(rows,key=lambda r:(objective(r,name),abs(r['timing_offset_frames']),
        abs(r['gap_frames']-48),r['gap_frames'],r['timing_offset_frames'],r['execution_cache_key'])) if rows else None

def selection(case,records,name):
    row=choose(records,name)
    examined=records
    if name in ('Fixed48','Fixed24'):
        examined=[r for r in records if r['timing_offset_frames']==0 and r['gap_frames']==(48 if name=='Fixed48' else 24)]
    elif name in ('LocalShortest','BalancedEqual'):
        examined=sorted([r for r in records if r['timing_offset_frames']==0],key=lambda r:r['gap_frames'])
        if name=='LocalShortest' and row is not None:
            examined=[r for r in examined if r['gap_frames']<=row['gap_frames']]
    return dict(case_key=case['key'],role=case['role'],policy=name,
        record_key=row['key'] if row else None,method='G1',
        gap_frames=row['gap_frames'] if row else None,
        timing_offset_frames=row['timing_offset_frames'] if row else None,
        accepted=bool(row and original_pass(row)),constructible=bool(row and row['methods']['G1'].get('constructible')),
        retention_guard_passed=bool(row and row['retention_guard_passed']),
        objective=objective(row,name) if row else None,
        candidate_population=len(records),original_pass_candidates=sum(original_pass(r) for r in records),
        rejection_reason=None if row else 'NO_ORIGINAL_PASS_OR_MUSIC_PROXY_UNAVAILABLE',
        guard_used_for_selection=False,fallback_used=False,
        trials=[dict(gap_frames=r['gap_frames'],offset=r['timing_offset_frames']) for r in examined],
        comparison_scope='EXPOSED_LOCAL_DIAGNOSTIC_NOT_ARTISTIC_OR_NATIVE_QUALIFICATION')
