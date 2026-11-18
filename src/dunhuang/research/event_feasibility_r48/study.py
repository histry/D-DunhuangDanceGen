"""Freeze 139 candidates and test real multi-slot clocks without recutting sources."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,multi_warp

ROOT=Path('outputs/event_feasibility_r48_01')
CONFIG=Path(__file__).with_name('config.json')

def windows(length, size=256, stride=128):
    if length<2 or not 0<stride<=size:raise ValueError('Real positive source support')
    starts=list(range(0,max(length-size,0)+1,stride))
    if starts[-1]+size<length:starts.append(max(0,length-size))
    return [[a,min(a+size,length)] for a in sorted(set(starts))]

def compile_clock(unit, interval, beats, cfg):
    mapped=multi_warp(unit['source_interval'],interval,unit['phases'],beats)
    if mapped is None:return None
    warp=np.asarray(mapped['time_warp']);ticks=np.arange(*interval)
    source=np.interp(ticks,warp[:,1],warp[:,0]);a,b=unit['source_interval']
    assert source.min()>=a and source.max()<=b-1
    blocks=[]
    for lo,hi in windows(len(ticks),cfg['window_frames'],cfg['window_stride']):
        blocks.append(dict(target_interval=[int(ticks[lo]),int(ticks[hi-1])+1],source_positions=source[lo:hi].tolist(),valid_frames=hi-lo,padded_observations=0,phase_frames=[p['frame'] for p in unit['phases'] if source[lo]<=p['frame']<=source[hi-1]]))
    return dict(**mapped,windows=blocks,source_memory_interval=[a,b],whole_event_preserved=True)

def run(out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh R48 source contract')
    cfg=read(CONFIG);path=Path('outputs/event_coverage_r47_01/index/result.json');old=read(path);rows=[];slots=[]
    for song in range(3):
        root=Path('outputs/complete_r36_01/planning');music=read(root/('song%d_music.json'%song));plan=read(root/('song%d_plan.json'%song));events=plan['events']
        for i in range(len(events)):
            for count in (1,2,3):
                if i+count<=len(events):slots.append(dict(song=song,slot_indices=list(range(i,i+count)),interval=[events[i]['music_interval'][0],events[i+count-1]['music_interval'][1]],beats=music['beats'],music_sha256=sha(root/('song%d_music.json'%song))))
    for u in old['units']:
        a,b=u['source_interval'];lo,hi=u['support_component'];assert lo<=a<b<=hi
        variants=[]
        for s in slots:
            if not .75<=(s['interval'][1]-s['interval'][0]-1)/(b-a-1)<=1.33:continue
            clock=compile_clock(u,s['interval'],s['beats'],cfg)
            if clock:variants.append(dict(song=s['song'],slots=s['slot_indices'],target_interval=s['interval'],music_sha256=s['music_sha256'],clock=clock))
        rows.append(dict(event_uid=u['unit_uid'],source_uid=u['source_uid'],source_sha256=u['source_sha256'],native_sha256=u['native_sha256'],source_interval=[a,b],source_validity=dict(accepted_component=[lo,hi],unknown_frames_used=0),phase_evidence=u['phases'],context_left=u['left_context'],context_right=u['right_context'],legal_duration_range_frames=[int(np.ceil(.75*(b-a-1)))+1,int(np.floor(1.33*(b-a-1)))+1],clock_variants=variants,bridge_admission='PAIR_SPECIFIC_NOT_INFERRED_FROM_DURATION',root_observation_scope='UNKNOWN_WORLD_ROOT',contact_observation_scope='UNKNOWN',semantic_truth='UNKNOWN',stage_evidence_truncated=u['stage_evidence_truncated'],can_route=False))
    assert len(rows)==139
    write(out/'result.json',dict(schema='EventCandidateV2_R48_RESEARCH',parent_index_sha256=sha(path),config=cfg,events=rows,actual_music_slots=len(slots),source_frames_covered=sum(u['duration_frames'] for u in old['units']),unknown_source_frames_excluded=sum(s['unknown_quality_frames'] for s in old['sources']),legacy499_replaced=False,can_route=False))
    print('R48_EVENT_CONTRACT',len(rows),sum(bool(u['clock_variants']) for u in rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
