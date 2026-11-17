"""Immutable subwindow indices. Parent events, physical sources and roles stay intact."""
from pathlib import Path
from collections import defaultdict
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,identity
from dunhuang.research.paired_event_r41.events import validate_roles


def relation(a,b,cfg):
    if a['role']!=b['role'] or a['raw_motion_sha256']==b['raw_motion_sha256']:return 'UNKNOWN'
    A=a['audio_frame_interval'];B=b['audio_frame_interval'];inter=max(0,min(A[1],B[1])-max(A[0],B[0]));union=max(A[1],B[1])-min(A[0],B[0]);same=a['audio_sha256']==b['audio_sha256']
    if same and inter/max(union,1e-9)>=cfg['positive_iou']:return 'CLOCK_POSITIVE'
    if not same or max(A[0],B[0])-min(A[1],B[1])>=cfg['negative_gap_frames']:return 'CLOCK_NEGATIVE'
    return 'UNKNOWN'


def build(out):
    from dunhuang.research.paired_event_r41.population import Population
    out=Path(out)
    if out.exists():raise ValueError('Fresh benchmark population')
    out.mkdir(parents=True);cfg=read(Path(__file__).with_name('config.json'));pop=Population(cfg['data']);rows=[]
    for i,parent in enumerate(pop.rows):
        z=pop.unit(i);a,b=parent['core_slice']
        for length in cfg['lengths']:
            if b-a<length:continue
            starts=sorted(set(list(range(a,b-length+1,max(1,length//2)))+[b-length]))
            for start in starts:
                source_start=parent['context_interval'][0]+start;audio_start=parent['context_audio_frame_interval'][0]+start
                rows.append(dict(uid=identity(dict(parent=parent['event_uid'],start=source_start,length=length,version='R43_RETRIEVAL_SUBWINDOW_V1')),parent_index=i,parent_event_uid=parent['event_uid'],source_uid=parent['source_uid'],recording_uid=parent['recording_uid'],raw_motion_sha256=parent['raw_motion_sha256'],audio_sha256=parent['audio_sha256'],source_group=parent['source_group'],choreography_uid=parent['choreography_uid'],role=parent['role'],frames=length,slice=[start,start+length],source_interval=[source_start,source_start+length],audio_frame_interval=[audio_start,audio_start+length],music_scalar=float(z['music'][start:start+length,0].mean()),motion_scalar=float(z['motion'][start:start+length,264].mean()),label_scope='PHYSICAL_CLOCK_SUBWINDOW_NOT_SEMANTIC_EVENT',semantic_truth='UNKNOWN'))
    validate_roles(rows);train=np.array([[r['music_scalar'],r['motion_scalar']] for r in rows if r['role']=='train']);mu=train.mean(0);sd=train.std(0).clip(.0001);queries=[];missing=[]
    for role in ('train','val'):
        groups=sorted({r['source_group'] for r in rows if r['role']==role})[:cfg['maximum_'+role+'_music_groups']]
        for group in groups:
            sources=sorted({r['raw_motion_sha256'] for r in rows if r['role']==role and r['source_group']==group})[:cfg['sources_per_music_group']]
            for source in sources:
                for length in cfg['lengths']:
                    options=sorted([i for i,r in enumerate(rows) if r['raw_motion_sha256']==source and r['frames']==length],key=lambda i:(rows[i]['source_interval'],rows[i]['uid']))
                    if options:queries.extend(options[int(k)] for k in np.unique(np.linspace(0,len(options)-1,min(len(options),cfg['queries_per_source_length'])).round().astype(int)))
                    else:missing.append(dict(role=role,group=group,raw_motion_sha256=source,frames=length,reason='NO_COMPLETE_REAL_SUBWINDOW'))
    tasks=[];unavailable=[]
    for q in queries:
        row=rows[q]
        if len({r['audio_sha256'] for r in rows if r['role']==row['role'] and r['frames']==row['frames']})<2:
            unavailable.append(dict(query=q,pool='ALL',reason='NO_REAL_COUNTERFACTUAL_AUDIO_DONOR'));continue
        candidates=[i for i,r in enumerate(rows) if r['role']==row['role'] and r['frames']==row['frames'] and r['raw_motion_sha256']!=row['raw_motion_sha256']];positive=[i for i in candidates if relation(row,rows[i],cfg)=='CLOCK_POSITIVE'];negative=[i for i in candidates if relation(row,rows[i],cfg)=='CLOCK_NEGATIVE'];positive=sorted(positive,key=lambda i:rows[i]['uid'])[:cfg['maximum_positives']]
        # Only the query music and candidate motion scalar enter deployable hard selection.
        distance=lambda i:abs((row['music_scalar']-mu[0])/sd[0]-(rows[i]['motion_scalar']-mu[1])/sd[1])
        for pool in ('same_music','energy_matched','cross_music'):
            neg=[i for i in negative if (rows[i]['audio_sha256']==row['audio_sha256'] if pool=='same_music' else rows[i]['audio_sha256']!=row['audio_sha256'])]
            if pool=='energy_matched':neg=[i for i in neg if distance(i)<=1.]
            neg=sorted(neg,key=lambda i:(distance(i) if pool!='same_music' else rows[i]['uid'],rows[i]['uid']))[:cfg['maximum_negatives']]
            if not positive or not neg:unavailable.append(dict(query=q,pool=pool,positive_count=len(positive),negative_count=len(neg)));continue
            ids=sorted(positive+neg,key=lambda i:rows[i]['uid']);tasks.append(dict(uid=identity(dict(query=row['uid'],pool=pool)),query=q,candidates=ids,positive=[i in positive for i in ids],role=row['role'],source_group=row['source_group'],pool=pool,frames=row['frames'],scope='PHYSICAL_AUDIO_CLOCK_RANKING_NOT_ARTISTIC_COMPATIBILITY'))
    value=dict(config=cfg,rows=rows,queries=queries,missing_query_support=missing,tasks=tasks,unavailable=unavailable,source_manifest_sha256=sha(Path(cfg['data'])/'manifest.json'),normalizer=pop.manifest['normalizer'],energy_calibration=dict(mean=mu.tolist(),std=sd.tolist(),scope='TRAIN_SUBWINDOW_SCALARS_ONLY'),target_motion_activity_used_in_candidate_filter=False,same_recording_candidates_excluded=True,parent_events_unchanged=True,test_tensor_reads=0,optimizer_steps=0,can_route=False,operational_qualified=False);write(out/'manifest.json',value);print('R43_BENCHMARK',len(rows),len(queries),len(tasks),len(unavailable),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();build(a.out)
