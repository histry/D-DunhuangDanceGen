"""Same-population paired event ranking, adding energy correlation to frozen Local."""
from pathlib import Path
import copy
import numpy as np
from dunhuang.io import checked
from dunhuang.research.complete_r36.core import read,write,relative_fk
from dunhuang.research.event_alignment_r49.common import Frozen,audio_signals,multiscale,interpolate,RADIUS
from dunhuang.research.event_alignment_r49.evaluate import summaries
from .common import fresh


def run(out):
    out=fresh(out);net=Frozen();old=read('outputs/event_alignment_r49_val/result.json');source={s['source_uid']:s for s in net.manifest['sources'] if s['role']=='val'};train=[s for s in net.manifest['sources'] if s['role']=='train']
    for field in ('source_group','audio_sha256','raw_motion_sha256','recording_uid'):
        assert not ({s[field] for s in source.values()} & {s[field] for s in train})
    units={r['event_uid']:r for r in net.manifest['rows'] if r['role']=='val'};activity={};audio={}
    for s in source.values():
        with np.load(checked(s['source_arrays'])) as d:R=d['R']
        P=relative_fk(R,net.joints,net.parents);activity[s['source_uid']]=np.linalg.norm(np.diff(P,axis=0,prepend=P[:1])*30,axis=-1).mean(-1)
        if s['audio_sha256'] not in audio:audio[s['audio_sha256']]=audio_signals(checked(s['audio']))
    result={}
    for name,tasks in old['events'].items():
        rows=[];balanced=[]
        for task in tasks:
            qrow=units[task['query']];query_source=source[qrow['source_uid']];a,b=qrow['source_interval'];start=(a+b-90)//2+query_source['audio_offset_frames'];q=np.arange(start,start+90)
            sig=audio[task['audio']];other=next(k for k in sorted(audio) if k!=task['audio']);wrong=audio[other]
            if q.min()<0 or q.max()>min(len(sig['energy']),len(wrong['energy']))-1:continue
            e=np.interp(q,np.arange(len(sig['energy'])),sig['energy']);w=np.interp(q,np.arange(len(wrong['energy'])),wrong['energy'])
            multi=[];swap=[];sources=[]
            for uid in task['candidate_events']:
                r=units[uid];lo,hi=r['source_interval'];s=(lo+hi-90)//2;v=activity[r['source_uid']][s:s+90];multi.append(multiscale(e,v));swap.append(multiscale(w,v));sources.append(r['raw_motion_sha256'])
            row=copy.deepcopy(task);local=row['scores']['Content'];music_swap=row['scores']['MusicSwap']
            row['scores']['Local']=local;row['scores']['Full']=(np.array(local)+.1*np.array(multi)).tolist();row['scores']['FullMusicSwap']=(np.array(music_swap)+.1*np.array(swap)).tolist();row['candidate_recordings']=sources;row['warp_cost']=[0.]*len(sources)
            assert sources[0] not in sources[1:]
            rows.append(row)
            ids=[];seen=set()
            for i,recording in enumerate(sources):
                if recording not in seen:ids.append(i);seen.add(recording)
            if len(ids)>=2:
                item=copy.deepcopy(row);item['scores']={k:[v[i] for i in ids] for k,v in row['scores'].items()};item['candidate_events']=[row['candidate_events'][i] for i in ids];item['candidate_recordings']=[sources[i] for i in ids];balanced.append(item)
        result[name]=dict(tasks=rows,summary=summaries(rows,actual='Full'),source_balanced_tasks=balanced,source_balanced_summary=summaries(balanced,actual='Full'))
        print('R51_PAIRED',name,len(rows),flush=True)
    write(out/'result.json',dict(tasks=result,model=net.binding(),role='HISTORICALLY_EXPOSED_VAL',independent_new_music_groups=0,identity_warp='Equal 90-frame physical crops; warp penalty identically zero',training_admitted=False,can_route=False))
