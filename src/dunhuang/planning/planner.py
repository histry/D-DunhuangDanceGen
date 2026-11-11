import numpy as np
from dunhuang.io import checked
from .event_plan import validate
from scipy.spatial.transform import Rotation

def timeline(audio,frames):
    import soundfile as sf
    import librosa
    x,sr=sf.read(audio,always_2d=True);x=x.mean(1)
    _,beat_times=librosa.beat.beat_track(y=x.astype(np.float32),sr=sr,units='time')
    beats=np.unique(np.rint(np.asarray(beat_times)*30).astype(int))
    n=max(1,round(frames/90))
    edges=[0]
    for point in np.linspace(0,frames,n+1)[1:-1]:
        options=beats[(beats>=point-10)&(beats<=point+10)&(beats>=edges[-1]+50)]
        edges.append(int(min(options,key=lambda b:abs(b-point))) if len(options) else int(round(point)))
    edges.append(frames)
    energy=np.array([np.sqrt(np.mean(x[int(i*sr/30):max(int((i+1)*sr/30),int(i*sr/30)+1)]**2)) for i in range(frames)])
    energy/=max(float(energy.max()),1e-8)
    return dict(fps=30,frames=frames,beats=beats[beats<frames].tolist(),phrase_method='approximately 3s subdivisions snapped to detected beats',
        slots=[dict(start=a,end=b,density=float(energy[a:b].mean())) for a,b in zip(edges[:-1],edges[1:])])

def plan(catalog,timing,audio_sha256,policy='heuristic',top_k=5,beam_width=3,reranker=None,condition_mode='temporal'):
    if policy not in ('heuristic','learned_shadow','learned_qualified'):raise ValueError('Unknown retrieval policy')
    if policy=='learned_qualified' and (reranker is None or not reranker.decision['enabled']):
        raise ValueError('Learned retrieval lacks bound validation qualification')
    trace=[]
    if policy=='learned_shadow' and reranker is None:raise ValueError('Shadow mode requires explicitly bound models and query features')
    beams=[(0.,[])]
    for slot in timing['slots']:
        options=[]
        for entry in catalog['entries']:
            qualification='soft_reference' if condition_mode=='adapter' else 'temporal8'
            if entry['feasibility'].get(qualification)!='PASS':continue
            start,end=slot['start'],slot['end'];a,b=entry['source_interval'];center=entry['center']
            ratio=(end-start)/(b-a)
            if not .75<=ratio<=1.33:continue
            tick=start+round((center-a)*ratio)
            beats=[t for t in timing['beats'] if abs(t-tick)<=4]
            if beats:tick=min(beats,key=lambda t:abs(t-tick))
            if tick-12<start or tick+12>=end:continue
            warp=[[a,start],[center-12,tick-12],[center+12,tick+12],[b,end]]
            rates=np.diff(np.array(warp)[:,1])/np.diff(np.array(warp)[:,0])
            if np.any(rates<.75) or np.any(rates>1.33):continue
            event=dict(entry,music_interval=[start,end],time_warp=warp,
                primitive=dict(source_frame=center,music_frame=int(tick),joints=list(range(12,20)),radius=12),
                boundary_context=dict(left_source_frames=[a,min(a+4,b)],right_source_frames=[max(a,b-4),b],confidence='SOURCE_ONLY'),
                scores=dict(heuristic=-abs(entry['activity_percentile']-slot['density']),global_score=None,sync_score=None,preference=None))
            if reranker is not None and policy!='heuristic':
                event['scores'].update(reranker.scores(entry,slot,local=False))
            options.append(event)
        if not options:raise ValueError('No legal event duration/phase for slot '+str(slot))
        # Keep source diversity in the short candidate list before global search.
        recalled=None
        if reranker is not None and policy!='heuristic':
            recalled=sorted(options,key=lambda e:(-e['scores']['global_score'],e['event_uid']))[:top_k]
            for event in recalled:event['scores'].update(reranker.scores(event,slot,local=True))
            if policy=='learned_qualified':options=recalled
        def utility(e):
            v=e['scores']
            return v['heuristic'] if policy!='learned_qualified' else v['global_score']+.25*v['sync_score']
        ordered=sorted(options,key=lambda e:(-utility(e),e['event_uid']))
        trace.append(dict(music_interval=[slot['start'],slot['end']],candidates=[dict(event_uid=e['event_uid'],scores=e['scores']) for e in (recalled if recalled is not None else ordered[:top_k])]))
        chosen=[]
        for uid in dict.fromkeys(e['source_uid'] for e in ordered):
            chosen.append(next(e for e in ordered if e['source_uid']==uid))
        chosen=sorted(chosen,key=lambda e:-utility(e))[:top_k]
        expanded=[]
        for cost,sequence in beams:
            for event in chosen:
                repeat=sum(e['event_uid']==event['event_uid'] for e in sequence)
                same_source=sum(e['source_uid']==event['source_uid'] for e in sequence)
                transition=0.
                if sequence:
                    relative=np.swapaxes(np.array(sequence[-1]['exit']),-1,-2)@np.array(event['entry'])
                    transition=float(Rotation.from_matrix(relative).magnitude().mean())
                total=cost-utility(event)+.3*repeat+.04*same_source+.25*transition
                expanded.append((total,sequence+[event]))
        beams=sorted(expanded,key=lambda pair:(pair[0],[e['event_uid'] for e in pair[1]]))[:beam_width]
    cost,events=beams[0]
    result=dict(schema='event_plan_v1',fps=30,frames=timing['frames'],audio_sha256=audio_sha256,
        retrieval_policy=policy,retrieval_qualified=policy=='learned_qualified',ranking_trace=trace,time_scale_limits=[.75,1.33],events=events,
        planning_cost=cost,top_k=top_k,beam_width=beam_width,
        controlled_frames=len(set(t for e in events for t in range(e['primitive']['music_frame']-11,e['primitive']['music_frame']+12))),
        native_feasibility='UNKNOWN')
    return validate(result,audio_sha256,timing['frames'])
