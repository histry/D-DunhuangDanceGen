import numpy as np
from dunhuang.io import checked

def validate(plan, audio_sha256, frames):
    if plan['schema']!='event_plan_v1' or plan['fps']!=30 or plan['frames']!=frames or plan['audio_sha256']!=audio_sha256:
        raise ValueError('Event plan clock or audio differs')
    end=0
    for event in plan['events']:
        a,b=event['music_interval']
        if a!=end or not a<b<=frames:raise ValueError('Event plan must cover the exact clock without gaps')
        warp=np.asarray(event['time_warp'],float)
        if warp.ndim!=2 or warp.shape[1]!=2 or not np.isfinite(warp).all() or np.any(np.diff(warp,axis=0)<=0):
            raise ValueError('Invalid monotone phase map')
        if warp[0,1]!=a or warp[-1,1]!=b:raise ValueError('Time map endpoints differ')
        rates=np.diff(warp[:,1])/np.diff(warp[:,0])
        low,high=plan['time_scale_limits']
        if np.any(rates<low-1e-6) or np.any(rates>high+1e-6):raise ValueError('Time map speed outside registered limits')
        checked(event['reference'])
        if event['feasibility']['native']!='UNKNOWN':raise ValueError('Plan cannot pre-admit generated physical quality')
        end=b
    if end!=frames:raise ValueError('Incomplete song plan')
    return plan

def anchors(plan):
    result=[]
    for event in plan['events']:
        data=np.load(checked(event['reference']),allow_pickle=False)
        index=event['primitive']['source_frame']
        result.append(dict(frame=event['primitive']['music_frame'],source_frame=index,
            source=event['reference'],pose=data[index].tolist(),joints=event['primitive']['joints'],
            catalog_id=event['event_uid']))
    return result
