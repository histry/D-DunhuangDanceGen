import numpy as np
from dunhuang.io import checked

def validate(plan, audio_sha256, frames):
    if plan['schema']!='event_plan_v1' or plan['fps']!=30 or plan['frames']!=frames or plan['audio_sha256']!=audio_sha256:
        raise ValueError('Event plan clock or audio differs')
    if plan['time_scale_limits']!=[.75,1.33]:raise ValueError('Registered time-scale limits differ')
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
        data=np.load(checked(event['reference']),allow_pickle=False,mmap_mode='r')
        if data.ndim!=2 or data.shape[1]!=151:raise ValueError('Invalid source motion shape')
        lo,hi=event['source_interval']
        if type(lo) is not int or type(hi) is not int or not 0<=lo<hi<=len(data):
            raise ValueError('Invalid source interval')
        if warp[0,0]!=lo or warp[-1,0]!=hi:raise ValueError('Source time map endpoints differ')
        primitive=event['primitive'];index=primitive['source_frame'];tick=primitive['music_frame']
        if type(index) is not int or type(tick) is not int or not lo<=index<hi or not a<=tick<b:
            raise ValueError('Primitive outside planned interval')
        if abs(float(np.interp(index,warp[:,0],warp[:,1]))-tick)>1e-6:
            raise ValueError('Primitive clock differs from time map')
        radius=primitive['radius']
        if radius!=12 or index-radius<lo or index+radius>=hi or tick-radius<a or tick+radius>=b:
            raise ValueError('Primitive context outside planned interval')
        joints=primitive['joints']
        if not joints or len(set(joints))!=len(joints) or any(type(j) is not int or not 12<=j<=21 for j in joints):
            raise ValueError('Unsupported primitive joint contract')
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

def candidate_identity(event,model_identity=None):
    import hashlib,json
    payload={k:event[k] for k in ('event_uid','source_interval','music_interval','time_warp','reference')}
    payload['expression']=event.get('expression',dict(amplitude=1.,mode='identity_source_reference'))
    payload['condition_schema']='soft_event_reference_v1'
    payload['score_models']=model_identity
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def semantic_plan_id(plan):
    """Execution identity excludes lineage, scores and filesystem locations."""
    import hashlib,json
    events=[]
    for event in plan['events']:
        row={k:event[k] for k in ('event_uid','source_uid','source_interval','music_interval','time_warp','primitive')}
        row['reference_sha256']=event['reference']['sha256']
        row['expression']=event.get('expression',dict(amplitude=1.,mode='identity_source_reference'))
        events.append(row)
    value=dict(schema='execution_plan_identity_v1',audio_sha256=plan['audio_sha256'],
        fps=plan['fps'],frames=plan['frames'],events=events)
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def revised_plan(plan,parent,feedback,action,changed_slots):
    import copy
    result=copy.deepcopy(plan)
    result['lineage']=dict(parent_semantic_id=semantic_plan_id(parent),feedback=feedback,
        action=action,changed_slots=sorted(changed_slots),event_globally_infeasible=False)
    result['semantic_id']=semantic_plan_id(result)
    return result
