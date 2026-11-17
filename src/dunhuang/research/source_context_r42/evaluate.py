"""Independent saved-output qualification; no contact or cultural truth is inferred."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.transition_r37.evaluate import decode,world_fk
from dunhuang.research.transition_r37.geometry import c2_envelope
from .post import frozen_rts


def run(data,generation,out,label,official=False):
    data=Path(data);generation=Path(generation);out=Path(out)
    if out.exists():raise ValueError('Fresh independent qualification')
    z=np.load(generation/'arrays.npz');target=np.load(data/'arrays.npz');meta=read(generation/'result.json');J=z['skeleton_J22'];parents=z['parents'];w=c2_envelope(256,[104,152]);records=[];rt=None
    if official:
        import torch
        from dunhuang.research.bridge_r38.study import runtime
        from dunhuang.research.coverage_r34.common import forbid_optimizers
        rt=runtime(read('src/dunhuang/research/bridge_r38/config.json'));forbid_optimizers()
    for row in meta['records']:
        key=row['key'];ek=key.split('__')[0];truth=target[ek+'_truth'];R,root=decode(truth);Q=relative_fk(R,J,parents);source_activity=float(np.sqrt(np.mean((np.diff(Q[104:152],axis=0)*30)**2)));metrics={};anchor=z[key+'_Geometry_139']
        for name in row['methods']+['FrozenRTS']:
            values=[]
            candidate=frozen_rts(z[key+'_Frozen_139'],anchor) if name=='FrozenRTS' else z[key+'_'+name+'_139']
            for i,x in enumerate(candidate):
                assert np.isfinite(x).all();P,pr=decode(x);A,ar=decode(anchor[i]);angle=Rotation.from_matrix((A.swapaxes(-1,-2)@P).reshape(-1,3,3)).magnitude().reshape(256,22)
                assert (angle<=.6*w[:,None]+3e-6).all(),(key,name,'body_budget')
                assert (np.linalg.norm(pr-ar,axis=-1)<=.5*w+3e-6).all(),(key,name,'root_budget')
                assert np.array_equal(x[w==0],anchor[i,w==0]);world=world_fk(P,pr,J,parents)
                if name!='FrozenRTS':assert abs(world-z[key+'_'+name+'_22'][i]).max()<2e-6
                if official:
                    with torch.no_grad():actual=rt.f(torch.tensor(x[None],device='cuda'),rt.s)[0,:,:22].cpu().numpy()
                    assert abs(actual-world).max()<2e-6
                points=relative_fk(P,J,parents);e=points[101:155]-Q[101:155];er=(pr-root)[101:155];s={f'body_D{k}':float(np.sqrt(np.mean((np.diff(e,n=k,axis=0)*30**k)**2))) for k in range(4)}
                s.update({f'root_D{k}':float(np.sqrt(np.mean((np.diff(er,n=k,axis=0)*30**k)**2))) if k or row['root_position_observed'] else None for k in range(3)})
                s['activity_ratio_to_truth']=float(np.sqrt(np.mean((np.diff(points[104:152],axis=0)*30)**2))/max(source_activity,1e-8));s['world_foot_speed_p95_mps']=float(np.quantile(np.linalg.norm(np.diff(world[:,[10,11]],axis=0)*30,axis=-1),.95));s['protected_exact']=True;values.append(s)
            metrics[name]=values
        records.append(dict(key=key,role=row['role'],domain=row['domain'],recording_uid=row['recording_uid'],source_interval=row['source_interval'],root_position_observed=row['root_position_observed'],target_music=row['target_music'],methods=metrics))
    if rt is not None:rt.check();rt.close()
    held=[r for r in records if r['role']=='held' and r['domain']=='ChangE' and r['target_music'] is None];groups={}
    for uid in sorted({r['recording_uid'] for r in held}):
        part=[r for r in held if r['recording_uid']==uid];groups[uid]={}
        for name in part[0]['methods']:
            groups[uid][name]={k:float(np.mean([s[k] for r in part for s in r['methods'][name] if s[k] is not None])) if any(s[k] is not None for r in part for s in r['methods'][name]) else None for k in part[0]['methods'][name][0] if k!='protected_exact'}
    gate={}
    for uid,v in groups.items():
        a,b=v[label+'RTS'],v['FrozenRTS'];gate[uid]=dict(body_D0=a['body_D0']<=1.05*b['body_D0'],body_D3=a['body_D3']<=b['body_D3'],activity=.85<=a['activity_ratio_to_truth']<=1.15)
    result=dict(label=label,records=records,held_change_recording_means=groups,registered_body_gate=gate,body_gate_pass_recordings=sum(all(v.values()) for v in gate.values()),body_gate_total_recordings=len(gate),official_fk_executed=official,data_sha256=sha(data/'arrays.npz'),generation_sha256=sha(generation/'arrays.npz'),whole_song_admitted=False,whole_song_generations=0,native24='NOT_EVALUABLE_22_OF24',contact_truth='UNKNOWN_FOOT_SPEED_IS_NOT_SLIDING_TRUTH',operational_qualified=False);write(out,result);print('R42_QUALIFY',label,result['body_gate_pass_recordings'],len(gate),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--generation',required=True);p.add_argument('--out',required=True);p.add_argument('--label',required=True);p.add_argument('--official',action='store_true');a=p.parse_args();run(a.data,a.generation,a.out,a.label,a.official)
