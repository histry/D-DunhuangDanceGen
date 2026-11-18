"""R46 observed-only bridge reconstruction and measured source-clock capacity."""
import argparse,subprocess,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import sixd
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.reperformance_r44.archive import Arrays
from dunhuang.research.reperformance_r44.motion import decode,world,kinematics
from dunhuang.research.bridge_r38.geometry import observed_bridge,polynomial

def angular(A,B):return Rotation.from_matrix((A.swapaxes(-1,-2)@B).reshape(-1,3,3)).magnitude().reshape(A.shape[:-2])

def reconstruct(raw,reference,gap):
    hidden=np.asarray(reference).copy();hidden[slice(*gap)]=np.nan
    R=observed_bridge(hidden,gap);out=raw.copy();out[:,7:]=sixd(R,row=True).reshape(len(raw),132)
    visible=np.ones(len(raw),bool);visible[slice(*gap)]=False;out[visible,7:]=raw[visible,7:]
    return out,R

def capacity(maps):
    rows=[]
    for m in maps:
        bank=m['bank'];allowed=[p for p in bank if p['eligible']];non=[p for p in allowed if not p['identity']];selected=next(p for p in bank if p['pattern_id']==m['pattern_id']);best=max(allowed,key=lambda p:(p['score'],-p['pattern_id']))
        assert best['pattern_id']==selected['pattern_id']
        reason='NONIDENTITY_SELECTED' if not selected['identity'] else 'NO_DETECTED_STAGE' if not m['phases'] else 'NO_ELIGIBLE_NONIDENTITY' if not non else 'FIXED_MUSIC_SCORE_PREFERS_IDENTITY'
        rows.append(dict(interval=m['interval'],stages=len(m['phases']),detector_status=m['evidence']['status'],maximum_source_speed=m['evidence'].get('maximum_speed'),unique_maps=len(bank),eligible_maps=len(allowed),nonidentity_eligible=len(non),selected_identity=selected['identity'],reason=reason,activity_range=[min(p['ratios']['D1'] for p in allowed),max(p['ratios']['D1'] for p in allowed)],maximum_displacement_frames=max(max(abs(np.array(p['target'])-p['source'])) for p in allowed),scope='MEASURED_KINEMATIC_CLOCK_CAPACITY_NOT_CULTURAL_EXPRESSIVITY'))
    return rows

def run(out):
    out=Path(out);cfg=read(Path(__file__).with_name('config.json'));parent=Path(cfg['parent']);meta=read(parent/'main/result.json');z=Arrays(parent/'main/arrays.npz')
    if out.exists():raise ValueError('Fresh R46 audit')
    out.mkdir(parents=True);J=z['skeleton_J22'];parents=z['parents'];arr=dict(skeleton_J22=J,parents=parents);rows=[];gap=cfg['gap'];visible=np.ones(256,bool);visible[slice(*gap)]=False
    for old in meta['records']:
        key=old['key'];raw=z[key+'_Geometry_139'];ref=z[key+'_reference_R'];G=decode(raw[0])[0];truth=relative_fk(ref,J,parents);P=relative_fk(G,J,parents);rebuilt,R=reconstruct(raw[0],ref,gap)
        # Invariance test: deliberately poison all unavailable hidden values.
        poisoned=ref.copy();poisoned[slice(*gap)]=Rotation.from_rotvec(np.ones((48*22,3))*1.7).as_matrix().reshape(48,22,3,3);again,_=reconstruct(raw[0],poisoned,gap);assert np.array_equal(rebuilt,again)
        methods={'Geometry':raw.copy(),'ObservedRebuild':np.stack([reconstruct(v,ref,gap)[0] for v in raw]),'Phase':z[key+'_Phase_139']};continuous=old['source']['domain']!='ChangE-cross'
        delta=angular(ref,G);cross=not continuous;ids=[101,102,103,152,153,154];anchor=ref[103];rv=Rotation.from_matrix((anchor.swapaxes(-1,-2)[None]@ref[ids]).reshape(-1,3,3)).as_rotvec().reshape(6,22,3);chart=np.full((256,22,3),np.nan);chart[ids]=rv;path=polynomial(chart,gap)
        shape=lambda X:dict(shoulder_wrist_m=np.linalg.norm(X[:,[20,21]]-X[:,[16,17]],axis=-1).tolist(),wrist_separation_m=np.linalg.norm(X[:,20]-X[:,21],axis=-1).tolist(),ankle_separation_m=np.linalg.norm(X[:,7]-X[:,8],axis=-1).tolist(),torso_extension_m=np.linalg.norm(X[:,12]-X[:,0],axis=-1).tolist())
        diagnostics=dict(hidden_used_in_constructor=False,hidden_poison_invariance=True,source_body_max_visible_error_rad=float(delta[visible].max()),rebuild_vs_geometry_max_angle_rad=float(angular(R,G).max()),gap_rebuild_vs_geometry_rmse_m=float(np.sqrt(np.mean((relative_fk(R,J,parents)[104:152]-P[104:152])**2))),gap_target_kind='CONTINUOUS_RECORDING_HIDDEN_TRUTH' if continuous else 'CROSS_SOURCE_NO_UNIQUE_BRIDGE_TRUTH',gap_so3_error_mean_rad=float(delta[104:152].mean()) if continuous else None,frame117_so3_mean_rad=float(delta[117].mean()) if continuous else None,frame117_fk_rmse_m=float(np.sqrt(np.mean((P[117]-truth[117])**2))) if continuous else None,endpoint_joint_angle_rad=angular(ref[103],ref[152]).tolist(),endpoint_log_velocity_per_frame=np.stack([(3*rv[2]-4*rv[1]+rv[0])/2,(-3*rv[3]+4*rv[4]-rv[5])/2]).tolist(),endpoint_log_acceleration_per_frame2=np.stack([rv[2]-2*rv[1]+rv[0],rv[5]-2*rv[4]+rv[3]]).tolist(),max_quintic_chart_rad=float(np.linalg.norm(path[104:152],axis=-1).max()),frame117_chart_rad=np.linalg.norm(path[117],axis=-1).tolist(),source_shape=shape(truth),geometry_shape=shape(P),bone_length_max_error_m=float(max(abs(np.linalg.norm(P[:,j]-P[:,int(parents[j])],axis=-1)-np.linalg.norm(J[j]-J[int(parents[j])])).max() for j in range(1,22))),source_root_truth='UNKNOWN_FOR_CHANGE; REFERENCE_DISPLAY_ROOT_RELATIVE',model_root_bounds_m=[raw[:,:,4:7].min(axis=(0,1)).tolist(),raw[:,:,4:7].max(axis=(0,1)).tolist()],capacity=capacity(old['phase_maps']),diagnostic_frames=cfg['diagnostic_frames'],frames={str(t):dict(geometry_local_xyz=P[t].tolist(),reference_local_xyz=truth[t].tolist(),body_angle_rad=delta[t].tolist() if continuous else None) for t in cfg['diagnostic_frames']})
        source={k:v for k,v in old['source'].items() if k in ('recording_uid','domain','role','start','raw_sha256','source_interval','evaluation_key','binding','music_identity','historical_exposure')};source['audit']=diagnostics
        metrics={};curves={}
        for method,values in methods.items():
            xyz=np.stack([world(v,J,parents) for v in values]);arr[key+'_'+method+'_139']=values;arr[key+'_'+method+'_22']=xyz;arr[key+'_'+method+'_clock']=np.arange(256,dtype=float);metrics[method]=kinematics(values[0],J,parents);curves[method]=np.r_[np.zeros(3),np.sqrt(np.mean(np.diff(xyz,n=3,axis=1)**2,axis=(0,2,3)))*27000].tolist()
        arr[key+'_reference_R']=ref;shutil.copyfile(parent/'main'/old['audio'],out/old['audio']);rows.append(dict(key=key,source=source,seeds=old['seeds'],audio=old['audio'],methods=list(methods),metrics=dict(methods=metrics,curves=curves),phase_maps=old['phase_maps'],music=old['music'],scope='R46 Bridge responsibility audit. Geometry/Phase frozen; ObservedRebuild reads only visible SO3 endpoints. Source gap is evaluation only for continuous recordings; cross-source gap has no unique truth. Same camera/scale, no neural sampling.'))
        print('R46_BRIDGE',key,'rebuild_angle',diagnostics['rebuild_vs_geometry_max_angle_rad'],'frame117_error',diagnostics['frame117_so3_mean_rad'],'capacity',[c['reason'] for c in diagnostics['capacity']],flush=True)
    np.savez_compressed(out/'arrays.npz',**arr);write(out/'result.json',dict(records=rows,config=dict(cfg,registered_video_sources=['female_FeiTian','female_meditation','male_pipa_2']),parent_arrays_sha256=sha(parent/'main/arrays.npz'),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),training_steps=0,new_neural_samples=0,operational_qualified=False));z.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
