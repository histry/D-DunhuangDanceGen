"""Frozen matched experiment and read-only diagnostics on existing samples."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dunhuang.conditioning import source_neighborhoods
from dunhuang.contracts import matrices, MotionAdapter
from dunhuang.io import checked, pinned, sha, write
from dunhuang.pipeline import run

def read(path):
    return json.loads(path.read_text())

def distribution(array):
    if not array.size:
        return None
    return dict(max=float(array.max()),p95=float(np.percentile(array,95)))

def angle_difference(a,b):
    relative = a.swapaxes(-1,-2) @ b
    return Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude().reshape(relative.shape[:-2])

def region_stats(norm, region):
    return dict(inside=distribution(norm[region]),outside=distribution(norm[~region]))

def evidence(path,q):
    motion = np.load(path/'native.npy')
    fk = np.load(path/'native_fk.npy')
    rotations = matrices(motion[:,7:].reshape(-1,24,6)).astype(float)
    delta = rotations[:-1].swapaxes(-1,-2) @ rotations[1:]
    omega = Rotation.from_matrix(delta.reshape(-1,3,3)).as_rotvec().reshape(-1,24,3)*30
    times = np.arange(len(fk))
    ticks = np.array([a['frame'] for a in q['anchors']])
    near = np.min(abs(times[:,None]-ticks),axis=1)<=3
    normalized = np.load(path/'normalized.npy')
    raw = np.load(path/'windows.npy')
    coverage = np.sum([((times>=75*i)&(times<75*i+150)) for i in range(len(raw))],axis=0)
    d1 = np.linalg.norm(np.diff(fk,axis=0),axis=-1)
    d2 = np.linalg.norm(np.diff(fk,n=2,axis=0),axis=-1)*900
    d3 = np.linalg.norm(np.diff(fk,n=3,axis=0),axis=-1)*27000
    source,_ = source_neighborhoods(len(fk),q['anchors'],12)
    errors=[]
    for target in source:
        joints=target['joints']
        truth=matrices(target['pose'][7:].reshape(24,6)).astype(float)
        errors.append(angle_difference(rotations[target['frame'],joints],truth[joints]).mean())
    audit=read(path/'physical.json')
    for name, values in [('joint_velocity_mps',d1*30),('joint_acceleration_mps2',d2),('joint_jerk_mps3',d3)]:
        for stat,measurement in distribution(values).items():
            if not np.isclose(measurement,audit['audit'][name+'_'+stat],rtol=1e-4,atol=.002):
                raise ValueError('Saved FK derivatives disagree with Native: '+name+'_'+stat)

    from motion.physics import derive_contacts_np
    from motion.config import MotionGenerationConfig
    from motion_geometry import physical as native
    from motion_geometry.smpl24 import FOOT_JOINTS
    contacts,_,_,_=derive_contacts_np(motion,MotionGenerationConfig(),precomputed_joints=fk)
    feet=fk[:,list(FOOT_JOINTS)]
    height=native.median_filter_bool_np(feet[...,1]<=np.percentile(feet[...,1],5)+.055,native._odd_window(1/12,30))
    states=native.classify_support_states_np(fk,contacts.astype(bool),fps=30,height_support=height,support_policy='final_fail_closed')
    static=states==native.STATIC_SUPPORT
    speed=native._foot_speed_mps(feet,30)
    measured=np.percentile(speed[static],95) if static.any() else 0
    if abs(measured-audit['audit']['foot_skate_mps_p95'])>1e-5:
        raise ValueError('Native support population mismatch')
    intervals=[]
    for k,j in enumerate(FOOT_JOINTS):
        edges=np.diff(np.r_[False,static[:,k],False].astype(int))
        intervals.extend(dict(joint=int(j),start=int(a),end_exclusive=int(b),speed=distribution(speed[a:b,k])) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
    boundary=np.abs(normalized)>=1-1e-6
    generation=read(path/'generation.json')
    residual=generation['sampler'].get('raw_denoiser_anchor_max_abs_by_step')
    peak=np.unravel_index(d3.argmax(),d3.shape)
    angular_speed=np.linalg.norm(omega,axis=-1)
    angular_acc=np.linalg.norm(np.diff(omega,axis=0),axis=-1)*30
    result=dict(frames=len(fk),windows=len(raw),motion_sha256=sha(path/'native.npy'),
        mean_center_angle_rad=float(np.mean(np.asarray(errors)[12::25])),mean_trajectory_angle_rad=float(np.mean(errors)),
        anchor_step_m=float(d1[np.r_[ticks-1,ticks]].max()),anchor_neighborhood_D1_m=region_stats(d1,near[1:]),
        angular_speed_rad_s=region_stats(angular_speed,near[1:]),angular_acceleration_rad_s2=region_stats(angular_acc,near[2:]),
        acceleration_mps2=region_stats(d2,near[2:]),jerk_mps3=distribution(d3),jerk_anchor=region_stats(d3,near[3:]),
        jerk_overlap=region_stats(d3,(coverage==2)[3:]),jerk_peak_frame=int(peak[0]+3),jerk_peak_joint=int(peak[1]),
        normalized_overlap_max=float(abs(normalized[1:,:75]-normalized[:-1,75:]).max()) if len(raw)>1 else None,
        raw_overlap_max=float(abs(raw[1:,:75]-raw[:-1,75:]).max()) if len(raw)>1 else None,
        normalized_boundary_fraction=float(boundary.mean()),upper8_boundary_near=float(boundary[:,:,79:127][np.stack([np.isin(np.arange(150)+75*i,times[near]) for i in range(len(raw))])].mean()),
        final_prediction_residual=residual[-1] if residual else None,
        static_intervals=intervals,skate_p95=float(measured),native_pass=audit['absolute']['ok'],reasons=audit['absolute']['reasons'],
        native_metrics={key:audit['audit'][key] for key in audit['limits'] if key in audit['audit']},
        noise_sha256=generation['sampler'].get('noise_sha256'))
    return result

def census(q, scaler):
    from motion.retrieval_generation import decode_continuous_source
    root=Path(q['asset_root'])
    catalog_file=root/'outputs/unpaired_data_protocol_20261015_ZY1C32/catalog/motion_catalog.json'
    catalog=read(catalog_file)
    result=[]
    for source in catalog['source_records']:
        raw=checked(dict(path=str(root/'assets/motion/smpl_official_14'/Path(source['raw_receipt']['path']).name),sha256=source['source_sha256']))
        n=source['raw_frames']
        motion,_=decode_continuous_source(raw,source['discovery_metadata'],(0,n),known_rows=np.ones(n,bool))
        edge=MotionAdapter.to_edge(motion)
        affine=edge[:,79:139]*scaler['scale'][79:139]+scaler['bias'][79:139]
        valid=(abs(affine)<=1+32*np.finfo(np.float32).eps).all(1)
        covered=np.zeros(n,bool)
        legal_centers=np.zeros(n,bool)
        for event in catalog['events']:
            if event['source_uid']!=source['source_uid']:continue
            a,b=event['core_frame_start'],event['core_frame_end']
            covered[a:b]=True
            if b-a>=25:legal_centers[a+12:b-12]=True
        spans=np.convolve(valid.astype(int),np.ones(25,int),'same')==25
        result.append(dict(source=source['source_uid'],catalog_frames=int(covered.sum()),supported_frames=int((valid&covered).sum()),
            eligible_centers=int(legal_centers.sum()),supported_centers=int((legal_centers&spans).sum())))
    return dict(catalog_sha256=sha(catalog_file),sources=result)

def compare(target,duration,q,historical_hard_hash):
    rows={}
    for mode in ('custom_none','anchored','temporal'):
        print(duration,mode,flush=True)
        directory=target/(duration+'_'+mode)
        run(dict(q,mode=mode,overlap_policy='mean',trajectory_radius=12,trajectory_strength=.55,
                 save_scaler=duration=='5s' and mode=='custom_none',output=str(directory)))
        rows[mode]=evidence(directory,q)
    a,b,c=rows.values()
    if len({item['noise_sha256'] for item in rows.values()})!=1:
        raise ValueError('Noise differs across conditions')
    if b['motion_sha256']!=historical_hard_hash:
        raise ValueError('Historical hard replay mismatch')
    checks=dict(center_better_than_uncontrolled=c['mean_center_angle_rad']<a['mean_center_angle_rad'],
        trajectory_better_than_uncontrolled=c['mean_trajectory_angle_rad']<a['mean_trajectory_angle_rad'],
        displacement_down_20pct=c['anchor_step_m']<=.8*b['anchor_step_m'],
        jerk_peak_down_20pct=c['jerk_mps3']['max']<=.8*b['jerk_mps3']['max'],
        jerk_p95_no_worse=c['jerk_mps3']['p95']<=b['jerk_mps3']['p95'],
        skate_no_worse=c['skate_p95']<=b['skate_p95'])
    write(target/(duration+'_comparison.json'),dict(rows=rows,checks=checks,advance=all(checks.values()),native_pass=c['native_pass']))
    return all(checks.values())

def main():
    target=Path(sys.argv[1]).resolve()
    target.mkdir(parents=True,exist_ok=False)
    history=ROOT/'outputs'
    def request(duration):
        return read(history/'supported_upper8'/duration/'request.json')
    first=request('5s')
    sys.path.insert(0,str(pinned(first['legacy_root'],first['legacy_commit'])))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    prior={}
    for duration in ('5s','15s','whole'):
        q=request(duration)
        for name,path in [('official',history/'experiment_v1'/('baseline_'+duration)),('hard',history/'supported_upper8'/duration)]:
            prior[duration+'_'+name]=evidence(path,q)
    write(target/'saved_diagnostics.json',prior)
    for duration in ('5s','15s','whole'):
        q=request(duration)
        qualifies=compare(target,duration,q,prior[duration+'_hard']['motion_sha256'])
        if duration=='5s':
            write(target/'upper10_census.json',census(q,np.load(target/'5s_custom_none/scaler.npz')))
        if duration=='5s' and not qualifies:break

if __name__=='__main__':main()
