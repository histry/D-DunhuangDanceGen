"""Frozen B3 FK attribution, root-only bound and bounded joint response study."""
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from b2 import ROOT, read, write, pinned, sha, matrices, MotionAdapter, angle_difference
from dunhuang.contracts import sixd
from dunhuang.retrieval import primitive_index

def direction(x,a,b,joints):
    r=matrices(x[a:b,7:].reshape(-1,24,6)).astype(float)
    delta=np.zeros((b-a,24,3))
    for j in joints:
        base=r[len(r)//2,j]
        tangent=Rotation.from_matrix(base.T@r[:,j]).as_rotvec()
        target=base@Rotation.from_rotvec(gaussian_filter1d(tangent,2,axis=0)).as_matrix()
        delta[:,j]=Rotation.from_matrix(r[:,j].swapaxes(-1,-2)@target).as_rotvec()
    delta*=np.minimum(1.,.1/np.maximum(np.linalg.norm(delta,axis=-1),1e-12))[...,None]
    return delta

def apply(x,a,b,delta,translation,amount):
    out=x.copy()
    envelope=np.sin(np.linspace(0,np.pi,b-a))**4
    envelope[[0,-1]]=0.
    r=matrices(x[a:b,7:].reshape(-1,24,6)).astype(float)
    for j in np.flatnonzero(np.any(delta!=0,axis=(0,2))):
        v=delta[:,j]*envelope[:,None]*amount
        changed=envelope>0
        rows=np.arange(a,b)[changed]
        out[rows,7+6*j:13+6*j]=sixd(r[changed,j]@Rotation.from_rotvec(v[changed]).as_matrix())
    out[a:b,4:7]+=translation*envelope[:,None]*amount
    return out

def main(target):
    target.mkdir(parents=True,exist_ok=False)
    parent=ROOT/'outputs/b2_whole_f06bc86/whole_temporal'
    q=read(parent/'request.json')
    sys.path.insert(0,str(pinned(q['legacy_root'],q['legacy_commit'])))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    from motion.geometry import fk_24_np
    from motion.physics import audit_motion_np, derive_contacts_np
    from motion.config import MotionGenerationConfig
    from motion_geometry import physical as native
    from motion_geometry.smpl24 import FOOT_JOINTS
    from contracts.physical_quality import evaluate_physical_audit
    x=np.load(parent/'native.npy');fk=np.load(parent/'native_fk.npy')
    import subprocess
    write(target/'provenance.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        request=q,parent_sha256=sha(parent/'native.npy'),fk_sha256=sha(parent/'native_fk.npy'),
        windows_sha256=sha(parent/'windows.npy'),fixed_amplitudes=[.25,.5],continuation_amplitude=.75,
        rotation_direction_cap_rad=.1,translation_direction_cap_m=.02,anchor_trajectory_cap_rad=.02))

    before=read(parent/'physical.json')
    rotations=matrices(x[:,7:].reshape(-1,24,6)).astype(float)
    def audit(value):
        pos=fk_24_np(value)
        measured=audit_motion_np(value,precomputed_joints=pos)
        gate=evaluate_physical_audit(measured)
        if gate['limits']!=before['absolute']['limits']:raise ValueError('Threshold drift')
        return pos,measured,gate
    contacts,_,_,_=derive_contacts_np(x,MotionGenerationConfig(),precomputed_joints=fk)
    feet=fk[:,list(FOOT_JOINTS)]
    height=native.median_filter_bool_np(feet[...,1]<=np.percentile(feet[...,1],5)+.055,native._odd_window(1/12,30))
    states=native.classify_support_states_np(fk,contacts.astype(bool),fps=30,height_support=height,support_policy='final_fail_closed')
    static=states==native.STATIC_SUPPORT
    speed=native._foot_speed_mps(feet,30)
    if not np.isclose(np.percentile(speed[static],95),before['audit']['foot_skate_mps_p95']):raise ValueError('Population mismatch')
    velocity=np.zeros((len(x),4,2));velocity[1:]=np.diff(feet[:,:,[0,2]],axis=0)*30
    count=static.sum(1)
    optimum=-np.sum(velocity*static[:,:,None],axis=1)/np.maximum(count[:,None],1)
    residual=velocity+optimum[:,None]
    write(target/'root_only.json',dict(population=int(static.sum()),multi_support_frames=int((count>1).sum()),
        before_sum_squared=float(np.sum(velocity[static]**2)),minimum_sum_squared=float(np.sum(residual[static]**2)),
        optimum_residual_speed_p95=float(np.percentile(np.linalg.norm(residual[static],axis=-1),95)),
        scope='Instantaneous unbounded root-XZ minimum L2 on frozen static population. P95 at L2 optimum is not a P95 lower bound.'))
    intervals=[]
    for k,j in enumerate(FOOT_JOINTS):
        edges=np.diff(np.r_[False,static[:,k],False].astype(int))
        for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            intervals.append(dict(joint=int(j),column=k,start=int(a),end=int(b),frames=int(b-a),
                p95=float(np.percentile(speed[a:b,k],95)),peak=float(speed[a:b,k].max()),
                positions=feet[a:b,k].tolist(),speeds=speed[a:b,k].tolist()))
    write(target/'support_intervals.json',intervals)
    jerk=np.linalg.norm(np.diff(fk,n=3,axis=0),axis=-1)*27000
    peak,joint=np.unravel_index(jerk.argmax(),jerk.shape);peak+=3
    stencil=np.arange(peak-3,peak+1)
    windows=np.load(parent/'windows.npy')
    contributors=[]
    for i,window in enumerate(windows):
        local=stencil-i*75
        if local.min()>=0 and local.max()<150:
            positions=fk_24_np(MotionAdapter.to_native(window.copy()))
            value=float(np.linalg.norm(np.diff(positions[local,joint],n=3,axis=0))*27000)
            contributors.append(dict(window=i,start=i*75,local=local.tolist(),joint_jerk=value,positions=positions[local,joint].tolist()))
    ancestors=[0,3,6,9,14,17,19,21,23]
    write(target/'attribution.json',dict(parent_sha256=sha(parent/'native.npy'),peak_frame=int(peak),joint=int(joint),
        global_stencil=stencil.tolist(),assembled_jerk=float(jerk[peak-3,joint]),complete_window_stencils=contributors,
        covering_windows_by_frame=[[i for i in range(len(windows)) if 0<=t-75*i<150] for t in stencil],
        root_positions=x[stencil,4:7].tolist(),ancestors=ancestors,
        ancestor_steps_rad=angle_difference(rotations[stencil[:-1]][:,ancestors],rotations[stencil[1:]][:,ancestors]).tolist()))
    eligible=read(ROOT/'outputs/b2_phrase_90eab5b/eligible.json')
    groups=primitive_index(eligible)
    write(target/'primitives.json',dict(centers=len(eligible),event_groups=len(groups),source_count=len({r['source_uid'] for r in groups}),
        overlapping_clusters=sum(len(r['clusters']) for r in groups),groups=groups,
        scope='Overlapping intervals within recorded event; independence beyond recording not established'))
    stance=max((s for s in intervals if s['frames']>=12),key=lambda s:s['p95'])
    domains=[('hand_root_trunk',max(0,peak-18),min(len(x),peak+19),[0,3,6,9]),
             ('hand_arm',max(0,peak-18),min(len(x),peak+19),[14,17,19,21]),
             ('support_leg',stance['start'],stance['end'],[1,2,4,5,7,8,10,11])]
    domains=[(name,int(a),int(b),joints) for name,a,b,joints in domains]
    candidates=[];positive=[]
    def save(label,proposal,a,b,joints,amount):
        pos,metrics,gate=audit(proposal)
        anchor_change=max(float(angle_difference(rotations[t['frame'],t['joints']],
            matrices(proposal[t['frame'],7:].reshape(24,6))[t['joints']]).max()) for t in q['anchors'])
        trajectory_change=0.
        for anchor in q['anchors']:
            t=anchor['frame'];joints=anchor['joints']
            proposed=matrices(proposal[t-12:t+13,7:].reshape(25,24,6))[:,joints]
            trajectory_change=max(trajectory_change,float(angle_difference(rotations[t-12:t+13,joints],proposed).max()))
        new_failures=sorted(set(gate['reasons'])-set(before['absolute']['reasons']))
        protected=not new_failures and anchor_change<=.02 and trajectory_change<=.02
        major=metrics['joint_jerk_mps3_max']<before['audit']['joint_jerk_mps3_max']*.99 or metrics['foot_skate_mps_p95']<before['audit']['foot_skate_mps_p95']*.99
        folder=target/label;folder.mkdir()
        np.save(folder/'native.npy',proposal);np.save(folder/'native_fk.npy',pos)
        write(folder/'physical.json',dict(audit=metrics,absolute=gate))
        row=dict(case=label,span=[a,b],joints=joints,amount=amount,anchor_max_change_rad=anchor_change,trajectory_max_change_rad=trajectory_change,
            protected=protected,major_improved=major,native_pass=gate['ok'],new_failures=new_failures,
            metrics={k:metrics[k] for k in before['absolute']['limits'] if k in metrics},reasons=gate['reasons'])
        return row
    for name,a,b,joints in domains:
        delta=direction(x,a,b,joints)
        translation=np.zeros((b-a,3))
        if name=='support_leg':
            original=fk[a:b,stance['joint']]
            goal=np.median(original,axis=0)
            jac=[]
            for j in joints:
                for axis in range(3):
                    perturbed=x[a:b].copy()
                    r=matrices(perturbed[:,7:].reshape(-1,24,6)).astype(float)
                    tangent=np.zeros((b-a,3));tangent[:,axis]=.001
                    perturbed[:,7+6*j:13+6*j]=sixd(r[:,j]@Rotation.from_rotvec(tangent).as_matrix())
                    jac.append((fk_24_np(perturbed)[:,stance['joint']]-original)/.001)
            J=np.stack(jac,axis=-1)
            solved=np.linalg.solve(J@J.transpose(0,2,1)+.01*np.eye(3),goal-original)
            v=np.einsum('tij,ti->tj',J,solved)
            delta[:]=0
            for k,j in enumerate(joints):delta[:,j]=gaussian_filter1d(v[:,3*k:3*k+3],2,axis=0)
            delta*=np.minimum(1.,.1/np.maximum(np.linalg.norm(delta,axis=-1),1e-12))[...,None]
            translation[:,[0,2]]=gaussian_filter1d((goal-original)[:,[0,2]],2,axis=0)
            translation*=np.minimum(1.,.02/np.maximum(np.linalg.norm(translation,axis=-1),1e-12))[:,None]
        for amount in (.25,.5):
            proposal=apply(x,a,b,delta,translation,amount)
            row=save(name+'_'+str(amount),proposal,a,b,joints,amount)
            candidates.append(row)
            if row['protected'] and row['major_improved']:positive.append((row,delta,translation))
    continuation=None
    if positive:
        row,delta,translation=min(positive,key=lambda pair:pair[0]['metrics']['joint_jerk_mps3_max'])
        a,b=row['span']
        continuation=save('continuation',apply(x,a,b,delta,translation,.75),a,b,row['joints'],.75)
    write(target/'summary.json',dict(candidates=candidates,positive_directions=len(positive),continuation=continuation,
        training=False,neural_forward_calls=0,full_generation_qualified=False))
    print('B3_COMPLETE',len(positive),flush=True)

if __name__=='__main__':main(Path(sys.argv[1]).resolve())
