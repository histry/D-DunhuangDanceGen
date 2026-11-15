"""Full-Core E1-W map, phase-driven sparse cues, matched masks across events."""
import numpy as np
from dunhuang.contracts import matrices
from dunhuang.generation.soft_reference import resample_native
from dunhuang.planning.time_map import source_positions

def compile_events(events,sources,T,cfg):
    if len(events)!=2 or len(sources)!=2 or not 0<cfg["core_start"]<cfg["core_start"]+T<256:
        raise ValueError("Two complete legal events inside native window")
    start=cfg["core_start"];values=[];phases=[];cue=np.zeros(256,bool)
    for event,source in zip(events,sources):
        if event["observed_joints"]!=list(range(22)) or event["root_translation_observed"] is not False or event.get("contact_observed") is not False:
            raise ValueError("Observed SMPL22 and unknown Root/contact required")
        positions=source_positions(event,np.arange(T))
        native=resample_native(source,positions)
        R=matrices(native[:,7:].reshape(T,24,6))[:,:22]
        rot=R[...,:2,:].reshape(T,132).astype(np.float32)
        x=np.zeros((256,139),np.float32);x[:,7:]=np.r_[np.repeat(rot[:1],start,axis=0),rot,np.repeat(rot[-1:],256-start-T,axis=0)]
        values.append(x);selected=[];warp=np.array(event["time_warp"])
        for kind in cfg["phase_types"]:
            candidates=[p for p in event["full_phases"] if p["type"]==kind and not p.get("evidence_truncated_by_core",False)]
            if not candidates:continue
            p=max(candidates,key=lambda p:(p["score"],-p["frame"]))
            tick=start+int(round(float(np.interp(p["frame"],warp[:,0],warp[:,1]))))
            tick=min(start+T-1,max(start,tick));a=max(start,tick-cfg["cue_radius"]);b=min(start+T,tick+cfg["cue_radius"]+1)
            cue[a:b]=True;selected.append(dict(p,target_frame=tick,label_scope="KINEMATIC_PROXY_NOT_SEMANTIC_TRUTH"))
        if len(selected)<3:raise ValueError("Insufficient independent phase types; no silent fallback")
        phases.append(selected)
    cue[start:start+2]=True;cue[start+T-2:start+T]=True
    if cue.sum()>=T//2:raise ValueError("Sparse cue budget exceeded")
    mask=np.zeros((256,139),np.float32);mask[cue,7:]=1.
    return np.stack(values),mask,phases

def angular_errors(motion,reference,joints=None):
    from scipy.spatial.transform import Rotation
    P=matrices(motion[:,7:].reshape(len(motion),22,6),row=True)
    G=matrices(reference[:,7:].reshape(len(reference),22,6),row=True)
    if joints is not None:P=P[:,joints];G=G[:,joints]
    return Rotation.from_matrix((G.swapaxes(-1,-2)@P).reshape(-1,3,3)).magnitude().reshape(len(motion),-1)

def frequency_readout(points,fps=30):
    p=np.asarray(points,float);p=p-p[:,:1]
    centered=p-p.mean(0,keepdims=True);window=np.hanning(len(p))[:,None,None]
    spectrum=np.fft.rfft(centered*window,axis=0)
    f=np.fft.rfftfreq(len(p),1/fps);energy=(abs(spectrum)**2).sum((1,2))*(2*np.pi*f)**2
    total=max(float(energy.sum()),1e-30)
    return dict(low_frequency_velocity_energy_fraction=float(energy[f<=3].sum()/total),
        high_frequency_velocity_energy_fraction=float(energy[f>6].sum()/total),
        bands_hz=dict(low=[0,3],high=[6,15]),root_relative=True,window="Hann after temporal mean removal",motion_modified=False)
