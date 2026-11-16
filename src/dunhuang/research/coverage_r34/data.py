"""Actual same-recording observations, extending the old Core clock without output selection."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices
from dunhuang.io import checked
from dunhuang.generation.soft_reference import resample_native
from dunhuang.planning.time_map import source_positions
from .common import *

def extend_clock(core,length=256,start=64):
    core=np.asarray(core,np.float64)
    if core.shape!=(96,) or not np.isfinite(core).all() or (np.diff(core)<=0).any():raise ValueError("Strict observed Core clock required")
    out=np.empty(length,np.float64);out[start:start+96]=core
    out[:start]=core[0]+(np.arange(start)-start)*(core[1]-core[0])
    out[start+96:]=core[-1]+np.arange(1,length-start-95)*(core[-1]-core[-2])
    return out

def observed(cfg,rt):
    _,_,_,z=population_data();p=read("outputs/lodge_reference_r24_01/main/protocol.json")
    assets=read("outputs/event_plan_01/prepare/assets.json");origins=read(checked(assets["catalog_source"]))["source_records"]
    bodies=[];local=[];halos=[];records=[]
    for i,key in enumerate(("A","B")):
        plan=read(checked(p["plans"][key]));event=plan["events"][0];origin=next(r for r in origins if r["source_uid"]==event["source_uid"])
        raw=np.load(checked(origin["raw_receipt"]));native=np.load(checked(event["reference"]))
        assert len(raw["poses"])==len(native)==origin["raw_frames"] and float(raw["mocap_framerate"])==30
        pos=extend_clock(source_positions(event,np.arange(96)));halo=pos[0]-(pos[1]-pos[0])
        if halo<0 or pos[-1]>len(native)-1:raise ValueError("Same recording lacks registered surrounding observations; no holds/loops/replacement")
        indices=np.arange(int(np.floor(halo)),int(np.ceil(pos[-1]))+1)
        original=Rotation.from_rotvec(raw["poses"][indices,:66].reshape(-1,3)).as_matrix().reshape(-1,22,3,3)
        inherited=matrices(native[indices,7:].reshape(-1,24,6))[:,:22]
        raw_delta=float(Rotation.from_matrix((original.swapaxes(-1,-2)@inherited).reshape(-1,3,3)).magnitude().max())
        assert raw_delta<1e-5 and np.isfinite(original).all()
        sampled=resample_native(native,np.r_[halo,pos]);R=matrices(sampled[:,7:].reshape(-1,24,6))[:,:22]
        body=R[...,:2,:].reshape(-1,132).astype(np.float32)
        assert np.max(abs(body[65:161]-z["source139"][i,64:160,7:]))<1e-5
        gauge=torch.zeros(1,256,139,device="cuda");gauge[0,:,7:]=torch.tensor(body[1:],device="cuda")
        with torch.no_grad():points=rt.f(gauge,rt.s)[0,:,:22].cpu().numpy()
        points=points-points[:,:1]
        target=np.full((256,139),np.nan,np.float32);target[:,7:]=body[1:]
        assert not np.isfinite(target[:,:7]).any()
        bodies.append(body[1:]);halos.append(body[0]);local.append(points)
        records.append(dict(source_uid=event["source_uid"],recording_uid=event["recording_uid"],event_uid=event["event_uid"],
            raw_receipt=origin["raw_receipt"],native_reference=event["reference"],source_sha256=event["source_sha256"],
            old_event_interval=event["source_interval"],source_positions=pos.tolist(),halo_source_position=float(halo),
            continuous_raw_interval=[int(np.floor(halo)),int(np.ceil(pos[-1]))+1],raw_native_angle_max_rad=raw_delta,
            clock_policy="EXACT_OLD_CORE; ENDPOINT_OBSERVED_STEP_EXTENSION; SAME_RECORDING_NO_HOLDS_LOOPS_OR_EVENT_SPLICES",
            source_body_observed_frames=256,Root_C4_truth="UNKNOWN",dancer_identity=origin["discovery_metadata"]["dancer_id_status"],
            source_role="SAME_TWO_R33_ADAPTED_RECORDINGS; EXPANDED_TIME_NOT_NEW_POPULATION"))
    return dict(body=np.stack(bodies),halo=np.stack(halos),local_points=np.stack(local),legacy_body=z["source139"][:,64:160,7:],legacy_points=z["source22"][:,64:160]),records

def packet(arrays,conditions,coverage,mode="correct"):
    ids=("female_FeiTian","male_pipa_1");select=[1-max(e,0) if mode=="swapped" else max(e,0) for m,e in conditions]
    drops=[e<0 for m,e in conditions];uids=[ids[e] for e in select]
    if coverage:
        return full_packet(arrays["body"][select],uids,mode="wrong_clock" if mode=="wrong_clock" else "correct",drops=drops,halo=arrays["halo"][select])
    return source(list(arrays["legacy_body"][select]),"change",mode="wrong_clock" if mode=="wrong_clock" else "correct",uids=uids,drops=drops)
