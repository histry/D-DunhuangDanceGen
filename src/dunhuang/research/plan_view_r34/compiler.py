"""Selected event + adjacent planned edges, on an explicit shared music/window clock."""
import numpy as np
import torch
from dunhuang.contracts import matrices
from dunhuang.generation.soft_reference import resample_native
from dunhuang.planning.time_map import source_positions
from dunhuang.research.reference_r26.reference import features
from dunhuang.research.coverage_r34.common import WindowSource
from dunhuang.research.coverage_r34.context_contract import WindowContext
from .common import *
from dunhuang.planning.event_plan import semantic_plan_id

def native_body(event,times):
    native=np.load(checked(event["reference"]));pos=source_positions(event,np.asarray(times))
    sample=resample_native(native,pos);R=matrices(sample[:,7:].reshape(-1,24,6))[:,:22]
    return R[...,:2,:].reshape(-1,132).astype(np.float32),pos

def choose(plan):
    for i,e in enumerate(plan["events"][1:-1],1):
        a,b=e["music_interval"];start=a-64
        if start>=0 and start+256<=plan["frames"] and b-start<=240:return i
    raise ValueError("Plan has no complete internal256 research window")

def compile_window(plan,index,audio_sha):
    events=plan["events"];event=events[index];a,b=event["music_interval"];start=a-64
    if not 0<index<len(events)-1 or start<0 or start+256>plan["frames"] or b-start>248:raise ValueError("Unsupported window; no truncated neighbor substitution")
    body,pos=native_body(event,np.arange(a,b));n=len(body);raw=np.load(checked(event["reference"]))
    halo_pos=max(0.,pos[0]-(pos[1]-pos[0]));halo=resample_native(raw,np.array([halo_pos]))
    haloR=matrices(halo[:,7:].reshape(-1,24,6))[:,:22];haloBody=haloR[...,:2,:].reshape(1,132).astype(np.float32)
    f,_=features(np.r_[haloBody,body]);f=f[1:]
    target_ticks=np.r_[np.arange(a-8,a),np.arange(b,b+8)]
    context_body=[];context_ids=[]
    for tick in target_ticks:
        neighbor=next(e for e in events if e["music_interval"][0]<=tick<e["music_interval"][1])
        assert neighbor["music_interval"]!=event["music_interval"]
        x,_=native_body(neighbor,np.array([tick]));context_body.append(x[0]);context_ids.append(neighbor["event_uid"])
    mapped=np.clip(np.arange(256)-64,0,n-1).astype(np.float32);control=(np.arange(256)>=64)&(np.arange(256)<64+n)
    return dict(body=body,features=f,source_positions=pos,mapped=mapped,control=control,context_body=np.stack(context_body),context_indices=target_ticks-start),dict(
        schema="selected_event_lodge_window_v1",song_audio_sha256=audio_sha,event_uid=event["event_uid"],source_sha256=event["source_sha256"],source_uid=event["source_uid"],
        plan_id=semantic_plan_id(plan),candidate_uid=event["candidate_uid"],source_interval=event["source_interval"],music_interval=[a,b],time_warp=event["time_warp"],
        source_memory_frames=n,source_control_local_interval=[64,64+n],window_start=start,window_frames=256,fps=30,
        neighbor_event_uids=context_ids,neighbor_provenance="PLANNED_BODY_ONLY; NO_GENERATED_HISTORY",Root_C4_truth="UNKNOWN",
        link_id=identity(audio_sha,event),generation_identity=identity(audio_sha,event,start),
        first_reference_velocity_observed=bool(halo_pos<pos[0]),source_not_extended_over_neighbor_events=True,context_branch_training="R34_NOT_STARTED; ZERO_OUTPUT",
        usage="UNQUALIFIED_PLAN_CONSUMER_SHORT_WINDOW_PREVIEW; NOT_CONNECTED_OR_WHOLE_SONG_QUALIFICATION")

def packets(arrays,binding,batch=2):
    n=len(arrays["body"]);t=np.arange(n)/max(n-1,1);clock=np.stack((np.sin(2*np.pi*t),np.cos(2*np.pi*t)),-1).astype(np.float32)
    tiled=lambda x:torch.tensor(np.repeat(x[None],batch,axis=0),device="cuda")
    source=WindowSource(dict(features=tiled(arrays["features"]),valid=torch.ones(batch,n,22,dtype=torch.bool,device="cuda"),
        clock=tiled(clock),ref_phase=torch.zeros(batch,n,8,device="cuda"),query_phase=torch.zeros(batch,256,8,device="cuda"),
        mapped_source=tiled(arrays["mapped"]),control_mask=tiled(arrays["control"]).bool()),tuple([binding["source_uid"]]*batch))
    values=torch.full((batch,16,139),float("nan"),device="cuda");values[:,:,7:]=tiled(arrays["context_body"])
    valid=torch.isfinite(values);indices=tiled(arrays["context_indices"]).long();side=torch.tensor([0]*8+[1]*8,device="cuda")[None].repeat(batch,1)
    origin=torch.full((batch,),binding["window_start"],device="cuda",dtype=torch.long);gap=torch.tensor(binding["source_control_local_interval"],device="cuda")[None].repeat(batch,1)
    context=WindowContext(values,valid,indices,side,torch.full_like(side,2),gap,origin,indices+origin[:,None],"COMMON_CANONICAL_AXES_BODY_ONLY_NO_WORLD_ROOT")
    return source.validate(),context.validate()
