"""Event responsibility includes editable edges; context remains outside the gap."""
import numpy as np
import torch
from dunhuang.research.plan_view_r34.compiler import compile_window,packets,native_body
from dunhuang.research.coverage_r34.context_contract import WindowContext
from .graph import zones

def compile(plan,index,audio_sha):
    arrays,binding=compile_window(plan,index,audio_sha)
    e=plan["events"][index];z=zones(e);start=binding["window_start"]
    a,b=z["full_event"];left=z["entry_edit"][1]-a;right=b-z["exit_edit"][0]
    gap=[a-left,b+right]
    ticks=np.r_[np.arange(gap[0]-8,gap[0]),np.arange(gap[1],gap[1]+8)]
    if ticks.min()<start or ticks.max()>=start+256:raise ValueError("Editing context exceeds actual256 clock")
    bodies=[];uids=[]
    for t in ticks:
        neighbor=next(x for x in plan["events"] if x["music_interval"][0]<=t<x["music_interval"][1])
        if neighbor["event_uid"]==e["event_uid"] and neighbor["music_interval"]==e["music_interval"]:raise ValueError("Gap content leaked into context")
        x,_=native_body(neighbor,np.array([t]));bodies.append(x[0]);uids.append(neighbor["event_uid"])
    arrays["context_body"]=np.stack(bodies);arrays["context_indices"]=ticks-start
    binding.update(context_gap_local=[x-start for x in gap],neighbor_event_uids=uids,protected_core=z["protected_core"],entry_edit=z["entry_edit"],exit_edit=z["exit_edit"],
        write_interval=gap,context_intervals=[[gap[0]-8,gap[0]],[gap[1],gap[1]+8]],
        source_observation_scope=e["source_interval"],source_time_correspondence=[a,b],
        source_residual_scope="GLOBAL_CONTENT_ATTENTION; TIME_BIAS_ONLY_IN_EVENT; NOT_A_RESIDUAL_GATE",
        context_branch_training="R35_CONTINUOUS_AIST_AND_SAME_RECORDING_PARTIAL_BODY",
        usage="TWO_BOUNDARY_WINDOW; ONLY_REGISTERED_WRITE_INTERVAL_RESPONSIBLE; NOT_COMPLETE_THREE_EVENTS_OR_WHOLE_SONG")
    return arrays,binding

def make_packets(arrays,binding,batch=2):
    src,old=packets(arrays,binding,batch)
    gap=torch.tensor(binding["context_gap_local"],device="cuda",dtype=torch.long)[None].repeat(batch,1)
    ctx=WindowContext(old.values,old.valid,old.frame_index,old.side,old.provenance,gap,old.window_origin,old.global_frame_index,old.coordinate_uid)
    return src,ctx.validate()
