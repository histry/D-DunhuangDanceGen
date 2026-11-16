"""Pinned source populations, explicit routing, and original native sampling math."""
from pathlib import Path
import numpy as np
import torch
from .contracts import SourcePacket,ContextPacket
from dunhuang.research.content_r32.common import population,reference,paired_schedule,core_body,core_points
from dunhuang.research.state_r27.common import read,sha,write,fresh,commit,digest,forbid_optimizers,same_arrays,readout_check,vector
from dunhuang.research.state_r27.trace import row_noise,record_sampler,trace_arrays
from dunhuang.research.lodge_reference_r24.noise import common_noise

def population_data():
    return population(read("src/dunhuang/research/content_r32/config.json"))

def source(bodies,domain,mode="correct",uids=None,drops=None,usage="inference"):
    if uids is None:raise ValueError("Source identities cannot be invented by the controller")
    p=reference(bodies,mode,drops)
    return SourcePacket(domain,p,tuple(uids),usage)

def sample(rt,music,*,source_packet=None,context_packet=None,seeds=None,common_seed=None,trace=False):
    if (seeds is None)==(common_seed is None):raise ValueError("Exactly one explicit noise protocol required")
    rt.c.clear_all()
    if source_packet is not None:rt.c.set_source(source_packet)
    if context_packet is not None:rt.c.set_context(context_packet)
    torch.manual_seed(27041 if seeds is not None else common_seed);torch.cuda.manual_seed_all(27041 if seeds is not None else common_seed)
    zeros=torch.zeros(len(music),256,139,device=music.device)
    protocol=row_noise(seeds,device=str(music.device)) if seeds is not None else common_noise(common_seed,batch=len(music),device=str(music.device))
    if trace:
        with torch.no_grad(),protocol as nr,record_sampler(rt.d,selected=(999,519,119,19)) as state:
            pred=rt.d.inpaint_soft_ddim(tuple(zeros.shape),music,None,constraint=dict(value=zeros,mask=zeros))
    else:
        with torch.no_grad(),protocol as nr:pred=rt.d.inpaint_soft_ddim(tuple(zeros.shape),music,None,constraint=dict(value=zeros,mask=zeros))
        state=None
    rt.c.clear_all();assert nr["draws"]==50
    return pred,nr,state

def observed_context(normalized):
    indices=torch.tensor(list(range(56,64))+list(range(160,168)),device=normalized.device)
    values=normalized[:,indices].clone();batch=len(values)
    side=torch.tensor([0]*8+[1]*8,device=values.device,dtype=torch.long)[None].expand(batch,-1)
    return ContextPacket(values,torch.ones_like(values,dtype=torch.bool),indices[None].expand(batch,-1),side,torch.zeros_like(side),"SHARED_MODEL0_WINDOW")
