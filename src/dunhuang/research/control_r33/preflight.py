"""Actual sampling parity for general/change/NoRef/zero-context; fresh change states."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .runtime import Runtime
from .common import *
from .inventory import compile_holdouts
from dunhuang.research.time_r29.model import mapping

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();pop,ledger,data,z=population_data();rt=Runtime(cfg);arrays={};noise=[];calls=0;trajectories=0
    prior31=np.load(Path(cfg["r31"])/"evaluation/arrays.npz");prior32=np.load(Path(cfg["r32"])/"evaluation/arrays.npz")
    for role in ("train","holdout"):
        cases=read("src/dunhuang/research/content_r32/config.json")["train_cases" if role=="train" else "holdout_cases"]
        rows=[data[c] for c in cases];music=torch.tensor(np.stack([r["music"] for r in rows]),device="cuda")
        for mode in ("correct","wrong_clock","swapped","none"):
            for k,seeds in enumerate(read("src/dunhuang/research/time_r29/config.json")["real_eval_seeds"]):
                packet=None if mode=="none" else source([core_body(r) for r in rows],"aist",mode,uids=[f"aist_{c}" for c in cases])
                pred,nr,_=sample(rt,music,source_packet=packet,seeds=seeds)
                assert np.array_equal(pred.cpu().numpy(),prior31[f"{role}_CanonicalIO_{mode}_{k}_normalized"])
                arrays[f"{role}_{mode}_{k}"]=pred.cpu().numpy();calls+=1;trajectories+=4;noise.append(nr)
        if role=="train":
            y=rt.n.normalize(torch.tensor(np.stack([r["model_raw"] for r in rows]),device="cuda"));context=observed_context(y)
            for mode in ("correct","none"):
                packet=source([core_body(r) for r in rows],"aist",uids=[f"aist_{c}" for c in cases]) if mode=="correct" else None
                pred,nr,_=sample(rt,music,source_packet=packet,context_packet=context,seeds=read("src/dunhuang/research/time_r29/config.json")["real_eval_seeds"][0])
                assert np.array_equal(pred.cpu().numpy(),arrays[f"train_{mode}_0"])
                arrays["zero_context_"+mode]=pred.cpu().numpy();calls+=1;trajectories+=4;noise.append(nr)
    conditions=[(m,event) for m in range(2) for event in (-1,0,1)]
    music=torch.tensor(np.stack([z["music"][m] for m,event in conditions]),device="cuda")
    uids=["female_FeiTian","male_pipa_1"]
    for mode in ("correct","wrong_clock","swapped"):
        events=[1-max(ev,0) if mode=="swapped" else max(ev,0) for m,ev in conditions]
        packet=source([z["source139"][e,64:160,7:] for e in events],"change",uids=[uids[e] for e in events],drops=[e<0 for m,e in conditions])
        packet.values["mapped_source"]=mapping(6,"cuda",wrong=mode=="wrong_clock")
        pred,nr,tr=sample(rt,music,source_packet=packet,common_seed=24041,trace=mode=="correct")
        assert np.array_equal(pred.cpu().numpy(),prior32[f"change_ChangE_{mode}_normalized"])
        arrays["change_"+mode]=pred.cpu().numpy();calls+=1;trajectories+=6;noise.append(nr)
        if tr is not None:np.savez_compressed(out/"change_states.npz",**trace_arrays(tr))
    # Planned pose has NO trustworthy first7; the context encoder sees only declared components.
    values=torch.full((6,16,139),float("nan"),device="cuda");valid=torch.zeros_like(values,dtype=torch.bool)
    values[:,:,7:]=torch.tensor(z["source139"][0,64:80,7:],device="cuda")[None];valid[:,:,7:]=True
    indices=torch.arange(160,176,device="cuda")[None].expand(6,-1);side=torch.ones_like(indices)
    planned=ContextPacket(values,valid,indices,side,torch.full_like(indices,2),"SHARED_MODEL0_WINDOW")
    packet=source([z["source139"][max(ev,0),64:160,7:] for m,ev in conditions],"change",uids=[uids[max(ev,0)] for m,ev in conditions],drops=[ev<0 for m,ev in conditions])
    pred,nr,_=sample(rt,music,source_packet=packet,context_packet=planned,common_seed=24041)
    assert np.array_equal(pred.cpu().numpy(),arrays["change_correct"]);arrays["zero_planned_context"]=pred.cpu().numpy();calls+=1;trajectories+=6;noise.append(nr)
    held,records,provenance=compile_holdouts(cfg,rt,z["music"]);np.savez_compressed(out/"held_observations.npz",**held)
    rt.check();frozen=rt.frozen;params=dict(general=sum(p.numel() for p in rt.c.general.parameters()),change=sum(p.numel() for p in rt.c.change.parameters()),context=sum(p.numel() for p in rt.c.context_branch.parameters()))
    rt.close();np.savez_compressed(out/"arrays.npz",**arrays);assert calls==22 and trajectories==96
    result=dict(general_R31_ref_and_NoRef_exact=True,change_R32_exact=True,zero_context_exact=True,planned_missing_first7_safe=True,
        single_backbone_instances=1,reference_routes="EXPLICIT_CATALOG_DOMAIN_PER_WINDOW; NOT_SHARED_ADAPTER_GENERALIZATION",
        context_trained=False,phase_trained=False,frozen=frozen,parameters=params,held_records=records,held_provenance=provenance,
        sampler_calls=calls,generated_trajectories=trajectories,optimizer_steps=0,operational_qualified=False,full_native_status="NOT_EVALUABLE",whole_song_generations=0,noise=noise)
    if replay:
        assert result==read(Path(replay)/"result.json")
        for p in out.glob("*.npz"):same_arrays(p,Path(replay)/p.name)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,files={p.name:sha(p) for p in out.glob("*.npz")}))
    print("R33_PREFLIGHT",calls,trajectories,params,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)
