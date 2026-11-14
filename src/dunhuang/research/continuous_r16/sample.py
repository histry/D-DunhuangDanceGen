"""Registered C0/C1 150-frame sampling; no hidden motion targets in C1 conditions."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.contracts import MotionAdapter,matrices,sixd
from dunhuang.sampling import constrained_ddim
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.masked_denoiser.common import setup,make,checkpoint_load,tensor_hash
from dunhuang.research.boundary_context.common import masks
from .contract import patch_control,align_donor

def run(cfg,preflight,out,replay=None):
    out=fresh(out);forbid_optimizers();pre=read(Path(preflight)/"protocol.json");bindings=pre["pairs"]
    assets,_,_,_,_,data,model,diffusion=setup(read(cfg["r13_config"]));wrapper=make(read(cfg["r13_config"]),model)
    checkpoint=read(Path(cfg["r13_root"])/"main/result.json")["checkpoint"]
    checkpoint_load(read(cfg["r13_config"]),wrapper,checkpoint,assets)
    digest=wrapper.digest()
    for p in wrapper.parameters():p.requires_grad_(False)
    records=[]
    for i,b in enumerate(bindings):
        z=np.load(checked(b["arrays"]));other=np.load(checked(bindings[(i+1)%len(bindings)]["arrays"]))
        origin=z["native"][0,4:7].copy();origin[1]=0
        wrong=align_donor(other["native"][:150],z["native"][:150])
        wrong=MotionAdapter.to_edge(wrong,origin)*data.scale+data.bias
        truth=torch.tensor(z["encoded"][:150][None],device="cuda",dtype=torch.float32)
        for arm in cfg["arms"]:
            folder=out/(str(i)+"_"+arm);folder.mkdir()
            music=z["music"][:150] if arm!="C1_wrong_music" else other["music"][:150]
            c=torch.tensor(music[None],device="cuda")
            if arm=="C0":
                hard,_=masks(truth,60,90);known=hard.clone();value=torch.where(known,truth,torch.zeros_like(truth));guidance="hard"
            else:
                value,known=patch_control(z["values"],z["known"],z["edge"],z["event"],wrong,arm)
                if not np.isfinite(value[known]).all() or (abs(value[known])>1+32*np.finfo(np.float32).eps).any():raise ValueError("Unsupported condition, never clipped")
                value=torch.tensor(value[None],device="cuda");known=torch.tensor(known[None],device="cuda")
                hard=torch.zeros_like(known);guidance="none"
            raw=[];state_hashes=[];original=diffusion.model_predictions
            def forward(x,c,t,*args,**kwargs):
                before=torch.cuda.get_rng_state()
                p=wrapper(x,c,t,value,known)
                assert torch.equal(before,torch.cuda.get_rng_state())
                raw.append(p.cpu().numpy().copy());state_hashes.append(tensor_hash(x))
                return diffusion.predict_noise_from_start(x,t,p),p
            diffusion.model_predictions=forward
            try:
                torch.manual_seed(cfg["seed"]+i);torch.cuda.manual_seed_all(cfg["seed"]+i)
                with torch.no_grad():sample,receipt=constrained_ddim(diffusion,c,torch.where(hard,value,torch.zeros_like(value)),hard,guidance=guidance)
            finally:diffusion.model_predictions=original
            file=folder/"trace.npz";np.savez_compressed(file,raw=np.concatenate(raw),sample=sample.cpu().numpy(),
                condition=value.cpu().numpy(),known=known.cpu().numpy(),hard=hard.cpu().numpy())
            write(folder/"sampler.json",dict(receipt,x_t_hashes=state_hashes,contract="30-frame graft" if arm=="C0" else "all150 frames generated; no hard/pasted source poses",
                soft_observed_coordinates=int(known.sum()),seed=cfg["seed"]+i))
            if replay:
                old=Path(replay)/folder.name;same_arrays(file,old/"trace.npz")
                assert read(folder/"sampler.json")==read(old/"sampler.json")
            records.append(dict(case=i,arm=arm,trace=ref(file),sampler=ref(folder/"sampler.json")))
            print("R16_SAMPLE",i,arm,flush=True)
    assert wrapper.digest()==digest
    result=dict(bindings=records,full_window_sampler_calls=len(records),steps_per_call=50,
        optimizer_steps=0,model_unchanged=True,new_whole_song_generations=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,preflight=ref(Path(preflight)/"protocol.json"),
        checkpoint=checkpoint,donor_indices=[(i+1)%len(bindings) for i in range(len(bindings))],hidden_targets_as_conditions=False))
    if replay:
        assert [(b["case"],b["arm"]) for b in records]==[(b["case"],b["arm"]) for b in read(Path(replay)/"result.json")["bindings"]]

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--preflight",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.preflight,a.out,a.replay_from)
