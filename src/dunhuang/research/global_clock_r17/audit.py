"""Independent frozen Mask factors and music-policy output response before training."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.sampling import constrained_ddim
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.masked_denoiser.common import tensor_hash
from .contract import factor_conditions,metadata
from .engine import load,tensors,observed,global_sample

def run(cfg,data_root,out,replay=None):
    out=fresh(out);forbid_optimizers();population=read(Path(data_root)/"result.json")
    records=[r for r in population["records"] if r["role"]=="eval"]
    wrapper,diffusion,data,predict,_=load(cfg);digest=wrapper.digest();factors=[];music_records=[]
    old=read(Path(cfg["r16_root"])/"preflight/protocol.json")["pairs"]
    for record in records:
        i=record["case"];z,y,m=tensors(record);oldz=np.load(checked(old[i]["arrays"]))
        for position in ("mid","ends"):
            for hard_on in (False,True):
                for event_on in (False,True):
                    name=str(i)+"_"+position+"_"+str(int(hard_on))+"_"+str(int(event_on));folder=out/name;folder.mkdir()
                    v,k,h=factor_conditions(z["encoded"][:150],position,hard_on,event_on,oldz["event"])
                    value=torch.tensor(v[None],device="cuda");known=torch.tensor(k[None],device="cuda");hard=torch.tensor(h[None],device="cuda")
                    raw=[];ledger=[];original=diffusion.model_predictions
                    def forward(x,c,t,*args,**kwargs):
                        p=wrapper(x,c,t,value,known);raw.append(p.detach().cpu().numpy().copy())
                        ledger.append(dict(t=int(t[0]),x_t=tensor_hash(x),raw=tensor_hash(p)))
                        return diffusion.predict_noise_from_start(x,t,p),p
                    diffusion.model_predictions=forward
                    try:
                        torch.manual_seed(cfg["seed"]+i);torch.cuda.manual_seed_all(cfg["seed"]+i)
                        with torch.no_grad():sample,receipt=constrained_ddim(diffusion,m[:1],torch.where(hard,value,0),hard,guidance="hard" if hard_on else "none")
                    finally:diffusion.model_predictions=original
                    np.savez_compressed(folder/"trace.npz",sample=sample[0].cpu().numpy(),raw_final=raw[-1][0],condition=v,known=k,hard=h)
                    write(folder/"sampler.json",dict(receipt,steps=ledger))
                    if replay:
                        other=Path(replay)/name;same_arrays(folder/"trace.npz",other/"trace.npz");assert read(folder/"sampler.json")==read(other/"sampler.json")
                    factors.append(dict(case=i,position=position,hard=hard_on,event=event_on,trace=ref(folder/"trace.npz"),sampler=ref(folder/"sampler.json")))
        value,known,hard=observed(y,"free");meta=torch.tensor(metadata()[None],device="cuda")
        for policy,mus in (("owned",m),("independent",torch.tensor(z["original_music"],device="cuda"))):
            name=str(i)+"_music_"+policy;folder=out/name;folder.mkdir()
            arrays,receipt=global_sample(diffusion,predict,mus,value,known,hard,meta,cfg["seed"]+100+i)
            np.savez_compressed(folder/"trace.npz",**arrays);write(folder/"sampler.json",receipt)
            if replay:
                other=Path(replay)/name;same_arrays(folder/"trace.npz",other/"trace.npz");assert receipt==read(other/"sampler.json")
            music_records.append(dict(case=i,policy=policy,trace=ref(folder/"trace.npz"),sampler=ref(folder/"sampler.json")))
        print("R17_FROZEN_AUDIT",i,flush=True)
    assert wrapper.digest()==digest;predict.close()
    write(out/"result.json",dict(factors=factors,music=music_records,optimizer_steps=0,model_unchanged=True,
        sampler_calls=len(factors)+len(music_records),r16_gate_unchanged=True,operational_qualified=False))
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,data=ref(Path(data_root)/"protocol.json"),
        factors="Position2 x boundary execution2 x event source2; all150 generated, no parent graft",
        training_admission="Only authenticated continuous data/numerical interfaces; R16 gate is not promoted",
        music_policy="Pre-registered owned225; independent crop output effect is descriptive, not used to choose policy"))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--data",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.data,a.out,a.replay_from)
