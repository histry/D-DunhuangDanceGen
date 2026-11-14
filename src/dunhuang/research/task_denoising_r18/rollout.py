"""Full global generation with explicit interventions; immutable generated-history task."""
import argparse,hashlib
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.global_clock_r17.engine import tensors,observed
from dunhuang.research.global_clock_r17.contract import metadata
from dunhuang.research.masked_denoiser.common import tensor_hash
from .engine import load,population

def sample(cfg,d,predict,m,v,k,h,meta,seed,intervention="correct"):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);noise_hash=hashlib.sha256()
    def draw(x):noise_hash.update(x.detach().cpu().numpy().tobytes());return x
    x=draw(torch.randn_like(v));times=list(reversed(torch.linspace(-1,999,51).int().tolist()))
    steps=[];controls={}
    with torch.no_grad():
        for step,(t,nxt) in enumerate(zip(times[:-1],times[1:])):
            ts=torch.tensor([t],device=x.device);p,windows=predict(x,m,v,k,meta,ts,intervention)
            if step in (24,49):
                for control in cfg["controls"]:
                    q=p if control==intervention else predict(x,m,v,k,meta,ts,control)[0]
                    controls[str(t)+"_"+control]=q[0].cpu().numpy()
            steps.append(dict(t=t,raw_sha256=tensor_hash(p),x_t_sha256=tensor_hash(x),
                overlap_max=float(abs(windows[0,75:]-windows[1,:75]).max())))
            clean=torch.where(h,v,p)
            if nxt<0:x=clean;continue
            alpha=d.alphas_cumprod[t];na=d.alphas_cumprod[nxt]
            eps=(x-alpha.sqrt()*clean)/(1-alpha).sqrt()
            sigma=((1-alpha/na)*(1-na)/(1-alpha)).sqrt()
            noise=draw(torch.randn_like(x))
            x=na.sqrt()*clean+(1-na-sigma**2).clamp_min(0).sqrt()*eps+sigma*noise
            x=torch.where(h,na.sqrt()*v+(1-na).sqrt()*noise,x)
    assert torch.equal(x[h],v[h])
    return dict(sample=x[0].cpu().numpy(),raw_final=p[0].cpu().numpy(),condition=v[0].cpu().numpy(),
        known=k[0].cpu().numpy(),hard=h[0].cpu().numpy(),**controls),dict(steps=steps,noise_sha256=noise_hash.hexdigest(),
        intervention=intervention,all_edge_calls_local150=True,output_pose_paste=False)

def run(cfg,root,out,stage,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();pre,fit,ev=population(cfg)
    rows=[r for r in fit if r["case"] in cfg["tiny_cases"]] if stage=="tiny" else ev
    bindings=[]
    for arm,mode in (("D0",None),("D1","self"),("D2","unique")):
        reference=read(root/stage/mode/"result.json")["checkpoint"] if mode else None
        w,d,data,predict,head=load(cfg,mode,reference,adapt=bool(mode));before=w.digest()
        for row in rows:
            z,y,m=tensors(row);i=row["case"];meta=torch.tensor(metadata()[None],device="cuda")
            for task in cfg["tasks"]:
                prefix=None;prefix_ref=None
                if task=="progressive":
                    prefix_ref=ref(out/(str(i)+"_D0_free_correct")/"trace.npz")
                    prefix=np.load(checked(prefix_ref))["sample"][:75].copy()
                v,k,h=observed(y,task,prefix=prefix)
                for control in (("correct","content") if mode=="unique" and task!="progressive" else ("correct",)):
                    arrays,receipt=sample(cfg,d,predict,m,v,k,h,meta,87041+i*100+cfg["tasks"].index(task),control)
                    name=f"{i}_{arm}_{task}_{control}";folder=out/name;folder.mkdir()
                    path=folder/"trace.npz";np.savez_compressed(path,**arrays)
                    write(folder/"sampler.json",dict(receipt,prefix=prefix_ref))
                    if task=="free":assert not arrays["known"][:,:13].any()
                    if task=="progressive":
                        assert np.array_equal(arrays["sample"][:75,4:],prefix[:,4:]) and not arrays["known"][75:,:13].any()
                    if replay:
                        old=Path(replay)/name;same_arrays(path,old/"trace.npz")
                        a=read(folder/"sampler.json");b=read(old/"sampler.json");ap=a.pop("prefix");bp=b.pop("prefix")
                        assert a==b
                        if ap:assert ap["sha256"]==bp["sha256"]
                    bindings.append(dict(role=row["role"],case=i,arm=arm,task=task,control=control,trace=ref(path),sampler=ref(folder/"sampler.json")))
                    print("R18_ROLLOUT",stage,name,flush=True)
        assert w.digest()==before;predict.close()
    write(out/"result.json",dict(bindings=bindings,sampler_calls=len(bindings),optimizer_steps=0,
        stage=stage,new_whole_song_generations=0,operational_qualified=False))
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,stage=stage,
        parent_data=ref(Path(cfg["parent"])/"data/protocol.json"),prefix="Shared D0 generated first75, all committed",
        teacher_forcing_evaluation=False))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True)
    p.add_argument("--stage",choices=["tiny","main"],required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.root,a.out,a.stage,a.replay_from)
