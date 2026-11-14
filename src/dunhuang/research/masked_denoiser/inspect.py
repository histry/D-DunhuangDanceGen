"""Post-training read-only q-vs-own-policy, clipping, and raw-to-hard Root audit."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.contracts import Q
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.boundary_context.common import masks
from .common import setup,make,checkpoint_load
from .metrics import qmetrics
from dunhuang.research.condition_dependency.codec import origin

def run(cfg,root,out,replay=None):
    out=fresh(out);root=Path(root);forbid_optimizers()
    stages=["tiny"]+(["main"] if (root/"main/result.json").exists() else []);records=[];chains=[]
    for stage in stages:
        assets,_,tiny,monitor,_,data,model,diffusion=setup(cfg);wrapper=make(cfg,model)
        tr=read(root/stage/"result.json");checkpoint_load(cfg,wrapper,tr["checkpoint"],assets);before=wrapper.digest()
        bindings=[b for b in read(root/(stage+"_rollout")/"result.json")["bindings"] if b["arm"]=="MCD" and b["driver"]=="correct"]
        for b in bindings:
            i=b["case"];y,music=data.batch([b["row"]]);k,_=masks(y,60,90);z=np.load(checked(b["trace"]))
            target=y[0].cpu().numpy();old=data.native(b["row"])
            for j,timestep in enumerate(z["t"]):
                t=torch.tensor([timestep],device="cuda",dtype=torch.long)
                policy=torch.tensor(z["xt"][j:j+1],device="cuda")
                torch.manual_seed(cfg["seed"]+80000+i*4+j);torch.cuda.manual_seed_all(cfg["seed"]+80000+i*4+j)
                qt=diffusion.q_sample(y,t)
                for scope,xt in (("own_policy",policy),("fresh_q",qt)):
                    with torch.no_grad():raw=wrapper(xt,music,t,y,k,clip=False);p=raw.clamp(-1,1)
                    if scope=="own_policy":assert np.array_equal(p[0].cpu().numpy(),z["prediction"][j,0])
                    records.append(dict(stage=stage,case=i,t=int(timestep),scope=scope,
                        known_root_clipped_fraction=float((raw[:,:,4:7][k[:,:,4:7]].abs()>1).float().mean()),
                        known_pose_clipped_fraction=float((raw[:,:,7:][k[:,:,7:]].abs()>1).float().mean()),
                        clipped=qmetrics(p[0].cpu().numpy(),target,dict(scale=data.scale,bias=data.bias)),
                        unclipped_known_root_rmse_m=float((((raw[:,:,4:7]-y[:,:,4:7])/torch.tensor(data.scale[4:7],device="cuda"))[k[:,:,4:7]]).square().mean().sqrt())))
            sample=z["sample"][0];raw=z["raw"][-1]
            assert np.array_equal(sample[60:90],raw[60:90])
            assert abs(sample[k[0].cpu().numpy()]-target[k[0].cpu().numpy()]).max()<=2e-6
            r=((raw[:,4:7]-data.bias[4:7])/data.scale[4:7])@Q+origin(old)
            hard=((sample[:,4:7]-data.bias[4:7])/data.scale[4:7])@Q+origin(old)
            native=np.load(root/(stage+"_native")/(str(i)+"_MCD_correct")/"native.npy")
            assert abs(native[60:90,4:7]-hard[60:90]).max()<=2e-6
            for t in (60,90):
                edge=59 if t==60 else 90
                def step(x,t):return float(np.linalg.norm(x[t]-x[t-1]))
                chains.append(dict(stage=stage,case=i,seam=t,raw_internal_root_step_m=step(r,t),
                    raw_known_root_error_m=float(np.linalg.norm(r[edge]-old[edge,4:7])),
                    after_hard_root_step_m=step(hard,t),after_native_root_step_m=step(native[:,4:7],t),
                    post_codec_difference_m=abs(step(hard,t)-step(native[:,4:7],t))))
        assert wrapper.digest()==before
    result=dict(records=records,root_chains=chains,optimizer_steps=0,new_neural_generations=0,
        raw_unknown_matches_sample_exact=True,own_policy_prediction_recomputed_exact=True,
        no_post_codec_root_jump=all(c["post_codec_difference_m"]<=2e-6 for c in chains),operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,
        purpose="predeclared post-training diagnosis; no model/loss/budget choice",fresh_q_seed_rule="seed+80000+case*4+registered_state_index"))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R13_CLIP_AND_CHAIN",len(records),len(chains),result["no_post_codec_root_jump"],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)
