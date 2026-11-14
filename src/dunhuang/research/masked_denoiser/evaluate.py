"""Fixed 50-step Train readout; saved final weights only; no optimizer."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.motion_bridge.model import Injection,MotionBridge,music_tokens
from .common import setup,make,checkpoint_load
from .rollout import sample,replay_arrays

def run(cfg,training,out,stage="tiny",replay=None):
    out=fresh(out);forbid_optimizers()
    assets,_,tiny,monitors,_,data,model,diffusion=setup(cfg);wrapper=make(cfg,model)
    trained=read(Path(training)/"result.json");checkpoint_load(cfg,wrapper,trained["checkpoint"],assets);before=wrapper.digest();frozen=wrapper.digest(True)
    population=tiny if stage=="tiny" else monitors
    bindings=[]
    for i,b in enumerate(population):
        y,music=data.batch([b["row"]]);wrong,_=data.batch([population[(i+1)%len(population)]["row"]])
        drivers=("correct","removed","wrong_left","wrong_right") if i<2 else ("correct",)
        for driver in drivers:
            folder=out/(str(i)+"_MCD_"+driver)
            result=sample(diffusion,wrapper,y,music,wrong,b["seed"],cfg,folder,driver=driver)
            assert result["receipt"]["noise_sha256"]==b["noise_sha256"]
            bindings.append(dict(case=i,arm="MCD",driver=driver,row=b["row"],trace=ref(folder/"trace.npz"),seed=b["seed"]))
            if replay:replay_arrays(folder,Path(replay)/folder.name)
            print("R13_ROLLOUT",stage,i,"MCD",driver,flush=True)
    assert wrapper.digest()==before and wrapper.digest(True)==frozen
    # Frozen B3 is a historical baseline, not an equal-training-data architecture ablation.
    _,_,_,_,_,_,oldmodel,olddiff=setup(cfg)
    r12=read(Path(cfg["r12_root"])/"B3/result.json");p=torch.load(checked(r12["checkpoint"]),map_location="cpu",weights_only=True)
    bridge=MotionBridge(p["state"]["scale"],p["state"]["bias"],p["state"]["inv_rms"],"B3",.1).cuda().eval();bridge.load_state_dict(p["state"],strict=True)
    for parameter in bridge.parameters():parameter.requires_grad_(False)
    hook=Injection(oldmodel,p["site"])
    def predict(x,c,t,v,k):return hook.predict(x,c,t,bridge(v,k,x,music_tokens(oldmodel,c),t))
    for i,b in enumerate(population):
        y,music=data.batch([b["row"]]);wrong,_=data.batch([population[(i+1)%len(population)]["row"]])
        folder=out/(str(i)+"_R12B3_correct");sample(olddiff,predict,y,music,wrong,b["seed"],cfg,folder)
        bindings.append(dict(case=i,arm="R12B3",driver="correct",row=b["row"],trace=ref(folder/"trace.npz"),seed=b["seed"]))
        if replay:replay_arrays(folder,Path(replay)/folder.name)
        print("R13_ROLLOUT",stage,i,"R12B3",flush=True)
    hook.close()
    write(out/"result.json",dict(bindings=bindings,optimizer_steps=0,new_neural_generations=len(bindings),operational_qualified=False))
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,training=ref(Path(training)/"result.json"),model=trained["checkpoint"],
        baseline=r12["checkpoint"],population_role="Train fit memorization" if stage=="tiny" else "heldout Train music groups",
        independent_cf_rollouts="First two cases: correct/removed/wrong-left/wrong-right; fixed correct hard anchors; all 50 steps",
        same_state_controls="Six denoiser-input controls on four fixed correct-driven states; do not generalize to all 50 steps"))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--training",required=True);p.add_argument("--out",required=True);p.add_argument("--stage",choices=["tiny","main"],default="tiny");p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.training,a.out,a.stage,a.replay_from)
