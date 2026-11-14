"""Read-only Train-population feature-scale probe of final saved adapters; no selection."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.boundary_context.common import Data,fixed_rows,load_model,digest,masks
from .balanced import make
from .train import shifted_context

def run(training,out):
    training=Path(training);out=fresh(out);arms=["C0","C1"] if (training/"C0").exists() else ["C2"];cfg=read(training/arms[0]/"protocol.json")["config"];pf=Path(cfg["preflight"])
    assets=read(checked(read(pf/"protocol.json")["assets"]));data=Data(np.load(checked(assets["scaler"])))
    rows=fixed_rows(read(pf/"rows.json"),"train",2,cfg["data_seed"]);target,music=data.batch(rows)
    model,diffusion,_=load_model(assets);before=digest(model);known,_=masks(target,60,90);t=torch.tensor([900,999],device="cuda")
    blocks=dict(root_position=(4,7),rotation=(7,151),observation_mask=(151,302),root_velocity=(306,309),angular_velocity=(309,453),root_acceleration=(457,460),angular_acceleration=(460,604),position=(604,605),diffusion_time=(605,606))
    records=[]
    torch.manual_seed(cfg["seed"]+331);torch.cuda.manual_seed_all(cfg["seed"]+331)
    with torch.no_grad():
        encoded=model.cond_encoder(model.abs_pos_encoding(model.cond_projection(music)))
        xt=diffusion.q_sample(target,t)
        unconditional=model(xt,music,t,cond_drop_prob=1.)
        for arm in arms:
            payload=torch.load(checked(read(training/arm/"result.json")["checkpoint"]),map_location="cpu",weights_only=True)
            adapter=make(payload,data.scale,data.bias).cuda().eval();adapter.load_state_dict(payload["state"],strict=True)
            features,_=adapter.features(target,known,t);scales={}
            for name,(a,b) in blocks.items():
                f=features[:,:,a:b];part=f@adapter.input.weight[:,a:b].T
                scales[name]=dict(input_rms=float(f.square().mean().sqrt()),input_linear_contribution_rms=float(part.square().mean().sqrt()))
            v,_,supported=shifted_context(target,known,60,90,torch.as_tensor(data.scale,device="cuda"),0,cfg["root_shift_m"])
            actual=adapter(target,known,encoded,t);shifted=adapter(v,known,encoded,t);wrong=adapter(target.roll(1,0),known,encoded,t)
            state={"value":target}
            handle=model.cond_encoder.register_forward_hook(lambda m,a,encoded:encoded+adapter(state["value"],known,encoded,t))
            conditional=model(xt,music,t,cond_drop_prob=0.);raw=unconditional+2*(conditional-unconditional)
            state["value"]=v
            cfconditional=model(xt,music,t,cond_drop_prob=0.);cfraw=unconditional+2*(cfconditional-unconditional)
            handle.remove()
            scale=torch.as_tensor(data.scale[4:7],device="cuda")
            clip=dict(root_fraction=float((abs(raw[:,60:90,4:7])>=1).float().mean()),rotation_fraction=float((abs(raw[:,60:90,7:])>=1).float().mean()),
                root_delta_before_clip_rms_m=float((((cfraw-raw)[:,60:90,4:7]/scale)**2).mean().sqrt()),
                root_delta_after_clip_rms_m=float((((cfraw.clamp(-1,1)-raw.clamp(-1,1))[:,60:90,4:7]/scale)**2).mean().sqrt()))
            records.append(dict(arm=arm,blocks=scales,clip=clip,root_shift_supported=supported,residual_rms=float(actual.square().mean().sqrt()),
                root_shift_residual_delta_rms=float((shifted-actual).square().mean().sqrt()),wrong_residual_delta_rms=float((wrong-actual).square().mean().sqrt())))
    assert before==digest(model)
    write(out/"result.json",dict(records=records,optimizer_steps=0,neural_generations=0,used_for_model_selection=False,
        interpretation="feature scale and fixed-weight sensitivity on two registered Train windows; large blocks suggest a hypothesis, not causal proof or tuning authorization"))
    write(out/"protocol.json",dict(code_commit=commit(),rows=rows,training=str(training),no_validation_selection=True))
    print("R10_FEATURE_PROBE",records,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--training",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(a.training,a.out)
