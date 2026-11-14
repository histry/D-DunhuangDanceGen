"""Selected-parameter gradient compatibility; finite probes restored byte-exactly."""
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.contracts import Q
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.masked_denoiser.common import setup,make,checkpoint_load
from dunhuang.research.motion_bridge.model import rotations
from .common import population

def fk(normalized,scale,bias,origin,parents,offsets):
    x=(normalized-bias)/scale
    R=rotations(x[:,:,7:].reshape(-1,150,24,6));q=torch.tensor(Q,device=x.device,dtype=x.dtype)
    local=[q.T@R[:,:,0]]+[R[:,:,j] for j in range(1,24)]
    world=[];positions=[]
    for j,parent in enumerate(parents):
        if j==0:world.append(local[0]);positions.append(x[:,:,4:7]@q+origin[:,None])
        else:
            world.append(world[parent]@local[j])
            positions.append(positions[parent]+(world[parent]@offsets[j,:,None]).squeeze(-1))
    return torch.stack(positions,dim=2)

def losses(pred,target,scale,bias,origin,parents,offsets,known):
    raw=(pred-bias)/scale;gt=(target-bias)/scale
    root=(((raw[:,:,4:7]-gt[:,:,4:7])/.1).square())[known[:,:,4:7]].mean()
    P=rotations(raw[:,60:90,7:].reshape(-1,30,24,6));G=rotations(gt[:,60:90,7:].reshape(-1,30,24,6))
    pose=(P-G).square().sum((-1,-2)).mean()/.3**2
    edit=torch.zeros_like(known);edit[:,60:90,4:]=True
    # The actual candidate FK uses protected real outside frames. Raw Root loss above does not.
    graft=torch.where(edit,pred,target);points=fk(graft,scale,bias,origin,parents,offsets)
    truth=fk(target,scale,bias,origin,parents,offsets)
    feet=[7,8,10,11];dv=points[:,[60,90]][:,:,feet]-points[:,[59,89]][:,:,feet]
    dg=truth[:,[60,90]][:,:,feet]-truth[:,[59,89]][:,:,feet]
    foot=((dv-dg)/.1).square().sum(-1).max()
    jerk=torch.diff(points,n=3,dim=1)*30**3
    centers=torch.arange(147,device=pred.device)+1.5
    band=((centers>=57)&(centers<=63))|((centers>=87)&(centers<=93))
    peak=jerk[:,band].square().sum(-1).max()/1620**2
    return dict(root=root,source_pose=pose,foot_step=foot,peak_jerk=peak)

def run(cfg,audit,out,replay=None):
    out=fresh(out);forbid_optimizers();oldcfg=read(cfg["r13_config"])
    assets,_,_,_,_=population(cfg)
    _,_,_,_,_,data,model,diffusion=setup(oldcfg);wrapper=make(oldcfg,model)
    saved=read(Path(cfg["r13_root"])/"main/result.json");checkpoint_load(oldcfg,wrapper,saved["checkpoint"],assets)
    before=wrapper.digest();parameters=wrapper.selected();original=[p.detach().clone() for p in parameters]
    skeleton=np.load(checked(read(Path(audit)/"protocol.json")["skeleton"]))
    parents=[int(x) for x in skeleton["parents"]];offsets=torch.tensor(skeleton["offsets"],device="cuda",dtype=torch.float32)
    scale=torch.tensor(data.scale,device="cuda");bias=torch.tensor(data.bias,device="cuda")
    bindings=[b for b in saved["self_caches"] if b["update"]==250][:cfg["conflict_cases"]]
    parity=np.load(Path(audit)/"train_parity.npz");records=[]
    from dunhuang.research.boundary_context.common import masks
    for i,b in enumerate(bindings):
        y,music=data.batch([b["row"]]);known,_=masks(y,60,90);old=data.native(b["row"]);o=old[0,4:7].copy();o[1]=0
        origin=torch.tensor(o[None],device="cuda");z=np.load(checked(b["trace"]))
        assert np.max(abs(fk(y,scale,bias,origin,parents,offsets)[0].cpu().numpy()-parity[str(i)]))<=1e-5
        for timestep in cfg["conflict_timesteps"]:
            j=int(np.flatnonzero(z["t"]==timestep)[0]);xt=torch.tensor(z["xt"][j:j+1],device="cuda");t=torch.tensor([timestep],device="cuda",dtype=torch.long)
            p=wrapper(xt,music,t,y,known);parts=losses(p,y,scale,bias,origin,parents,offsets,known)
            grads=[];norms=[]
            for index,(name,value) in enumerate(parts.items()):
                g=torch.autograd.grad(value,parameters,retain_graph=index<3,allow_unused=True)
                flat=torch.cat([(torch.zeros_like(v) if w is None else w.detach()).ravel() for v,w in zip(parameters,g)])
                assert torch.isfinite(flat).all();norm=flat.norm();norms.append(float(norm));grads.append(flat/norm.clamp(min=1e-20))
            gs=torch.stack(grads);gram=(gs@gs.T).cpu().numpy().astype(float);active=np.flatnonzero(np.asarray(norms)>1e-12)
            w=np.zeros(4)
            if len(active):
                A=gram[np.ix_(active,active)]
                sol=minimize(lambda v:float(v@A@v),np.ones(len(active))/len(active),jac=lambda v:2*A@v,
                    constraints={"type":"eq","fun":lambda v:v.sum()-1,"jac":lambda v:np.ones_like(v)},
                    bounds=[(0.,1.)]*len(active),method="SLSQP",options={"maxiter":200,"ftol":1e-12})
                if not sol.success:raise ValueError("Common-direction simplex solve failed")
                w[active]=sol.x
            direction=-(torch.tensor(w,device="cuda",dtype=gs.dtype)@gs);rms=direction.square().mean().sqrt()
            baseline={n:float(v.detach()) for n,v in parts.items()};trials=[]
            for budget in cfg["parameter_probe_rms"]:
                delta=direction/rms.clamp(min=1e-20)*budget;cursor=0
                with torch.no_grad():
                    for param,base in zip(parameters,original):
                        count=param.numel();param.copy_(base+delta[cursor:cursor+count].reshape_as(param));cursor+=count
                    after=losses(wrapper(xt,music,t,y,known),y,scale,bias,origin,parents,offsets,known)
                    actual={n:float(v) for n,v in after.items()}
                    for param,base in zip(parameters,original):param.copy_(base)
                trials.append(dict(parameter_rms=budget,losses=actual,all_nonregression=all(actual[k]<=v*(1+1e-5)+1e-10 for k,v in baseline.items())))
            assert wrapper.digest()==before
            records.append(dict(case=i,role="Train-fit cached policy250 states evaluated by frozen final400 model",t=timestep,
                losses=baseline,gradient_norms=dict(zip(parts,norms)),normalized_gradient_gram=gram.tolist(),
                simplex_weights=w.tolist(),normalized_first_order_descent=(gram@w).tolist(),trials=trials))
        print("R14_CONFLICT",i,flush=True)
    result=dict(records=records,optimizer_steps=0,finite_parameter_interventions_restored_exact=True,backbone_unchanged=True,
        selected_parameter_dimension=sum(p.numel() for p in parameters),output_space_only=False,new_neural_generations=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,r13_model=saved["checkpoint"],
        audit=ref(Path(audit)/"protocol.json"),data_scope="Eight existing Train-fit cached policy250 windows; no old monitor or Val used for directions",
        labels="paired targets used for training-objective diagnostics only; no contact/support truth"))
    if replay:assert result==read(Path(replay)/"result.json")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--audit",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.audit,a.out,a.replay_from)
