"""No weight updates: matched L2 input interventions on saved R14 self-policy states."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.contracts import MotionAdapter
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.boundary_context.common import masks
from dunhuang.research.masked_denoiser.common import setup,make,checkpoint_load
from dunhuang.research.boundary_state_r14.readout import donors
from .common import source
from .geometry import input_mask,shuffled_target
from .objectives import parts,fk

def run(cfg,preflight,out,replay=None):
    out=fresh(out);forbid_optimizers();assets,bindings,_,_=source(cfg)
    oldcfg=read(read(cfg["r14_config"])["r13_config"])
    _,_,_,_,_,data,model,_=setup(oldcfg);wrapper=make(oldcfg,model)
    checkpoint=read(Path(read(cfg["r14_config"])["r13_root"])/"main/result.json")["checkpoint"]
    checkpoint_load(oldcfg,wrapper,checkpoint,assets);digest=wrapper.digest()
    for p in wrapper.parameters():p.requires_grad_(False)
    rows=[b["row"] for b in bindings];mapping=donors(rows,cfg["seed"])
    sk=np.load(checked(read(Path(preflight)/"protocol.json")["skeleton"]));parents=list(map(int,sk["parents"]))
    offsets=torch.tensor(sk["offsets"],device="cuda",dtype=torch.float32)
    scale=torch.tensor(data.scale,device="cuda");bias=torch.tensor(data.bias,device="cuda")
    parity=np.load(checked(read(Path(preflight)/"protocol.json")["target_fk"]));records=[];refs={}
    for i,b in enumerate(bindings):
        y,music=data.batch([b["row"]])
        known,_=masks(y,60,90);native=data.native(b["row"]);o=native[0,4:7].copy();o[1]=0
        origin=torch.tensor(o[None],device="cuda")
        shuffled=shuffled_target(native,data.native(rows[int(mapping[i])]))
        wrong=torch.tensor((MotionAdapter.to_edge(shuffled,o)*data.scale+data.bias)[None],device="cuda",dtype=torch.float32)
        assert np.max(abs(fk(y,scale,bias,origin,parents,offsets)[0].cpu().numpy()-parity["capacity_train_"+str(i)]))<=1e-5
        z=np.load(b["trace"]);saved={}
        for timestep in cfg["timesteps"]:
            j=int(np.flatnonzero(z["t"]==timestep)[0]);initial=torch.tensor(z["xt"][j:j+1],device="cuda")
            t=torch.tensor([timestep],device="cuda",dtype=torch.long)
            with torch.no_grad():base=wrapper(initial,music,t,y,known)
            # Exact frozen R14 J0 compatibility is required.
            expected=np.load(Path(cfg["r14_root"])/"capacity"/("case_"+str(i))/"probes.npz")["baseline_"+str(timestep)]
            assert np.array_equal(base[0].cpu().numpy(),expected)
            saved["J0_"+str(timestep)]=base[0].cpu().numpy()
            for arm in cfg["arms"][1:]:
                x=initial.clone().requires_grad_(True);pred=wrapper(x,music,t,y,known)
                target=wrong if arm=="J4" else y
                # J4 shuffles only oracle unknown labels. Real hard context is always y.
                target=torch.where(known,y,target)
                loss_parts=parts(pred,target,scale,bias,origin,parents,offsets)
                value=loss_parts["root"]+(loss_parts["world_foot"] if arm in ("J3","J4") else 0)
                grad,=torch.autograd.grad(value,x)
                permission=torch.tensor(input_mask(tuple(x.shape),arm,cfg["leg_joints"]),device="cuda")
                grad=torch.where(permission,grad,torch.zeros_like(grad));norm=grad.norm()
                direction=-grad/norm.clamp(min=1e-20)
                eps=cfg["derivative_epsilon"]
                def objective(p):
                    ls=parts(p,target,scale,bias,origin,parents,offsets)
                    return ls["root"]+(ls["world_foot"] if arm in ("J3","J4") else 0)
                with torch.no_grad():
                    high=objective(wrapper(x+eps*direction,music,t,y,known))
                    low=objective(wrapper(x-eps*direction,music,t,y,known))
                    numeric=(high-low)/(2*eps);analytic=(grad*direction).sum()
                    relative=float(abs(numeric-analytic)/torch.maximum(abs(numeric),abs(analytic)).clamp(min=1e-6))
                    trials=[]
                    for k,fraction in enumerate(cfg["line_fractions"]):
                        delta=direction*cfg["root_input_l2"]*fraction
                        assert not delta[~permission].any()
                        q=wrapper(x+delta,music,t,y,known)
                        saved[arm+"_"+str(timestep)+"_"+str(k)]=q[0].cpu().numpy()
                        trials.append(dict(trial=k,fraction=fraction,input_l2=float(delta.norm()),objective=float(objective(q))))
                records.append(dict(case=i,arm=arm,t=timestep,gradient_norm=float(norm),objective_before=float(value.detach()),
                    derivative_analytic=float(analytic),derivative_numeric=float(numeric),derivative_relative_error=relative,trials=trials,
                    unknown_target_scope="TRAIN_PAIRED_ORACLE",input_coordinates=int(permission.sum())))
        path=out/("case_"+str(i)+".npz");np.savez_compressed(path,**saved);refs[str(i)]=ref(path)
        if replay:same_arrays(path,Path(replay)/path.name)
        print("R15_PROBES",i,flush=True)
    assert wrapper.digest()==digest
    result=dict(records=records,probes=refs,model_unchanged=True,optimizer_steps=0,new_full_window_generations=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,rows=rows,donors=mapping.tolist(),
        checkpoint=checkpoint,target_role="Train capacity oracle; NEVER inference labels",trace_inputs=[ref(b["trace"]) for b in bindings]))
    if replay:
        previous=read(Path(replay)/"result.json")
        assert result["records"]==previous["records"]

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--preflight",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.preflight,a.out,a.replay_from)
