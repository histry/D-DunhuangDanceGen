"""Reuse R29 finite decoders; preserve local geometry without inventing Root truth."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import *
from dunhuang.research.time_r29.readout import Decoder,save,restore as load_decoder,objective
from dunhuang.research.state_r27.common import write,fresh,commit,vector,readout_check,forbid_optimizers,same_arrays

def run(cfg,out,replay=None):
    out=fresh(out)
    if replay:forbid_optimizers()
    pop,ledger,data,z=population(cfg);d,n,s,f,e,info,b=runtime(cfg);frozen=fingerprint(d,b)
    rows=[r for r in pop if r["domain"]=="aist"]+[r for r in pop if r["domain"]=="change"]
    bodies=[core_body(r["data"]) for r in rows];ref=reference(bodies)
    with torch.no_grad():
        b.set_reference(**ref);mem=b.reference.context[0]
        inputs=dict(raw=torch.cat((ref["features"].flatten(2),ref["valid"].float(),ref["clock"]),-1).detach(),
            memory=torch.cat((mem[:,1:],mem[:,:1].expand(-1,96,-1)),-1).detach())
    b.clear();y=torch.tensor(np.stack(bodies),device="cuda");arrays={};models={};allchecks=[]
    for domain in ("aist","change"):
        indices=[i for i,r in enumerate(rows) if r["domain"]=="aist" and r["role"]=="train"] if domain=="aist" else [i for i,r in enumerate(rows) if r["domain"]=="change"]
        assert len(indices)==(16 if domain=="aist" else 2)
        for view,x in inputs.items():
            name=domain+"_"+view;folder=out/name;folder.mkdir()
            torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"]);model=Decoder(x.shape[-1],cfg["readout_width"]).cuda();initial=digest(model)
            obs=[];actual=[];curves=[]
            def measure(step):
                with torch.no_grad():pred=model(x)
                arrays[f"{name}_{step}"]=pred.cpu().numpy()
                return dict(update=step,records=[dict(uid=r["uid"],role=r["role"],domain=r["domain"],fitted=i in indices,
                    metrics=local_metrics(pred[i],bodies[i],core_points(r["data"]),s,f),prediction_outside_box_fraction=float((pred[i].abs()>1).float().mean()))
                    for i,r in enumerate(rows)])
            curves.append(measure(0))
            if not replay:
                save(model,folder/"initial.pt");opt=torch.optim.AdamW(model.parameters(),lr=cfg["readout_lr"])
                for step in range(cfg["readout_steps"]):
                    watch=step in cfg["readout_snapshots"]
                    if watch:save(model,folder/f"{step}_pre.pt");before=vector(model).clone()
                    opt.zero_grad(set_to_none=True);loss=objective(model,x[indices],y[indices]);loss.backward()
                    gradient=float(sum(p.grad.square().sum() for p in model.parameters()).sqrt())
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                    if watch:
                        save(model,folder/f"{step}_post.pt");obs.append(dict(step=step,loss=float(loss.detach()),gradients={"decoder":gradient},update_norm=float((vector(model)-before).norm())))
                    if step+1 in (300,600):save(model,folder/f"weights_{step+1}.pt");curves.append(measure(step+1))
                save(model,folder/"decoder.pt");steps=cfg["readout_steps"]
            else:
                old=read(Path(replay)/"result.json")["models"][name];assert initial==old["initial_digest"]
                for fn,h in old["files"].items():assert sha(Path(replay)/name/fn)==h
                for item in old["observations"]:
                    step=item["step"];load_decoder(model,Path(replay)/name/f"{step}_pre.pt");before=vector(model).clone();model.zero_grad(set_to_none=True)
                    loss=objective(model,x[indices],y[indices]);loss.backward();gradient=float(sum(p.grad.square().sum() for p in model.parameters()).sqrt())
                    load_decoder(model,Path(replay)/name/f"{step}_post.pt");value=dict(step=step,loss=float(loss.detach()),gradients={"decoder":gradient},update_norm=float((vector(model)-before).norm()))
                    allchecks.append(readout_check(value,item));actual.append(value)
                for curve in old["curves"][1:]:
                    load_decoder(model,Path(replay)/name/f"weights_{curve['update']}.pt");current=measure(curve["update"]);assert current==curve;curves.append(current)
                load_decoder(model,Path(replay)/name/"decoder.pt");steps=0;obs=old["observations"]
            models[name]=dict(initial_digest=initial,final_digest=digest(model),parameters=sum(p.numel() for p in model.parameters()),
                input_channels=x.shape[-1],optimizer_steps=steps,curves=curves,observations=obs,actual_readouts=actual,
                training_uids=[rows[i]["uid"] for i in indices],files=old["files"] if replay else {p.name:sha(p) for p in folder.glob("*.pt")})
            print("R32_READOUT",name,steps,flush=True)
    def means(name):
        r=[x["metrics"] for x in models[name]["curves"][-1]["records"] if x["domain"]=="change"]
        return {k:float(np.mean([v[k] for v in r])) for k in ("body22_rad","pelvis_wrist_rmse_m")}
    raw=means("change_raw");memory=means("change_memory")
    raw_ok=raw["body22_rad"]<.20 and raw["pelvis_wrist_rmse_m"]<.10
    memory_gap=memory["body22_rad"]>max(.20,raw["body22_rad"]+.10) or memory["pelvis_wrist_rmse_m"]>max(.10,raw["pelvis_wrist_rmse_m"]+.05)
    assert_frozen(d,b,frozen);assert all(p.grad is None for p in b.parameters());b.close()
    np.savez_compressed(out/"arrays.npz",**arrays)
    result=dict(models=models,replay_checks=allchecks,decoder_optimizer_steps=sum(x["optimizer_steps"] for x in models.values()),
        generator_optimizer_steps=0,change_readout=dict(raw=raw,memory=memory),encoding_repair_indicated=raw_ok and memory_gap,
        partial_training_readout_admitted=raw_ok and not memory_gap,Root_translation_only_FK_gauge=True,
        scope="FINITE_TWO_EXPOSED_PRIMITIVE_DECODABILITY_NOT_NEW_SOURCE_GENERALIZATION_OR_GENERATION",operational_qualified=False)
    if replay:
        same_arrays(out/"arrays.npz",Path(replay)/"arrays.npz")
        assert all(models[k]["final_digest"]==read(Path(replay)/"result.json")["models"][k]["final_digest"] for k in models)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,arrays_sha256=sha(out/"arrays.npz")))
    print("R32_READOUT_GATE",raw_ok,memory_gap,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)
