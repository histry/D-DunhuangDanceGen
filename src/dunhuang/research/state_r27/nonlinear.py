"""Conditional nonlinear memory probes; these are not motion generators."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import *
from .train import dataset

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out)
    assert read(root/"decision.json")["route"]=="capacity"
    if replay:forbid_optimizers()
    d,n,s,f,e,info,adapter=runtime(cfg);z=dataset(root);target=torch.tensor(z["target"],device="cuda")
    rows=[];arrays={};snapshots=[];actual=[];steps=0
    for label,path in [("R26_C1",Path(cfg["r26"])/"models/C1/adapter.pt"),("P",root/"models/P/adapter.pt")]:
        adapter.load_state_dict(torch.load(path,map_location="cpu",weights_only=True)["state"],strict=True)
        with torch.no_grad():
            adapter.set_reference(**packet(list(z["raw139"][:,64:160,7:])))
            memory=adapter.context[0][:,1:].detach();adapter.clear()
        torch.manual_seed(cfg["seed"]+900);torch.cuda.manual_seed_all(cfg["seed"]+900)
        decoder=nn.Sequential(nn.Linear(64,256),nn.SiLU(),nn.Linear(256,132)).cuda()
        folder=out/label;folder.mkdir()
        def save(name):torch.save({k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()},folder/name)
        def restore_decoder(name):decoder.load_state_dict(torch.load(name,map_location="cpu",weights_only=True),strict=True)
        def report(update):
            with torch.no_grad():
                pred=decoder(memory);full=target.clone();full[:,64:160,7:]=pred
                values=decoded_metrics(full,target,n,s,f)
                rows.append(dict(memory=label,update=update,mse=float((pred-target[:,64:160,7:]).square().mean()),metrics=values))
                arrays[f"{label}_{update}"]=pred.cpu().numpy()
        report(0)
        if not replay:
            optimizer=torch.optim.AdamW(decoder.parameters(),lr=cfg["nonlinear_learning_rate"])
            save("initial.pt")
            for i in range(cfg["nonlinear_steps"]):
                watch=i in (0,299,599);before=torch.cat([p.detach().flatten() for p in decoder.parameters()]).clone() if watch else None
                if watch:save(f"{i}_pre.pt")
                optimizer.zero_grad(set_to_none=True);loss=(decoder(memory)-target[:,64:160,7:]).square().mean();loss.backward()
                g=float(sum(p.grad.square().sum() for p in decoder.parameters()).sqrt()) if watch else None
                optimizer.step();steps+=1
                if watch:
                    save(f"{i}_post.pt");after=torch.cat([p.detach().flatten() for p in decoder.parameters()])
                    snapshots.append(dict(memory=label,step=i,loss=float(loss.detach()),gradients=dict(decoder=g),update_norm=float((after-before).norm())))
                if i+1 in (200,600):save(f"weights_{i+1}.pt");report(i+1)
            save("decoder.pt")
        else:
            old=read(Path(replay)/"result.json")
            for item in [x for x in old["observations"] if x["memory"]==label]:
                i=item["step"];restore_decoder(Path(replay)/label/f"{i}_pre.pt")
                before=torch.cat([p.detach().flatten() for p in decoder.parameters()]).clone();decoder.zero_grad(set_to_none=True)
                loss=(decoder(memory)-target[:,64:160,7:]).square().mean();loss.backward();g=float(sum(p.grad.square().sum() for p in decoder.parameters()).sqrt())
                restore_decoder(Path(replay)/label/f"{i}_post.pt");after=torch.cat([p.detach().flatten() for p in decoder.parameters()])
                row=dict(memory=label,step=i,loss=float(loss.detach()),gradients=dict(decoder=g),update_norm=float((after-before).norm()))
                readout_check(row,item);actual.append(row)
            for update in (200,600):restore_decoder(Path(replay)/label/f"weights_{update}.pt");report(update)
            restore_decoder(Path(replay)/label/"decoder.pt")
        assert all(p.grad is None for p in adapter.parameters())
        print("R27_NONLINEAR_PROBE",label,flush=True)
    np.savez_compressed(out/"arrays.npz",**arrays)
    result=dict(records=rows,observations=snapshots if not replay else old["observations"],actual_readouts=actual,
        neural_optimizer_steps=steps,generator_optimizer_steps=0,decoder_is_not_a_generator=True,teacher_targets_are_model_predictions=True,
        files={str(p.relative_to(out)):sha(p) for p in out.rglob("*.pt")} if not replay else old["files"])
    if replay:
        assert rows==old["records"];same_arrays(out/"arrays.npz",Path(replay)/"arrays.npz")
        for name,h in old["files"].items():assert sha(Path(replay)/name)==h
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg))
    adapter.close()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)
