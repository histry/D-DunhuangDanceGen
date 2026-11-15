"""Real frozen Lodge graph, trained source/phase interventions, and null-reference identity."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import write,sha
from .common import runtime,dataset,packet,install,restore,read,fresh,commit,forbid_optimizers,same_arrays
from .model import digest

def run(cfg,root,out,trained=False,replay=None):
    out=fresh(out);forbid_optimizers();d,n,s,f,e,info,adapter=runtime(cfg);adapter.attach(d.model);pop,data=dataset(root)
    cases=pop["train_cases"][:2];y=torch.tensor(np.stack([data[c]["target"] for c in cases]),device="cuda")
    music=torch.tensor(np.stack([data[c]["music"] for c in cases]),device="cuda");t=torch.tensor([519,519],device="cuda")
    torch.manual_seed(cfg["seed"]+80);xt=d.q_sample(y,t)
    adapter.clear()
    with torch.no_grad():baseline=d.model(xt,music,None,t,cond_drop_prob=0)
    records=[];arrays={"baseline":baseline.cpu().numpy()};frozen=digest(d.model)
    for arm in cfg["arms"]:
        adapter.phase=arm=="phase"
        if trained:
            assert sha(Path(root)/f"models/{arm}/adapter.pt")==read(Path(root)/f"models/{arm}/result.json")["checkpoint_hashes"]["adapter.pt"]
            restore(adapter,Path(root)/f"models/{arm}/adapter.pt",cfg,arm)
        for mode in ("correct","removed","reference_swapped","phase_removed","phase_wrong"):
            donor=cases[::-1] if mode=="reference_swapped" else cases
            p=packet([data[c]["raw139"][64:160,7:] for c in donor],drops=[mode=="removed"]*2,
                control="removed" if mode=="phase_removed" else "wrong" if mode=="phase_wrong" else "correct")
            install(adapter,p);adapter.capture=True
            with torch.no_grad():pred=d.model(xt,music,None,t,cond_drop_prob=0)
            arrays[arm+"_"+mode]=pred.cpu().numpy()
            if mode=="removed" or not trained:assert torch.equal(pred,baseline)
            records.append(dict(arm=arm,mode=mode,output_rms_change_from_baseline=float((pred-baseline).square().mean().sqrt()),layers=dict(adapter.readouts)))
            adapter.clear()
    assert digest(d.model)==frozen
    path=out/"readouts.npz";np.savez_compressed(path,**arrays)
    result=dict(records=records,zero_initialization_exact=not trained,null_reference_exact=True,trained=trained,optimizer_steps=0)
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(path,Path(replay)/"readouts.npz")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,checkpoint=info))
    adapter.close()
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--trained",action="store_true");p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.trained,a.replay_from)
