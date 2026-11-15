"""Read-only reconstruction/physical gradient directions on nine actual pre states."""
import argparse
from pathlib import Path
import torch
from dunhuang.research.branch_r22.engine import load,cache_data,state,weights,model_meta
from dunhuang.research.branch_r22.objective import build
from dunhuang.research.objective_r19.common import restore,parameters,gradient
from .common import inputs,finish,read,fresh,forbid_optimizers

def direction(a,b):
    na=a.double().norm();nb=b.double().norm();dot=(a.double()*b.double()).sum()
    return dict(first_norm=float(na),second_norm=float(nb),dot=float(dot),
        cosine=float(dot/(na*nb).clamp_min(1e-30)),second_over_first=float(nb/na.clamp_min(1e-30)))
def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();root,pc,pop=inputs(cfg);records=[]
    for arm in cfg["gradient_arms"]:
        w,d,data,p,h=load(pc,root,arm);cache=cache_data(pc,root,d,data);ow,tw=weights(pc)
        model=read(root/f"models/{arm}/result.json")
        for step in cfg["gradient_steps"]:
            item=next(x for x in model["update_observations"] if x["step"]==step);restore(pc,w,h,item["pre"]);before=model_meta(w,h)
            ledger=model["ledger"][step];s,_,_=state(pc,cache,ledger["case"],ledger["t"],ledger["variant"],ledger["pattern"],ledger["seed"])
            pred,terms,counts,active,total=build(pc,p,cache,s,ow,tw,arm)
            ps=parameters(w,h);recon=active["unknown"]+active["observed"];physical={n:x for n,x in active.items() if n not in ("unknown","observed")}
            gr=gradient(recon,ps);gp=gradient(sum(physical.values()),ps)
            records.append(dict(arm=arm,step=step,case=ledger["case"],t=ledger["t"],pre=item["pre"],
                reconstruction_vs_physics=direction(gr,gp),per_physical={n:direction(gr,gradient(v,ps)) for n,v in physical.items()},
                optimizer_executed=False,model_unchanged=model_meta(w,h)==before))
            assert records[-1]["model_unchanged"]
        p.close()
    finish(out,dict(records=records,actual_pre_snapshots=9,optimizer_steps=0),cfg,replay)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.out,a.replay_from)
