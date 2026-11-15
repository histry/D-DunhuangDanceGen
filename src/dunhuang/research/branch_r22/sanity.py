"""Real initialization, gradients and unchanged inference, before optimization."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,fresh,forbid_optimizers,same_arrays,commit
from dunhuang.research.objective_r19.common import parameters,gradient
from dunhuang.research.context_r21.objective import build as old_build
from dunhuang.research.context_r21.sampler import sample
from .engine import load,cache_data,state,weights,model_meta
from .objective import build

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    w,d,data,p,h=load(cfg,root,"G0");cache=cache_data(cfg,root,d,data);ow,tw=weights(cfg)
    s,_,_=state(cfg,cache,cache["population"]["train_probe"][0],519)
    before=model_meta(w,h);records={}
    for arm in cfg["arms"]:
        pred,terms,counts,active,obj=build(cfg,p,cache,s,ow,tw,arm)
        raw=list(p.branches);raw_grad=torch.autograd.grad(obj,raw,retain_graph=True)
        record=dict(objective=float(obj.detach()),unknown=float(terms["unknown"].detach()),guided_unknown=float(terms["guided_unknown"].detach()),
            raw_branch_gradient_norms=[float(x.norm()) for x in raw_grad],parameter_gradient_norm=float(gradient(obj,parameters(w,h)).norm()))
        assert all(np.isfinite(record["raw_branch_gradient_norms"])) and min(record["raw_branch_gradient_norms"])>0
        assert record["parameter_gradient_norm"]>0
        if arm=="G0":
            old=old_build(cfg,p,cache,s,ow,tw,balanced=True)
            assert torch.equal(obj,old[-1])
            assert torch.equal(gradient(obj,parameters(w,h)),gradient(old[-1],parameters(w,h)))
        records[arm]=record
    assert len({r["guided_unknown"] for r in records.values()})==1
    # Capturing live raw branches must have no effect on inference.
    p.capture_branches=False
    with torch.no_grad():pa,_=p(s[-1],s[1],s[3],s[4],s[6],s[7])
    p.capture_branches=True
    with torch.no_grad():pb,_=p(s[-1],s[1],s[3],s[4],s[6],s[7])
    assert torch.equal(pa,pb)
    a,ra=sample(cfg,cache,p,s,cfg["seed"]+88000)
    b,rb=sample(cfg,cache,p,s,cfg["seed"]+88000)
    assert ra==rb
    for k in a:assert np.array_equal(a[k],b[k])
    np.savez_compressed(out/"sample.npz",**a)
    assert model_meta(w,h)==before
    result=dict(readouts=records,neutral_sampler_calls=2,capture_prediction_exact=True,optimizer_steps=0,model=before,code_commit=commit())
    if replay:
        assert result==read(Path(replay)/"result.json");same_arrays(out/"sample.npz",Path(replay)/"sample.npz")
    write(out/"result.json",result);p.close();print("R22_SANITY",records,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)
