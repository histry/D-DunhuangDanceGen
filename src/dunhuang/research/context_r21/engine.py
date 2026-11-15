"""One frozen warm source, matched ledgers, explicit optional branch diagnostics."""
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha
from dunhuang.research.source_capacity.common import read
from dunhuang.research.temporal_r20.common import load as old_load,grid
from dunhuang.research.temporal_r20.trace import TracedPredictor,blend
from dunhuang.research.objective_r19.common import restore,parameters,gradient,group_norm
from dunhuang.research.objective_r19.objective import reconstruction
from dunhuang.research.global_clock_r17.model import fk
from dunhuang.research.global_clock_r17.contract import metadata
from .contract import conditions,intervention

class Predictor(TracedPredictor):
    def __init__(self,old):
        old.telemetry.close();super().__init__(old.base)
        self.capture_branches=False;self.branches=[]
        self.branch_handle=self.wrapper.net.final_layer.register_forward_hook(self.capture)
    def capture(self,module,args,out):
        if self.capture_branches:self.branches.append(out)
    def __call__(self,*a,**kw):
        self.branches=[];return super().__call__(*a,**kw)
    def close(self):self.branch_handle.remove();super().close()

def layout(arm):return "split" if arm in ("B0","B2") else "joint"
def balanced(arm):return arm not in ("B0","B1")
def source(cfg,root,arm):
    return read(Path(root)/"models/B3/result.json")["checkpoint"] if arm in ("Q","M","F") else None

def load(cfg,root,arm,reference=None):
    oldcfg=read(Path(cfg["r20_root"])/"interface/protocol.json")["config"]
    seal=read("reports/r20_release/r20_receipt.json")
    assert sha(Path(cfg["r20_root"])/"interface/protocol.json")==seal["evidence"]["interface/protocol.json"]
    w,d,data,old,h=old_load(oldcfg,layout(arm));p=Predictor(old)
    reference=source(cfg,root,arm) if reference is None else reference
    if reference:restore(cfg,w,h,reference)
    return w,d,data,p,h

def cache_data(cfg,root,d,data):
    pre=read(Path(root)/"data/protocol.json");population=read(Path(root)/"data/result.json")
    parity=np.load(checked(pre["parity"]));parents=list(map(int,parity["parents"]))
    cache=dict(rows=population["records"],population=population,parents=parents,offsets=torch.tensor(parity["offsets"],device="cuda"),
        scale=torch.tensor(data.scale,device="cuda"),bias=torch.tensor(data.bias,device="cuda"),diffusion=d,cache={},
        proxy_foot_ids=cfg["proxy"]["foot_joints"],support_scales=cfg["support_scales"],assets=pre["assets"])
    for row in population["records"]:
        z=dict(np.load(checked(row["arrays"])));y=torch.tensor(z["encoded"][None],device="cuda")
        music=torch.tensor(np.stack((z["music"][:150],z["music"][75:])),device="cuda");origin=torch.tensor(z["origin"][None],device="cuda")
        with torch.no_grad():error=float(abs(fk(y,cache["scale"],cache["bias"],origin,parents,cache["offsets"])[0].cpu().numpy()-parity[str(row["case"])]).max())
        assert error<=1e-5,("FK parity",row["case"],error)
        proxy={n:torch.tensor(z[n][None],device="cuda") for n in ("proxy_static","proxy_swing","proxy_anchor","proxy_segment","proxy_floor")}
        cache["cache"][row["case"]]=(z,y,music,origin,proxy)
    return cache

def state(cfg,cache,case,t,variant=0,pattern=0,seed=None,control="correct"):
    z,y,m,o,proxy=cache["cache"][case];v,k,h,_=conditions(y,cfg,variant,pattern,True)
    original=k.clone()
    if control!="correct":
        rows=cache["population"]["train_probe"] if case in cache["population"]["train_cases"] else cache["population"]["dev_cases"]
        donor=cache["cache"][rows[(rows.index(case)+1)%len(rows)]][1]
        v,k,h=intervention(v,k,h,donor,control,cfg,variant)
    meta=torch.tensor(metadata()[None],device="cuda");ts=torch.tensor([t],device="cuda")
    seed=cfg["seed"]+case*1000+t if seed is None else seed
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    xt=cache["diffusion"].q_sample(y,ts)
    return (y,m,o,v,k,h,meta,ts,xt),original,proxy

def schedule(cfg,population,arm):
    follow=arm in ("Q","M","F");cases=population["follow_cases"] if follow else population["train_cases"]
    length=cfg["follow_steps"] if follow else cfg["steps"];rng=np.random.default_rng(cfg["seed"]);order=[];g=grid();times=[]
    while len(times)<length//2:times.extend(np.array(g)[rng.permutation(len(g))].tolist())
    for epoch in range((length+len(cases)-1)//len(cases)):
        for c in np.array(cases)[rng.permutation(len(cases))]:
            step=len(order)
            if step==length:return order
            t=int(times[step//2]) if step%2 else int(rng.integers((step//2%10)*100,(step//2%10+1)*100))
            order.append(dict(step=step,case=int(c),t=t,variant=0 if follow else epoch%len(cfg["contexts"]),
                pattern=0 if follow else epoch//len(cfg["contexts"])%len(cfg["observation_joints"]),
                seed=cfg["seed"]+10000+step,use_cache=arm=="M" and step%2==1))
    return order

def weights(cfg):
    oldcfg=read(Path(cfg["r20_root"])/"interface/protocol.json")["config"]
    path=Path(cfg["r20_root"])/"temporal_calibration/result.json"
    assert sha(path)==read("reports/r20_release/r20_receipt.json")["evidence"]["temporal_calibration/result.json"]
    return read(Path(oldcfg["parent"])/"calibration/result.json")["weights"],read(path)["weights"]

def branch_readout(predict,y,known,w=None,head=None):
    assert len(predict.branches)==2
    result={}
    for name,raw in zip(("unconditional_music","conditional_music"),predict.branches):
        p=blend(raw.clamp(-1,1));terms,_=reconstruction(p,y,known)
        result[name]=dict(unknown=float(terms["unknown"].detach()),observed=float(terms["observed"].detach()))
        if w is not None:result[name]["unknown_gradient_norm"]=group_norm(gradient(terms["unknown"],parameters(w,head)),sum(x.numel() for x in w.selected()))
    return result

def probes(cfg,cache,predict,controls=False,dev=False):
    cases=cache["population"]["train_probe"]+(cache["population"]["dev_cases"] if dev else [])
    rows=[];arrays={}
    variants=cfg["eval_variants"] if dev else [0]
    with torch.no_grad():
        for case in cases:
            for variant in variants:
                times=grid() if variant==0 else [999,519,119,19]
                for t in times:
                    s,original,_=state(cfg,cache,case,t,variant=variant)
                    y,m,o,v,k,h,meta,ts,xt=s
                    predict.capture_branches=t in (999,519,119,19)
                    pred,_=predict(xt,m,v,k,meta,ts);terms,counts=reconstruction(pred,y,original)
                    row=dict(case=case,variant=variant,t=t,unknown=float(terms["unknown"]),observed=float(terms["observed"]),counts=counts)
                    if predict.capture_branches:
                        arrays[str(case)+"_"+str(variant)+"_"+str(t)]=pred[0].cpu().numpy();row["branches"]=branch_readout(predict,y,original)
                    if controls and t in (999,519,119,19):
                        row["controls"]={}
                        for control in ("left_wrong","right_wrong","both_absent"):
                            cs,_,_=state(cfg,cache,case,t,variant=variant,control=control)
                            cp,_=predict(cs[-1],cs[1],cs[3],cs[4],cs[6],cs[7]);loss,_=reconstruction(cp,y,original)
                            row["controls"][control]=float(loss["unknown"])
                    rows.append(row)
    predict.capture_branches=False
    return rows,arrays
