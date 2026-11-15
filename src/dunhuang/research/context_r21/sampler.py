"""Unchanged global225 reverse update; CFG-preclip labels and actual hard domains."""
import hashlib
import torch
from dunhuang.research.masked_denoiser.common import tensor_hash
from dunhuang.research.objective_r19.objective import reconstruction
from dunhuang.research.global_clock_r17.model import fk
from dunhuang.research.temporal_r20.objective import stage_metrics as legacy_metrics
from .contract import regions

def domain_metrics(x,cache,origin,hard):
    result=legacy_metrics(x,cache,origin)
    legacy=result.pop("fk_jerk")
    result["legacy_r20_measurement"]=legacy
    if legacy is None:return dict(result,hard_domains=None)
    points=fk(x,cache["scale"],cache["bias"],origin,cache["parents"],cache["offsets"])
    jerk=(torch.diff(points,n=3,dim=1)*30**3).norm(dim=-1)
    domains={}
    for name,mask in regions(hard[:,:,4:].all(-1),3).items():
        a=jerk[mask]
        domains[name]=dict(templates=int(mask.sum()),max=float(a.max()) if a.numel() else None,
            p95=float(torch.quantile(a.flatten(),.95)) if a.numel() else None)
    result["hard_domains"]=domains
    return result

def sample(cfg,cache,predict,s,seed,keep_xt=False,detailed=False,score_mask=None):
    y,m,o,v,k,h,meta,ts,initial=s
    score_mask=k if score_mask is None else score_mask
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);noise=hashlib.sha256()
    def draw(x):noise.update(x.detach().cpu().numpy().tobytes());return x
    x=draw(torch.randn_like(v));times=list(reversed(torch.linspace(-1,999,51).int().tolist()))
    arrays={};rows=[];q=predict.telemetry;q.grad_enabled=False;predict.capture_branches=False
    with torch.no_grad():
        for t,nxt in zip(times[:-1],times[1:]):
            q.enabled=detailed and t in cfg["stage_snapshots"]
            if keep_xt:arrays["xt_"+str(t)]=x[0].cpu().numpy()
            pred,windows=predict(x,m,v,k,meta,torch.tensor([t],device=x.device))
            clean=torch.where(h,v,pred);terms,_=reconstruction(pred,y,score_mask)
            row=dict(t=t,x_t_sha256=tensor_hash(x),raw_sha256=tensor_hash(pred),unknown=float(terms["unknown"]))
            if q.enabled:
                stages={}
                for name,key in (("music_unconditional","unc"),("music_conditional","cond"),("cfg_preclip","preclip"),("cfg_clipped","clipped")):
                    local=predict.last[key]
                    from dunhuang.research.temporal_r20.trace import blend
                    merged=blend(local)
                    arrays[str(t)+"_"+name]=local.cpu().numpy();stages[name]=domain_metrics(merged,cache,o,h)
                stages["hard_preserved"]=domain_metrics(clean,cache,o,h)
                arrays[str(t)+"_hard_preserved"]=clean[0].cpu().numpy()
                row.update(stages=stages,saturation=predict.last["saturation"],layer_forward=predict.last["layer_forward"])
            rows.append(row)
            if nxt<0:x=clean;continue
            alpha=cache["diffusion"].alphas_cumprod[t];na=cache["diffusion"].alphas_cumprod[nxt]
            eps=(x-alpha.sqrt()*clean)/(1-alpha).sqrt();sigma=((1-alpha/na)*(1-na)/(1-alpha)).sqrt()
            z=draw(torch.randn_like(x))
            x=na.sqrt()*clean+(1-na-sigma**2).clamp_min(0).sqrt()*eps+sigma*z
            x=torch.where(h,na.sqrt()*v+(1-na).sqrt()*z,x)
    q.enabled=False
    assert torch.equal(x[h],v[h])
    arrays.update(sample=x[0].cpu().numpy(),raw_final=pred[0].cpu().numpy(),condition=v[0].cpu().numpy(),known=k[0].cpu().numpy(),hard=h[0].cpu().numpy())
    return arrays,dict(records=rows,noise_sha256=noise.hexdigest(),hard_exact=True,
        layout="joint" if predict.adapter.memory_mode=="joint" else "split",cfg_weight=2,clamp=[-1,1],
        condition_sha256=tensor_hash(v),known_sha256=tensor_hash(k),cache_t=[r["t"] for r in rows] if keep_xt else [],
        stage_label="cfg_preclip is AFTER CFG, BEFORE clamp; not pre-CFG",optimizer_steps=0)
