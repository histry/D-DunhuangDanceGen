"""Single225 reverse state. All stages are observed without changing CFG or clamp."""
import hashlib
import torch
from dunhuang.research.masked_denoiser.common import tensor_hash
from .trace import blend
from .objective import stage_metrics

def sample(cfg,d,predict,s,seed,cache,detailed=False,keep_xt=False,policy="mean",score_known=None):
    y,m,origin,v,k,h,meta,t,xt=s
    score_mask=k if score_known is None else score_known
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);noise_hash=hashlib.sha256()
    def draw(x):noise_hash.update(x.detach().cpu().numpy().tobytes());return x
    x=draw(torch.randn_like(v));times=list(reversed(torch.linspace(-1,999,51).int().tolist()))
    arrays={};records=[];predict.policy=policy;q=predict.telemetry;q.enabled=detailed;q.grad_enabled=False
    with torch.no_grad():
        for step,(t,nxt) in enumerate(zip(times[:-1],times[1:])):
            if keep_xt:arrays["xt_"+str(t)]=x[0].cpu().numpy()
            ts=torch.tensor([t],device=x.device);pred,windows=predict(x,m,v,k,meta,ts)
            clean=torch.where(h,v,pred)
            errors=(pred[:,:,4:]-y[:,:,4:]).square()
            unknown=.5*(errors[:,:,:3][~score_mask[:,:,4:7]].mean()+errors[:,:,3:][~score_mask[:,:,7:]].mean())
            row=dict(t=t,x_t_sha256=tensor_hash(x),raw_sha256=tensor_hash(pred),unknown=float(unknown),
                overlap_max=float(abs(windows[0,75:]-windows[1,:75]).max()) if len(windows)==2 else 0.)
            if detailed:
                last=predict.last;stages={}
                for name in ("unc","cond","preclip","clipped"):
                    w=last[name]
                    stages[name+"_A"]=stage_metrics(w[:1],cache,origin,0)
                    if len(w)==2:stages[name+"_B"]=stage_metrics(w[1:],cache,origin,75)
                    stages[name+"_merged"]=stage_metrics(blend(w,policy),cache,origin)
                    if t in cfg["stage_snapshots"]:arrays[str(t)+"_"+name]=w.cpu().numpy()
                stages["merged"]=stage_metrics(pred,cache,origin);stages["preserved"]=stage_metrics(clean,cache,origin)
                row.update(stages=stages,saturation=last["saturation"],layer_forward=last["layer_forward"])
                if t in cfg["stage_snapshots"]:
                    arrays[str(t)+"_merged"]=pred[0].cpu().numpy();arrays[str(t)+"_preserved"]=clean[0].cpu().numpy()
            records.append(row)
            if nxt<0:x=clean;continue
            alpha=d.alphas_cumprod[t];na=d.alphas_cumprod[nxt]
            eps=(x-alpha.sqrt()*clean)/(1-alpha).sqrt();sigma=((1-alpha/na)*(1-na)/(1-alpha)).sqrt()
            noise=draw(torch.randn_like(x))
            x=na.sqrt()*clean+(1-na-sigma**2).clamp_min(0).sqrt()*eps+sigma*noise
            x=torch.where(h,na.sqrt()*v+(1-na).sqrt()*noise,x)
    q.enabled=False;predict.policy="mean"
    assert torch.equal(x[h],v[h])
    arrays.update(sample=x[0].cpu().numpy(),raw_final=pred[0].cpu().numpy(),condition=v[0].cpu().numpy(),
        known=k[0].cpu().numpy(),hard=h[0].cpu().numpy())
    return arrays,dict(records=records,noise_sha256=noise_hash.hexdigest(),blend=policy,all_edge_calls_local150=predict.adapter.memory_mode!="joint",single_call_joint225=predict.adapter.memory_mode=="joint",
        final_hard_exact=True,model_state_is_global225=True,backbone_frames=225 if predict.adapter.memory_mode=="joint" else 150,cache_t=[r["t"] for r in records] if keep_xt else [])
