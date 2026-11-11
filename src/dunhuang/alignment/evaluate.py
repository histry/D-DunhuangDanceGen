import numpy as np
import torch

def retrieval(model,rows,arrays,ids,device):
    music=[];motion=[]
    with torch.no_grad():
        for start in range(0,len(ids),64):
            ix=ids[start:start+64]
            m=torch.as_tensor(arrays['music'][ix],device=device).float()
            d=torch.as_tensor(arrays['motion'][ix,15:135],device=device).float()
            mm=torch.as_tensor(arrays['mm'][ix],device=device);dm=torch.as_tensor(arrays['dm'][ix,15:135],device=device)
            music.append(model.music(m,mm)[0].cpu());motion.append(model.motion(d,dm)[0].cpu())
    scores=(torch.cat(music)@torch.cat(motion).T).numpy()
    result={}
    for domain in ('finedance','aistpp'):
        selected=[j for j,i in enumerate(ids) if rows[i]['domain']==domain]
        values=[];random=[];group={}
        for j in selected:
            # Multiple motions for the same real audio are legitimate global positives.
            pool=[k for k in selected if rows[ids[k]]['audio_sha256']==rows[ids[j]]['audio_sha256'] or rows[ids[k]]['group']!=rows[ids[j]]['group']]
            positive=np.array([rows[ids[k]]['audio_sha256']==rows[ids[j]]['audio_sha256'] for k in pool])
            order=np.argsort(-scores[j,pool]);r=int(np.flatnonzero(positive[order])[0])+1
            values.append([r<=1,r<=5,1/r]);random.append(float(positive.mean()))
            group.setdefault(rows[ids[j]]['group'],[]).append([r<=1,1/r])
        if not values:continue
        group_values=np.array([np.mean(v,0)[0] for v in group.values()])
        rng=np.random.default_rng(117)
        ci=np.percentile([rng.choice(group_values,len(group_values),replace=True).mean() for _ in range(500)],[2.5,97.5])
        result[domain]=dict(recall1_group_bootstrap95=ci.tolist(),queries=len(values),groups=len(group),recall1=float(np.mean(values,0)[0]),
            recall5=float(np.mean(values,0)[1]),mrr=float(np.mean(values,0)[2]),chance=float(np.mean(random)),
            group_metrics={k:np.mean(v,0).tolist() for k,v in group.items()})
    return result

SHIFTS=(-15,-8,0,8,15)

def sync_logits(model,music,motion,mm,dm):
    b=len(music)
    m,mt,mv=model.music(music,mm)
    d=torch.stack([motion[:,15+s:135+s] for s in SHIFTS],1)
    mask=torch.stack([dm[:,15+s:135+s] for s in SHIFTS],1)
    _,dt,dv=model.motion(d.reshape(b*5,120,74),mask.reshape(b*5,120,74))
    mt=mt[:,None].expand(-1,5,-1,-1).reshape(b*5,60,128)
    valid=mv[:,None].expand(-1,5,-1).reshape(b*5,60)&dv
    return model.sync_score(mt,dt,valid).reshape(b,5)

def synchronization(model,rows,arrays,ids,device):
    logits=[]
    with torch.no_grad():
        for start in range(0,len(ids),16):
            ix=ids[start:start+16]
            args=[torch.as_tensor(arrays[k][ix],device=device) for k in ('music','motion','mm','dm')]
            logits.append(sync_logits(model,*args).cpu().numpy())
    logits=np.concatenate(logits)
    output={}
    for domain in ('finedance','aistpp'):
        selected=[j for j,i in enumerate(ids) if rows[i]['domain']==domain]
        z=logits[selected];negative=z[:,[0,1,3,4]]
        wins=(z[:,2,None]>negative).mean(1)
        grouped={}
        for j,w in zip(selected,wins):grouped.setdefault(rows[ids[j]]['group'],[]).append(float(w))
        grouped={k:float(np.mean(v)) for k,v in grouped.items()}
        rng=np.random.default_rng(117)
        means=list(grouped.values())
        ci=np.percentile([rng.choice(means,len(means),replace=True).mean() for _ in range(500)],[2.5,97.5])
        output[domain]=dict(queries=len(selected),groups=len(grouped),aligned_vs_shift_win=float(wins.mean()),
            offset_choice_accuracy=float((z.argmax(1)==2).mean()),chance=.2,
            margin=float((z[:,2,None]-negative).mean()),per_shift_win={str(s/30):float((z[:,2]>z[:,k]).mean()) for k,s in enumerate(SHIFTS) if s!=0},group_wins=grouped,group_bootstrap95=ci.tolist())
    return output


class EnergyBaseline:
    """Per-window energy-envelope correlation on the same 30 Hz observations."""
    @staticmethod
    def encode(values,mask,channel):
        signal=values[:,:,channel][:,::2]
        valid=mask[:,:,channel][:,::2]
        mean=(signal*valid).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)
        vector=torch.nn.functional.normalize((signal-mean)*valid,dim=-1)
        return vector,None,valid
    def music(self,x,mask):
        return self.encode(x,mask,0)
    def motion(self,x,mask):
        return self.encode(x,mask,66)
