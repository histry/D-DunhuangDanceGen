"""Five original objectives on identical Train batches, with isolated virtual gradient steps.
Virtual clone interventions are recorded separately from optimizer/training updates.
"""
from pathlib import Path
import copy
import numpy as np
import torch
from dunhuang.io import write
from .event_study import score,shift_loss,donor_indices
from .event_metrics import multi_positive_loss,proxy_retrieval
from .event_weak_labels import weak_loss
from .event_perturb import batch as synthetic_batch,supervised_path_loss
from .clock_experiment import digest
from .content_discrimination import template_paths,pair_result,fixed_score

NAMES=("retrieval","shift","partial","null","weak")

def gradients(losses,parameters):
    results=[]
    for name in NAMES:
        g=torch.autograd.grad(losses[name],parameters,retain_graph=True,allow_unused=True)
        results.append([torch.zeros_like(p) if v is None else v for p,v in zip(parameters,g)])
    return results

def cosine_matrix(vectors):
    result=[]
    for a in vectors:
        row=[]
        for b in vectors:
            den=a.norm()*b.norm()
            row.append(float((a@b)/den) if den>1e-12 else None)
        result.append(row)
    return result

def objectives(head,bank,queries,dense,old_config,donors,seed):
    candidate=bank["candidates"][queries].cpu().numpy();n=len(queries);count=candidate.shape[1]
    values,_=score(head,"strict_path",bank,np.repeat(queries,count),candidate.reshape(-1))
    retrieval=multi_positive_loss(values.reshape(n,count),bank["positive"][queries],bank["valid"][queries],old_config["score_temperature"])
    shift=shift_loss(head,"strict_path",dense,queries,old_config)
    synthetic=synthetic_batch(dense["motion"],queries,[donors[i] for i in queries],seed)
    truth,_=supervised_path_loss(head,synthetic,"cuda")
    _,own=score(head,"strict_path",bank,queries,queries)
    weak=weak_loss(own,bank["weak_target"][queries],bank["weak_weight"][queries])
    return dict(retrieval=retrieval,shift=shift,partial=truth["partial"],null=truth["null"],weak=weak)

def ranking(head,bank_numpy,bank,rows,queries):
    candidates=bank_numpy["candidates"][queries];n,width=candidates.shape
    q=np.repeat(queries,width);c=candidates.reshape(-1)
    r=pair_result(head,bank_numpy,q,c)
    null=float(r["null_cost"].detach());paths=template_paths(bank_numpy,q,c,head.band_seconds,null)
    fixed=fixed_score(r,paths,head.band_seconds)["mean_content"].reshape(n,width)
    values=r["score"].reshape(n,width)
    p=bank["positive"][queries];v=bank["valid"][queries]
    def margin(x):
        return float((x.masked_fill(~p,-1e4).max(-1).values-x.masked_fill(~(v&~p),-1e4).max(-1).values).mean())
    report=dict(path_margin=margin(values),clock_template_content_margin=margin(fixed),
        ranking=proxy_retrieval(values.detach().cpu().numpy(),p.cpu().numpy(),v.cpu().numpy()))
    return report

def run(head,bank_numpy,rows,dense,source,old_config,config,out):
    out=Path(out);out.mkdir();head.requires_grad_(True);head.null_logit.requires_grad_(False)
    before=digest(head);parameters=[head.music.weight,head.motion.weight]
    bank={k:torch.as_tensor(v,device="cuda") for k,v in bank_numpy.items()}
    schedule=np.load(Path(source)/"training_schedule.npy");donors=donor_indices(rows)
    records=[];saved={};virtual_steps=0
    for index in config["gradient_batches"]:
        queries=schedule[index]
        if any(rows[int(i)]["role"]!="train" for i in queries):raise ValueError("Gradient audit cannot inspect Val")
        losses=objectives(head,bank,queries,dense,old_config,donors,config["seed"]+index)
        g=gradients(losses,parameters);flat=[torch.cat([x.flatten() for x in v]).detach() for v in g]
        baseline=ranking(head,bank_numpy,bank,rows,queries)
        record=dict(schedule_row=index,queries=len(queries),gradient_cosine=cosine_matrix(flat),losses={k:float(v.detach()) for k,v in losses.items()},objectives={})
        for name,parts,vector in zip(NAMES,g,flat):
            saved[str(index)+"_"+name]=vector.cpu().numpy()
            norms=[float(x.detach().norm()) for x in parts];r=dict(music_projection_norm=norms[0],motion_projection_norm=norms[1],total_norm=float(vector.norm()),before=baseline)
            if vector.norm()>1e-12:
                clone=copy.deepcopy(head)
                with torch.no_grad():
                    for p,grad in zip((clone.music.weight,clone.motion.weight),parts):p.add_(grad,alpha=-config["diagnostic_step"])
                    after=ranking(clone,bank_numpy,bank,rows,queries)
                    after_loss=objectives(clone,bank,queries,dense,old_config,donors,config["seed"]+index)[name]
                r.update(after=after,own_loss_after=float(after_loss),step=config["diagnostic_step"]);virtual_steps+=1
            else:r.update(after=None,reason="zero gradient / unsupported weak labels in this batch")
            record["objectives"][name]=r
        records.append(record);print("LOCAL_GRADIENT_BATCH",index,flush=True)
    if digest(head)!=before:raise ValueError("Gradient audit altered the frozen checkpoint")
    np.savez_compressed(out/"gradients.npz",**saved)
    report=dict(schema="local_loss_gradient_audit_v1",objectives=list(NAMES),records=records,
        train_only=True,null_cost_frozen=True,projection_dimensions=[144,144],checkpoint_unchanged=True,
        optimizer_calls=0,diagnostic_virtual_clone_updates=virtual_steps,
        scope="one SGD-like hypothetical step on independent clones at registered step; not deployed training or Val selection")
    write(out/"result.json",report);return report
