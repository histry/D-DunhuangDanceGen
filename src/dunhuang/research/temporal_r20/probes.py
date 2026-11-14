"""Fixed-input bilateral interventions; compare the same originally-unknown middle."""
import torch
from dunhuang.research.global_clock_r17.contract import metadata
from dunhuang.research.objective_r19.objective import reconstruction
from .conditions import endpoint,CONTROLS
from .common import grid

def run(cfg,cache,predict,full=True):
    arrays={};scores=[]
    times=grid() if full else [999,519,119,19]
    with torch.no_grad():
        for case in cfg["train_cases"]:
            z,y,m,o=cache["cache"][case];donor=cache["cache"][1-case][1]
            for t in times:
                torch.manual_seed(cfg["seed"]+50000+case*1000+t);torch.cuda.manual_seed_all(cfg["seed"]+50000+case*1000+t)
                ts=torch.tensor([t],device="cuda");xt=cache["diffusion"].q_sample(y,ts)
                meta=torch.tensor(metadata()[None],device="cuda")
                correct=None
                for control in CONTROLS if t in (999,519,119,19) else ("correct",):
                    v,k,h,original=endpoint(y,donor,control)
                    p,_=predict(xt,m,v,k,meta,ts);terms,counts=reconstruction(p,y,original)
                    arrays[f"{case}_{t}_{control}"]=p[0].cpu().numpy()
                    if control=="correct":correct=p
                    row=dict(case=case,t=t,control=control,unknown=float(terms["unknown"]),observed=float(terms["observed"]),
                        alpha=float(cache["diffusion"].alphas_cumprod[t]),counts=counts,
                        raw_edge_root_rmse_m=float((((p[:,:,4:7]-y[:,:,4:7])/cache["scale"][4:7])[original[:,:,4:7]]).square().mean().sqrt()))
                    if control!="correct":
                        interval=slice(8,75) if control.startswith("right") else slice(150,217)
                        row.update(far_side_exact_equal=bool(torch.equal(p[:,interval,4:],correct[:,interval,4:])),
                            far_root_response_m=float(((p[:,interval,4:7]-correct[:,interval,4:7])/cache["scale"][4:7]).square().mean().sqrt()),
                            far_motion_response_normalized=float((p[:,interval,4:]-correct[:,interval,4:]).square().mean().sqrt()))
                    scores.append(row)
    return arrays,scores
