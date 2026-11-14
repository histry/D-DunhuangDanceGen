"""Train-only matched analytical readouts; not denoiser training or extra information."""
import argparse,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.boundary_context.common import Data
from .common import population
from .states import extract,design,label,reference,FEATURE_DIM

def donors(rows,seed):
    order=sorted(range(len(rows)),key=lambda i:(hashlib.sha256((str(seed)+rows[i]["source_group"]).encode()).hexdigest(),
        rows[i]["raw_motion"]["sha256"],rows[i]["start_frame"]))
    groups=[rows[i]["source_group"] for i in order];n=max(groups.count(g) for g in set(groups))
    if 2*n>len(rows):raise ValueError("Cannot derange groups while preserving population")
    source=np.roll(order,n);mapping=np.empty(len(rows),int)
    for i,j in zip(order,source):mapping[i]=j
    assert all(rows[i]["source_group"]!=rows[j]["source_group"] for i,j in enumerate(mapping))
    return mapping

def qualification(scores,cfg):
    reference={r["group"]:r for r in scores if r["family"]=="A0"};shuffle={r["group"]:r for r in scores if r["family"]=="A4"}
    result={}
    for family in ("A1","A2","A3"):
        rows=[r for r in scores if r["family"]==family]
        wins=sum(r["root_rmse_m"]<reference[r["group"]]["root_rmse_m"] for r in rows)
        root=np.mean([r["root_rmse_m"] for r in rows]);base=np.mean([reference[r["group"]]["root_rmse_m"] for r in rows])
        wrong=np.mean([shuffle[r["group"]]["root_rmse_m"] for r in rows]);rotation=np.mean([r["root_orientation_rad"] for r in rows])
        refrot=np.mean([reference[r["group"]]["root_orientation_rad"] for r in rows]);g=cfg["representation_gate"]
        result[family]=dict(groups=len(rows),group_wins=wins,root_relative_gain=float(1-root/base),shuffle_relative_gain=float(1-root/wrong),
            orientation_change_rad=float(rotation-refrot),qualified=bool(len(rows)==cfg["head_holdout_groups"] and wins>=g["minimum_group_wins"] and root<=base*(1-g["root_improvement"])
                and root<=wrong*(1-g["shuffle_improvement"]) and rotation<=refrot+g["orientation_nonregression_rad"]))
    return result

def run(cfg,out,replay=None):
    out=fresh(out);assets,fit,holdout,groups,_=population(cfg);rows=fit+holdout;n=len(fit)
    data=Data(np.load(checked(assets["scaler"])));motions=[data.native(r) for r in rows]
    features=[extract(x) for x in motions];Y=np.stack([label(x) for x in motions])
    permutation=np.r_[donors(fit,cfg["seed"]),n+donors(holdout,cfg["seed"]+1)]
    # All difficulty thresholds are fitted on FIT, before scoring heldout groups.
    difficulty=np.asarray([[np.linalg.norm(f[2][:3]),np.linalg.norm(f[2][15:18]),max(np.linalg.norm(f[2][9:12]),np.linalg.norm(f[2][12:15]))] for f in features])
    cuts=np.quantile(difficulty[:n],[1/3,2/3],axis=0)
    records=[];models={};predictions={};scores=[]
    for family in cfg["head_families"]:
        X=np.stack([design(*f,family,donor=(features[permutation[i]][2],features[permutation[i]][3])) for i,f in enumerate(features)])
        if replay:
            model=dict(np.load(checked(read(Path(replay)/"result.json")["models"][family])))
        else:
            mean=X[:n].mean(0);std=X[:n].std(0);std=np.where(std>1e-6,std,1.)
            z=(X[:n]-mean)/std;ym=Y[:n].mean(0)
            gram=z@z.T+cfg["head_ridge"]*n*np.eye(n)
            coef=z.T@np.linalg.solve(gram,Y[:n]-ym)
            model=dict(mean=mean,std=std,coef=coef,target_mean=ym)
        pred=((X[n:]-model["mean"])/model["std"])@model["coef"]+model["target_mean"]
        f=out/(family+".npz");np.savez_compressed(f,**model);models[family]=ref(f);predictions[family]=pred
        for j,row in enumerate(holdout):
            x=motions[n+j];p,R=reference(x,pred[j]);gt=__import__("dunhuang.contracts",fromlist=["matrices"]).matrices(x[60:90,7:13])
            orientation=float(Rotation.from_matrix((gt.swapaxes(-1,-2)@R).reshape(-1,3,3)).magnitude().mean())
            records.append(dict(row=j,family=family,group=row["source_group"],recording=row["recording_uid"],
                root_rmse_m=float(np.sqrt(np.mean((p-x[60:90,4:7])**2))),root_orientation_rad=orientation,
                difficulty=difficulty[n+j].tolist(),difficulty_bins=[int(np.searchsorted(cuts[:,k],difficulty[n+j,k])) for k in range(3)]))
        for group in sorted(set(r["source_group"] for r in holdout)):
            rr=[r for r in records if r["family"]==family and r["group"]==group]
            scores.append(dict(family=family,group=group,windows=len(rr),root_rmse_m=float(np.mean([r["root_rmse_m"] for r in rr])),
                root_orientation_rad=float(np.mean([r["root_orientation_rad"] for r in rr]))))
        print("R14_READOUT",family,len(fit),len(holdout),flush=True)
    q=qualification(scores,cfg);selected=next((f for f in ("A1","A2","A3") if q[f]["qualified"]),None)
    np.savez_compressed(out/"predictions.npz",**predictions)
    result=dict(records=records,group_scores=scores,models=models,qualification=q,selected_family=selected,
        reference_capacity_study_admitted=selected is not None,analytical_heads_fitted=0 if replay else 5,
        parameters_per_head=FEATURE_DIM*270+270,neural_optimizer_steps=0,fit_windows=n,holdout_windows=len(holdout),
        fit_groups=len(groups)-cfg["head_holdout_groups"],holdout_groups=cfg["head_holdout_groups"],difficulty_cuts=cuts.tolist(),
        operational_qualified=False,independent_information_claimed=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,fit=fit,holdout=holdout,
        donor_indices=permutation.tolist(),normalization_role="fit only",old_monitor_and_val_tensor_reads=0,
        label_scope="paired hidden Root translation and SO3 residuals used only for fitting/scoring; not inference conditions",
        baseline="A0 matched raw observed-motion/mask ridge; frozen R13 generation remains a separate baseline"))
    if replay:
        old=read(Path(replay)/"result.json")
        for key in ("records","group_scores","qualification","selected_family","parameters_per_head"):assert result[key]==old[key]
        from dunhuang.research.source_tasks.common import same_arrays
        same_arrays(out/"predictions.npz",Path(replay)/"predictions.npz")
        for f in cfg["head_families"]:same_arrays(out/(f+".npz"),checked(old["models"][f]))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.out,a.replay_from)
