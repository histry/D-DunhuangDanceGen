"""Full Native reports and a separate protected Train-only capacity gate."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.boundary_context.common import Data
from dunhuang.research.state_bridge.common import native_environment
from dunhuang.research.masked_denoiser.metrics import physical
from .common import source
from .geometry import response,metrics,protected,qualification

def run(cfg,preflight,probes,out,replay=None):
    out=fresh(out);forbid_optimizers();assets,rows,_,_=source(cfg);native_environment(assets)
    scaler=np.load(checked(assets["scaler"]));data=Data(scaler)
    targets=np.load(checked(read(Path(preflight)/"protocol.json")["target_fk"]))
    ledger=read(Path(probes)/"result.json");directions={(v["case"],v["arm"],v["t"]):v for v in ledger["records"]}
    states=[];groups=[];all_trials=[];native_reads=0
    for i,b in enumerate(rows):
        source_motion=data.native(b["row"]);target_fk=targets["capacity_train_"+str(i)]
        z=np.load(checked(ledger["probes"][str(i)]));selected={}
        def score(name):
            nonlocal native_reads
            x,fk,audit,m=physical(z[name],source_motion,scaler);m.update(metrics(x,fk,source_motion,target_fk))
            folder=out/(str(i)+"_"+name);folder.mkdir()
            np.save(folder/"native.npy",x);np.save(folder/"fk.npy",fk);write(folder/"physical.json",audit)
            if replay:
                other=Path(replay)/folder.name
                assert np.array_equal(x,np.load(other/"native.npy")) and np.array_equal(fk,np.load(other/"fk.npy"))
                assert audit==read(other/"physical.json")
            native_reads+=1
            return x,m
        for timestep in cfg["timesteps"]:
            base,m=score("J0_"+str(timestep));selected["J0",timestep]=dict(accepted=False,metrics=m,protection={})
            states.append(dict(case=i,arm="J0",t=timestep,accepted=False,metrics=m))
            for arm in cfg["arms"][1:]:
                direction=directions[i,arm,timestep];choice=None;candidate_records=[]
                derivative_ok=direction["derivative_relative_error"]<=.05 or abs(direction["derivative_numeric"]-direction["derivative_analytic"])<=1e-4
                for k,fraction in enumerate(cfg["line_fractions"]):
                    name=arm+"_"+str(timestep)+"_"+str(k);x,cm=score(name)
                    trust=response(x,base,cfg);guard=protected(cm,m,cfg)
                    objective_ok=direction["trials"][k]["objective"]<direction["objective_before"]
                    accepted=derivative_ok and trust["within_radius"] and objective_ok and all(guard.values())
                    row=dict(case=i,arm=arm,t=timestep,trial=k,fraction=fraction,response=trust,protection=guard,
                        objective_decreased=objective_ok,numeric_gradient_ok=derivative_ok,eligible=bool(accepted),metrics=cm)
                    candidate_records.append(row)
                    if accepted and choice is None:choice=row
                all_trials.extend(candidate_records)
                result=dict(case=i,arm=arm,t=timestep,accepted=choice is not None,
                    selected_trial=choice["trial"] if choice else None,
                    metrics=choice["metrics"] if choice else m,
                    protection=choice["protection"] if choice else {k:False for k in protected(m,m,cfg)},
                    scope="Train oracle line-search; baseline fallback is not a success")
                selected[arm,timestep]=result;states.append(result)
        for arm in cfg["arms"]:
            rr=[selected[arm,t] for t in cfg["timesteps"]]
            group=dict(case=i,source_group=b["row"]["source_group"],arm=arm,accepted=all(x["accepted"] for x in rr),
                metrics={k:float(np.mean([x["metrics"][k] for x in rr])) for k in ("root_rmse_m","feet_rmse_m")},
                protection={k:all(x["protection"].get(k,False) for x in rr) for k in protected(m,m,cfg)} if arm!="J0" else {})
            groups.append(group)
        print("R15_NATIVE",i,flush=True)
    gate=qualification(groups,cfg)
    result=dict(states=states,groups=groups,trials=all_trials,qualification=gate,native_readouts=native_reads,
        original_population_retained=True,optimizer_steps=0,new_full_window_generations=0,new_whole_song_generations=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,
        local_gate_scope="Train-oracle protected single-state capacity; not generation or generalization",
        formal_gate="Unchanged Native absolute acceptance; immutable exclusions never relabeled feasible"))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R15_LOCAL_CAPACITY",gate["local_capacity_passed"],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--preflight",required=True);p.add_argument("--probes",required=True)
    p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.preflight,a.probes,a.out,a.replay_from)
