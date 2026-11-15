"""Frozen seen/new-reference by raw/canonical representation experiment; no training."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .representation import packet,rotations
from dunhuang.research.state_r27.common import read,sha,write,fresh,forbid_optimizers,runtime,digest,restore,decoded_metrics,commit,same_arrays
from dunhuang.research.state_r27.trace import sample
from dunhuang.research.state_r27.qualification import sampling_gate
from .input_contract import foot_proxy

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    r27=Path(cfg["r27"]);assert sha(r27/"artifacts_manifest.json")==cfg["r27_manifest_sha256"]
    manifest=read(r27/"artifacts_manifest.json")["files"]
    for name in ("teachers/teacher.npz","models/P/adapter.pt"):assert sha(r27/name)==manifest[name]["sha256"]
    seen=dict(np.load(r27/"teachers/teacher.npz"))
    d,n,s,f,e,info,a=runtime(read("src/dunhuang/research/state_r27/config.json"));frozen=digest(d.model)
    music=torch.tensor(seen["music"],device="cuda")
    value,noise,_=sample(d,a,None,music,cfg["new_teacher_seeds"]);calls=1
    raw=n.unnormalize(value.clone());points=f(raw,s)[:,:,:22]
    new=dict(target=value.cpu().numpy(),raw139=raw.cpu().numpy(),points22=points.cpu().numpy(),music=seen["music"])
    arrays={f"new_{k}":v for k,v in new.items()};records=[];parities=[];memories=[];noises=[];base={}
    for population,z in (("seen",seen),("new_reference",new)):
        target=torch.tensor(z["target"],device="cuda")
        baseline,receipt,_=sample(d,a,None,music,cfg["factorial_noise_seeds"]);calls+=1;base[population]=baseline
        refs=list(z["raw139"][:,64:160,7:]);wrong=[refs[i] for i in (1,0,3,2)]
        for representation in ("raw","canonical"):
            transformed=[rotations(x,representation) for x in refs]
            copy=z["raw139"].copy();copy[:,64:160,7:]=np.stack(transformed)
            with torch.no_grad():p=f(torch.tensor(copy,device="cuda"),s)[:,:,:22].cpu().numpy()
            error=float(abs(p-z["points22"]).max());assert error<2e-5
            parities.append(dict(population=population,representation=representation,physical_reference_fk_max_m=error))
        for arm in ("R26_C1","P"):
            if arm=="P":restore(a,r27/"models/P/adapter.pt")
            else:
                v=torch.load(Path(cfg["r26"])/"models/C1/adapter.pt",map_location="cpu",weights_only=True)["state"];a.load_state_dict(v,strict=True)
            for representation in (("raw","canonical") if arm=="P" else ("canonical",)):
                # A paired-state memory/injection readout does not stand in for actual sampling.
                ref=packet(refs,representation);a.set_reference(**ref);memory=a.context[0].detach().clone();a.capture=True
                ts=torch.full((4,),519,device="cuda",dtype=torch.long)
                torch.manual_seed(cfg["seed"]);xt=d.q_sample(target,ts,torch.randn_like(target))
                with torch.no_grad():d.model(xt,music,None,ts,cond_drop_prob=0)
                memories.append(dict(population=population,arm=arm,representation=representation,
                    memory_rms=float(memory.square().mean().sqrt()),memory_sha256=__import__("hashlib").sha256(memory.cpu().numpy().tobytes()).hexdigest(),
                    injection_readouts=dict(a.readouts)))
                a.capture=False;a.clear()
                for mode,source in (("correct",refs),("wrong",wrong)):
                    pred,receipt,_=sample(d,a,packet(source,representation),music,cfg["factorial_noise_seeds"]);calls+=1
                    stem=f"{population}_{arm}_{representation}_{mode}";arrays[stem+"_normalized"]=pred.cpu().numpy()
                    with torch.no_grad():xyz=f(n.unnormalize(pred.clone()),s)[:,:,:22]
                    arrays[stem+"_points"]=xyz.cpu().numpy()
                    errors=decoded_metrics(pred,target,n,s,f)
                    for i,er in enumerate(errors):
                        records.append(dict(population=population,arm=arm,representation=representation,mode=mode,teacher=i,
                            metrics=er,foot_proxy=foot_proxy(xyz[i].cpu().numpy()),target_scope="MODEL_GENERATED_FULL139_NOT_CONTACT_TRUTH"))
                    noises.append(dict(population=population,arm=arm,representation=representation,mode=mode,noise=receipt))
                    print("R28_FACTORIAL",population,arm,representation,mode,flush=True)
            assert digest(d.model)==frozen
        arrays[population+"_baseline_normalized"]=baseline.cpu().numpy()
        arrays[population+"_baseline_points"]=f(n.unnormalize(baseline.clone()),s)[:,:,:22].cpu().numpy()
        for i,er in enumerate(decoded_metrics(baseline,target,n,s,f)):
            records.append(dict(population=population,arm="baseline",representation="none",mode="none",teacher=i,metrics=er))
    # Paired-reference direction and NoRef/amplitude readouts; one new-noise group is a diagnostic, not the R27 two-group gate.
    cells=[]
    for population in ("seen","new_reference"):
        for representation in ("raw","canonical"):
            tests=[]
            for i in range(4):
                def row(mode):return next(r for r in records if (r["population"],r["arm"],r["representation"],r["mode"],r["teacher"])==(population,"P",representation,mode,i))
                correct=row("correct")["metrics"];wrong=row("wrong")["metrics"]
                none=next(r["metrics"] for r in records if (r["population"],r["arm"],r["teacher"])==(population,"baseline",i))
                values=dict(body_gain=correct["body22_rad"]<=.75*none["body22_rad"],body_reference=correct["body22_rad"]<wrong["body22_rad"],
                    local_reference=correct["pelvis_wrist_rmse_m"]<wrong["pelvis_wrist_rmse_m"],
                    activity=correct["body_activity_ratio"] is not None and .6<=correct["body_activity_ratio"]<=1.6,
                    extent=correct["arm_range_ratio"] is not None and .6<=correct["arm_range_ratio"]<=1.6)
                tests.append(dict(teacher=i,tests=values,passed=all(values.values())))
            cells.append(dict(population=population,representation=representation,conditions=tests,passed_conditions=sum(x["passed"] for x in tests)))
    result=dict(records=records,cells=cells,reference_parities=parities,memory_and_injection=memories,noise=noises,new_teacher_noise=noise,
        new_teacher_seeds=cfg["new_teacher_seeds"],sampler_calls=calls,generated_trajectories=4*calls,
        new_teacher_source_never_used_for_adaptation=True,new_reference_same_music_as_seen=True,optimizer_steps=0,
        representation_intervention_does_not_change_target=True,full_native_status="NOT_EVALUABLE",operational_qualified=False)
    path=out/"arrays.npz";np.savez_compressed(path,**arrays);write(out/"result.json",result)
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,arrays_sha256=sha(path),checkpoint=info))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(path,Path(replay)/path.name)
    a.close();print("R28_FACTORIAL_DONE",[(r["population"],r["representation"],r["passed_conditions"]) for r in cells],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");args=p.parse_args()
    run(read(args.config),args.root,args.out,args.replay_from)
