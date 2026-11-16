"""One pre-registered16-record/8-group coverage experiment, independent of Native release."""
import argparse,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .common import runtime,packet
from dunhuang.research.state_r27.common import read,write,sha,checked,fresh,commit,forbid_optimizers,matrices,digest,checkpoint,restore,vector,gradient_norms,readout_check,same_arrays
from dunhuang.research.contract_r28.input_contract import alignment_height,height_features,translate
from dunhuang.research.contract_r28.train import objective,curves
from dunhuang.research.state_r27.qualification import fitting_gate

def prepare(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();assert read(root/"coverage_admission.json")["coverage_admitted"]
    parent=Path(cfg["r26"]);pop=read(parent/"data/result.json");allrows=pop["records"]
    bycase={r["case"]:r for r in allrows};original={bycase[c]["source_group"]:c for c in cfg["train_cases"]}
    excluded={bycase[c]["source_group"] for c in cfg["holdout_cases"]}
    key=lambda text:hashlib.sha256((str(cfg["seed"])+text).encode()).hexdigest()
    groups=list(original)+sorted({r["source_group"] for r in allrows if r["role"]=="train"}-set(original)-excluded,key=key)[:4]
    selected=[]
    for g in groups:
        candidates=sorted([r for r in allrows if r["role"]=="train" and r["source_group"]==g],key=lambda r:key(r["recording_uid"]))
        if g in original:
            first=bycase[original[g]];selected.extend([first,next(r for r in candidates if r["case"]!=first["case"])])
        else:selected.extend(candidates[:2])
    assert len(selected)==16 and len({r["recording_uid"] for r in selected})==16
    assert not {r["source_group"] for r in selected}&excluded
    plan=dict(groups=groups,cases=[r["case"] for r in selected],excluded_groups=sorted(excluded),rule="Fixed UID hash, original four groups retained; no quality filtering")
    write(out/"cohort_plan.json",plan) # Sealed before reading new performance or constructing optimizer.
    d,n,s,f,e,info,a=runtime(cfg);a.close();manifest=read(parent/"artifacts_manifest.json")["files"];records=[]
    for row in selected:
        name="data/"+row["arrays"];assert sha(parent/name)==manifest[name]["sha256"]
        old=dict(np.load(parent/name));native=old["native"];full=np.load(checked(row["source_native"]))
        first=full[:1];R=matrices(first[:,7:].reshape(1,24,6))[:,:22]
        rawfirst=np.concatenate((np.zeros((1,4),np.float32),first[:,4:7],R[...,:2,:].reshape(1,132)),1).astype(np.float32)
        with torch.no_grad():p=f(torch.tensor(rawfirst[None],device="cuda"),s)[0,:,:22].cpu().numpy()
        h=alignment_height(p[0,[10,11],1])
        R=matrices(native[:,7:].reshape(256,24,6))[:,:22]
        raw=np.concatenate((np.zeros((256,4),np.float32),native[:,4:7],R[...,:2,:].reshape(256,132)),1).astype(np.float32)
        raw=translate(raw,np.array([-raw[0,4],-h,-raw[0,6]],np.float32))
        with torch.no_grad():
            p=f(torch.tensor(raw[None],device="cuda"),s)[0,:,:22].cpu().numpy()
            raw[:,:4]=height_features(p);target=n.normalize(torch.tensor(raw[None],device="cuda"))
        path=out/f"{row['case']}.npz";np.savez_compressed(path,model_raw=raw,model_target=target[0].cpu().numpy(),model_points=p,music=old["music"])
        records.append(dict(case=row["case"],recording_uid=row["recording_uid"],source_group=row["source_group"],arrays=path.name,sha256=sha(path),C4_scope="HEIGHT_DERIVED_NOT_MEASURED_CONTACT"))
        if replay:same_arrays(path,Path(replay)/path.name)
    result=dict(records=records,plan=plan,optimizer_steps=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,checkpoint=info,plan_sha256=sha(out/"cohort_plan.json")))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R29_COVERAGE_DATA",16,8,flush=True)

def train(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out)
    if replay:forbid_optimizers()
    assert read(root/"coverage_admission.json")["coverage_admitted"]
    pop=read(root/"coverage_data/result.json");data={r["case"]:dict(np.load(root/"coverage_data"/r["arrays"])) for r in pop["records"]}
    for r in pop["records"]:assert sha(root/"coverage_data"/r["arrays"])==r["sha256"]
    cases=pop["plan"]["cases"];rng=np.random.default_rng(cfg["seed"])
    per_time={t:np.asarray(cases)[rng.permutation(16)].tolist() for t in cfg["sampler_times"]};visits=Counter();ledger=[]
    for i in range(1200):
        t=cfg["sampler_times"][(i*17)%50];start=4*(visits[t]%4);visits[t]+=1
        ledger.append(dict(step=i,cases=per_time[t][start:start+4],seed=cfg["training_noise_seed"]+i,t=t))
    assert set(Counter(c for r in ledger for c in r["cases"]).values())=={300}
    assert set(Counter((c,r["t"]) for r in ledger for c in r["cases"]).values())=={6}
    d,n,s,f,e,info,a=runtime(cfg);initial=digest(a);frozen=digest(d.model);probes=[data[c] for c in cfg["train_cases"]]
    cs=[curves(cfg,d,a,n,s,f,probes,"model",0,out)];history=[];obs=[];actual=[];checks=[]
    def batch(row):
        v=[data[c] for c in row["cases"]]
        return torch.tensor(np.stack([x["model_target"] for x in v]),device="cuda"),torch.tensor(np.stack([x["music"] for x in v]),device="cuda"),packet(v)
    if not replay:
        (out/"snapshots").mkdir();checkpoint(a,out/"initial.pt",cfg,"COVER",0);opt=torch.optim.AdamW(a.parameters(),lr=cfg["learning_rate"])
        for row in ledger:
            i=row["step"];watch=i in cfg["snapshot_steps"]
            if watch:before=vector(a).clone();checkpoint(a,out/f"snapshots/{i}_pre.pt",cfg,"COVER",i)
            y,m,ref=batch(row);opt.zero_grad(set_to_none=True);loss,parts=objective(d,a,y,m,ref,row["t"],row["seed"]);loss.backward();g=gradient_norms(a) if watch else None
            torch.nn.utils.clip_grad_norm_(a.parameters(),1.,error_if_nonfinite=True);opt.step();history.append(dict(row,loss=float(loss.detach()),parts=parts))
            if watch:
                checkpoint(a,out/f"snapshots/{i}_post.pt",cfg,"COVER",i+1);obs.append(dict(step=i,loss=float(loss.detach()),parts=parts,gradients=g,update_norm=float((vector(a)-before).norm())))
            if i+1 in cfg["curve_updates"]:
                checkpoint(a,out/f"weights_{i+1}.pt",cfg,"COVER",i+1);cs.append(curves(cfg,d,a,n,s,f,probes,"model",i+1,out))
            if (i+1)%200==0:print("R29_COVERAGE_TRAIN",i+1,float(loss.detach()),flush=True)
        checkpoint(a,out/"adapter.pt",cfg,"COVER",1200);steps=1200
    else:
        old=read(Path(replay)/"result.json");assert initial==old["initial_digest"]
        for name,h in old["files"].items():assert sha(Path(replay)/name)==h
        for item in old["observations"]:
            i=item["step"];row=ledger[i];y,m,ref=batch(row);restore(a,Path(replay)/f"snapshots/{i}_pre.pt");before=vector(a).clone();a.zero_grad(set_to_none=True)
            loss,parts=objective(d,a,y,m,ref,row["t"],row["seed"]);loss.backward();g=gradient_norms(a);restore(a,Path(replay)/f"snapshots/{i}_post.pt")
            current=dict(step=i,loss=float(loss.detach()),parts=parts,gradients=g,update_norm=float((vector(a)-before).norm()))
            checks.append(readout_check(current,item));actual.append(current);write(out/f"actual_{i}.json",current)
        for oldcurve in old["curves"][1:]:
            step=oldcurve["update"];restore(a,Path(replay)/f"weights_{step}.pt");c=curves(cfg,d,a,n,s,f,probes,"model",step,out);assert c==oldcurve;cs.append(c)
            same_arrays(out/f"curve_{step}.npz",Path(replay)/f"curve_{step}.npz")
        restore(a,Path(replay)/"adapter.pt");history=old["history"];obs=old["observations"];steps=0
    assert digest(d.model)==frozen and all(p.grad is None for p in d.model.parameters())
    result=dict(arm="COVER",parameters=sum(p.numel() for p in a.parameters()),optimizer_steps=steps,initial_digest=initial,final_digest=digest(a),frozen_digest=frozen,
        history=history,curves=cs,observations=obs,actual_readouts=actual,replay_checks=checks,per_target_presentations=300,recordings=16,music_groups=8,
        source_fit_on_original_four=fitting_gate(cs[-1]["records"],cs[0]["records"]),files=old["files"] if replay else {str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.suffix in (".pt",".npz")},
        same_total_budget_not_equal_per_source_dose=True,full_native_status="NOT_EVALUABLE",operational_qualified=False)
    if replay:assert result["final_digest"]==old["final_digest"]
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),checkpoint=info,cohort_plan_sha256=sha(root/"coverage_data/cohort_plan.json")));a.close()
    print("R29_COVERAGE_DONE",steps,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--task",choices=["data","train"],required=True);p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    (prepare if a.task=="data" else train)(read(a.config),a.root,a.out,a.replay_from)
