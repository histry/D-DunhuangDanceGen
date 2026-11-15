"""Fixed exposed-source transfer after Train-only routing; no selection on these outputs."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import *
from .trace import sample as row_sample
from dunhuang.research.lodge_adapter_r25.sample import sample as common_sample
from dunhuang.research.lodge_adapter_r25.replay import change_conditions
from dunhuang.research.lodge_adapter_r25.metrics import source_metrics

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    admission=read(root/"release_admission.json");models=admission["models"]
    assert admission["transfer_selection_allowed"] is False
    pop,rows,data=sources(cfg);cases=[r["case"] for r in rows]
    r24=Path(cfg["r24"]);old=dict(np.load(checked(read(r24/"main/protocol.json")["arrays"])))
    hist=dict(np.load(Path(cfg["r26"])/"samples/arrays.npz"))
    manifest=read(Path(cfg["r26"])/"artifacts_manifest.json")["files"]
    assert sha(Path(cfg["r26"])/"samples/arrays.npz")==manifest["samples/arrays.npz"]["sha256"]
    diag=read(root/"diagnostic/protocol.json")
    frozen_arrays={}
    for mode in ("none","correct","wrong"):
        name=("baseline_none" if mode=="none" else "C1_"+mode)+".npz"
        assert sha(root/"diagnostic"/name)==diag["files"][name]
        frozen_arrays[mode]=dict(np.load(root/"diagnostic"/name))["final"]
    d,n,s,f,e,info,a=runtime(cfg);frozen=digest(d.model);arrays={};records=[];noise=[];calls=0
    mask=np.zeros(256,bool);mask[64:160]=True
    def measure(dataset,arm,mode,value,refs,rpoints,items):
        raw=n.unnormalize(value.clone());points=f(raw,s)[:,:,:22];stem=f"{dataset}_{arm}_{mode}"
        for suffix,arr in (("normalized",value),("139",raw),("22",points)):arrays[stem+"_"+suffix]=arr.cpu().numpy()
        for i,item in enumerate(items):
            er=[metrics(raw[i].cpu().numpy(),points[i].cpu().numpy(),x,q) for x,q in zip(refs,rpoints)]
            legacy=[source_metrics(raw[i].cpu().numpy(),x,points[i].cpu().numpy(),q,mask) for x,q in zip(refs,rpoints)]
            records.append(dict(dataset=dataset,arm=arm,mode=mode,item=item,errors=er,legacy=legacy,
                quality=motion_metrics(raw[i].cpu().numpy(),points[i].cpu().numpy(),np.zeros(256,bool)),
                source_truth_scope="EXPOSED_PAIRED_RECORDING" if dataset=="aist" else "RETIMED_CHANGE_REFERENCE_NOT_NEW_MUSIC_TRUTH"))
    music=torch.tensor(np.stack([data[c]["music"] for c in cases]),device="cuda")
    paired_refs=[data[c]["raw139"] for c in cases];paired_points=[data[c]["source22"] for c in cases]
    for mode,v in frozen_arrays.items():measure("aist","baseline" if mode=="none" else "R26_C1",mode,torch.tensor(v,device="cuda"),paired_refs,paired_points,cases)
    conditions=change_conditions()
    for arm in ("C0","C1"):
        for mode in ("correct","swapped"):
            if arm=="C0" and mode=="swapped":continue
            v=hist[f"change_{arm}_{mode}_24041_normalized"]
            measure("change","R26_"+arm,mode,torch.tensor(v,device="cuda"),old["source139"],old["source22"],conditions)
    for arm in models:
        a.close();a=ReferenceModel(arm=="P2").cuda();restore(a,root/f"models/{arm}/adapter.pt");a.attach(d.model)
        for mode in ("correct","wrong","none"):
            ref=None if mode=="none" else packet([data[c]["raw139"][64:160,7:] for c in (cases if mode=="correct" else cases[::-1])])
            v,receipt,_=row_sample(d,a,ref,music,cfg["diagnostic_seeds"]);calls+=1
            if mode=="none":assert np.array_equal(v.cpu().numpy(),frozen_arrays["none"])
            measure("aist",arm,mode,v,paired_refs,paired_points,cases);noise.append(dict(dataset="aist",arm=arm,mode=mode,noise=receipt))
        cmusic=torch.tensor(np.stack([old["music"][m] for m,ev in conditions]),device="cuda")
        for mode in ("correct","swapped"):
            ref=packet([old["source139"][1-max(ev,0) if mode=="swapped" else max(ev,0),64:160,7:] for m,ev in conditions],drops=[ev<0 for m,ev in conditions])
            v,receipt=common_sample(d,a,ref,cmusic,cfg["transfer_seed"],arm);calls+=1
            for i in (0,3):assert np.array_equal(v[i].cpu().numpy(),old["24041_normalized"][i])
            measure("change",arm,mode,v,old["source139"],old["source22"],conditions);noise.append(dict(dataset="change",arm=arm,mode=mode,noise=receipt))
        assert digest(d.model)==frozen
        print("R27_TRANSFER",arm,flush=True)
    def get(arm,mode,m,event):
        return next(r for r in records if r["dataset"]=="change" and (r["arm"],r["mode"],r["item"])==(arm,mode,[m,event]))
    comparisons=[]
    for arm in ["R26_C1"]+models:
        for m in range(2):
            events=[]
            for ev in range(2):
                r=get(arm,"correct",m,ev);w=get(arm,"swapped",m,ev);b=get("R26_C0","correct",m,ev);base=get("R26_C0","correct",m,-1)
                x=r["legacy"][ev];wrong=w["legacy"][ev];be=b["legacy"][ev];q=r["quality"];qn=base["quality"]
                tests={k+"_gain":x[k]<=.95*be[k] for k in ("body22_rad","wrist_relative_shoulder_rmse_m","double_arm_wrist_relation_rmse_m")}
                tests.update(correct_body=x["body22_rad"]<wrong["body22_rad"],correct_wrist=x["wrist_relative_shoulder_rmse_m"]<wrong["wrist_relative_shoulder_rmse_m"],
                    activity=q["activity_mps"]>=.8*qn["activity_mps"],global_jerk=q["jerk_p95_mps3"]<=1.1*qn["jerk_p95_mps3"],
                    core_jerk=q["regions"]["core_interior_unprompted"]["p95"]<=1.1*qn["regions"]["core_interior_unprompted"]["p95"])
                events.append(dict(event=ev,tests=tests,passed=all(tests.values())))
            comparisons.append(dict(arm=arm,music=m,events=events,passed=all(x["passed"] for x in events)))
    result=dict(models=models,records=records,comparisons=comparisons,noise=noise,sampler_calls=calls,generated_trajectories=18*len(models),
        reused_trajectories=24,optimizer_steps=0,no_reference_exact=True,reference_state_write=False,selection_on_transfer=False,
        subset_scope="2 historical ChangE music groups, seed24041; not full R26 4-group qualification",
        human_reviewed=False,operational_qualified=False,full_native_status="NOT_EVALUABLE",whole_song_generations=0)
    path=out/"arrays.npz";np.savez_compressed(path,**arrays);write(out/"result.json",result)
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,admission_sha256=sha(root/"release_admission.json"),arrays_sha256=sha(path),checkpoint=info))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(path,Path(replay)/path.name)
    a.close();print("R27_TRANSFER_DONE",[(r["arm"],r["music"],r["passed"]) for r in comparisons],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)
