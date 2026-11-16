"""Few fixed conditional states: descriptive attention and explicitly diagnostic FK controls."""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from .common import runtime,rows,packet
from .dynamics import decomposition
from dunhuang.research.state_r27.common import read,write,sha,commit,fresh,forbid_optimizers,restore,metrics,same_arrays
from dunhuang.research.state_r27.trace import sample,trace_arrays

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();records=[];controls=[];arrays={}
    d,n,s,f,e,info,a=runtime(cfg,0)
    values,_=rows(cfg,root,"aist","train");music=torch.tensor(np.stack([v["music"] for v in values]),device="cuda")
    restore(a,Path(cfg["r28"])/"models/T1/adapter.pt")
    final,noise,trace=sample(d,a,packet(values),music,cfg["real_eval_seeds"][0],trace=True,selected=cfg["times"])
    old=np.load(Path(cfg["r28"])/"evaluation/arrays.npz")
    assert np.array_equal(final.cpu().numpy(),old["train_T1_correct_0_normalized"])
    np.savez_compressed(out/"actual_B0.npz",final=final.cpu().numpy(),**trace_arrays(trace))
    actuals={"B0":dict(np.load(out/"actual_B0.npz")),"B1":dict(np.load(root/"evaluation_aist/actual_B1.npz"))}
    assert sha(root/"evaluation_aist/actual_B1.npz")==read(root/"evaluation_aist/protocol.json")["trace_files"]["actual_B1.npz"]
    for arm in ("B0","B1"):
        a.beta=0 if arm=="B0" else cfg["beta"];restore(a,Path(cfg["r28"])/"models/T1/adapter.pt" if arm=="B0" else root/"models/B1/adapter.pt")
        a.record_attention=True;state=actuals[arm]
        for t in cfg["times"]:
            ts=torch.full((4,),t,device="cuda",dtype=torch.long)
            xt=torch.tensor(state["xt"][state["times"].tolist().index(t)],device="cuda")
            for mode in ("correct","wrong_clock","swapped","none"):
                a.clear();a.attention_readouts={}
                if mode!="none":a.set_reference(**packet(values,mode))
                with torch.no_grad():pred=d.model(xt,music,None,ts,cond_drop_prob=0)
                if mode=="correct":assert np.array_equal(pred.cpu().numpy(),state[f"p_{t}_cond"])
                attention=copy.deepcopy(a.attention_readouts)
                if mode=="none":assert not attention
                arrays[f"{arm}_{t}_{mode}"]=pred.cpu().numpy()
                records.append(dict(arm=arm,t=t,mode=mode,attention=attention,scope="ACTUAL_REVERSE_STATE_FROM_CORRECT_REFERENCE_TRAJECTORY; CONDITIONAL_BRANCH; ATTENTION_DESCRIPTIVE",
                    correct_prediction_exact=mode=="correct",analysed_cases=cfg["train_cases"][:2]))
    a.record_attention=False
    z=np.load(root/"evaluation_aist/arrays.npz");parent=np.load(Path(cfg["r28"])/"evaluation/arrays.npz")
    for arm in ("NoRef","B0","B1"):
        raw=(parent["train_baseline_none_0_139"][0] if arm=="NoRef" else z["train_B0_correct_0_139"][0] if arm=="B0" else z["train_B1_correct_0_139"][0]).copy()
        for label,joints in (("actual",[]),("hold_Root_orientation",[0]),("hold_Root_spine",[0,3,6,9]),("hold_Root_spine_shoulders",[0,3,6,9,13,14,16,17])):
            changed=raw.copy();rot=changed[:,7:].reshape(256,22,6)
            for j in joints:rot[:,j]=rot[64,j]
            with torch.no_grad():p=f(torch.tensor(changed[None],device="cuda"),s)[0,:,:22].cpu().numpy()
            controls.append(dict(arm=arm,control=label,held_joints=joints,held_frame=64,dynamics=decomposition(p),
                scope="FK_COUNTERFACTUAL_DIAGNOSTIC_NOT_SAMPLED_OR_ACCEPTED_MOTION_NOT_ADDITIVE_CAUSAL_ATTRIBUTION"))
            arrays[f"FK_{arm}_{label}"]=p
    # The raw positive control really contains source pose. This oracle is not a learned decoder or generator.
    from dunhuang.research.contract_r28.representation import rotations
    oracle=[]
    for i,v in enumerate(values[:2]):
        raw=v["model_raw"].copy();raw[64:160,7:]=rotations(raw[64:160,7:])
        with torch.no_grad():p=f(torch.tensor(raw[None],device="cuda"),s)[0,:,:22].cpu().numpy()
        oracle.append(dict(case=cfg["train_cases"][i],metrics=metrics(raw,p,v["model_raw"],v["model_points"]),scope="DIRECT_RAW_REFERENCE_GEOMETRY_ORACLE_NOT_TRAINED_GENERATION"))
    result=dict(attention_records=records,FK_controls=controls,raw_reference_oracle=oracle,optimizer_steps=0,sampler_calls=1,generated_trajectories=4,noise=noise,
        global_token_neutral=True,Core_outside_clock_neutral=True,operational_qualified=False)
    path=out/"arrays.npz";np.savez_compressed(path,**arrays);write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),arrays_sha256=sha(path),checkpoint=info,actual_B0_sha256=sha(out/"actual_B0.npz")))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(path,Path(replay)/path.name);same_arrays(out/"actual_B0.npz",Path(replay)/"actual_B0.npz")
    a.close();print("R29_ATTENTION_AND_FK_PROBES",len(records),len(controls),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)
