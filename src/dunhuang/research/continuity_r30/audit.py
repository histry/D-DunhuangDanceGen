"""Source-matched physical readouts and a bounded actual-state propagation diagnostic."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import runtime,values,parent,checked_parent
from .metrics import measure
from dunhuang.research.time_r29.common import packet
from dunhuang.research.state_r27.common import read,write,sha,fresh,commit,forbid_optimizers,same_arrays,restore
from dunhuang.research.state_r27.trace import sample,trace_arrays

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();p=parent(cfg);d,n,s,f,e,info,a=runtime(cfg)
    records=[];arrays={}
    prior={"aist":[("B0","evaluation_aist","B0"),("B1","evaluation_aist","B1"),("R0","evaluation_coverage","COVER")],
           "fine":[("F0","evaluation_fine","F0"),("F1","evaluation_fine","F1")]}
    for domain in ("aist","fine"):
        for role in ("train","holdout"):
            data,items=values(cfg,domain,role)
            for i,row in enumerate(data):
                raw=row["model_raw"];points=row["model_points"];prefix=f"{domain}_{role}_{i}"
                with torch.no_grad():
                    normalized=n.normalize(torch.tensor(raw[None],device="cuda").clone())
                    roundtrip=n.unnormalize(normalized.clone());rp=f(roundtrip,s)[0,:,:22].cpu().numpy()
                arrays[prefix+"_source22"]=points;arrays[prefix+"_roundtrip22"]=rp
                records.append(dict(domain=domain,role=role,item=items[i],arm="Source",seed=None,
                    result=measure(raw,points,raw,points),scope="UNCLIPPED_SOURCE_MODEL0_OFFICIAL22_FK; NOT_NATIVE24"))
                records.append(dict(domain=domain,role=role,item=items[i],arm="roundtrip",seed=None,
                    result=measure(roundtrip[0].cpu().numpy(),rp,raw,points),scope="ACTUAL_OFFICIAL_NORMALIZER_ROUNDTRIP"))
                for arm,folder,oldarm in prior[domain]:
                    old=np.load(checked_parent(cfg,folder+"/arrays.npz"))
                    for k in range(2):
                        stem=f"{role}_{oldarm}_correct_{k}"
                        records.append(dict(domain=domain,role=role,item=items[i],arm=arm,seed=k,
                            result=measure(old[stem+"_139"][i],old[stem+"_22"][i],raw,points),scope="FROZEN_REAL_TRAJECTORY"))
    # Exactly one new COVER correct trajectory batch; saved R29 first-seed result is the parity target.
    data,items=values(cfg,"aist","train");music=torch.tensor(np.stack([v["music"] for v in data]),device="cuda")
    restore(a,checked_parent(cfg,"models/COVER/adapter.pt"))
    seeds=read("src/dunhuang/research/time_r29/config.json")["real_eval_seeds"][0]
    final,noise,trace=sample(d,a,packet(data),music,seeds,trace=True,selected=cfg["times"])
    old=np.load(checked_parent(cfg,"evaluation_coverage/arrays.npz"))
    assert np.array_equal(final.cpu().numpy(),old["train_COVER_correct_0_normalized"])
    np.savez_compressed(out/"actual_R0.npz",final=final.cpu().numpy(),**trace_arrays(trace))
    states={"B1":dict(np.load(checked_parent(cfg,"evaluation_aist/actual_B1.npz"))),
            "R0":dict(np.load(out/"actual_R0.npz"))}
    probes=[]
    for arm,t in [("B1",19)]+[("R0",t) for t in cfg["times"]]:
        restore(a,checked_parent(cfg,"models/"+("B1" if arm=="B1" else "COVER")+"/adapter.pt"))
        a.detail=True;a.intervention=None;a.set_reference(**packet(data))
        saved=states[arm];xt=torch.tensor(saved["xt"][saved["times"].tolist().index(t)],device="cuda")
        ts=torch.full((4,),t,device="cuda",dtype=torch.long)
        with torch.no_grad():prediction=d.model(xt,music,None,ts,cond_drop_prob=0)
        assert np.array_equal(prediction.cpu().numpy(),saved[f"p_{t}_cond"])
        normal=a.details;stem=f"{arm}_{t}";arrays[stem+"_prediction"]=prediction.cpu().numpy()
        layer_readouts=[]
        for layer,z in normal.items():
            for key in ("attention","weighted_v","residual"):arrays[stem+"_"+layer+"_"+key]=z[key][0,64:160]
            att=z["attention"][0,130:132,1:];mass=att.sum(-1)
            expected=(att*np.arange(96)).sum(-1)/mass
            layer_readouts.append(dict(layer=int(layer),expected_source_time=expected.tolist(),expected_jump=float(np.diff(expected)[0]),
                local_mass=mass.tolist(),weighted_v_pair_jump_rms=float(np.sqrt(np.mean((z["weighted_v"][0,131]-z["weighted_v"][0,130])**2))),
                residual_pair_jump_rms=float(np.sqrt(np.mean((z["residual"][0,131]-z["residual"][0,130])**2)))))
        a.intervention=(7,131)
        with torch.no_grad():
            changed=d.model(xt,music,None,ts,cond_drop_prob=0)
            raw=n.unnormalize(prediction.clone());cr=n.unnormalize(changed.clone())
            points=f(raw,s)[0,:,:22].cpu().numpy();cp=f(cr,s)[0,:,:22].cpu().numpy()
        a.intervention=None;arrays[stem+"_held_reading_prediction"]=changed.cpu().numpy()
        for name,v in (("points",points),("held_reading_points",cp)):arrays[stem+"_"+name]=v
        delta=(changed[0]-prediction[0]).cpu().numpy()
        probes.append(dict(arm=arm,t=t,case=items[0],frames=[130,131],layers=layer_readouts,correct_prediction_exact=True,
            Core_prediction_delta_rms=float(np.sqrt(np.mean(delta[64:160]**2))),
            Root_prediction_delta_rms=float(np.sqrt(np.mean(delta[64:160,4:7]**2))),
            body_prediction_delta_rms=float(np.sqrt(np.mean(delta[64:160,7:]**2))),
            normal=measure(raw[0].cpu().numpy(),points,data[0]["model_raw"],data[0]["model_points"]),
            held_reading=measure(cr[0].cpu().numpy(),cp,data[0]["model_raw"],data[0]["model_points"]),
            scope="SINGLE_STATE_LAYER7_FRAME131_WEIGHTED_V_HELD_AT_FRAME130; NOT_FULL_SAMPLING_OR_JERK_CAUSAL_FRACTION"))
    z=probes[0];last=next(v for v in z["layers"] if v["layer"]==7)
    admitted=abs(last["expected_jump"])>=8 and last["weighted_v_pair_jump_rms"]>1e-4 and last["residual_pair_jump_rms"]>1e-4 and z["Core_prediction_delta_rms"]>1e-4
    result=dict(records=records,propagation=probes,reading_control_admitted=admitted,
        admission_rule="Fixed B1 case92 t19 layer7: clock jump>=8; V/residual/Core prediction RMS effects>1e-4; bounded operator experiment, not main Jerk cause",
        sampler_calls=1,generated_trajectories=4,optimizer_steps=0,noise=noise,operational_qualified=False,whole_song_generations=0,full_native_status="NOT_EVALUABLE")
    a.close();path=out/"arrays.npz";np.savez_compressed(path,**arrays)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,checkpoint=info,arrays_sha256=sha(path)))
    if replay:
        assert result==read(Path(replay)/"result.json")
        for name in ("arrays.npz","actual_R0.npz"):same_arrays(out/name,Path(replay)/name)
    print("R30_AUDIT",len(records),"READING_ADMITTED",admitted,flush=True)
if __name__=="__main__":
    q=argparse.ArgumentParser();q.add_argument("--config",required=True);q.add_argument("--out",required=True);q.add_argument("--replay-from");a=q.parse_args()
    run(read(a.config),a.out,a.replay_from)
