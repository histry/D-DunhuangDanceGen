"""Matched 2x2 source-supervision/time-support experiment; frozen single EDGE."""
import argparse,copy,json,subprocess
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.generation.targets import digest
from dunhuang.generation.source_following import following_loss,hard_keyframe_inputs
from dunhuang.research.source_identifiability.g2 import Runtime,physical,source_error
from dunhuang.sampling import constrained_ddim

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def null_certificate(rt,uid):
    rt.batch_self([uid]);geometry=rt.state["geometry"];music=torch.zeros(len(geometry),150,512,device="cuda")
    with torch.no_grad():
        delta=rt.adapter(geometry,music,visibility_policy="observed")
        null=rt.adapter(torch.zeros_like(geometry),music,visibility_policy="observed")
    if torch.count_nonzero(null)!=0 or not torch.equal(delta-null,delta):raise AssertionError("Observed NULL contract changed")
    return dict(null_delta_exact_zero=True,subtracting_null_is_identity=True,new_centering_variant_added=False)

def train(rt,manifest,prepared,out,cfg,seed):
    index=rt.index;uids=manifest["train_uids"];schedule=dict(np.load(prepared/"schedule.npz"));masks=dict(np.load(prepared/"supervision.npz"))
    initial=copy.deepcopy(rt.initial);initial_digest=digest(rt.adapter);states={};certificates={};updates=0
    scale=torch.as_tensor(rt.scale,device="cuda");bias=torch.as_tensor(rt.bias,device="cuda")
    for arm in cfg["arms"]:
        rt.adapter.load_state_dict(initial);rt.adapter.requires_grad_(True);rt.adapter.train()
        if digest(rt.adapter)!=initial_digest:raise AssertionError("Matched initialization differs")
        folder=out/arm;folder.mkdir();history=[];optimizer=torch.optim.AdamW(rt.adapter.parameters(),lr=cfg["learning_rate"])
        expanded=arm in ("M1","M3","M3_shuffle");actual=arm in ("M2","M3","M3_shuffle")
        for step,ids in enumerate(schedule["rows"]):
            selected=[uids[int(i)] for i in ids]
            wrong=[uids[int(i)] for i in schedule["donors"][step]] if arm=="M3_shuffle" else None
            target,music,geometry,reference,observed,keys=rt.batch_self(selected,wrong)
            if expanded:keys=torch.from_numpy(np.stack([masks[uid+"__expanded"] for uid in selected])).cuda()
            t=torch.as_tensor(schedule["ddim_t" if actual else "old_t"][step],device="cuda")
            generator=torch.Generator(device="cuda").manual_seed(seed+step)
            noise=torch.randn(target.shape,generator=generator,device="cuda")
            predicted=rt.model(rt.diffusion.q_sample(target,t,noise),music,t,cond_drop_prob=0.)
            loss,components=rt.loss(target,predicted,geometry)
            follow=following_loss(predicted,reference,observed,keys,scale,bias,"same_recording_pose_exact")
            loss=loss+cfg["following_weight"]*follow
            if not torch.isfinite(loss):raise ValueError("Nonfinite G2-M loss")
            optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(rt.adapter.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError("Nonfinite G2-M gradients")
            optimizer.step();updates+=1
            if (step+1)%50==0:
                history.append(dict(step=step+1,total=float(loss.detach()),following=float(follow.detach()),gradient_norm=float(norm),**components))
                write(folder/"history.json",history);print("G2M_TRAIN",arm,step+1,float(loss.detach()),flush=True)
        states[arm]=copy.deepcopy(rt.adapter.state_dict());rt.adapter.eval();rt.adapter.requires_grad_(False)
        certificates[arm]=null_certificate(rt,uids[0])
        torch.save(dict(state=states[arm],mode="cross",condition_schema="native_event_music_v1",visibility_policy="observed",
            condition_policy="completion_upper8",base_sha256=rt.assets["checkpoint"]["sha256"],task_binding=read(prepared/"protocol.json")["task_binding"],
            protocol_sha256=sha(out/"protocol.json"),selected_step=cfg["steps"],arm=arm,operational_qualified=False),folder/"adapter.pt")
    return states,certificates,updates

def evaluate(rt,states,manifest,prepared,out,cfg):
    stage_masks=dict(np.load(prepared/"supervision.npz"));records=[];preferences=[];noise_hashes={};total=0
    for case,pair in enumerate(manifest["validation_pairs"]):
        uids=[pair["a"],pair["b"]];target,music,_,_,_,_=rt.batch_self([uids[0]])
        if not np.array_equal(rt.self_data(uids[0])[1],rt.self_data(uids[1])[1]):raise ValueError("Source exchange changes actual music")
        refs=[rt.self_data(uid)[2] for uid in uids];conditions=[rt.source(uid) for uid in uids]
        for s,c in enumerate(conditions):
            folder=out/("contract_"+str(case)+"_"+str(s));folder.mkdir();write(folder/"plan.json",c[2]);write(folder/"contract.json",c[3])
        for seed in cfg["generation_seeds"]:
            folder=out/("case_"+str(case)+"_seed_"+str(seed));folder.mkdir();outputs={};data={};hashes=[]
            controls=[("bypass",None,None)]+[(arm+"_"+str(s),arm,s) for arm in cfg["arms"] for s in (0,1)]+[("hard_"+str(s),None,s) for s in (0,1)]
            for name,arm,s in controls:
                rt.batch_self([uids[0]],[uids[s or 0]])
                if arm:rt.adapter.load_state_dict(states[arm])
                rt.adapter.eval();rt.adapter.requires_grad_(False);rt.state.update(enabled=arm is not None,capture=False)
                values=torch.zeros_like(target);mask=torch.zeros_like(target,dtype=torch.bool)
                hard=name.startswith("hard_")
                if hard:
                    actual=torch.from_numpy(rt.self_data(uids[s])[0])[None].cuda()
                    values,mask=hard_keyframe_inputs(actual,conditions[s][1].cuda())
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                with torch.no_grad():sample,sampler=constrained_ddim(rt.diffusion,music,values,mask,guidance="hard" if hard else "none")
                motion,metrics=physical(rt,sample,music,folder/name,rt.assets);total+=1
                outputs[name]=motion;data[name]=metrics;hashes.append(sampler["noise_sha256"])
                errors={}
                for j,label in enumerate(("A","B")):
                    known=conditions[j][0].known[0,:,79:127].reshape(150,8,6).all(-1).numpy()
                    expanded=stage_masks[uids[j]+"__expanded"]
                    errors[label]=dict(full=source_error(motion,refs[j],np.ones((150,8),bool)),
                        provided=source_error(motion,refs[j],known),hidden=source_error(motion,refs[j],~known),
                        keys=source_error(motion,refs[j],conditions[j][1][0].numpy()),stages=source_error(motion,refs[j],expanded))
                records.append(dict(case=case,seed=seed,control=name,**metrics,errors=errors,sampler=sampler))
                noise_hashes[str(case)+"_"+str(seed)+"_"+name]=sampler["noise_sha256"]
                print("G2M_GENERATED",case,seed,name,flush=True)
            if len(set(hashes))!=1:raise AssertionError("Matched counterfactual noise differs")
            base=next(r for r in records if r["case"]==case and r["seed"]==seed and r["control"]=="bypass")
            for arm in cfg["arms"]+["hard"]:
                a=next(r for r in records if r["case"]==case and r["seed"]==seed and r["control"]==arm+"_0")
                b=next(r for r in records if r["case"]==case and r["seed"]==seed and r["control"]==arm+"_1")
                pa=a["errors"]["B"]["full"]-a["errors"]["A"]["full"];pb=b["errors"]["A"]["full"]-b["errors"]["B"]["full"]
                gain=(base["errors"]["A"]["full"]+base["errors"]["B"]["full"]-a["errors"]["A"]["full"]-b["errors"]["B"]["full"])/2
                activity=min(a["dynamics"]["upper8_angular_speed_mean_radps"],b["dynamics"]["upper8_angular_speed_mean_radps"])/max(base["dynamics"]["upper8_angular_speed_mean_radps"],1e-12)
                preferences.append(dict(case=case,seed=seed,arm=arm,source_a_preference=pa,source_b_preference=pb,mean_preference=(pa+pb)/2,
                    gain=gain,activity_retention=activity,hidden_direction_A=b["errors"]["A"]["hidden"]-a["errors"]["A"]["hidden"],
                    hidden_direction_B=a["errors"]["B"]["hidden"]-b["errors"]["B"]["hidden"],
                    hidden_A_vs_null=base["errors"]["A"]["hidden"]-a["errors"]["A"]["hidden"],
                    hidden_B_vs_null=base["errors"]["B"]["hidden"]-b["errors"]["B"]["hidden"],
                    source_following_pass=pa>.001 and pb>.001 and gain>0 and activity>=.9))
    return dict(generations=total,native_evaluations=total,native_passes=sum(r["native_ok"] for r in records),records=records,preferences=preferences,noise_hashes=noise_hashes,
        following_pass_counts={arm:sum(p["source_following_pass"] for p in preferences if p["arm"]==arm) for arm in cfg["arms"]+["hard"]},
        evaluation_cases=len(manifest["validation_pairs"])*len(cfg["generation_seeds"]),new_blind_source=False,operational_qualified=False,g3_started=False)

def run(prepared,config,out,replay_from=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);prepared=Path(prepared)
    p=read(prepared/"protocol.json");manifest=read(checked(p["manifest"]));checked(p["supervision"]);checked(p["schedule"])
    assets=read(checked(p["assets"]));cfg=config["g2"]
    if p["config"]!=config:raise ValueError("Registered source/task/training configuration differs")
    if not p["gate"]["qualified"]:
        write(out/"protocol.json",dict(code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),config=config,
            qualified_task=ref(prepared/"protocol.json"),optimizer_calls=0,gate_closed=True))
        write(out/"result.json",dict(status="NOT_STARTED_G2M_TASK_NOT_QUALIFIED",qualification=p["gate"],optimizer_updates=0,generations=0,operational_qualified=False))
        if replay_from:write(out/"replay.json",dict(status="CLOSED_G2M_GATE_REPLAYED",optimizer_calls=0,regenerated=0))
        return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False)
    index={r["window_uid"]:r for r in manifest["rows"]};rt=Runtime(assets,index,manifest["initial_adapter"],manifest["audio_contract"])
    protocol=dict(schema="g2m_factorial_v1",code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),config=config,
        qualified_task=ref(prepared/"protocol.json"),task_binding=p["task_binding"],base=assets["checkpoint"],scaler=assets["scaler"],
        initial_adapter=manifest["initial_adapter"],initial_parameter_digest=digest(rt.adapter),frozen_parameter_digest=rt.before,
        adapter_parameters=sum(v.numel() for v in rt.adapter.parameters()),factors=dict(M0=["original_keys","50_to_949"],M1=["observed_stages","50_to_949"],
            M2=["original_keys","actual_ddim_support"],M3=["observed_stages","actual_ddim_support"],M3_shuffle=["observed_stages","actual_ddim_support","source_permuted"]),
        selected_step=cfg["steps"],checkpoint_selection="fixed_last",operational_qualified=False)
    write(out/"protocol.json",protocol)
    try:
        if replay_from:
            def forbid(*args,**kwargs):raise RuntimeError("G2-M replay optimizer forbidden")
            torch.optim.AdamW=forbid;torch.optim.Adam=forbid;torch.optim.SGD=forbid
            states={}
            for arm in cfg["arms"]:
                payload=torch.load(Path(replay_from)/arm/"adapter.pt",map_location="cuda",weights_only=True)
                if payload["protocol_sha256"]!=sha(Path(replay_from)/"protocol.json") or payload["task_binding"]!=p["task_binding"]:raise ValueError("Saved model/task binding differs")
                states[arm]=payload["state"]
            certificates={}
            for arm,state in states.items():
                rt.adapter.load_state_dict(state);rt.adapter.eval();rt.adapter.requires_grad_(False)
                certificates[arm]=null_certificate(rt,manifest["train_uids"][0])
            if certificates!=read(Path(replay_from)/"result.json")["null_centering"]:raise ValueError("NULL algebra certificate differs")
            updates=0
        else:states,certificates,updates=train(rt,manifest,prepared,out,cfg,config["seed"])
        rt.adapter.requires_grad_(False);rt.adapter.zero_grad(set_to_none=True);rt.adapter.eval()
        result=evaluate(rt,states,manifest,prepared,out,cfg)
        result.update(status="COMPLETED_G2M_FIXED_BUDGET",optimizer_updates=updates if not replay_from else read(Path(replay_from)/"result.json")["optimizer_updates"],
            replay_optimizer_updates=0,null_centering=certificates,task_scope=cfg["scope"],formal_data_v3_training_qualified=False)
        write(out/"result.json",result)
        if replay_from:
            old=Path(replay_from)
            if read(old/"result.json")!=result:raise ValueError("G2-M scientific metrics replay differs")
            count=0
            for file in old.glob("case_*/*/native.npy"):
                folder=file.parent.relative_to(old)
                for name in ("native.npy","native_fk.npy","normalized.npy"):
                    if not np.array_equal(np.load(old/folder/name),np.load(out/folder/name)):raise ValueError("G2-M regenerated arrays differ")
                    count+=1
                if read(old/folder/"physical.json")!=read(out/folder/"physical.json"):raise ValueError("G2-M Native replay differs")
            write(out/"replay.json",dict(status="FRESH_G2M_GENERATION_REPLAY",optimizer_calls=0,regenerated=result["generations"],arrays_verified=count))
        print("G2M_COMPLETED",flush=True)
    finally:rt.close()
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("prepared","config","out"):p.add_argument("--"+k,required=True)
    p.add_argument("--replay-from");a=p.parse_args();run(a.prepared,read(a.config),a.out,a.replay_from)
