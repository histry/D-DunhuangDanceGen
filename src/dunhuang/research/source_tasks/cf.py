"""Matched real-pair source CF study; original frozen EDGE and Native."""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.generation.targets import digest
from dunhuang.generation.source_following import following_loss
from dunhuang.research.source_identifiability.g2 import Runtime
from dunhuang.research.necessity.g2m import evaluate,null_certificate
from .common import read,commit,fresh,forbid_optimizers,source_protocol
from .math import cf_loss
from .source_effect import compute

def train(rt,manifest,q,out,config):
    cfg=config["g2"];pairs=read(checked(q["pairs"]))["train"];plan=dict(np.load(checked(q["schedule"])))
    if not np.array_equal(plan["alpha"],rt.diffusion.alphas_cumprod.cpu().numpy()):raise ValueError("Actual diffusion alpha differs from support audit")
    masks=dict(np.load(checked(read(checked(q["previous_preflight"]))["supervision"])))
    states={};certs={};updates=0;initial=copy.deepcopy(rt.initial);initial_id=digest(rt.adapter)
    scale=torch.as_tensor(rt.scale,device="cuda");bias=torch.as_tensor(rt.bias,device="cuda")
    for arm in cfg["arms"]:
        rt.adapter.load_state_dict(initial);rt.adapter.train();rt.adapter.requires_grad_(True)
        if digest(rt.adapter)!=initial_id:raise ValueError("Matched initial adapter differs")
        folder=out/arm;folder.mkdir();history=[];optimizer=torch.optim.AdamW(rt.adapter.parameters(),lr=cfg["learning_rate"])
        for step,(pid,tick,anchor) in enumerate(zip(plan["pairs"][:,0],plan["times"],plan["anchors"])):
            pair=pairs[int(pid)];uids=[pair["a"],pair["b"]];wrong=uids[::-1] if arm=="C3" else None
            target,music,geometry,reference,observed,_=rt.batch_self(uids,wrong)
            if not torch.equal(music[0],music[1]):raise ValueError("Paired training music must be exactly equal")
            keys=torch.from_numpy(np.stack([masks[u+"__expanded"] for u in uids])).cuda()
            valid=torch.from_numpy(masks[uids[0]+"__known"]&masks[uids[1]+"__known"]).cuda()
            generator=torch.Generator(device="cuda").manual_seed(config["seed"]+step)
            noise=torch.randn((1,150,151),generator=generator,device="cuda").expand(2,-1,-1)
            t=torch.full((2,),int(tick),device="cuda",dtype=torch.long)
            clean=target if arm=="C0" else target[int(anchor):int(anchor)+1].expand(2,-1,-1)
            noisy=rt.diffusion.q_sample(clean,t,noise)
            if arm!="C0" and not torch.equal(noisy[0],noisy[1]):raise ValueError("Counterfactual branches must share their actual state")
            y=rt.model(noisy,music,t,cond_drop_prob=0.)
            loss,components=rt.loss(target,y,geometry)
            follow=following_loss(y,reference,observed,keys,scale,bias,"same_recording_pose_exact")
            cf,margin=cf_loss(y,reference,valid,scale,bias,cfg)
            loss=loss+cfg["following_weight"]*follow+(cfg["cf_weight"]*cf if arm in ("C2","C3") else 0)
            if step==0 and arm in ("C2","C3"):
                gradients=torch.autograd.grad(cf,tuple(rt.adapter.parameters()),retain_graph=True,allow_unused=True)
                value=float(sum(v.square().sum() for v in gradients if v is not None).sqrt())
                if not np.isfinite(value) or value<=0:raise ValueError("CF objective has no finite nonzero adapter gradient")
                write(folder/"cf_gradient.json",dict(first_cf_gradient_norm=value,margin_squared_rad=float(margin),shared_noisy_state=True))
            if not torch.isfinite(loss):raise ValueError("Nonfinite CF objective")
            optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(rt.adapter.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError("Nonfinite CF gradients")
            optimizer.step();updates+=1
            if (step+1)%50==0:
                history.append(dict(step=step+1,total=float(loss.detach()),following=float(follow.detach()),cf=float(cf.detach()),margin_squared_rad=float(margin),**components))
                write(folder/"history.json",history);print("SOURCE_CF_TRAIN",arm,step+1,float(loss.detach()),flush=True)
        rt.adapter.eval();rt.adapter.requires_grad_(False);states[arm]=copy.deepcopy(rt.adapter.state_dict())
        certs[arm]=null_certificate(rt,manifest["train_uids"][0])
        torch.save(dict(state=states[arm],mode="cross",condition_schema="native_event_music_v1",visibility_policy="observed",
            condition_policy="completion_upper8",base_sha256=rt.assets["checkpoint"]["sha256"],task_binding=q["task_binding"],
            protocol_sha256=sha(out/"protocol.json"),selected_step=cfg["steps"],arm=arm,operational_qualified=False),folder/"adapter.pt")
    return states,certs,updates

def run(config,qualified,out,replay_from=None):
    out=fresh(out);q=read(Path(qualified)/"protocol.json");root,p,m=source_protocol(config);cfg=config["g2"]
    if q["config"]!=config:raise ValueError("Registered CF config changed")
    protocol=dict(code_commit=commit(),config=config,qualified=ref(Path(qualified)/"protocol.json"),task_binding=q["task_binding"],
        initial_adapter=m["initial_adapter"],base=read(config["assets"])["checkpoint"],selected_step=cfg["steps"],
        C0="M3 objective on SAME pair population/time support, two self q states",C1="Same real pair, one common q mixture state",
        C2="C1 plus bidirectional CF hinge on common observed SO(3)",C3="C2 with A/B source exchange; exact within-pair source marginal",
        optimizer_calls=0 if replay_from else None,operational_qualified=False)
    write(out/"protocol.json",protocol)
    if not q["gate"]["qualified"]:
        result=dict(status="NOT_STARTED_REAL_PAIR_OR_Q_SUPPORT_UNQUALIFIED",optimizer_updates=0,generations=0,operational_qualified=False);write(out/"result.json",result)
        if replay_from:write(out/"replay.json",dict(status="CLOSED_CF_GATE_REPLAY",optimizer_calls=0,regenerated=0,arrays_verified=0))
        return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False)
    if replay_from:forbid_optimizers()
    rt=Runtime(read(config["assets"]),{r["window_uid"]:r for r in m["rows"]},m["initial_adapter"],m["audio_contract"])
    try:
        if replay_from:
            states={};certs={}
            for arm in cfg["arms"]:
                payload=torch.load(Path(replay_from)/arm/"adapter.pt",map_location="cuda",weights_only=True)
                if payload["task_binding"]!=q["task_binding"] or payload["protocol_sha256"]!=sha(Path(replay_from)/"protocol.json"):raise ValueError("Frozen CF model binding differs")
                states[arm]=payload["state"];rt.adapter.load_state_dict(states[arm]);rt.adapter.eval()
                certs[arm]=null_certificate(rt,m["train_uids"][0])
            updates=0
        else:states,certs,updates=train(rt,m,q,out,config)
        result=evaluate(rt,states,m,root/"preflight",out,cfg)
        result.update(status="COMPLETED_FIXED_CF_BUDGET",optimizer_updates=updates if not replay_from else read(Path(replay_from)/"result.json")["optimizer_updates"],
            replay_optimizer_calls=0,null_centering=certs,formal_data_v3_training_qualified=False)
        masks=dict(np.load(checked(p["supervision"])))
        result["source_response"]=compute(out,m,masks,result)
        write(out/"result.json",result)
        if replay_from:
            old=Path(replay_from)
            if read(old/"result.json")!=result:raise ValueError("CF generation metrics differ")
            count=0
            for file in old.glob("case_*/*/native.npy"):
                folder=file.parent.relative_to(old)
                for name in ("normalized.npy","native.npy","native_fk.npy"):
                    if not np.array_equal(np.load(old/folder/name),np.load(out/folder/name)):raise ValueError("CF actual generated arrays differ")
                    count+=1
                if read(old/folder/"physical.json")!=read(out/folder/"physical.json"):raise ValueError("CF Native differs")
            write(out/"replay.json",dict(status="FRESH_CF_GENERATION_REPLAY",optimizer_calls=0,regenerated=result["generations"],arrays_verified=count))
        print("SOURCE_CF_COMPLETED",result["following_pass_counts"],flush=True)
    finally:rt.close()
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("config","qualified","out"):p.add_argument("--"+k,required=True)
    p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.qualified,a.out,a.replay_from)
