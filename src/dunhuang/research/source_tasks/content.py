"""Fixed-candidate content experiment; new normalization and matched parameter budget."""
import argparse,collections,copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.generation.targets import digest
from .common import read,commit,fresh,forbid_optimizers,same_arrays
from .content_model import ContentModel

def ranks(scores,c,i):
    positive=scores[i][c["positive"][i]].max()
    return 1+int((scores[i][c["valid"][i]&~c["positive"][i]]>=positive-1e-7).sum())

def metrics(scores,baseline,c,rows,cfg,available=None):
    ids=[i for i,r in enumerate(rows) if r["role"]=="val" and c["eligible"][i] and (available is None or available[i])]
    groups=collections.defaultdict(list);recordings={rows[i]["recording_uid"] for i in ids}
    for i in ids:
        rank=ranks(scores,c,i);old=ranks(baseline,c,i)
        groups[rows[i]["source_group"]].append([float(rank==1),1/rank,float(rank==1)-float(old==1),1/rank-1/old])
    if len(ids)<30 or len(recordings)<15 or len(groups)<2:return dict(status="NOT_EVALUABLE",queries=len(ids),music_groups=len(groups))
    means=np.array([np.mean(groups[g],0) for g in sorted(groups)]);rng=np.random.default_rng(cfg["bootstrap_seed"])
    draws=np.stack([means[rng.integers(0,len(means),len(means))].mean(0) for _ in range(cfg["bootstrap_replicates"])])
    lower,upper=np.quantile(draws,[.025,.975],axis=0)
    return dict(status="EXPOSED_DEVELOPMENT_OFFLINE_CONTENT",queries=len(ids),recordings=len(recordings),music_groups=len(groups),
        group_R1=float(means[:,0].mean()),group_MRR=float(means[:,1].mean()),
        delta_R1=float(means[:,2].mean()),delta_MRR=float(means[:,3].mean()),
        delta95=[[float(lower[2]),float(lower[3])],[float(upper[2]),float(upper[3])]],
        gain_lower_bounds_positive=bool(lower[2]>0 and lower[3]>0),fresh_blind=False,semantic_event_ground_truth=False)

def encode_val(model,arrays,rows):
    ids=[i for i,r in enumerate(rows) if r["role"]=="val"];lookup=np.full(len(rows),-1,int);lookup[ids]=np.arange(len(ids))
    banks={}
    with torch.no_grad():
        for branch,mask in (("music","mm"),("motion","dm")):
            parts=[[],[],[]]
            for start in range(0,len(ids),32):
                selected=ids[start:start+32]
                values=torch.from_numpy(arrays[branch][selected]).cuda();known=torch.from_numpy(arrays[mask][selected]).cuda()
                result=getattr(model,branch)(values,known)
                for a,b in zip(parts,result):a.append(b)
            banks[branch]=tuple(torch.cat(v) for v in parts)
    return banks,lookup

def score_val(model,banks,lookup,c,rows,cfg,swap=None):
    scores=np.zeros(c["valid"].shape,np.float32)
    ids=[i for i,r in enumerate(rows) if r["role"]=="val" and c["eligible"][i] and (swap is None or swap[i]>=0)]
    with torch.no_grad():
        for start in range(0,len(ids),128):
            qs=np.array(ids[start:start+128]);mi=qs if swap is None else swap[qs];di=c["candidates"][qs]
            music=tuple(v[torch.as_tensor(lookup[mi],device="cuda")] for v in banks["music"])
            motion=tuple(v[torch.as_tensor(lookup[di],device="cuda")] for v in banks["motion"])
            scores[qs]=model.score(music,motion,cfg["local_weight"]).cpu().numpy()
    return scores

def run(config,prepared,out,replay_from=None):
    out=fresh(out);p=read(Path(prepared)/"protocol.json");cfg=config["content"]
    if p["config"]!=config:raise ValueError("Content task config differs")
    manifest=read(checked(p["manifest"]));rows=manifest["rows"];checked(p["normalizer"]);checked(p["observations"]);checked(p["schedule"])
    write(out/"protocol.json",dict(code_commit=commit(),config=config,prepared=ref(Path(prepared)/"protocol.json"),
        task_binding=p["task_binding"],normalizer=p["normalizer"],checkpoint_selection="fixed_last",
        encoder_initialization="same new random weights; no historical checkpoint inherited",dropout=0.,
        dual="shared encoder weights at both physical time grids; more FLOPs, same parameters",operational_qualified=False))
    if not p["gate"]["qualified"]:
        result=dict(status="NOT_STARTED_CONTENT_TASK_UNQUALIFIED",optimizer_updates=0,operational_qualified=False);write(out/"result.json",result)
        if replay_from:write(out/"replay.json",dict(status="CLOSED_CONTENT_GATE_REPLAY",optimizer_calls=0))
        return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False)
    if replay_from:forbid_optimizers()
    arrays=dict(np.load(p["observations"]["path"]));plan=dict(np.load(p["schedule"]["path"]))
    cs={task:dict(np.load(checked(manifest["candidate_files"][task]))) for task in ("time","association")}
    torch.manual_seed(config["seed"]);torch.cuda.manual_seed_all(config["seed"])
    initial_model=ContentModel("15");initial=copy.deepcopy(initial_model.state_dict());initial_id=digest(initial_model);parameters=sum(v.numel() for v in initial_model.parameters());del initial_model
    results={};updates=0
    for arm in cfg["arms"]:
        rate="dual" if arm=="dual_shuffle" else arm
        model=ContentModel(rate).cuda();model.load_state_dict(initial)
        if digest(model)!=initial_id or sum(v.numel() for v in model.parameters())!=parameters:raise ValueError("Rate comparison changed capacity/initialization")
        folder=out/arm;folder.mkdir()
        if replay_from:
            payload=torch.load(Path(replay_from)/arm/"content.pt",map_location="cuda",weights_only=True)
            if payload["task_binding"]!=p["task_binding"] or payload["normalizer_sha256"]!=p["normalizer"]["sha256"] or payload["protocol_sha256"]!=sha(Path(replay_from)/"protocol.json"):raise ValueError("Content checkpoint lineage differs")
            model.load_state_dict(payload["state"])
        else:
            model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"]);history=[]
            for step,qs in enumerate(plan["queries"]):
                task=("time","association")[int(plan["task"][step])];c=cs[task];di=c["candidates"][qs];mi=plan["wrong_music"][step] if arm=="dual_shuffle" else qs
                def tensor(name,ids):return torch.from_numpy(arrays[name][ids]).cuda()
                m=model.music(tensor("music",mi),tensor("mm",mi));flat=di.reshape(-1)
                d=model.motion(tensor("motion",flat),tensor("dm",flat))
                d=tuple(v.reshape(len(qs),4,*v.shape[1:]) for v in d)
                scores=model.score(m,d,cfg["local_weight"])/cfg["temperature"]
                valid=torch.from_numpy(c["valid"][qs]).cuda();scores=scores.masked_fill(~valid,-1e4)
                loss=F.cross_entropy(scores,torch.zeros(len(qs),device="cuda",dtype=torch.long))
                if not torch.isfinite(loss):raise ValueError("Nonfinite content objective")
                optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError("Nonfinite content gradient")
                optimizer.step();updates+=1
                if (step+1)%50==0:
                    history.append(dict(step=step+1,loss=float(loss.detach())));write(folder/"history.json",history);print("CONTENT_TRAIN",arm,step+1,float(loss.detach()),flush=True)
            torch.save(dict(state=model.state_dict(),rate=rate,task_binding=p["task_binding"],normalizer_sha256=p["normalizer"]["sha256"],
                protocol_sha256=sha(out/"protocol.json"),selected_step=cfg["steps"],parameters=parameters,operational_qualified=False),folder/"content.pt")
        model.eval();model.requires_grad_(False);banks,lookup=encode_val(model,arrays,rows);tasks={};saved={}
        for task,c in cs.items():
            scores=score_val(model,banks,lookup,c,rows,cfg);swap=score_val(model,banks,lookup,c,rows,cfg,arrays["swap"])
            tasks[task]=dict(vs_nuisance=metrics(scores,c["train_nuisance_linear"],c,rows,cfg),
                vs_clock=metrics(scores,c["relative_clock_partial_path"],c,rows,cfg),
                feature_swap_same_subset=metrics(swap,scores,c,rows,cfg,arrays["swap"]>=0),
                feature_swap_scope="Different music at same source clock grid; not energy preserving")
            saved[task]=scores;saved[task+"_feature_swap"]=swap
        np.savez_compressed(folder/"scores.npz",**saved);results[arm]=tasks
        del model,banks;torch.cuda.empty_cache()
    result=dict(status="COMPLETED_FIXED_CONTENT_BUDGET",parameters_per_arm=parameters,arms=results,
        optimizer_updates=updates if not replay_from else read(Path(replay_from)/"result.json")["optimizer_updates"],
        replay_optimizer_calls=0,test_tensor_reads=0,historical_checkpoint_reused=False,operational_qualified=False)
    write(out/"result.json",result)
    if replay_from:
        old=Path(replay_from)
        if read(old/"result.json")!=result:raise ValueError("Content metrics replay differs")
        for arm in cfg["arms"]:same_arrays(old/arm/"scores.npz",out/arm/"scores.npz")
        write(out/"replay.json",dict(status="FRESH_FROZEN_CONTENT_SCORE_REPLAY",optimizer_calls=0,arms_verified=len(cfg["arms"])))
    print("CONTENT_COMPLETED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("config","prepared","out"):p.add_argument("--"+k,required=True)
    p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.prepared,a.out,a.replay_from)
