"""A2-V3: separately registered tasks, Train-only support, pure operator baselines."""
import argparse,collections,json,subprocess
from pathlib import Path
import numpy as np
from dunhuang.io import sha,write
from dunhuang.alignment.audio_clock import music_sample
from dunhuang.alignment.partial_path import exact_path
from dunhuang.research.transfer_audit.contracts import validate_song_task
from .support import vectors,fit,build

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def populations(rows,c,cfg):
    result={}
    for role in ("train","val"):
        ids=[i for i,r in enumerate(rows) if r["role"]==role and c["eligible"][i]]
        n=len({rows[i]["recording_uid"] for i in ids});g=len({rows[i]["source_group"] for i in ids})
        result[role]=dict(queries=len(ids),recordings=n,groups=g,population_sufficient=len(ids)>=cfg["min_queries"] and n>=cfg["min_recordings"] and g>=cfg["min_"+role+"_groups"])
    return result

def cross_nuisance(stats,i,j):
    # Only music-side query and motion-side candidate. No target-motion selector distance.
    return np.array([abs(stats["counts"][i,0]/12-stats["counts"][j,1]/20),
        np.mean(abs(stats["quantiles"][i,0]-stats["quantiles"][j,1])),
        np.sum(abs(stats["hist"][i,0]-stats["hist"][j,1])),
        np.log(max(stats["energy"][i,0],0)+1e-6),
        np.log(max(stats["energy"][j,1],0)+1e-6),stats["onset_density"][i]],float)

def retrieval(score,c,rows,role,pop,cfg):
    ids=[i for i,r in enumerate(rows) if r["role"]==role and c["eligible"][i]]
    if not pop[role]["population_sufficient"]:return dict(status="NOT_EVALUABLE_INSUFFICIENT_POPULATION",queries=len(ids),R1=None,MRR=None)
    group_values=collections.defaultdict(list);r1=[];mrr=[]
    for i in ids:
        value=score[i];best=value[c["positive"][i]].max();rank=1+int((value[c["valid"][i]&~c["positive"][i]]>=best-1e-7).sum())
        r1.append(float(rank==1));mrr.append(1/rank);group_values[rows[i]["source_group"]].append((rank==1,1/rank))
    means=np.array([np.mean(group_values[g],axis=0) for g in sorted(group_values)])
    rng=np.random.default_rng(cfg["bootstrap_seed"]);draws=np.stack([means[rng.integers(0,len(means),len(means))].mean(0) for _ in range(cfg["bootstrap_replicates"])])
    return dict(status="EXPOSED_DEVELOPMENT_SOURCE_TIME_ONLY",queries=len(ids),R1=float(np.mean(r1)),MRR=float(np.mean(mrr)),
        group_balanced_R1=float(means[:,0].mean()),group_balanced_MRR=float(means[:,1].mean()),
        group_bootstrap95=np.quantile(draws,[.025,.975],axis=0).tolist(),independent_known_music_groups=len(means),
        semantic_labels=False,fresh_blind=False,tie_policy="pessimistic")

def operators(rows,bank,stats,c,pop,cfg):
    x=[];labels=[];weights=[];groups=collections.Counter(r["source_group"] for i,r in enumerate(rows) if r["role"]=="train" and c["eligible"][i])
    for i,r in enumerate(rows):
        if r["role"]!="train" or not c["eligible"][i]:continue
        slots=np.flatnonzero(c["valid"][i]);negative=len(slots)-1
        for slot in slots:
            j=c["candidates"][i,slot];x.append(cross_nuisance(stats,i,j));labels.append(float(slot==0))
            weights.append(.5/groups[r["source_group"]]/(1 if slot==0 else negative))
    if not x:return {},{},None
    x=np.stack(x);labels=np.asarray(labels);weights=np.asarray(weights);mean=np.average(x,axis=0,weights=weights)
    std=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=weights)),1e-6);design=np.c_[np.ones(len(x)),(x-mean)/std]
    coef=np.linalg.solve(design.T@(design*weights[:,None])+np.eye(7),design.T@(labels*weights))
    scores={k:np.zeros(c["valid"].shape,float) for k in ("relative_clock_partial_path","train_nuisance_linear","content_zero")}
    for i in np.flatnonzero(c["eligible"]):
        for slot in np.flatnonzero(c["valid"][i]):
            j=c["candidates"][i,slot];mt=bank["mt"][i][bank["mv"][i]];dt=bank["dt"][j][bank["dv"][j]]
            dist=abs(mt[:,None]-dt[None,:]);path=exact_path(.2*dist/.5,.5,.5,dist<=.5+1e-6)
            scores["relative_clock_partial_path"][i,slot]=-path["cost"]/(len(mt)+len(dt))
            scores["train_nuisance_linear"][i,slot]=np.r_[1,(cross_nuisance(stats,i,j)-mean)/std]@coef
    report={name:{role:retrieval(value,c,rows,role,pop,cfg) for role in ("train","val")} for name,value in scores.items()}
    return report,scores,dict(mean=mean.tolist(),std=std.tolist(),coefficients=coef.tolist(),scope="Train source-group balanced nuisance ridge; not SDMR")

def run(config,out,replay_from=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);root=Path(config["source_a2"]);replay=Path(config["source_a2_replay"])
    published=read("reports/transfer_audit_summary.json")
    for name in ("nodes.npz","nuisance.npz","manifest.json","train_statistics.json"):
        if sha(root/name)!=published["evidence"]["a2/"+name]:raise ValueError("Frozen A2 source seal differs")
    if read(replay/"replay.json")["status"]!="FRESH_V3_RAW_AUDIO_MOTION_CANDIDATE_REPLAY":raise ValueError("Raw observation replay absent")
    rows=read(root/"manifest.json")["rows"];validate_song_task(rows);bank=dict(np.load(root/"nodes.npz"));stats=dict(np.load(root/"nuisance.npz"))
    cfg=config["a2"];z=vectors(stats);train=np.array([r["role"]=="train" for r in rows]);model=fit(z,train,cfg["shrinkage"])
    np.savez_compressed(out/"support_model.npz",**model)
    protocol=dict(schema="a2_v3_train_only_support",code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        config=config,source_execution_commit=published["a2_execution_commit"],input_sha256={name:sha(root/name) for name in ("manifest.json","nodes.npz","nuisance.npz")},
        selector_features_for_scoring=False,encoder_training=False,test_tensor_reads=0,old_v2_thresholds_changed=False)
    write(out/"protocol.json",protocol);results={}
    available=bank["mv"].any(-1)&bank["dv"].any(-1);ticks=np.array([music_sample(r) for r in rows])
    for task in ("time","association"):
        c,support,dist=build(rows,ticks,z,model,cfg,task,available)
        pop=populations(rows,c,cfg);op,scores,linear=operators(rows,bank,stats,c,pop,cfg)
        np.savez_compressed(out/(task+".npz"),**c,selector_distance=dist,**scores)
        write(out/(task+"_nuisance_linear.json"),linear)
        results[task]=dict(populations=pop,train_support=support,operator_results=op,eligible=int(c["eligible"].sum()),
            label_scope="verified same-song source-time" if task=="time" else "verified data association, not artistic incompatibility",
            common_population_for_all_operators=True,encoder_comparison_started=False)
        print("A2_V3_TASK",task,json.dumps(pop),flush=True)
    result=dict(schema="a2_v3_support_results",tasks=results,windows=len(rows),old_v2_unchanged=True,
        train_statistics_fit_calls=1,nuisance_linear_fits=sum(bool(v["operator_results"]) for v in results.values()),
        optimizer_calls=0,encoder_forward_calls=0,operational_qualified=False)
    write(out/"result.json",result)
    if replay_from:
        old=Path(replay_from)
        if read(old/"result.json")!=result:raise ValueError("A2-V3 support replay differs")
        for name in ("support_model.npz","time.npz","association.npz"):
            a=dict(np.load(old/name));b=dict(np.load(out/name))
            if set(a)!=set(b) or any(not np.array_equal(a[k],b[k],equal_nan=True) for k in a):raise ValueError("A2-V3 operator arrays differ")
        write(out/"replay.json",dict(status="FRESH_TRAIN_SUPPORT_AND_OPERATOR_REPLAY",optimizer_calls=0,encoder_forward_calls=0))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.out,a.replay_from)
