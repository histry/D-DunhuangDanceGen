"""A2: audible-clock coverage and a separately registered source-association challenge."""
import argparse,collections,json,subprocess
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha,write
from dunhuang.alignment.audio_clock import music_sample,same_music_time,music_seconds
from dunhuang.alignment.conditional_candidates import validate_roles,nuisance_statistics
from dunhuang.alignment.dense_study import evaluate,VIEWS
from dunhuang.alignment.local_identifiability import frozen_head
from dunhuang.alignment.event_study import make_head
from dunhuang.alignment.content_discrimination import pair_result,template_paths,fixed_score,metrics
from dunhuang.alignment.event_metrics import multi_positive_loss
from dunhuang.alignment.sampling import group_schedule,coverage
from dunhuang.planning.time_map import motion_features74

def read(p):return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))

def stats_for(source,rows,bank,manifest):
    events={};audio={};motion={};em=[];ed=[]
    event_root=Path(manifest["source_population"]["path"]).parent.parent
    clock_root=Path(manifest["source_clock_manifest"]["path"]).parent/"audio"
    seals=read(event_root/"preflight.json")["files"]
    for r in rows:
        rk=r["raw_motion"]["sha256"];ak=r["audio"]["sha256"]
        if rk not in motion:motion[rk]=motion_features74(np.load(checked(r["source_native"])))[0][:,66]
        if ak not in audio:
            p=clock_root/(ak+".npz");audio[ak]=np.load(p)["values"][:,0]
            sidecar=event_root/"data/sidecars"/(ak+".json")
            if sha(sidecar)!=seals["data/sidecars/"+sidecar.name]:raise ValueError("Frozen audio event sidecar differs")
            events[ak]=read(sidecar)["events"]
        a=r["start_frame"]+15+round(r["audio_offset_seconds"]*30)
        em.append(audio[ak][a:a+120]);ed.append(motion[rk][r["start_frame"]:r["start_frame"]+150])
    return nuisance_statistics(rows,bank,dict(energy_music=np.stack(em),energy_motion=np.stack(ed)),events)

def candidates(rows,bank,stats,strict):
    validate_roles(rows);n=len(rows);c=np.repeat(np.arange(n)[:,None],4,axis=1);p=np.zeros((n,4),bool);v=p.copy();eligible=np.zeros(n,bool)
    ticks=np.array([music_sample(r) for r in rows]);role=np.array([r["role"] for r in rows]);groups=np.array([r["source_group"] for r in rows])
    songs=np.array([r["music_uid"] for r in rows]);choreo=np.array([r["choreography_uid"] for r in rows]);available=bank["mv"].any(-1)&bank["dv"].any(-1)
    evidence=[];intersections=collections.Counter()
    for i,r in enumerate(rows):
        if not available[i]:continue
        pool=np.flatnonzero((role==role[i])&(ticks==ticks[i])&(groups!=groups[i])&(songs!=songs[i])&(choreo!=choreo[i])&available)
        if not len(pool):continue
        count=(abs(stats["counts"][pool]-stats["counts"][i])<=strict["count_tolerance"]).all(-1)
        w1=abs(stats["quantiles"][pool]-stats["quantiles"][i]).mean(-1);hist=abs(stats["hist"][pool]-stats["hist"][i]).sum(-1)
        energy=abs(np.log((np.maximum(stats["energy"][pool],0)+1e-6)/(np.maximum(stats["energy"][i],0)+1e-6)))
        onset=abs(stats["onset_density"][pool]-stats["onset_density"][i])
        rules=np.stack((count,(w1<=strict["time_w1_seconds"]).all(-1),(hist<=strict["histogram_l1"]).all(-1),(energy<=strict["energy_log_ratio"]).all(-1),onset<=strict["onset_density_per_second"]),-1)
        codes=(~rules).astype(int)@np.array([1,2,4,8,16]);intersections.update(map(int,codes))
        ok=pool[rules.all(-1)];distance=w1.sum(-1)+hist.sum(-1)+energy.sum(-1)+onset
        if len(ok):
            rank=pool[np.argsort(distance,kind="stable")];accepted=set(ok);chosen=[j for j in rank if j in accepted][:3]
            c[i,0]=i;c[i,1:1+len(chosen)]=chosen;p[i,0]=True;v[i,:1+len(chosen)]=True;eligible[i]=True
            evidence.append(dict(query=i,negative_indices=chosen,scope="different verified data association; artistic suitability UNKNOWN"))
    populations={}
    for role_name in ("train","val"):
        ids=np.flatnonzero(eligible&(role==role_name));records=len({rows[i]["recording_uid"] for i in ids});components=len({rows[i]["source_group"] for i in ids})
        populations[role_name]=dict(queries=len(ids),recordings=records,components=components,population_sufficient=len(ids)>=strict["min_queries"] and records>=strict["min_recordings"] and components>=strict["min_"+role_name+"_components"])
    return dict(candidates=c,positive=p,valid=v,eligible=eligible),dict(populations=populations,rule_failure_intersections={str(k):int(v) for k,v in intersections.items()},evidence=evidence)

def clock_audit(rows,old_evidence):
    groups={};bad_positive=0;offsets=[];coverage_rows=[]
    for r in rows:
        offsets.append(r["audio_offset_seconds"]);groups.setdefault((r["music_uid"],r["start_frame"]),[]).append(r)
    for values in groups.values():bad_positive+=sum(not same_music_time(values[0],r) for r in values[1:])
    bad_negative=sum(same_music_time(rows[e["query"]],rows[e["candidate"]]) for e in old_evidence["evidence"])
    failures={role:collections.Counter() for role in ("train","val")}
    for e in old_evidence["evidence"]:
        code=sum((1<<i) for i,k in enumerate(("count","time_distribution","histogram","energy","onset")) if not e["rules"][k])
        failures[rows[e["query"]]["role"]][str(code)]+=1
    for key in ("music_uid","choreography_uid","source_group"):
        buckets={}
        for r in rows:buckets.setdefault(r[key],[]).append(r)
        for identity,values in buckets.items():
            times=sorted({music_sample(r) for r in values});coverage_rows.append(dict(kind=key,identity=identity,role=values[0]["role"],unique_audible_starts=len(times),start_seconds=times[0]/44100,end_seconds=times[-1]/44100+5))
    return dict(old_positive_actual_clock_disagreements=bad_positive,old_negative_same_actual_music_time=bad_negative,
        offset_seconds_min=min(offsets),offset_seconds_max=max(offsets),old_rule_failure_intersections={k:dict(v) for k,v in failures.items()},
        codebook=dict(count=1,time_distribution=2,histogram=4,energy=8,onset=16),new_blind_source=False),coverage_rows

def train(source,rows,bank,c,stats,proto,out,config,replay_from=None):
    gate=all(v["population_sufficient"] for v in proto["populations"].values())
    if not gate:return dict(status="NOT_STARTED_INSUFFICIENT_V2_POPULATION",optimizer_updates=0,results={})
    reference,payload=frozen_head(Path(source)/"path_synthetic/head.pt");null=reference.null_logit.detach().clone()
    ids=np.flatnonzero(c["eligible"]);templates=np.zeros((len(rows),4,12,20),bool)
    for start in range(0,len(ids),64):
        q=ids[start:start+64];cs=c["candidates"][q]
        templates[q]=template_paths(bank,np.repeat(q,4),cs.reshape(-1),reference.band_seconds,float(.05+.9*null.sigmoid())).reshape(len(q),4,12,20)
    np.save(out/"templates.npy",templates)
    schedule=group_schedule([dict(r,role=r["role"] if c["eligible"][i] else "excluded") for i,r in enumerate(rows)],config["steps"],config["batch"],config["seed"])
    np.save(out/"training_schedule.npy",schedule);results={}
    for name in config["arms"]:
        torch.manual_seed(config["seed"]);head=make_head("strict_path",payload["config"]).cuda()
        head.null_logit.data.copy_(null);head.null_logit.requires_grad_(False);view=dict(bank,music=bank[VIEWS[name]])
        folder=out/name;folder.mkdir()
        if replay_from:head.load_state_dict(torch.load(Path(replay_from)/name/"head.pt",map_location="cuda",weights_only=True)["state"])
        else:
            opt=torch.optim.AdamW([head.music.weight,head.motion.weight],lr=config["learning_rate"])
            for step,q in enumerate(schedule):
                cs=c["candidates"][q];score=fixed_score(pair_result(head,view,np.repeat(q,4),cs.reshape(-1)),templates[q].reshape(-1,12,20),head.band_seconds)["mean_content"].reshape(len(q),4)
                positive=c["positive"][q].copy();valid=c["valid"][q]
                if name=="tcn30_random":
                    rng=np.random.default_rng(config["seed"]+step)
                    for j in range(len(q)):
                        slots=np.flatnonzero(valid[j]);positive[j,slots]=np.roll(positive[j,slots],int(rng.integers(1,len(slots))))
                loss=multi_positive_loss(score,torch.as_tensor(positive,device="cuda"),torch.as_tensor(valid,device="cuda"),.1)
                if not torch.isfinite(loss):raise ValueError("A2 objective nonfinite")
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([head.music.weight,head.motion.weight],1.);opt.step()
                if (step+1)%100==0:print("A2_TRAIN",name,step+1,flush=True)
            torch.save(dict(state=head.state_dict(),operational_qualified=False),folder/"head.pt")
        result={}
        for role in ("train","val"):
            m,values=evaluate(head,view,rows,c,templates,role);np.save(folder/(role+"_scores.npy"),values);m["label_scope"]="verified source association; artistic compatibility UNKNOWN";result[role]=m
        val_ids=np.array([i for i,r in enumerate(rows) if r["role"]=="val" and c["eligible"][i]])
        zero=np.zeros((len(val_ids),4))
        result["content_zero"]=metrics(zero,c["positive"][val_ids],c["valid"][val_ids],[rows[i] for i in val_ids])
        clock_scores=[]
        with torch.no_grad():
            for start in range(0,len(val_ids),32):
                q=val_ids[start:start+32];cs=c["candidates"][q]
                z=pair_result(head,view,np.repeat(q,4),cs.reshape(-1),"clock")
                clock_scores.append(fixed_score(z,templates[q].reshape(-1,12,20),head.band_seconds)["score"].cpu().numpy().reshape(len(q),4))
        clock_scores=np.concatenate(clock_scores)
        np.save(folder/"clock_scores.npy",clock_scores)
        result["clock_only"]=metrics(clock_scores,c["positive"][val_ids],c["valid"][val_ids],[rows[i] for i in val_ids])
        fitting=np.array([r["role"]=="train" for r in rows]);center=stats["energy"][fitting].mean(0);scale=np.maximum(stats["energy"][fitting].std(0),1e-6)
        energy=(stats["energy"]-center)/scale
        energy_scores=-abs(energy[val_ids,0,None]-energy[c["candidates"][val_ids],1])
        np.save(folder/"energy_scores.npy",energy_scores)
        result["mean_energy_only"]=metrics(energy_scores,c["positive"][val_ids],c["valid"][val_ids],[rows[i] for i in val_ids])
        # Content substitution preserves the receiving time grid and observation counts.
        swapped=dict(view);swapped["music"]=view["music"].copy()
        for i in ids:
            donor=c["candidates"][i][1];a=np.flatnonzero(bank["mv"][i]);b=np.flatnonzero(bank["mv"][donor])
            if len(a)!=len(b):raise ValueError("Feature-swap count mismatch")
            swapped["music"][i,a]=view["music"][donor,b]
        m,values=evaluate(head,swapped,rows,c,templates,"val");np.save(folder/"feature_swap_scores.npy",values);result["feature_swap"]=m
        results[name]=result
    return dict(status="COMPLETED_FIXED_BUDGET_V2_SHADOW",optimizer_updates=0 if replay_from else config["steps"]*len(config["arms"]),results=results,operational_qualified=False)

def run(source,out,config,replay_from=None):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);source=Path(source)
    summary=read(source.parent/"public_summary.json")
    for name in ("a1/data/manifest.json","a1/data/nodes.npz","a1/data/conditional.npz"):
        if sha(source.parent/name)!=summary["evidence"][name]:raise ValueError("Frozen dense input seal differs")
    manifest=read(source/"data/manifest.json");rows=manifest["rows"];validate_roles(rows)
    bank=dict(np.load(source/"data/nodes.npz"));old=read(source/"data/candidate_evidence.json")
    audit,coverage_rows=clock_audit(rows,old);stats,_=stats_for(source,rows,bank,manifest);c,e=candidates(rows,bank,stats,config["strict"])
    write(out/"clock_coverage.json",coverage_rows);write(out/"candidate_evidence.json",e);np.savez_compressed(out/"candidates.npz",**c);np.savez_compressed(out/"nuisance.npz",**stats)
    protocol=dict(schema="content_identifiability_v2",code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),config=config,source_manifest=ref(source/"data/manifest.json"),source_nodes=ref(source/"data/nodes.npz"),populations=e["populations"],new_blind_source=False,label_scope="source association, not artistic incompatibility")
    write(out/"protocol.json",protocol)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    training=train(config["event_source"],rows,bank,c,stats,protocol,out,config,replay_from)
    result=dict(status="COMPLETED_A2_SUPPORT_AUDIT",windows=len(rows),clock=audit,populations=e["populations"],v2_rule_failure_intersections=e["rule_failure_intersections"],
        original_a1_thresholds_changed=False,new_v2_task=True,training=training,operational_qualified=False)
    if replay_from:
        original=Path(replay_from)
        for name in ("candidates","nuisance"):
            a=np.load(original/(name+".npz"));b=np.load(out/(name+".npz"))
            if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError("A2 fresh source audit replay differs")
        old_result=read(original/"result.json")
        if training["status"].startswith("COMPLETED"):result["training"]["optimizer_updates"]=old_result["training"]["optimizer_updates"]
        if result!=old_result:raise ValueError("A2 metrics replay differs")
        for p in original.glob("*/*scores.npy"):
            if not np.array_equal(np.load(p),np.load(out/p.relative_to(original))):raise ValueError("A2 head prediction replay differs")
        write(out/"replay.json",dict(status="FRESH_A2_SOURCE_AND_HEAD_REPLAY_VERIFIED",optimizer_calls=0))
    write(out/"result.json",result);print("A2_COMPLETED",json.dumps(e["populations"]),flush=True)

def main():
    p=argparse.ArgumentParser()
    for k in ("source","out","config"):p.add_argument("--"+k,required=True)
    p.add_argument("--replay-from");a=p.parse_args()
    if a.replay_from:
        def forbid(*args,**kwargs):raise RuntimeError("A2 replay optimizer forbidden")
        torch.optim.AdamW=forbid;torch.optim.Adam=forbid;torch.optim.SGD=forbid
    run(a.source,a.out,read(a.config),a.replay_from)
if __name__=="__main__":main()
