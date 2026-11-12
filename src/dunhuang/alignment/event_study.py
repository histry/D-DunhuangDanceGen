"""Pre-registered source-clock event retrieval, controls and representation probes.
No model-selected event labels. No Test extraction, Val selection, or Planner routing.
"""
import argparse,copy,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from dunhuang.io import sha,write,checked
from .clock_data import read,ref
from .clock_metrics import grouped,clock_report,correlation
from .clock_experiment import digest,paired_scores
from .model import RetrievalModel
from .sampling import group_schedule,coverage
from .transport import ClockNullTransport
from .partial_path import StrictPartialPath,exact_path
from .event_data import prepare
from .event_metrics import multi_positive_loss,proxy_retrieval,independent_metrics
from .event_targets import validate_anchors,train_lags
from .event_weak_labels import weak_loss
from .event_perturb import construct,batch as synthetic_batch,supervised_path_loss,metrics as synthetic_metrics

def make_head(mode,config,width=144):
    if mode=="sync_cold":return nn.Sequential(nn.Linear(width*4,128),nn.GELU(),nn.Linear(128,1))
    cls=ClockNullTransport if mode=="null_ot" else StrictPartialPath
    kwargs=dict(width=width,band_seconds=config["band_seconds"],epsilon=config["epsilon"],iterations=config["sinkhorn_iterations"])
    if mode=="null_ot":kwargs["allow_simultaneous"]=True
    return cls(**kwargs)

def score(head,mode,bank,queries,candidates,control=None):
    m=bank["music"][queries];d=bank["motion"][candidates]
    mv=bank["mv"][queries];dv=bank["dv"][candidates]
    mt=bank["mt"][queries];dt=bank["dt"][candidates]
    if control=="zero":
        m=torch.zeros_like(m);d=torch.zeros_like(d)
    elif control=="shuffle":
        # Keep event clocks, masks and population. Shuffle contents only at observed slots.
        m=m.clone();d=d.clone()
        for i in range(len(m)):
            ids=torch.nonzero(mv[i]).flatten();m[i,ids]=m[i,ids.flip(0)]
            ids=torch.nonzero(dv[i]).flatten();d[i,ids]=d[i,ids.flip(0)]
    elif control=="swap":
        # Circularly swap source contents while preserving the receiving clock/mask.
        # Interpolate donor slot index (not physical time); this is a destructive control.
        d=d.clone()
        donors=bank["donors"][candidates]
        for i,donor in enumerate(donors):
            source=bank["motion"][donor][bank["dv"][donor]]
            slots=torch.nonzero(dv[i]).flatten()
            values=torch.nn.functional.interpolate(source.T[None],size=len(slots),mode="linear",align_corners=True)[0].T
            d[i,slots]=values
    if mode=="sync_cold":
        distance=(mt[:,:,None]-dt[:,None,:]).abs().masked_fill(~dv[:,None,:],1e5)
        ix=distance.argmin(-1);chosen=d.gather(1,ix[...,None].expand(-1,-1,d.shape[-1]))
        x=torch.cat((m,chosen,m-chosen,m*chosen),-1)
        values=head(x).squeeze(-1)
        return (values*mv).sum(-1)/mv.sum(-1).clamp_min(1),None
    result=head(m,d,mv,dv,mt,dt)
    return result["score"],result

def matrix(head,mode,bank,ids,control=None):
    count=bank["candidates"].shape[1];values=[]
    for start in range(0,len(ids),24):
        q=ids[start:start+24]
        query=np.repeat(q,count);candidate=bank["candidates"][q].detach().cpu().numpy().reshape(-1)
        value,_=score(head,mode,bank,query,candidate,control)
        values.append(value.reshape(len(q),count).detach().cpu().numpy())
    return np.concatenate(values)

def evaluate(head,mode,bank,ids,rows):
    with torch.no_grad():
        values=matrix(head,mode,bank,ids)
        pos=bank["positive"][ids].cpu().numpy();valid=bank["valid"][ids].cpu().numpy()
        result=proxy_retrieval(values,pos,valid)
        wins=[]
        for s,p,v in zip(values,pos,valid):
            positive=s[p].max();negative=s[v&~p]
            wins.append(float(np.mean((positive>negative+1e-7)+.5*(np.abs(positive-negative)<=1e-7))))
        result["pair_win_grouped"]=grouped(wins,[rows[i] for i in ids])
        result["controls"]={};controls={}
        for control in ("zero","shuffle","swap"):
            controls[control]=matrix(head,mode,bank,ids,control)
            result["controls"][control]=dict(metrics=proxy_retrieval(controls[control],pos,valid),
                mean_abs_score_change=float(np.abs(values-controls[control]).mean()))
        return result,dict(scores=values,positive=pos,valid=valid,**controls)

def path_diagnostics(head,mode,bank,ids):
    records=[]
    if mode=="sync_cold":return records
    with torch.no_grad():
        for i in ids:
            _,r=score(head,mode,bank,[i],[i])
            if mode!="null_ot":
                d=head.decode(r);records.append(dict(matches=len(d["matches"]),hard_path_crossings=d["crossings"],
                    music_null_fraction=len(d["music_drops"])/int(bank["mv"][i].sum()),
                    motion_null_fraction=len(d["motion_drops"])/int(bank["dv"][i].sum()),cost=d["cost"]))
            else:
                p=r["plan"][0].cpu().numpy()
                # Any two positive edges with inverted indices: E[j] monotonic is insufficient.
                # Weighted crossing mass over all i<i', j>j', not an arbitrary thresholded accuracy.
                suffix=np.cumsum(np.cumsum(p[::-1],axis=0),axis=1)[::-1]
                crossing=sum(float(p[a,b]*suffix[a+1,b-1]) for a in range(len(p)-1) for b in range(1,p.shape[1]))
                records.append(dict(coverage=float(r["coverage"][0]),expected_position_violations=int(r["monotonic_violations"][0]),
                    crossing_mass=crossing,marginal_residual=float(r["marginal_residual"][0])))
    return records

def baselines(bank,ids,rows,source):
    count=bank["candidates"].shape[1];clock=[];nearest=[];energy=[]
    obs=np.load(Path(source)/"data/observations.npz")
    for i in ids:
        a=[];b=[];c=[]
        for j in bank["candidates"][i].cpu().tolist():
            m=bank["mt"][i][bank["mv"][i]].cpu().numpy();d=bank["dt"][j][bank["dv"][j]].cpu().numpy()
            distance=abs(m[:,None]-d[None,:])
            b.append(float(-distance.min(-1).mean()))
            p=exact_path(.2*distance/.5,.5,.5,distance<=.5)
            a.append(float((.5*(len(m)+len(d))-p["cost"])/(len(m)+len(d))))
            c.append(float(correlation(obs["energy_music"][i,::2],obs["energy_motion"][j,15:135:2])))
        clock.append(a);nearest.append(b);energy.append(c)
    positive=bank["positive"][ids].cpu().numpy();valid=bank["valid"][ids].cpu().numpy()
    return {name:dict(metrics=proxy_retrieval(np.array(values),positive,valid),scores=np.array(values))
        for name,values in (("clock_only_path",clock),("nearest_time",nearest),("energy",energy))}

def probes(data,rows,config,out,schedule,replay=False):
    arrays=dict(np.load(Path(data)/"probes.npz"));target=arrays.pop("target")[:,::2]
    train=np.array([i for i,r in enumerate(rows) if r["role"]=="train"])
    val=np.array([i for i,r in enumerate(rows) if r["role"]=="val"])
    tm=target[train].mean((0,1));ts=np.maximum(target[train].std((0,1)),1e-6)
    y=torch.as_tensor((target-tm)/ts,device="cuda")
    results={}
    for name,x in arrays.items():
        x=x if name=="transformer" else x[:,::2]
        mean=x[train].mean((0,1));std=x[train].std((0,1));std[std<1e-6]=1
        features=torch.as_tensor((x-mean)/std,device="cuda")
        torch.manual_seed(config["seed"]);model=nn.Linear(128,3).cuda()
        path=Path(out)/("probe_"+name+".pt")
        if replay:model.load_state_dict(torch.load(path,map_location="cuda",weights_only=True)["state"])
        else:
            optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"])
            for ids in schedule[:config["probe_steps"]]:
                loss=(model(features[ids])-y[ids]).square().mean();optimizer.zero_grad();loss.backward();optimizer.step()
            torch.save(dict(state=model.state_dict(),mean=mean.tolist(),std=std.tolist(),target_mean=tm.tolist(),target_std=ts.tolist(),
                parameters=sum(v.numel() for v in model.parameters()),steps=config["probe_steps"]),path)
        with torch.no_grad():prediction=(model(features[val]).cpu().numpy()*ts+tm)
        true=target[val]
        squared=(prediction-true)**2
        phase=np.abs(prediction.argmax(1)-true.argmax(1))/15
        results[name]=dict(parameters=387,steps=config["probe_steps"],kinematic_proxy_mse=float(squared.mean()),
            per_body_mse=squared.mean((0,1)).tolist(),peak_phase_mae_seconds=phase.mean(0).tolist(),
            peak_phase_scope="argmax of Cartesian FK speed on common 60-slot 15Hz grid; no semantic event correspondence",
            pair_grouped_mse=grouped(squared.mean((1,2)),[rows[i] for i in val]))
        if replay:
            old=np.load(Path(out)/("probe_"+name+".npz"))["prediction"]
            if not np.array_equal(old,prediction):raise ValueError("Fresh probe prediction differs")
        else:np.savez_compressed(Path(out)/("probe_"+name+".npz"),prediction=prediction,target=true)
    return dict(results=results,target="independent Native fixed-offset FK Cartesian upper/lower/root speed proxy",
        capacity="all four probes Linear(128,3)=387 parameters, same source schedule and 60-slot grid",
        padding="raw74 structural zero features are probe design, never zero-filled 151D motion targets",
        train_only_normalization=True,zero_optimizer_calls=replay)

def operator_audit(source,data,out):
    source=Path(source);a=np.load(Path(data)/"operator.npz");manifest=read(Path(data)/"manifest.json");rows=manifest["rows"]
    norm=manifest["norm"]["motion"];mean=np.asarray(norm["mean"]);std=np.asarray(norm["std"]);mask=a["mask"]
    old=a["old"];current=a["current"];delta=old-current
    report=dict(scope="same complete Train/Val windows; native60 increments then30 sampling versus native30 increments; no crop derivative origin",
        mean_abs_difference=float(np.abs(delta[mask]).mean()),max_abs_difference=float(np.abs(delta[mask]).max()),
        per_channel_rmse=np.sqrt(np.mean(delta**2,axis=(0,1))).tolist(),normalization_frozen=True)
    for name,x in (("native60_then30",old),("native30",current)):
        z=(x-mean)/std
        report[name]=dict(channel_mean=x.mean((0,1)).tolist(),channel_std=x.std((0,1)).tolist(),
            normalized_abs_above3_fraction=float((np.abs(z[mask])>3).mean()),normalized_abs_above6_fraction=float((np.abs(z[mask])>6).mean()))
    val=[i for i,r in enumerate(rows) if r["role"]=="val"]
    model=RetrievalModel().cuda().eval();payload=torch.load(checked(manifest["global_checkpoint"]),map_location="cuda",weights_only=True);model.load_state_dict(payload["state"])
    enc=np.load(source/"encodings.npz");mt=torch.as_tensor(enc["mt"][val],device="cuda");mv=torch.as_tensor(enc["mv"][val],device="cuda")
    dt=[];dv=[]
    with torch.no_grad():
        for start in range(0,len(val),24):
            ids=val[start:start+24];x=torch.as_tensor((((old[ids]-mean)/std)*mask[ids]).astype(np.float32),device="cuda");m=torch.as_tensor(mask[ids],device="cuda")
            shifts=(-15,-8,0,8,15);v=torch.stack([x[:,15+s:135+s] for s in shifts],1);w=torch.stack([m[:,15+s:135+s] for s in shifts],1)
            _,tokens,valid=model.motion(v.flatten(0,1),w.flatten(0,1));dt.append(tokens.reshape(len(ids),5,60,128));dv.append(valid.reshape(len(ids),5,60))
    olddt=torch.cat(dt);olddv=torch.cat(dv);newdt=torch.as_tensor(enc["dt"][val],device="cuda")
    cosine=torch.nn.functional.cosine_similarity(olddt,newdt,dim=-1)
    report["token_cosine_mean"]=float(cosine.mean());report["token_cosine_min"]=float(cosine.min())
    oldbank=dict(mt=mt,mv=mv,dt=olddt,dv=olddv);newbank=dict(mt=mt,mv=mv,dt=newdt,dv=olddv)
    report["frozen_heads"]={};measurements={}
    for mode in ("A","B","C"):
        checkpoint=torch.load(source/mode/"head.pt",map_location="cuda",weights_only=True)
        head=copy.deepcopy(model.sync) if mode=="A" else ClockNullTransport(band_seconds=checkpoint["config"]["band_seconds"],epsilon=checkpoint["config"]["epsilon"],iterations=checkpoint["config"]["sinkhorn_iterations"]).cuda()
        head.load_state_dict(checkpoint["state"]);values=[]
        with torch.no_grad():
            for bank in (oldbank,newbank):
                parts=[]
                for start in range(0,len(val),24):
                    ids=list(range(start,min(start+24,len(val))));scores,_=paired_scores(head,mode,bank,ids,ids);parts.append(scores.cpu().numpy())
                values.append(np.concatenate(parts))
        before,after=values
        report["frozen_heads"][mode]=dict(mean_absolute_score_change=float(np.abs(before-after).mean()),
            native60_then30=clock_report(before,[rows[i] for i in val]),native30=clock_report(after,[rows[i] for i in val]),
            checkpoint=ref(source/mode/"head.pt"))
        measurements[mode+"_old"]=before;measurements[mode+"_current"]=after
    np.savez_compressed(Path(out)/"operator_scores.npz",**measurements)
    return report

def preflight(source,assets,config,out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    if config["test_evaluations"]!=0 or config["planner_routing_enabled"] or config["lag_calibrated"] or config["lag_seconds"]!=0:raise ValueError("Registered Shadow protocol differs")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    if subprocess.check_output(["git","diff","HEAD","--"],text=True):raise ValueError("Dirty tracked code")
    protocol=dict(schema="event_identifiability_study_v1",code_commit=commit,config=config,source=ref(Path(source)/"result.json"),
        assets=ref(assets),native_generation_changed=False,test_evaluations=0,qualification="independent event labels and fresh source validation required")
    write(out/"protocol.json",protocol)
    fixture=construct(np.pad(np.eye(60,dtype=np.float32),((0,0),(0,84))),
        -np.pad(np.eye(60,dtype=np.float32),((0,0),(0,84))),config["seed"],n=config["synthetic_events"])
    distance=np.linalg.norm(fixture["music"][:,None]-fixture["motion"][None,:],axis=-1)
    allowed=np.abs(fixture["mt"][:,None]-fixture["dt"][None,:])<=config["band_seconds"]
    oracle=exact_path(np.where(distance<1e-6,0.,3.),.5,.5,allowed)
    predicted=np.full(config["synthetic_events"],-1,np.int64)
    for i,j in oracle["matches"]:predicted[i]=j
    if not np.array_equal(predicted,fixture["target"]):raise ValueError("Exact synthetic mechanism fixture failed")
    write(out/"mechanism_fixture.json",dict(status="EXACT_CONSTRUCTED_IDENTITIES_RECOVERED",
        matches=len(oracle["matches"]),deleted=2,inserted=2,scope="cost oracle algorithm fixture, not learned cross-modal quality"))
    data=prepare(source,out/"data",config,read(assets));rows=data["rows"]
    for name in ("events","probes","operator"):
        archive=np.load(out/"data"/(name+".npz"))
        for key in archive.files:
            if not np.isfinite(archive[key]).all():raise ValueError("Nonfinite preflight observations")
    prior=Path(source).parent/"replay/result.json"
    if read(prior).get("status")!="FRESH_CLOCK_ROOT_REPLAY_VERIFIED":raise ValueError("Prior frozen source replay missing")
    protocol["prior_fresh_replay"]=ref(prior);write(out/"protocol.json",protocol)
    bank=np.load(out/"data/events.npz");eligible=bank["eligible"]
    schedule_rows=[dict(r,role=r["role"] if eligible[i] else "excluded") for i,r in enumerate(rows)]
    schedule=group_schedule(schedule_rows,max(config["steps"],config["probe_steps"]),config["batch"],config["seed"])
    np.save(out/"training_schedule.npy",schedule)
    # Representation probes use all Train windows, independently from event detectability.
    probe_schedule=group_schedule(rows,config["probe_steps"],config["batch"],config["seed"])
    np.save(out/"probe_schedule.npy",probe_schedule)
    seal={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write(out/"preflight.json",dict(status="PREFLIGHT_PASSED_WEAK_SOURCE_CLOCK_ONLY",files=seal,
        eligible=dict(train=data["train_eligible"],val=data["val_eligible"]),schedule=coverage(schedule_rows,schedule),
        prior_schedule=data["old_schedule_coverage"],independent_labels=0,semantic_qualification_passed=False,configuration_frozen=True))
    print("EVENT_PREFLIGHT_PASSED",flush=True)

def run(source,out):
    out=Path(out);pre=read(out/"preflight.json")
    for p,h in pre["files"].items():
        if sha(out/p)!=h:raise ValueError("Sealed preflight file differs: "+p)
    protocol=read(out/"protocol.json");config=protocol["config"]
    if subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()!=protocol["code_commit"]:raise ValueError("Published execution commit differs")
    if not torch.cuda.is_available():raise ValueError("Registered bounded study needs CUDA")
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    manifest=read(out/"data/manifest.json");rows=manifest["rows"];data=dict(np.load(out/"data/events.npz"))
    bank={k:torch.as_tensor(v,device="cuda") for k,v in data.items()}
    bank["donors"]=torch.as_tensor(donor_indices(rows,data["mv"].any(-1)&data["dv"].any(-1)),device="cuda")
    ids=[i for i,r in enumerate(rows) if r["role"]=="val" and data["eligible"][i]]
    schedule=np.load(out/"training_schedule.npy");results={};inits={};start=time.monotonic()
    dense=load_dense(source)
    donors=donor_indices(rows)
    samples=synthetic_samples(dense,ids,donors,config)
    np.savez_compressed(out/"synthetic_truth.npz",**{k:np.stack([v[k] for v in samples]) for k in ("music","motion","mt","dt","target","source_token_indices","warp","inserted_indices")})
    baseline=baselines(bank,ids,rows,source)
    write(out/"baselines.json",{k:v["metrics"] for k,v in baseline.items()})
    np.savez_compressed(out/"baseline_scores.npz",**{k:v["scores"] for k,v in baseline.items()})
    for mode in config["arms"]:
        folder=out/mode;folder.mkdir();torch.manual_seed(config["seed"])
        head=make_head(mode,config).cuda();inits[mode]=dict(digest=digest(head),parameters=sum(p.numel() for p in head.parameters()))
        frozen,measure=evaluate(head,mode,bank,ids,rows);write(folder/"frozen_validation.json",frozen)
        np.savez_compressed(folder/"frozen_measurements.npz",**measure)
        optimizer=torch.optim.AdamW(head.parameters(),lr=config["learning_rate"]);history=[]
        auxiliary=mode in ("path_synthetic","path_weak","path_random_weak")
        for step,queries in enumerate(schedule[:config["steps"]],1):
            count=config["candidates"];q=np.repeat(queries,count);c=data["candidates"][queries].reshape(-1)
            scores,_=score(head,mode,bank,q,c)
            loss=multi_positive_loss(scores.reshape(len(queries),count),bank["positive"][queries],bank["valid"][queries],config["score_temperature"])
            parts=dict(retrieval=float(loss.detach()))
            if auxiliary:
                shift=shift_loss(head,mode,dense,queries,config)
                branch="motion" if step%2 else "music"
                synthetic=synthetic_batch(dense[branch],queries,[donors[i] for i in queries],config["seed"]+step*65537)
                truth,_=supervised_path_loss(head,synthetic,"cuda")
                loss=loss+config["shift_weight"]*shift+config["partial_weight"]*truth["partial"]+config["null_weight"]*truth["null"]
                parts.update(shift=float(shift.detach()),partial=float(truth["partial"].detach()),null=float(truth["null"].detach()))
            if mode in ("path_weak","path_random_weak"):
                _,own=score(head,mode,bank,queries,queries)
                weak=weak_loss(own,bank["weak_target"][queries],bank["weak_weight"][queries],mode=="path_random_weak")
                loss=loss+config["weak_weight"]*weak;parts["weak"]=float(weak.detach())
            if not torch.isfinite(loss):raise ValueError("Nonfinite event retrieval objective")
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
            if step%50==0:
                history.append(dict(step=step,loss=float(loss.detach()),components=parts));write(folder/"history.json",history);print("EVENT_TRAIN",mode,step,float(loss.detach()),flush=True)
        torch.save(dict(state=head.state_dict(),config=config,mode=mode,protocol_sha256=sha(out/"protocol.json"),data_sha256=sha(out/"data/events.npz"),
            step=config["steps"],qualified=False,supervision="official source-clock multi-positive retrieval, NOT independent event anchors"),folder/"head.pt")
        result,measure=evaluate(head,mode,bank,ids,rows);np.savez_compressed(folder/"measurements.npz",**measure)
        diagnostics=path_diagnostics(head,mode,bank,ids);write(folder/"path_diagnostics.json",diagnostics)
        result["independent_event_metrics"]=independent_metrics([],{},{})
        result["synthetic_shift"]=shift_evaluation(head,mode,dense,ids,config)
        if mode!="sync_cold":result["synthetic_partial"]=synthetic_metrics(head,samples,"null_ot" if mode=="null_ot" else "strict_path")
        write(folder/"validation.json",result);results[mode]=result
        print("EVENT_EVALUATED",mode,flush=True)
    assert all(inits[x]==inits["strict_path"] for x in config["arms"] if x!="sync_cold")
    probe=probes(out/"data",rows,config,out,np.load(out/"probe_schedule.npy"));write(out/"probes.json",probe)
    operator=operator_audit(source,out/"data",out);write(out/"operator_audit.json",operator)
    write(out/"initializations.json",inits)
    write(out/"result.json",dict(status="COMPLETED_EVENT_IDENTIFIABILITY_SHADOW",results=results,
        baselines=read(out/"baselines.json"),probes=probe,operator=operator,initializations=inits,
        steps_per_arm=config["steps"],source_rows=len(rows),validation_queries=len(ids),validation_components=len({rows[i]["source_group"] for i in ids}),
        test_evaluations=0,selection="fixed last; no Val tuning",operational_qualified=False,
        downstream_generation_executed=False,qualification_blockers=["NO_INDEPENDENT_EVENT_ANCHORS","NO_NEW_BLIND_SOURCE","ONLY_TWO_VAL_COMPONENTS"],
        seconds=time.monotonic()-start))
    print("EVENT_STUDY_COMPLETED",flush=True)


def load_dense(source):
    a=np.load(Path(source)/"encodings.npz")
    def pad(x):return np.pad(x,[(0,0)]*(x.ndim-1)+[(0,16)]).astype(np.float32)
    return dict(music=pad(a["mt"]),motion=pad(a["dt"][:,2]),shifts=pad(a["dt"]),
        mv=a["mv"],dv=a["dv"])

def donor_indices(rows,eligible=None):
    donors=[]
    for i,r in enumerate(rows):
        pool=[j for j,q in enumerate(rows) if q["role"]==r["role"] and q["source_group"]!=r["source_group"] and (eligible is None or eligible[j])]
        if not pool:raise ValueError("Distinct component donor unavailable")
        donors.append(pool[i%len(pool)])
    return donors

def synthetic_samples(dense,ids,donors,config):
    rng=np.random.default_rng(config["seed"]+9001)
    chosen=rng.choice(ids,config["synthetic_sequences"],replace=len(ids)<config["synthetic_sequences"])
    return [construct(dense["music" if k%2 else "motion"][i],
        dense["music" if k%2 else "motion"][donors[i]],config["seed"]+k+70001,n=config["synthetic_events"]) for k,i in enumerate(chosen)]

def shift_scores(head,mode,dense,ids,config):
    stride=config["shift_stride"];n=len(ids)
    m=torch.as_tensor(dense["music"][ids,::stride],device="cuda")
    d=torch.as_tensor(dense["shifts"][ids,:,::stride],device="cuda")
    mv=torch.as_tensor(dense["mv"][ids,::stride],device="cuda")
    dv=torch.as_tensor(dense["dv"][ids,:,::stride],device="cuda")
    m=m[:,None].expand(-1,5,-1,-1).flatten(0,1);mv=mv[:,None].expand(-1,5,-1).flatten(0,1)
    d=d.flatten(0,1);dv=dv.flatten(0,1)
    if mode=="sync_cold":
        x=torch.cat((m,d,m-d,m*d),-1);valid=mv&dv
        value=(head(x).squeeze(-1)*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    else:
        clock=torch.arange(m.shape[1],device="cuda")*stride/15
        value=head(m,d,mv,dv,clock,clock)["score"]
    return value.reshape(n,5)

def shift_loss(head,mode,dense,ids,config):
    scores=shift_scores(head,mode,dense,ids,config)
    target=torch.full((len(ids),),2,device="cuda",dtype=torch.long)
    return torch.nn.functional.cross_entropy(scores/config["score_temperature"],target)

def shift_evaluation(head,mode,dense,ids,config):
    values=[]
    with torch.no_grad():
        for a in range(0,len(ids),24):values.append(shift_scores(head,mode,dense,ids[a:a+24],config).cpu().numpy())
    scores=np.concatenate(values);choice=scores.argmax(-1);offsets=np.array([-15,-8,0,8,15])/30
    return dict(scope="known injected cross-modal source-clock offsets; not semantic event labels",
        offsets_seconds=offsets.tolist(),accuracy=float((choice==2).mean()),
        selected_offset_MAE_seconds=float(np.abs(offsets[choice]).mean()),
        correct_vs_shift_win_rate=float(((scores[:,2,None]>scores[:,[0,1,3,4]])+.5*(scores[:,2,None]==scores[:,[0,1,3,4]])).mean()),
        stride=config["shift_stride"],input_clock_grid_class_independent=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=("preflight","train"),required=True);ap.add_argument("--source",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--assets");ap.add_argument("--config");a=ap.parse_args()
    if a.phase=="preflight":preflight(a.source,a.assets,read(a.config),a.out)
    else:run(a.source,a.out)
if __name__=="__main__":main()
