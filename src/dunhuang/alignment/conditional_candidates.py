"""Strict nuisance matching, defined before scoring. Never relax when population is small."""
import numpy as np

def nuisance_statistics(rows,bank,observations,events):
    n=len(rows);counts=np.zeros((n,2),np.int64);quantiles=np.zeros((n,2,16));hist=np.zeros((n,2,4))
    energy=np.stack((observations["energy_music"].mean(-1),observations["energy_motion"][:,15:135].mean(-1)),-1)
    onset=np.zeros(n)
    for i,r in enumerate(rows):
        for side,(tk,mk) in enumerate((("mt","mv"),("dt","dv"))):
            t=bank[tk][i][bank[mk][i]];counts[i,side]=len(t)
            if len(t):
                quantiles[i,side]=np.quantile(t,np.linspace(0,1,16))
                hist[i,side]=np.histogram(t,bins=np.arange(5))[0]/len(t)
        # events here are full-source sidecars, not confidence-truncated event-bank nodes.
        a=r["start_frame"]+15+30*r["audio_offset_seconds"]
        onset[i]=sum(e["event_type"]=="onset" and a<=e["frame"]<a+120 for e in events[r["audio"]["sha256"]])/4
    if not all(np.isfinite(x).all() for x in (quantiles,hist,energy,onset)):raise ValueError("Nonfinite nuisance observations")
    return dict(counts=counts,quantiles=quantiles,hist=hist,energy=energy,onset_density=onset)

def match_statistics(stats,i,j,config):
    counts=np.abs(stats["counts"][i]-stats["counts"][j])
    w1=np.abs(stats["quantiles"][i]-stats["quantiles"][j]).mean(-1)
    histogram=np.abs(stats["hist"][i]-stats["hist"][j]).sum(-1)
    ratio=np.abs(np.log((np.maximum(stats["energy"][i],0)+1e-6)/(np.maximum(stats["energy"][j],0)+1e-6)))
    onset=abs(stats["onset_density"][i]-stats["onset_density"][j])
    rules=dict(count=bool((counts<=config["count_tolerance"]).all()),
        time_distribution=bool((w1<=config["time_w1_seconds"]).all()),
        histogram=bool((histogram<=config["histogram_l1"]).all()),
        energy=bool((ratio<=config["energy_log_ratio"]).all()),onset=bool(onset<=config["onset_density_per_second"]))
    return all(rules.values()),dict(rules=rules,time_w1=w1.tolist(),histogram_l1=histogram.tolist(),energy_log_ratio=ratio.tolist(),
        onset_density_difference=float(onset),selection_distance=float(w1.sum()+histogram.sum()+ratio.sum()+onset))

def build_candidates(rows,bank,stats,config):
    n=len(rows);width=4;candidates=np.repeat(np.arange(n)[:,None],width,axis=1)
    positive=np.zeros((n,width),bool);valid=positive.copy();eligible=np.zeros(n,bool);evidence=[]
    failures={k:0 for k in ("count","time_distribution","histogram","energy","onset")}
    for i,r in enumerate(rows):
        if r["role"] not in ("train","val"):raise ValueError("No Test population permitted")
        if not bank["mv"][i].any() or not bank["dv"][i].any():continue
        positives=[i]+[j for j,s in enumerate(rows) if j!=i and s["role"]==r["role"] and
            s["music_uid"]==r["music_uid"] and s["choreography_uid"]==r["choreography_uid"] and
            s["start_frame"]==r["start_frame"] and bank["mv"][j].any() and bank["dv"][j].any()]
        negative=[]
        for j,s in enumerate(rows):
            if s["role"]!=r["role"] or s["music_uid"]!=r["music_uid"] or s["choreography_uid"]!=r["choreography_uid"] or abs(s["start_frame"]-r["start_frame"])<120:continue
            if not bank["mv"][j].any() or not bank["dv"][j].any():continue
            accepted,detail=match_statistics(stats,i,j,config)
            for key,ok in detail["rules"].items():failures[key]+=not ok
            evidence.append(dict(query=i,candidate=j,accepted=accepted,**detail))
            if accepted:negative.append((detail["selection_distance"],j))
        if not negative:continue
        selected=positives[:2]+[j for _,j in sorted(negative)[:2]]
        candidates[i,:len(selected)]=selected;positive[i,:min(len(positives),2)]=True;valid[i,:len(selected)]=True;eligible[i]=True
    populations={}
    for role in ("train","val"):
        ids=[i for i,r in enumerate(rows) if r["role"]==role and eligible[i]]
        components=len({rows[i]["source_group"] for i in ids});recordings=len({rows[i]["recording_uid"] for i in ids})
        sufficient=(len(ids)>=config["min_queries"] and components>=config["min_"+role+"_components"] and recordings>=config["min_recordings"])
        populations[role]=dict(queries=len(ids),components=components,recordings=recordings,population_sufficient=sufficient)
    return dict(candidates=candidates,positive=positive,valid=valid,eligible=eligible),dict(populations=populations,
        comparison_count=len(evidence),failed_rule_counts=failures,relaxed=False,rules=config,evidence=evidence,
        label_scope="official source-window association, not event semantics",same_song=True,same_choreography=True)

def validate_roles(rows):
    identities={}
    for r in rows:
        if r["role"] not in ("train","val"):raise ValueError("Test assets forbidden")
        for key in ("source_group","recording_uid","music_uid","choreography_uid"):
            pair=(key,r[key])
            if pair in identities and identities[pair]!=r["role"]:raise ValueError("Associated source crosses roles")
            identities[pair]=r["role"]
