"""Fixed detector/repetition evidence, never model-generated correspondence truth."""
import numpy as np

def weak_labels(rows,events,band=.5):
    n=len(rows);mcap=max(len(e["music"]) for e in events);dcap=max(len(e["motion"]) for e in events)
    target=np.zeros((n,mcap,dcap),np.float32);weight=np.zeros((n,mcap),np.float32);evidence=[]
    associations={}
    for i,r in enumerate(rows):
        if r["role"]=="train":
            key=(r["music_uid"],r["choreography_uid"],r["start_frame"])
            associations.setdefault(key,[]).append(i)
    states={"PSEUDO_MATCH":0,"PSEUDO_NULL":0,"UNKNOWN":0}
    for i,r in enumerate(rows):
        if r["role"]!="train":continue
        key=(r["music_uid"],r["choreography_uid"],r["start_frame"])
        peers=[j for j in associations[key] if rows[j]["recording_uid"]!=r["recording_uid"]]
        for a,m in enumerate(events[i]["music"]):
            support=[]
            for b,d in enumerate(events[i]["motion"]):
                gap=abs(m["relative_seconds"]-d["relative_seconds"])
                same=[any(e["event_type"]==d["event_type"] and e["body_part"]==d["body_part"] and
                    abs(e["relative_seconds"]-d["relative_seconds"])<=.1 for e in events[j]["motion"]) for j in peers]
                repeat=float(np.mean(same)) if same else 0.
                clock=max(0.,1-gap/band)
                salience=float(np.sqrt(m["confidence"]*d["confidence"]))
                consensus=bool(m.get("detector_agreement",False) and d.get("detector_agreement",False))
                q=clock*salience*repeat
                accepted=(gap<=band and len(peers)>=1 and repeat>=.5 and consensus and
                    m["event_type"]!="silence" and m["confidence"]>=.4 and d["confidence"]>=.4)
                support.append(dict(motion_index=b,clock=clock,salience=salience,repeat=repeat,
                    repeat_recordings=len(peers),total_recordings=1+len(peers),detector_consensus=consensus,uncalibrated_score=q,accepted=accepted))
                if accepted:target[i,a,b]=q
            if target[i,a].sum()>0:
                weight[i,a]=min(.5,float(target[i,a].max()));target[i,a]/=target[i,a].sum();state="PSEUDO_MATCH"
            else:state="UNKNOWN"
            # Absence of a detected movement is insufficient evidence of semantic NULL.
            # Explicit PSEUDO_NULL remains empty until a coverage protocol supports it.
            states[state]+=1
            evidence.append(dict(window_uid=r["window_uid"],music_index=a,state=state,support=support,
                human_reviewed=False,scope="detector/repeated-capture weak candidate; not independent truth"))
    return target,weight,dict(schema="weak_event_correspondence_v1",states=states,evidence=evidence,
        pseudo_null_disabled_reason="No independent evidence that a music event must lack a dance response",
        no_model_predictions_used=True,validation_labels_generated=False)

def weak_loss(result,target,weight,randomized=False):
    import torch
    cost=result["cost"];allowed=result["allowed"];null=result["null_cost"]
    target=target.to(cost)
    if randomized:
        target=target.clone()
        # Fixed step RNG is set by the caller. Preserve mass/counts within admissible support.
        for b in range(len(target)):
            for i in range(target.shape[1]):
                ids=torch.nonzero(allowed[b,i]).flatten()
                if len(ids)>1: target[b,i,ids]=target[b,i,ids[torch.randperm(len(ids),device=ids.device)]]
    logits=torch.cat((-cost.masked_fill(~allowed,1e4),(-2*null).expand(len(cost),cost.shape[1],1)),-1)/.15
    logp=torch.log_softmax(logits,-1)[...,:cost.shape[2]]
    valid=weight>0
    return (-(target*logp).sum(-1)*weight).sum()/weight.sum().clamp_min(1e-8) if valid.any() else cost.sum()*0
