"""Train-only correlated nuisance support. Selector data never becomes a score."""
import hashlib
import numpy as np

def key(value):return hashlib.sha256(str(value).encode()).hexdigest()

def vectors(stats):
    blocks=[np.asarray(stats["counts"],float),np.asarray(stats["quantiles"]).reshape(-1,32),
        np.asarray(stats["hist"]).reshape(-1,8),np.log(np.maximum(stats["energy"],0)+1e-6),
        np.asarray(stats["onset_density"])[:,None]]
    return np.concatenate(blocks,axis=1)

def fit(z,train,shrinkage):
    x=np.asarray(z)[np.asarray(train)]
    if len(x)<3 or not np.isfinite(x).all():raise ValueError("Finite Train nuisance observations required")
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);u=(x-mean)/scale
    covariance=u.T@u/max(len(x)-1,1);eig,rotation=np.linalg.eigh((1-shrinkage)*covariance+shrinkage*np.eye(x.shape[1]))
    matrix=(rotation*(1/np.sqrt(np.maximum(eig,1e-6))))@rotation.T
    return dict(mean=mean,scale=scale,matrix=matrix,correlation=covariance)

def transform(z,model):return ((z-model["mean"])/model["scale"])@model["matrix"]

def candidate_pool(rows,ticks,i,task,config,pool_ids=None):
    r=rows[i];pool_ids=range(len(rows)) if pool_ids is None else pool_ids
    if task=="time":
        return [j for j in pool_ids if (s:=rows[j])["role"]==r["role"] and s["music_uid"]==r["music_uid"] and
            s["choreography_uid"]==r["choreography_uid"] and abs(ticks[j]-ticks[i])>=config["min_time_difference_frames"]*1470]
    bucket={}
    for j in pool_ids:
        s=rows[j]
        if s["role"]==r["role"] and ticks[j]==ticks[i] and s["music_uid"]!=r["music_uid"] and s["choreography_uid"]!=r["choreography_uid"]:
            bucket.setdefault(s["music_uid"],[]).append(j)
    groups=sorted(bucket,key=lambda song:key((r["window_uid"],song)))[:config["donor_music_groups"]]
    return [j for song in groups for j in sorted(bucket[song],key=lambda j:key(rows[j]["window_uid"]))[:config["per_music_cap"]]]

def build(rows,ticks,z,model,config,task,available):
    n=len(rows);white=transform(z,model);index={}
    bucket=lambda i:(rows[i]["role"],rows[i]["music_uid"],rows[i]["choreography_uid"]) if task=="time" else (rows[i]["role"],ticks[i])
    for i in range(n):index.setdefault(bucket(i),[]).append(i)
    pools=[candidate_pool(rows,ticks,i,task,config,index[bucket(i)]) if available[i] else [] for i in range(n)]
    pools=[[j for j in pool if available[j]] for pool in pools]
    nearest=[]
    for i,r in enumerate(rows):
        if r["role"]=="train" and pools[i]:nearest.append(float(np.linalg.norm(white[pools[i]]-white[i],axis=-1).min()))
    cutoff=float(np.quantile(nearest,config["nearest_quantile"])) if nearest else None
    radial=float(np.quantile(np.linalg.norm(white[[r["role"]=="train" for r in rows]],axis=-1),config["radial_quantile"]))
    candidates=np.repeat(np.arange(n)[:,None],4,axis=1);positive=np.zeros((n,4),bool);valid=positive.copy();eligible=np.zeros(n,bool);distance=np.full((n,4),np.nan)
    for i,pool in enumerate(pools):
        if not pool or cutoff is None or np.linalg.norm(white[i])>radial:continue
        d=np.linalg.norm(white[pool]-white[i],axis=-1);chosen=[(float(v),j) for v,j in zip(d,pool) if v<=cutoff and np.linalg.norm(white[j])<=radial]
        # Source-time task and association task each have known recording labels.
        chosen=sorted(chosen,key=lambda pair:(pair[0],key(rows[pair[1]]["window_uid"])))[:3]
        if not chosen:continue
        ids=[i]+[j for v,j in chosen];candidates[i,:len(ids)]=ids;positive[i,0]=True;valid[i,:len(ids)]=True;eligible[i]=True
        distance[i,:len(ids)]=[0]+[v for v,j in chosen]
    return dict(candidates=candidates,positive=positive,valid=valid,eligible=eligible),dict(cutoff=cutoff,radial=radial,train_nearest_count=len(nearest),
        calibration="Train only; same fixed donor-music inventory budget for both roles",selector_distance_not_a_predictor=True),distance
