"""Geometry, nuisance-conditioned source-window AUC, and fixed-path rescoring."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from dunhuang.io import write
from .content_discrimination import VARIANTS,pair_result,template_paths,fixed_score,metrics,fit_normalization

def distributions(x):
    x=np.asarray(x);return dict(count=int(x.size),mean=float(x.mean()) if x.size else None,
        quantiles=np.percentile(x,[0,5,25,50,75,95,100]).tolist() if x.size else None)

def spectrum(x):
    x=np.asarray(x,np.float64);centered=x-x.mean(0);cov=centered.T@centered/max(len(x)-1,1)
    e=np.maximum(np.linalg.eigvalsh(cov),0);total=e.sum();p=e/max(total,1e-12);nz=p>0
    return dict(samples=len(x),width=x.shape[-1],norm=distributions(np.linalg.norm(x,axis=-1)),
        mean_vector_norm=float(np.linalg.norm(x.mean(0))),covariance_trace=float(total),
        effective_rank=float(np.exp(-(p[nz]*np.log(p[nz])).sum())) if total>0 else 0.,
        participation_rank=float(total**2/max((e**2).sum(),1e-12)),
        top12_variance_fraction=p[::-1][:12].tolist())

def deduplicate(bank,rows,events,side,width,role):
    name="music" if side=="music" else "motion";mask="mv" if side=="music" else "dv"
    seen=set();values=[];owners=[]
    for i,r in enumerate(rows):
        if r["role"]!=role:continue
        for j,e in enumerate(events[i][side]):
            if not bank[mask][i,j]:continue
            key=(r["audio" if side=="music" else "raw_motion"]["sha256"],e["frame"])
            if width==144:key+=(e["event_type"],e["body_part"])
            if key in seen:continue
            seen.add(key);values.append(bank[name][i,j,:width]);owners.append(i)
    return np.asarray(values),np.asarray(owners)

def hubness(m,d,limit,seed):
    rng=np.random.default_rng(seed)
    mi=np.sort(rng.choice(len(m),min(limit,len(m)),replace=False));di=np.sort(rng.choice(len(d),min(limit,len(d)),replace=False))
    x=torch.as_tensor(m[mi],device="cuda");y=torch.as_tensor(d[di],device="cuda")
    x=F.normalize(x,dim=-1);y=F.normalize(y,dim=-1);counts=np.zeros(len(y),int)
    with torch.no_grad():
        for a in range(0,len(x),256):
            ids=(x[a:a+256]@y.T).argmax(-1).cpu().numpy();counts+=np.bincount(ids,minlength=len(y))
    sorted_counts=np.sort(counts);n=len(counts);total=max(counts.sum(),1)
    return dict(query_tokens=len(x),candidate_tokens=len(y),max_indegree=int(counts.max()),
        top_one_percent_share=float(np.sort(counts)[-max(1,int(np.ceil(n*.01))):].sum()/total),
        unused_fraction=float((counts==0).mean()),gini=float(2*(np.arange(1,n+1)*sorted_counts).sum()/(n*total)-(n+1)/n),
        source_deduplicated=True,ground_truth_event_labels=False)

def geometry(bank,rows,events,head,config):
    result={}
    for role in ("train","val"):
        result[role]={}
        for width in (128,144):
            m,_=deduplicate(bank,rows,events,"music",width,role);d,_=deduplicate(bank,rows,events,"motion",width,role)
            result[role]["raw"+str(width)]=dict(music=spectrum(m),motion=spectrum(d),
                mean_direction_cosine=float(m.mean(0)@d.mean(0)/max(np.linalg.norm(m.mean(0))*np.linalg.norm(d.mean(0)),1e-12)),
                hubness=hubness(m,d,config["max_geometry_tokens"],config["seed"]))
            if width==144:
                m=m@head.music.weight.detach().cpu().numpy().T;d=d@head.motion.weight.detach().cpu().numpy().T
                result[role]["projected144"]=dict(music=spectrum(m),motion=spectrum(d),
                    hubness=hubness(m,d,config["max_geometry_tokens"],config["seed"]))
    return result

def evaluate(head,bank,rows,cohort,out,config,normalizations=None):
    out=Path(out);out.mkdir()
    ids=cohort["ids"];candidate=cohort["candidates"][ids];positive=cohort["positive"][ids];valid=cohort["valid"][ids]
    sufficient=cohort.get("sufficient",False)
    if not len(ids):
        report=dict(status="NOT_EVALUABLE_EMPTY_POPULATION",queries=0,variants={})
        write(out/"result.json",report);return report
    width=candidate.shape[1];queries=np.repeat(ids,width);targets=candidate.reshape(-1)
    null=float(.05+.9*head.null_logit.detach().sigmoid());templates=template_paths(bank,queries,targets,head.band_seconds,null)
    arrays=dict(queries=np.asarray(ids),candidates=candidate,positive=positive,valid=valid,clock_template=templates)
    reports={};full_paths=None;full_assign=None
    choices=[(v,v,None) for v in VARIANTS]+[(k,"full",v) for k,v in (normalizations or {}).items()]
    for name,variant,norm in choices:
        acc={k:[] for k in ("scores","similarity","cost","allowed","fixed_full_score","fixed_template_content","fixed_content","fixed_time","fixed_constant","assignment","null_fraction","raw128_mean_cos")}
        for start in range(0,len(queries),64):
            q=queries[start:start+64];c=targets[start:start+64]
            with torch.no_grad():
                r=pair_result(head,bank,q,c,variant,norm)
                assignment=np.full((len(q),bank["mv"].shape[1]),-1,np.int64);paths=np.zeros_like(r["allowed"].cpu().numpy())
                for b in range(len(q)):
                    decoded=head.decode(r,b)
                    for i,j in decoded["matches"]:assignment[b,i]=j;paths[b,i,j]=True
                if name=="full":
                    if full_paths is None:full_paths=np.zeros((len(queries),*paths.shape[1:]),bool);full_assign=np.zeros((len(queries),paths.shape[1]),np.int64)
                    full_paths[start:start+len(q)]=paths;full_assign[start:start+len(q)]=assignment
                locked=fixed_score(r,full_paths[start:start+len(q)],head.band_seconds)
                template=fixed_score(r,templates[start:start+len(q)],head.band_seconds)
                rawm=F.normalize(torch.as_tensor(bank["music"][q,:,:128],device="cuda"),dim=-1)
                rawd=F.normalize(torch.as_tensor(bank["motion"][c,:,:128],device="cuda"),dim=-1)
                rawsim=rawm@rawd.transpose(-1,-2);allowed=r["allowed"]
                rawmean=(rawsim*allowed).sum((-1,-2))/allowed.sum((-1,-2)).clamp_min(1)
                values=dict(scores=r["score"],similarity=r["similarity"],cost=r["cost"],allowed=allowed,
                    fixed_full_score=locked["score"],fixed_template_content=template["mean_content"],
                    fixed_content=locked["content"],fixed_time=locked["time"],fixed_constant=locked["constant"],
                    assignment=assignment,null_fraction=(assignment<0).sum(-1),raw128_mean_cos=rawmean)
                for k,v in values.items():acc[k].append(v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v)
        acc={k:np.concatenate(v) for k,v in acc.items()}
        for key,value in acc.items():arrays[name+"_"+key]=value
        sim_mean=(acc["similarity"]*acc["allowed"]).sum((-1,-2))/np.maximum(acc["allowed"].sum((-1,-2)),1)
        same=np.asarray([[rows[j]["music_uid"]==rows[i]["music_uid"] for j in candidate[a]] for a,i in enumerate(ids)])
        selected=[rows[i] for i in ids];rpt={}
        for key,score in (("optimized_path",acc["scores"]),("fixed_full_path",acc["fixed_full_score"]),("fixed_clock_template_content",acc["fixed_template_content"]),("mean_allowed_cosine",sim_mean),("raw128_mean_allowed_cosine",acc["raw128_mean_cos"])):
            score=score.reshape(len(ids),width)
            rpt[key]=dict(all_candidates=metrics(score,positive,valid,selected,sufficient),
                same_song=metrics(score,positive,valid&same,selected,sufficient))
        pairvalid=valid.reshape(-1);nodes=bank["mv"][queries]&pairvalid[:,None]
        rpt["path_change_fraction"]=float(((acc["assignment"]!=full_assign)&nodes).sum()/max(nodes.sum(),1))
        rpt["music_null_fraction"]=float(((acc["assignment"]<0)&nodes).sum()/max(nodes.sum(),1))
        permitted=acc["allowed"]&pairvalid[:,None,None]
        rpt["cosine_distribution"]=distributions(acc["similarity"][permitted]);rpt["cost_distribution"]=distributions(acc["cost"][permitted])
        rpt["edges_cheaper_than_two_drops_fraction"]=float((acc["cost"][permitted]<2*null).mean()) if permitted.any() else None
        if name!="full":rpt["optimized_score_intervention"]=distributions((acc["scores"]-arrays["full_scores"])[pairvalid])
        rpt["null_cost_fixed"]=null;rpt["clocks_masks_candidates_fixed"]=True;reports[name]=rpt
        print("LOCAL_COST_AUDIT",out.name,name,flush=True)
    arrays["full_hard_path"]=full_paths
    np.savez_compressed(out/"measurements.npz",**arrays)
    report=dict(schema="conditional_local_cost_audit_v1",queries=len(ids),population_sufficient=sufficient,variants=reports,
        labels="source-window association only; permitted matrix entries are not event-positive labels",
        fixed_path_scope="full-head hard path locked; clock-only geometric template independent of content",
        optimized_difference_scope="intervention, not additive causal content contribution",operational_qualified=False)
    write(out/"result.json",report);return report

def repeat_consistency(rows,enc,limit):
    groups={}
    for i,r in enumerate(rows):groups.setdefault((r["role"],r["music_uid"],r["choreography_uid"],r["start_frame"]),[]).append(i)
    results={}
    for role in ("train","val"):
        values=[]
        for key,ids in sorted(groups.items()):
            if key[0]!=role:continue
            for i in ids:
                peers=[j for j in ids if j>i and rows[j]["recording_uid"]!=rows[i]["recording_uid"] and rows[j]["raw_motion"]["sha256"]!=rows[i]["raw_motion"]["sha256"]]
                if not peers:continue
                j=peers[0]
                def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
                m=unit(enc["mt"][i]);d=unit(enc["dt"][i,2]);other=unit(enc["dt"][j,2])
                values.append((float((d*other).sum(-1).mean()),float(np.sqrt(np.mean((m@d.T-m@other.T)**2)))))
                if len(values)>=limit:break
            if len(values)>=limit:break
        results[role]=dict(capture_pairs=len(values),same_clock_motion_cosine=distributions([v[0] for v in values]),
            cross_modal_matrix_repeat_rms=distributions([v[1] for v in values]),music_repeat_is_same_audio_not_independent_evidence=True,
            scope="same music/choreography/source clock repeated capture proxy, not frame-exact human motion correspondence")
    return results
