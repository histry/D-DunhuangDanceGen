"""Descriptive review accounting; no independent model-performance inference."""
from collections import Counter,defaultdict
from itertools import combinations

def statistics(bindings,accepted,decisions):
    by_id={b["blind_item_id"]:b for b in bindings};by_item=defaultdict(list)
    for r in accepted:by_item[r["blind_item_id"]].append(r)
    population=Counter();reviewers=defaultdict(Counter);sources=defaultdict(Counter);music=defaultdict(Counter);tasks=defaultdict(Counter)
    for d in decisions:
        b=by_id[d["blind_item_id"]];population[d["state"]]+=1;sources[b["source_uid"]][d["state"]]+=1;tasks[b["task_type"]][d["state"]]+=1
        identity=b.get("music_identity");group=identity["work_uid"] if identity else "UNPAIRED_MOTION_TASK"
        music[group][d["state"]]+=1
    agreement=defaultdict(lambda:dict(a=[],b=[]))
    for item,rs in by_item.items():
        for r in rs:
            reviewers[r["reviewer_id"]][r["decision"]]+=1
            if r["decision"]=="UNSURE":reviewers[r["reviewer_id"]]["UNSURE_REASON_"+r["reason_code"]]+=1
        # Initial two original votes only; selected adjudications must not bias agreement.
        for a,b in combinations(sorted(rs[:2],key=lambda r:r["reviewer_id"]),2):
            k=(by_id[item]["task_type"],a["reviewer_id"],b["reviewer_id"])
            agreement[k]["a"].append(a["decision"]);agreement[k]["b"].append(b["decision"])
    scores=[]
    for (task,a,b),v in sorted(agreement.items()):
        n=len(v["a"]);ca=Counter(v["a"]);cb=Counter(v["b"]);observed=sum(x==y for x,y in zip(v["a"],v["b"]))/n
        expected=sum(ca[k]*cb[k] for k in set(ca)|set(cb))/(n*n)
        scores.append(dict(task_type=task,reviewer_a=a,reviewer_b=b,paired_items=n,agreement=observed,disagreement=1-observed,
            cohen_kappa=(observed-expected)/(1-expected) if expected<1 else None,unsure_is_a_retained_category=True))
    return dict(item_states=dict(population),by_source={k:dict(v) for k,v in sources.items()},by_music_group={k:dict(v) for k,v in music.items()},
        by_task={k:dict(v) for k,v in tasks.items()},by_reviewer={k:dict(v) for k,v in reviewers.items()},pairwise_agreement=scores,agreement_scope="first two original votes per item, excluding adjudication votes",
        missing_original_reviews=sum(max(0,2-len(by_item[b["blind_item_id"]])) for b in bindings),
        confidence_intervals="NOT_EVALUABLE_UNREGISTERED_INDEPENDENT_SAMPLING_PROTOCOL",model_performance_qualification=False)
