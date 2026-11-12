"""Source-balanced full rotations; no first-batch truncation when groups > batch."""
import numpy as np

def group_schedule(rows, steps, batch, seed):
    if steps < 1 or batch < 1: raise ValueError("Positive schedule dimensions required")
    groups={}
    for i,r in enumerate(rows):
        if r["role"]=="train": groups.setdefault(r["source_group"],[]).append(i)
    if not groups: raise ValueError("No training groups")
    rng=np.random.default_rng(seed); names=sorted(groups); queue=[]
    schedule=[]
    for _ in range(steps):
        ids=[]
        for _ in range(batch):
            if not queue: queue=list(rng.permutation(len(names)))
            group=names[queue.pop(0)]
            ids.append(int(rng.choice(groups[group])))
        schedule.append(ids)
    return np.asarray(schedule,dtype=np.int64)

def coverage(rows, schedule):
    from collections import Counter
    ids=np.asarray(schedule).reshape(-1)
    if any(i<0 or i>=len(rows) or rows[int(i)]["role"]!="train" for i in ids):
        raise ValueError("Schedule includes non-training rows")
    counts=Counter(rows[int(i)]["source_group"] for i in ids)
    names={r["source_group"] for r in rows if r["role"]=="train"}
    return dict(groups=len(names),visited_groups=len(counts),omitted_groups=len(names-set(counts)),
        samples=int(len(ids)),per_group_counts=sorted(counts.values()),unique_windows=len(set(map(int,ids))))
