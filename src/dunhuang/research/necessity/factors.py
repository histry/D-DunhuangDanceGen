"""Matched random numbers, phase masks and exact occurrence-preserving corruption."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def ddim_times():
    # Same int-truncation operator used by the registered 50-step sampler.
    return np.linspace(-1,999,51).astype(int)[::-1][:-1]

def timesteps(uniform,actual):
    u=np.asarray(uniform)
    if not np.isfinite(u).all() or (u<0).any() or (u>=1).any():raise ValueError("Uniform quantiles in [0,1) required")
    return np.sort(ddim_times())[(u*50).astype(int)] if actual else 50+(u*900).astype(int)

def phase_mask(phases,start,known,keys,radius):
    known=np.asarray(known);keys=np.asarray(keys)
    if known.shape!=(150,8) or keys.shape!=known.shape or known.dtype!=bool or keys.dtype!=bool:raise ValueError("Observed upper8 masks required")
    result=keys.copy()
    for phase in phases:
        if phase["type"]=="root_turn_proxy":continue
        center=phase["frame"]-start
        if 0<=center<150:result[max(0,center-radius):min(150,center+radius+1)]=True
    return result&known

def donors(schedule,rows,seed):
    flat=np.asarray(schedule).reshape(-1);n=len(flat);rng=np.random.default_rng(seed)
    allowed=np.array([[rows[int(a)]["source_group"]!=rows[int(b)]["source_group"] and
        rows[int(a)]["choreography_uid"]!=rows[int(b)]["choreography_uid"] and
        rows[int(a)]["recording_uid"]!=rows[int(b)]["recording_uid"] for b in flat] for a in flat],bool)
    cost=rng.random((n,n));cost[~allowed]=1e6
    i,j=linear_sum_assignment(cost)
    if not allowed[i,j].all():raise ValueError("No exact source-marginal derangement")
    result=flat[j]
    if sorted(result.tolist())!=sorted(flat.tolist()):raise AssertionError("Source occurrence marginal changed")
    return result.reshape(np.asarray(schedule).shape)
