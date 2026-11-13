"""Deterministic acoustic hierarchy; candidate phrases are not semantic labels."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
def timeline(legacy,music,cfg):
    frames=legacy["frames"];x=np.asarray(music)[:frames,18:50]
    if len(x)!=frames or x.shape[1]!=32 or not np.isfinite(x).all():raise ValueError("Full real music50 clock required")
    novelty=gaussian_filter1d(np.r_[0,np.abs(np.diff(x,axis=0)).mean(1)],3)
    novelty/=max(float(novelty.max()),1e-8)
    prominence=max(float(np.median(abs(novelty-np.median(novelty)))),1e-8)
    phrases=find_peaks(novelty,distance=cfg["phrase_distance"],prominence=prominence)[0]
    beats=set(legacy["beats"]);nodes=sorted(set([0,frames,max(0,frames-cfg["target_frames"])]+list(range(0,frames,cfg["grid_frames"]))+list(beats)+phrases.tolist()))
    cost={0:0.};parent={}
    phrase=set(int(t) for t in phrases)
    for b in nodes[1:]:
        options=[]
        for a in reversed(nodes[:nodes.index(b)]):
            duration=b-a
            if duration>cfg["max_frames"]:break
            if duration<cfg["min_frames"] or a not in cost:continue
            bonus=0 if b==frames else cfg["novelty_weight"]*novelty[b]+cfg["beat_weight"]*(b in beats)+cfg["phrase_weight"]*(b in phrase)
            options.append((cost[a]+((duration-cfg["target_frames"])/30)**2-bonus,a))
        if options:cost[b],parent[b]=min(options)
    if frames not in parent:raise ValueError("Registered acoustic slot graph has no complete path")
    ends=[frames]
    while ends[-1]:ends.append(parent[ends[-1]])
    ends=ends[::-1]
    # Preserve the legacy actual audio RMS/density clock via length-weighted slots.
    energy=np.zeros(frames)
    for slot in legacy["slots"]:energy[slot["start"]:slot["end"]]=slot["density"]
    result=dict(legacy,phrase_method="full-audio logmel novelty candidates, not independent musical phrase truth",
        phrase_candidates=phrases.tolist(),novelty=novelty.tolist(),slots=[dict(start=a,end=b,density=float(energy[a:b].mean())) for a,b in zip(ends[:-1],ends[1:])],
        slot_method="global bounded-duration acoustic/beat dynamic program",acoustic_cost=float(cost[frames]))
    return result
