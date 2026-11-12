"""Exact integer-frame feasible domains for the frozen four-knot primitive map.

This opt-in mathematical worker never relocates source anchors or changes V1.
All rate bounds use integer arithmetic: 3/4 <= delta_music/delta_source <= 133/100.
"""
from fractions import Fraction
from numbers import Integral
import numpy as np
from .time_map import source_positions

RADIUS=12

def integer(x):
    if isinstance(x,bool) or not isinstance(x,Integral):raise ValueError("Integer physical frame required")
    return int(x)

def interval(values):
    if len(values)!=2:raise ValueError("Two interval endpoints required")
    a,b=map(integer,values)
    if not 0<=a<b:raise ValueError("Nonnegative nonempty half-open interval required")
    return a,b

def domain(source_interval,music_interval,center):
    a,b=interval(source_interval);A,B=interval(music_interval)
    result=dict(schema="four_knot_integer_domain_v1",source_interval=[a,b],music_interval=[A,B],
        radius=RADIUS,rate_numerators=[75,133],rate_denominator=100,
        integer_domain=None,continuous_domain_100=None)
    def reject(reason):return dict(result,status=reason)
    if center is None:return reject("NO_SAFE_ANCHOR")
    c=integer(center);result["source_center"]=c
    if not 100*(B-A)>=75*(b-a) or not 100*(B-A)<=133*(b-a):return reject("DURATION_RATIO")
    if c-RADIUS<a or c+RADIUS>=b:return reject("SOURCE_CONTEXT")
    L,R=c-a-RADIUS,b-c-RADIUS
    if L<=0 or R<=0:return reject("ZERO_OR_NONMONOTONE_WARP")
    low=max(100*(A+RADIUS)+75*L,100*(B-RADIUS)-133*R,100*(A+RADIUS))
    high=min(100*(A+RADIUS)+133*L,100*(B-RADIUS)-75*R,100*(B-RADIUS-1))
    result.update(left_source_frames=L,right_source_frames=R,continuous_domain_100=[low,high])
    if low>high:return reject("EMPTY_CONTINUOUS_DOMAIN")
    lo,hi=-((-low)//100),high//100
    if lo>hi:return reject("NO_INTEGER_TICK")
    return dict(result,status="PASS",integer_domain=[lo,hi])

def solve(entry,slot,beats,mode="temporal"):
    if mode not in ("temporal","adapter"):raise ValueError("Unknown condition mode")
    d=domain(entry["source_interval"],[slot["start"],slot["end"]],entry.get("center"))
    if mode=="temporal" and entry.get("center") is not None and entry.get("normalization_support") is not True:
        return dict(d,status="NORMALIZER_SUPPORT",integer_domain=None,tick=None,time_warp=None)
    beats=sorted(set(integer(t) for t in beats))
    if any(t<0 for t in beats):raise ValueError("Negative beat time")
    if d["status"]!="PASS":return dict(d,tick=None,time_warp=None)
    a,b=d["source_interval"];A,B=d["music_interval"];c=d["source_center"];lo,hi=d["integer_domain"]
    preferred=A+round(Fraction((c-a)*(B-A),b-a))
    slot_beats=[t for t in beats if A<=t<B]
    candidates={min(hi,max(lo,t)) for t in slot_beats} if slot_beats else {min(hi,max(lo,preferred))}
    distance=lambda tick:min(abs(tick-t) for t in slot_beats) if slot_beats else 0
    tick=min(candidates,key=lambda t:(distance(t),abs(t-preferred),t))
    warp=[[a,A],[c-RADIUS,tick-RADIUS],[c+RADIUS,tick+RADIUS],[b,B]]
    array=np.asarray(warp,np.int64);steps=np.diff(array,axis=0)
    if np.any(steps<=0) or np.any(100*steps[:,1]<75*steps[:,0]) or np.any(100*steps[:,1]>133*steps[:,0]):
        raise AssertionError("Constructed map violates its exact domain")
    source_positions(dict(source_interval=[a,b],music_interval=[A,B],time_warp=warp),np.arange(A,B))
    return dict(d,tick=tick,time_warp=warp,linear_preferred_tick=preferred,
        selection_rule="min slot-beat distance, then linear displacement, then earlier integer tick",
        slot_beat_count=len(slot_beats),feasible_beat_count=sum(lo<=t<=hi for t in slot_beats),
        beat_distance_frames=distance(tick) if slot_beats else None,linear_shift_frames=tick-preferred)
