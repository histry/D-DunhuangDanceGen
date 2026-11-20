"""Frozen finite compensation domain; target postmaps preserve both edge jets.

High/Low use translated copies of the same C3 transition kernel. Distributed
splits each transition into two half-width, half-mass pieces: its first-rate
energy matches, but its curvature does NOT. It is a separate diagnostic unless
all registered dose criteria match. No semantic phase labels are manufactured.
"""
import itertools
import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq
from dunhuang.research.local_timing_r68 import warp as old

S=Polynomial([0,0,0,0,35,-84,70,-20])

def smooth(x,order=0):
    x=np.asarray(x,float)
    if order==0:return S(np.clip(x,0,1))
    return np.where((x>0)&(x<1),S.deriv(order)(np.clip(x,0,1)),0.)

def field(u,segments,order=0):
    u=np.asarray(u,float);out=np.zeros_like(u)
    for a,b,weight in segments:out+=weight*smooth((u-a)/(b-a),order)/(b-a)**order
    return out

def proposals(centre,activity):
    """Rank locations using only frozen baseline activity, before legality/quality."""
    result=[]
    for width in (54.,60.,66.):
        left_end=centre-8.;right_start=centre+8.
        def starts(lo,hi):
            return sorted(set([float(lo),float(hi)]+list(np.arange(lo,hi+1e-8,6.)))) if hi>=lo else []
        left=starts(3.,left_end-width);right=starts(right_start,236.-width)
        all_pairs=[]
        for l,r in itertools.product(left,right):
            seg=[[l,l+width,1.],[r,r+width,-1.]]
            density=np.abs(field(np.arange(240),seg,1))
            score=float(density@activity/max(density.sum(),1e-12))
            all_pairs.append(dict(width=width,segments=seg,activity=score))
        for kind,sign in [('High',-1),('Low',1)]:
            for rank,p in enumerate(sorted(all_pairs,key=lambda p:(sign*p['activity'],p['segments']))[:4]):
                result.append(dict(p,kind=kind,rank=rank))
        if left and right:
            half=width/2
            # Two nonoverlapping pieces per side, separated as widely as support allows.
            seg=[[3.,3.+half,.5],[left_end-half,left_end,.5],
                 [right_start,right_start+half,-.5],[236.-half,236.,-.5]]
            density=np.abs(field(np.arange(240),seg,1))
            result.append(dict(width=width,segments=seg,activity=float(density@activity/max(density.sum(),1e-12)),kind='Distributed',rank=0))
    return result

def solve(case,proposal,delta):
    baseline=old.baseline(case['time_warp'],case['target_interval'])
    base=old._setup(case['time_warp'],case['target_interval'])[0].copy();A=case['target_interval'][0];base[:,1]-=A
    segments=proposal['segments'];points=[]
    for a,b,_ in segments:points += [a,(a+b)/2,b]
    rates=[]
    for l,r in zip(base[:-1],base[1:]):
        sites=[l[1],r[1]]+[u for u in points if l[1]<=u<=r[1]]
        rates += [float((r[1]-l[1])/(r[0]-l[0])*(1+delta*field(u,segments,1))) for u in sites]
    proof=dict(min_rate=min(rates),max_rate=max(rates),limits=[.75,1.33],continuous_one_sided_extrema=True)
    result=dict(**proposal,delta=delta,proof=proof,status='NOT_FEASIBLE',q=None)
    if baseline['status']!='LEGAL' or not .75<=min(rates)<=max(rates)<=1.33:return dict(result,reason='FROZEN_RATE_LIMIT')
    t=np.arange(240,dtype=float);q0=baseline['q'];q=q0.copy()
    changed=abs(field(t,segments))>1e-12
    for i in np.flatnonzero(changed):
        u=brentq(lambda u:u+delta*field(u,segments)-t[i],0,239,xtol=1e-11)
        q[i]=np.interp(u,base[:,1],base[:,0])
    if not np.array_equal(q[:4],q0[:4]) or not np.array_equal(q[-4:],q0[-4:]):raise AssertionError('Protected edge changed')
    if not (np.diff(q)>0).all() or q[0]!=q0[0] or q[-1]!=q0[-1]:raise AssertionError('Support/order changed')
    energy={}
    for order in (1,2,3):
        antiderivative=(S.deriv(order)**2).integ();constant=float(antiderivative(1)-antiderivative(0))
        energy[str(order)]=sum(delta**2*weight**2*constant/(b-a)**(2*order-1) for a,b,weight in segments)
    dose=dict(target_offset_rms=float(np.sqrt(np.mean((delta*field(t,segments))**2))),
              source_shift_rms=float(np.sqrt(np.mean((q-q0)**2))),source_shift_max=float(abs(q-q0).max()),
              postmap_derivative_energy=energy,unchanged_edge_frames=4)
    return dict(result,status='LEGAL',reason=None,q=q,dose=dose,
                offset=delta*field(t,segments),compensation_mask=abs(field(t,segments,1))>1e-12,edited_mask=changed)

def relative_gap(a,b):return abs(a-b)/max(abs(a),abs(b),1e-12)

def match(a,b):
    fields=['target_offset_rms','source_shift_rms','source_shift_max']
    gaps={f:relative_gap(a['dose'][f],b['dose'][f]) for f in fields}
    gaps['rate_energy']=relative_gap(a['dose']['postmap_derivative_energy']['1'],b['dose']['postmap_derivative_energy']['1'])
    peak=abs(a['actual_peak_frame']-b['actual_peak_frame'])
    curvature=relative_gap(a['dose']['postmap_derivative_energy']['2'],b['dose']['postmap_derivative_energy']['2'])
    return dict(eligible=max(gaps.values())<=.05 and curvature<=.05 and peak<=1.,gaps=gaps,peak_gap_frames=peak,
                curvature_energy_gap=relative_gap(a['dose']['postmap_derivative_energy']['2'],b['dose']['postmap_derivative_energy']['2']),
                phase_semantics='UNKNOWN',budget_scope='RMS_MAX_CLOCK_SHIFT_FIRST_AND_SECOND_RATE_ENERGY; CURVATURE_MISMATCH_EXCLUDES_PRIMARY')
