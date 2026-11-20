"""R69 bounded proxy baselines, not reproductions of external systems.

Both solvers retain source order and fixed endpoints.  Their finite mixed
node domain contains the exact Uniform clock as well as integer target
positions.  Costs optimize registered proxies, never artistic quality.
"""
from fractions import Fraction
import hashlib
import math
import time

import numpy as np

FRAMES = 240
_RATE_LOWER, _RATE_UPPER = Fraction(3,4), Fraction(133,100)


def _real(v,name):
    if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.integer,np.floating)) or not math.isfinite(v):
        raise ValueError('Finite numeric '+name+' required')
    return float(v)


def _integer(v,name):
    x=_real(v,name)
    if x!=int(x):raise ValueError('Integer '+name+' required')
    return int(x)


def _interval(v,name,n=None):
    if not isinstance(v,(list,tuple,np.ndarray)) or len(v)!=2:
        raise ValueError('Actual half-open '+name+' interval required')
    a,b=[_integer(x,name+' frame') for x in v]
    if a<0 or b-a<2 or n is not None and b-a!=n:
        raise ValueError('Invalid actual '+name+' frame support')
    return a,b


def _vector(v,name,length=None):
    raw=np.asarray(v,dtype=object)
    if raw.ndim!=1 or any(isinstance(x,(bool,np.bool_,str)) or not isinstance(x,(int,float,np.integer,np.floating)) for x in raw):
        raise ValueError('Numeric one-dimensional '+name+' required')
    x=np.asarray(v,dtype=float)
    if not np.isfinite(x).all() or length is not None and len(x)!=length:
        raise ValueError('Finite full-coverage '+name+' required')
    return x


def _sha(x):
    a=np.ascontiguousarray(x,dtype='<f8')
    return hashlib.sha256(a.tobytes()).hexdigest()


def _config(band_frames,max_internal_nodes,deformation_weight,**other):
    band=_integer(band_frames,'band_frames');cap=_integer(max_internal_nodes,'max_internal_nodes')
    weight=_real(deformation_weight,'deformation_weight')
    if band<0 or cap<0 or weight<0:raise ValueError('Nonnegative fixed solver budgets required')
    return dict(band_frames=band,max_internal_nodes=cap,deformation_weight=weight,
        rate_limits_dt_ds=[.75,1.33],target_frames=FRAMES,
        candidate_domain='EXACT_UNIFORM_FRACTION_UNION_ALL_INTEGER_LOCAL_TARGET_FRAMES_IN_UNIFORM_BAND',
        continuous_domain_complete=False,tie_policy='EXACT_FLOAT_COST_THEN_LEXICOGRAPHIC_TARGET_PATH',**other)


def _base(method,source_interval,target_interval,config):
    a,b=_interval(source_interval,'source');A,B=_interval(target_interval,'target',FRAMES)
    rho=Fraction(FRAMES-1,b-a-1)
    r=dict(method=method,source_interval=[a,b],target_interval=[A,B],frames=FRAMES,
        config=config,uniform_dt_ds=float(rho),endpoint_rate_legal=bool(_RATE_LOWER<=rho<=_RATE_UPPER),
        map_direction='SOURCE_TO_TARGET_ACTUAL_FRAME_CENTERS',source_positions_authoritative=True,
        map_smoothness='PIECEWISE_LINEAR_NOT_C1_OR_C2_GUARANTEED',source_order_preserved=True,
        source_observation_scope='SOURCE_INTERVAL_ONLY_NO_MISSING_FRAME_INVENTION',
        phase_semantic_truth='UNKNOWN',human_reviewed=False,artistic_quality='NOT_EVALUABLE',
        Native24='NOT_EVALUABLE',can_route=False,fallback_used=False,
        counts=dict(dp_stages=0,states_reached=0,transition_pairs_considered=0,
            rate_rejected_transitions=0,legal_transitions=0,edge_objective_evaluations=0,
            node_objective_evaluations=0,legal_complete_paths=0))
    return r,rho


def _finish(result,started,status,reason=None):
    result.update(status=status,reason=reason,
        runtime_seconds=float(time.perf_counter()-started),
        runtime_scope='WALL_CLOCK_ONLY_REMOVE_TO_SEPARATE_RUNTIME_TABLE_BEFORE_SCIENTIFIC_SEAL')
    return result


def _options(nodes,source_interval,band):
    a,b=source_interval;S=b-a-1;options=[];linear=[];details=[]
    for k,s in enumerate(nodes):
        u=Fraction((s-a)*(FRAMES-1),S);linear.append(u)
        if k in (0,len(nodes)-1):
            own=[u];integer_count=1;merged=True
        else:
            lo=max(1,math.ceil(u-band));hi=min(FRAMES-2,math.floor(u+band))
            ints=[Fraction(i) for i in range(lo,hi+1)]
            merged=u in ints;own=sorted(set(ints+[u]));integer_count=len(ints)
        options.append(own)
        details.append(dict(source_frame=s,uniform_target_local=float(u),
            uniform_target_exact=[u.numerator,u.denominator],integer_candidate_count=integer_count,
            uniform_merged_with_integer=bool(merged),actual_candidate_count=len(own)))
    return options,linear,details


def _deformation(x0,x1,y0,y1,rho,total_source_span):
    rate=(y1-y0)/(x1-x0)
    return (x1-x0)/total_source_span*(float(rate/rho)-1.)**2


def _solve(result,nodes,options,linear,rho,node_cost,edge_cost):
    counts=result['counts'];counts['dp_stages']=len(nodes)-1
    # Each state stores the best path AND the number of all admissible paths
    # reaching that state.  Counting does not enumerate or claim a continuous domain.
    start=options[0][0]
    states={start:(float(node_cost(0,start)),0.,(start,),1)}
    counts['states_reached']=1
    for k in range(1,len(nodes)):
        current={}
        for y in options[k]:
            local_node=float(node_cost(k,y));trials=[];path_count=0
            for previous,(matching,regularizer,path,ways) in states.items():
                counts['transition_pairs_considered']+=1
                rate=(y-previous)/(nodes[k]-nodes[k-1])
                if not _RATE_LOWER<=rate<=_RATE_UPPER:
                    counts['rate_rejected_transitions']+=1;continue
                counts['legal_transitions']+=1;counts['edge_objective_evaluations']+=1
                cost=float(edge_cost(k,previous,y))
                reg=_deformation(nodes[k-1],nodes[k],previous,y,rho,nodes[-1]-nodes[0])
                new_match=matching+local_node+cost;new_reg=regularizer+reg
                total=new_match+result['config']['deformation_weight']*new_reg
                trials.append((total,path+(y,),new_match,new_reg))
                path_count+=ways
            if trials:
                _,path,matching,regularizer=min(trials,key=lambda row:(row[0],row[1]))
                current[y]=(matching,regularizer,path,path_count)
        states=current;counts['states_reached']+=len(states)
        if not states:return None
    terminal=options[-1][0]
    matching,regularizer,path,ways=states[terminal]
    counts['legal_complete_paths']=int(ways)
    return path,float(matching),float(regularizer)


def _materialize(result,nodes,options,linear,details,solved):
    path,matching,regularizer=solved;a,b=result['source_interval'];A,B=result['target_interval']
    local=np.asarray([float(v) for v in path]);source=np.asarray(nodes,dtype=float)
    selected_uniform=all(y==u for y,u in zip(path,linear))
    q=(np.linspace(a,b-1,FRAMES) if selected_uniform else
       np.interp(np.arange(FRAMES,dtype=float),local,source))
    if not np.isfinite(q).all() or not (np.diff(q)>0).all() or q[0]!=a or q[-1]!=b-1:
        raise AssertionError('Solver inverse lost actual monotone source support')
    exact_rates=[(path[k]-path[k-1])/(nodes[k]-nodes[k-1]) for k in range(1,len(nodes))]
    if not all(_RATE_LOWER<=rate<=_RATE_UPPER for rate in exact_rates):
        raise AssertionError('Solver materialized an illegal continuous piece')
    warp=np.c_[source,A+local]
    serialized=np.diff(warp[:,1])/np.diff(warp[:,0])
    reconstructed=np.interp(q,source,local)
    inverse_error=float(abs(reconstructed-np.arange(FRAMES)).max())
    if inverse_error>1e-8:raise AssertionError('Local inverse interpolation residual')
    result.update(time_warp=warp.tolist(),source_positions=q,q=q,
        source_nodes=nodes,source_node_options=details,
        target_node_options_local=[[float(y) for y in own] for own in options],
        selected_target_nodes_exact=[[v.numerator,v.denominator] for v in path],
        selected_uniform_path=bool(selected_uniform),
        source_positions_sha256_f64le=_sha(q),interpolation_read_support=sorted(set(
            np.floor(q).astype(int).tolist()+np.ceil(q).astype(int).tolist())),
        proof=dict(legal=True,rate_limits_dt_ds=[.75,1.33],
            source_to_target_rate_min=float(min(exact_rates)),source_to_target_rate_max=float(max(exact_rates)),
            inverse_rate_min=float(1/max(exact_rates)),inverse_rate_max=float(1/min(exact_rates)),
            rate_check='EXACT_RATIONAL_RATE_OF_EACH_CONTINUOUS_PIECE_NO_THRESHOLD_RELAXATION',
            source_to_target_rates_exact=[[r.numerator,r.denominator] for r in exact_rates],
            serialized_absolute_target_float_rate_min=float(serialized.min()),
            serialized_absolute_target_float_rate_max=float(serialized.max()),
            absolute_float_serialization_roundoff_is_not_a_new_physical_rate_bound=True,
            q_inverse_computed_on_local_target_clock=True,inverse_max_residual_frames=inverse_error,
            source_support_preserved=True,endpoint_centres_exact=True),
        objective=dict(matching=matching,deformation=regularizer,
            total=matching+result['config']['deformation_weight']*regularizer,
            deformation_definition='SOURCE_SPAN_WEIGHTED_MEAN((DT_DS/UNIFORM_DT_DS-1)^2)',
            scope='INTERNAL_OPTIMIZATION_PROXY_NOT_INDEPENDENT_QUALITY'))
    return result


def beat_dp(source_interval,target_interval,node_frames,node_weights,beat_frames,*,
            band_frames=15,max_internal_nodes=8,sigma_frames=3,deformation_weight=.03):
    """Finite Gaussian-beat DP using existing source nodes, not semantic labels."""
    started=time.perf_counter();sigma=_real(sigma_frames,'sigma_frames')
    if sigma<=0:raise ValueError('Positive Gaussian frame sigma required')
    config=_config(band_frames,max_internal_nodes,deformation_weight,sigma_frames=sigma,
        source_node_rule='MAX_WEIGHT_THEN_SOURCE_FRAME_CAP_BEFORE_TARGET_SCORE; ENDPOINTS_FIXED',
        matching_definition='SUM_RETAINED_NODE_NORMALIZED_WEIGHT*(1-MAX_GAUSSIAN_TO_IN_CLIP_BEAT)')
    result,rho=_base('BaselineBeatDP_FIXED_PROXY_ADAPTATION',source_interval,target_interval,config)
    a,b=result['source_interval'];A,B=result['target_interval']
    frames=_vector(node_frames,'source node frames')
    if any(v!=int(v) for v in frames) or len(frames) and (frames[0]<a or frames[-1]>b-1 or not (np.diff(frames)>0).all()):
        raise ValueError('Unique chronological original source nodes inside source interval required')
    weights=np.ones(len(frames)) if node_weights is None else _vector(node_weights,'source node weights',len(frames))
    if (weights<0).any():raise ValueError('Nonnegative source node weights required')
    beats=_vector(beat_frames,'target beat frames')
    if len(beats) and (beats[0]<0 or not (np.diff(beats)>0).all()):
        raise ValueError('Unique chronological nonnegative target beat observations required')
    used=beats[(beats>=A)&(beats<=B-1)]-A
    result['counts'].update(source_nodes_input=len(frames),target_beat_frames_input=len(beats),
        target_beat_frames_used=len(used),target_beat_frames_outside_clip=len(beats)-len(used))
    positive=[(int(s),float(weight)) for s,weight in zip(frames,weights) if weight>0]
    fixed=[p for p in positive if p[0] in (a,b-1)]
    internal=sorted([p for p in positive if a<p[0]<b-1],key=lambda row:(-row[1],row[0]))[:config['max_internal_nodes']]
    retained=sorted(fixed+internal);result['retained_source_events']=[dict(frame=s,weight=v) for s,v in retained]
    result['counts'].update(source_events_positive_weight=len(positive),retained_source_events=len(retained),
        retained_internal_source_events=len(internal),source_events_not_retained=len(positive)-len(retained))
    if not retained or not len(used):return _finish(result,started,'NO_EVENT','SOURCE_NODES_OR_IN_CLIP_BEATS_MISSING')
    if not result['endpoint_rate_legal']:return _finish(result,started,'NOT_FEASIBLE','ENDPOINT_RATE_OUTSIDE_FROZEN_LIMITS')
    nodes=sorted(set([a,b-1]+[s for s,v in retained]));options,linear,details=_options(nodes,result['source_interval'],config['band_frames'])
    denominator=sum(v for s,v in retained)
    if not math.isfinite(denominator):raise ValueError('Finite summed source node weight required')
    event_weights={s:v/denominator for s,v in retained}
    node_cache={}
    for k,own in enumerate(options):
        weight=event_weights.get(nodes[k],0.)
        for y in own:
            distance=float(np.min(abs(used-float(y))))
            node_cache[k,y]=weight*(1-math.exp(-distance*distance/(2*sigma*sigma)))
    result['counts']['node_objective_evaluations']=len(node_cache)
    solved=_solve(result,nodes,options,linear,rho,lambda k,y:node_cache[k,y],lambda k,x,y:0.)
    if solved is None:return _finish(result,started,'NOT_FEASIBLE','NO_LEGAL_PATH_IN_REGISTERED_FINITE_MIXED_DOMAIN')
    return _finish(_materialize(result,nodes,options,linear,details,solved),started,'LEGAL')


def rhythm_dtw(source_interval,target_interval,source_profile,target_profile,*,
               band_frames=15,max_internal_nodes=8,deformation_weight=.03):
    """Constrained rhythm-DTW-inspired piecewise-linear proxy adaptation.

    Profiles are normalized once over their complete original supports.
    The source profile is resampled as an observed scalar signal; this does
    NOT include the time-derivative chain-rule scaling of actual motion.
    """
    started=time.perf_counter();config=_config(band_frames,max_internal_nodes,deformation_weight,
        source_node_rule='ENDPOINTS_PLUS_ROUNDED_UNIFORM_SOURCE_GRID_MAX8_BY_DEFAULT',
        matching_definition='FIXED240_TARGET_FRAME_GLOBAL_TRAPEZOID_ERROR_WITH_EACH_FRAME_COUNTED_ONCE_DIVIDED_BY239',
        profile_chain_rule_scaling=False,external_system_reproduction=False,
        adaptation_scope='RHYTHM_DTW_IDEA_WITH_FIXED_PROXY_PROFILES_AND_LEGAL_PIECEWISE_LINEAR_CLOCK')
    result,rho=_base('BaselineRhythmDTW_CONSTRAINED_PROXY_ADAPTATION',source_interval,target_interval,config)
    a,b=result['source_interval'];S=b-a-1
    source=_vector(source_profile,'source rhythm profile',b-a)
    target=_vector(target_profile,'target rhythm profile',FRAMES)
    with np.errstate(over='ignore',invalid='ignore'):
        sm,tm=float(source.mean()),float(target.mean());ss,ts=float(source.std()),float(target.std())
    if not all(math.isfinite(v) for v in (sm,tm,ss,ts)):
        raise ValueError('Finite full-profile normalization statistics required')
    result['normalization']=dict(source_mean=sm,source_std=ss,target_mean=tm,target_std=ts,
        source_profile_sha256_f64le=_sha(source),target_profile_sha256_f64le=_sha(target),
        scope='ONCE_INDEPENDENTLY_OVER_COMPLETE_SOURCE_AND_ALL240_TARGET_FRAMES; NO_LOCAL_SEGMENT_REFIT',
        constant_std_epsilon=1e-12)
    if ss<=1e-12 or ts<=1e-12:
        return _finish(result,started,'NOT_EVALUABLE','CONSTANT_SOURCE_OR_TARGET_RHYTHM_PROFILE')
    if not result['endpoint_rate_legal']:return _finish(result,started,'NOT_FEASIBLE','ENDPOINT_RATE_OUTSIDE_FROZEN_LIMITS')
    zsource=(source-sm)/ss;ztarget=(target-tm)/ts
    nodes=np.unique(np.rint(np.linspace(a,b-1,config['max_internal_nodes']+2)).astype(int)).tolist()
    options,linear,details=_options(nodes,result['source_interval'],config['band_frames'])
    source_centres=np.arange(len(source),dtype=float)
    def edge_cost(k,y0,y1):
        lo,hi=float(y0),float(y1);x0,x1=nodes[k-1]-a,nodes[k]-a
        # One fixed global quadrature over the actual240 target frames.  A
        # segment owns [y0,y1), with the final segment also owning frame239.
        # Fractional nodes add no extra samples; integer shared nodes count once.
        first=math.ceil(y0)
        stop=(FRAMES if k==len(nodes)-1 else math.ceil(y1))
        t=np.arange(first,stop,dtype=float)
        q=x0+(t-lo)*(x1-x0)/(hi-lo)
        error=(np.interp(q,source_centres,zsource)-ztarget[t.astype(int)])**2
        weights=np.ones(len(t));weights[(t==0)|(t==FRAMES-1)]=.5
        return float(error@weights)/(FRAMES-1)
    solved=_solve(result,nodes,options,linear,rho,lambda k,y:0.,edge_cost)
    if solved is None:return _finish(result,started,'NOT_FEASIBLE','NO_LEGAL_PATH_IN_REGISTERED_FINITE_MIXED_DOMAIN')
    result=_materialize(result,nodes,options,linear,details,solved)
    errors=(np.interp(result['source_positions']-a,source_centres,zsource)-ztarget)**2
    weights=np.ones(FRAMES);weights[[0,-1]]=.5
    check=float(errors@weights)/(FRAMES-1)
    if abs(check-result['objective']['matching'])>1e-9:
        raise AssertionError('DP segment costs differ from actual complete-frame profile cost')
    result['objective']['actual240_profile_matching_check']=check
    return _finish(result,started,'LEGAL')
