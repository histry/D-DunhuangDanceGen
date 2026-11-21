"""Exact decimal-rational reachability at actual frame centers.

These certificates cover piecewise-linear clocks with endpoint/rate/window
constraints only. They never certify an interpolation family, G1 or physics.
"""
from fractions import Fraction
from .feasible_time_map import interval,integer

def rational(x):
    if isinstance(x,bool):raise ValueError('Boolean is not a physical time')
    try:return x if isinstance(x,Fraction) else Fraction(str(x))
    except (ValueError,TypeError,ZeroDivisionError) as e:raise ValueError('Finite rational time required') from e

def pair(x):return [x.numerator,x.denominator]

def domain(source_interval,target_frames,rates):
    a,b=interval(source_interval);T=integer(target_frames);l,u=map(rational,rates)
    if b-a<2 or T<2 or not 0<l<=u:raise ValueError('Two real frame centers and positive ordered rates required')
    return Fraction(a),Fraction(b-1),Fraction(T-1),l,u

def single(source_interval,anchor_source_frame,target_frames,requested_target_frame,tolerance=0,rates=(.75,1.33),additional_constraints=()):
    a,b,D,l,u=domain(source_interval,target_frames,rates);s=rational(anchor_source_frame);t=rational(requested_target_frame);tol=rational(tolerance)
    if not a<=s<=b or tol<0:raise ValueError('Supported source anchor and nonnegative tolerance required')
    lower=max(l*(s-a),D-u*(b-s));upper=min(u*(s-a),D-l*(b-s));duration_ok=l*(b-a)<=D<=u*(b-a)
    error=None if not duration_ok or lower>upper else max(lower-t,t-upper,Fraction(0));ok=error is not None and error<=tol
    scope='ENDPOINTS_MONOTONICITY_LOCAL_RATE_ONLY_ACTUAL_FRAME_CENTERS'
    result=dict(source_interval=[int(a),int(b+1)],anchor_source_frame=float(s),target_frames=int(D+1),rate_min=float(l),rate_max=float(u),requested_target_frame=float(t),target_tolerance_frames=float(tol),continuous_reachable_interval=None if error is None else [float(lower),float(upper)],reachable_interval_rational=None if error is None else [pair(lower),pair(upper)],minimum_target_error_lower_bound=None if error is None else float(error),error_lower_bound_rational=None if error is None else pair(error),rate_status='REACHABLE_BY_RATE' if ok else 'EXCLUDED_BY_RATE',status=('UNKNOWN_WITH_ADDITIONAL_CONSTRAINTS' if additional_constraints else 'REACHABLE_BY_RATE') if ok else 'EXCLUDED_BY_RATE',additional_constraints=list(additional_constraints),exact_arithmetic='RATIONAL_FROM_DECLARED_DECIMAL_PARAMETERS',scope=scope,witness_clock=None)
    if ok:
        chosen=min(upper,max(lower,t));knots={a:Fraction(0),s:chosen,b:D};w=sorted(knots.items());assert all(l*(y-x)<=v-z<=u*(y-x) for (x,z),(y,v) in zip(w,w[1:]));result['witness_clock']=[[float(x),float(y)] for x,y in w];result['witness_clock_rational']=[[pair(x),pair(y)] for x,y in w]
    return result

def multiple(source_interval,target_frames,anchors,rates=(.75,1.33),fixed_knots=(),additional_constraints=()):
    """Exact interval propagation on a chain, including simultaneous constraints.

    Each anchor is {source_frame, requested, tolerance}; optional IDs stay in
    the caller. Duplicate source coordinates are intersected, never relocated.
    """
    a,b,D,l,u=domain(source_interval,target_frames,rates);boxes={a:[Fraction(0),Fraction(0)],b:[D,D]};requests=[]
    def add(x,lo,hi):
        if not a<=x<=b:raise ValueError('Source constraint outside actual support')
        lo=max(Fraction(0),lo);hi=min(D,hi)
        if x in boxes:lo=max(lo,boxes[x][0]);hi=min(hi,boxes[x][1])
        boxes[x]=[lo,hi]
    for r in anchors:
        x=rational(r['source_frame']);t=rational(r['requested']);tol=rational(r.get('tolerance',0))
        if tol<0:raise ValueError('Nonnegative tolerance required')
        add(x,t-tol,t+tol);requests.append((x,t))
    for x,y in fixed_knots:
        x,y=rational(x),rational(y);add(x,y,y)
    xs=sorted(boxes);bounds=[boxes[x][:] for x in xs];failed=None
    for i in range(len(xs)):
        if i:
            dx=xs[i]-xs[i-1];bounds[i][0]=max(bounds[i][0],bounds[i-1][0]+l*dx);bounds[i][1]=min(bounds[i][1],bounds[i-1][1]+u*dx)
        if bounds[i][0]>bounds[i][1]:failed=i;break
    if failed is None:
        for i in range(len(xs)-2,-1,-1):
            dx=xs[i+1]-xs[i];bounds[i][0]=max(bounds[i][0],bounds[i+1][0]-u*dx);bounds[i][1]=min(bounds[i][1],bounds[i+1][1]-l*dx)
            if bounds[i][0]>bounds[i][1]:failed=i;break
    result=dict(rate_status='REACHABLE_BY_RATE' if failed is None else 'EXCLUDED_BY_RATE',status=('UNKNOWN_WITH_ADDITIONAL_CONSTRAINTS' if additional_constraints else 'REACHABLE_BY_RATE') if failed is None else 'EXCLUDED_BY_RATE',scope='JOINT_ENDPOINT_RATE_AND_REGISTERED_ANCHOR_WINDOWS_ONLY',additional_constraints=list(additional_constraints),failed_source_frame=None if failed is None else float(xs[failed]),projected_intervals=None,witness_clock=None,exact_arithmetic='RATIONAL_FROM_DECLARED_DECIMAL_PARAMETERS')
    if failed is not None:return result
    result['projected_intervals']=[dict(source_frame=float(x),target_interval=[float(v) for v in bound],target_interval_rational=[pair(v) for v in bound]) for x,bound in zip(xs,bounds)]
    pref={x:t for x,t in requests};ys=[]
    for i,x in enumerate(xs):
        lo,hi=bounds[i]
        if i:lo=max(lo,ys[-1]+l*(x-xs[i-1]));hi=min(hi,ys[-1]+u*(x-xs[i-1]))
        assert lo<=hi;want=pref.get(x,D*(x-a)/(b-a));ys.append(min(hi,max(lo,want)))
    assert ys[0]==0 and ys[-1]==D and all(l*(xs[i]-xs[i-1])<=ys[i]-ys[i-1]<=u*(xs[i]-xs[i-1]) for i in range(1,len(xs)))
    result['witness_clock']=[[float(x),float(y)] for x,y in zip(xs,ys)];result['witness_clock_rational']=[[pair(x),pair(y)] for x,y in zip(xs,ys)];return result
