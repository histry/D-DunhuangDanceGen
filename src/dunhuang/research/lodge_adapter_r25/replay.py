"""Only FP32-derived gradient-norm diagnostics use a declared relative precision."""
import math
GRADIENT_NORM_RTOL=1e-6

def verify_readout(actual,expected):
    a={k:v for k,v in actual.items() if k!="gradient_norms"}
    b={k:v for k,v in expected.items() if k!="gradient_norms"}
    if a!=b:raise AssertionError("Non-gradient readout must remain exact")
    ga=actual["gradient_norms"];gb=expected["gradient_norms"];assert ga.keys()==gb.keys()
    deltas={}
    for name,x in ga.items():
        y=gb[name];assert math.isfinite(x) and math.isfinite(y) and min(x,y)>=0
        if not math.isclose(x,y,rel_tol=GRADIENT_NORM_RTOL,abs_tol=0.):
            raise AssertionError("Gradient norm precision exceeded: "+name)
        deltas[name]=dict(actual=x,expected=y,absolute_error=abs(x-y),relative_error=abs(x-y)/max(abs(x),abs(y)) if x or y else 0.)
    return dict(non_gradient_fields_exact=True,gradient_norms_bitwise_exact=ga==gb,
        gradient_norm_rtol=GRADIENT_NORM_RTOL,gradient_norm_atol=0.,gradient_norm_deltas=deltas)


def change_conditions():
    """JSON-stable condition grid; null rows remain0 and3."""
    return [[m,event] for m in range(2) for event in (-1,0,1)]
