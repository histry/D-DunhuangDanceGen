"""Same six-dimensional physical projection plus two affine identity constraints."""
import numpy as np
from scipy.optimize import minimize
from dunhuang.research.protected_guidance.solver import constraints
def source_constraints(gains,z,margins):
    return gains@z-margins,gains
def project(a,gram,target,gains,margins,cfg):
    a=np.asarray(a,float);gram=np.asarray(gram,float);target=np.asarray(target,float);gains=np.asarray(gains,float);margins=np.asarray(margins,float);k=len(target)
    if a.shape!=(8250,k) or gram.shape!=(k,k) or gains.shape!=(2,k) or margins.shape!=(2,) or k>6:raise ValueError("Registered finite subspace dimensions required")
    if not all(np.isfinite(v).all() for v in (a,gram,target,gains,margins)) or np.any(margins<=0) or np.linalg.eigvalsh(gram).min()<=1e-9:raise ValueError("Finite independent basis and positive margins required")
    def values(z):
        p,j=constraints(a,gram,z,cfg["linear_safety_fraction"]);s,sj=source_constraints(gains,z,margins)
        return np.r_[p,s],np.concatenate([j,sj])
    fun=lambda z:.5*z@gram@z-target@z
    grad=lambda z:gram@z-target
    result=minimize(fun,np.zeros(k),jac=grad,method="SLSQP",constraints=dict(type="ineq",fun=lambda z:values(z)[0],jac=lambda z:values(z)[1]),
        options=dict(maxiter=cfg["max_solver_iterations"],ftol=cfg["solver_ftol"],disp=False))
    value,j=values(result.x);violation=float(max(0,-value.min()))
    accepted=bool(result.success and np.isfinite(result.x).all() and violation<=cfg["linear_tolerance"])
    z=result.x if accepted else np.zeros(k)
    return z,dict(status="SOLVED_FINITE_SUBSPACE" if accepted else "UNKNOWN_SOLVER_OR_LINEAR_FEASIBILITY",scipy_status=int(result.status),iterations=int(result.nit),
        max_linear_violation=violation,rank=k,linear_source_gains=(gains@z).tolist(),linear_source_margins=margins.tolist(),
        rejected_candidate_source_gains=(gains@result.x).tolist() if np.isfinite(result.x).all() else None,global_infeasibility_claimed=False,solver_message=str(result.message))
