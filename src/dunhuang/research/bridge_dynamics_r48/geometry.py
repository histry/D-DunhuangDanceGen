"""Six bounded scalar freedoms, no hidden pose or endpoint edits."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import relative_fk
from dunhuang.research.manifold_bridge_r47.geometry import endpoint_jets,coefficients,jet,polynomial_max

def path(R,parameters,gap=(104,152),certify=False):
    a,b=gap;T=b-a+1;left=R[a-1];right=R[b]
    delta=Rotation.from_matrix(left.swapaxes(-1,-2)@right).as_rotvec()
    anchors=np.stack([left@Rotation.from_rotvec(delta*t).as_matrix() for t in np.linspace(0,1,5)])
    times=np.linspace(a-1,b,5);out=R.copy();maximum=0.;residual=0.
    jets=np.array([endpoint_jets(R[:,j],gap) for j in range(22)])
    for k in range(4):
        dt=times[k+1]-times[k];r=Rotation.from_matrix(anchors[k].swapaxes(-1,-2)@anchors[k+1]).as_rotvec();cc=[]
        wl=jets[:,0] if k==0 else delta/T*parameters[k-1]
        al=jets[:,1] if k==0 else delta/T**2*parameters[k+2]
        wr=jets[:,2] if k==3 else delta/T*parameters[k]
        ar=jets[:,3] if k==3 else delta/T**2*parameters[k+3]
        for j in range(22):
            c=coefficients(r[j],wl[j],al[j],wr[j],ar[j],dt);cc.append(c)
            if certify:
                maximum=max(maximum,polynomial_max(c));q=jet(c,0,dt)+jet(c,1,dt)
                residual=max(residual,*[float(np.max(abs(x-y))) for x,y in zip(q,(wl[j],al[j],wr[j],ar[j]))])
        cc=np.stack(cc,axis=1);ids=np.arange(a,b);ids=ids[(ids>=times[k])&(ids<times[k+1])];u=(ids-times[k])/dt
        v=np.einsum('ti,ijk->tjk',u[:,None]**np.arange(6),cc)
        out[ids]=anchors[k][None]@Rotation.from_rotvec(v.reshape(-1,3)).as_matrix().reshape(-1,22,3,3)
    return out,dict(maximum_chart_rad=maximum,physical_boundary_jet_residual=residual,hidden_read=False,fixed_anchor_count=5)

def optimize(R,J,parents,reference,cfg):
    a,b=cfg['gap'];x0=np.r_[np.ones(3),np.zeros(3)];base,_=path(R,x0,(a,b));P0=relative_fk(base,J,parents);w=cfg['objective_weights']
    # Residuals depend only on endpoint-derived paths, never source hidden ground truth.
    def residual(x):
        Q,_=path(R,x,(a,b));P=relative_fk(Q[a-3:b+3],J,parents);B=P0[a-3:b+3]
        d2=np.diff(P,n=2,axis=0)*900;d3=np.diff(P,n=3,axis=0)*27000
        return np.r_[np.sqrt(w['D2'])*d2.ravel()/max(reference['D2'],1e-6)/np.sqrt(d2.size),np.sqrt(w['D3'])*d3.ravel()/max(reference['D3'],1e-6)/np.sqrt(d3.size),np.sqrt(w['shape'])*(P[:,[16,17,20,21]]-B[:,[16,17,20,21]]).ravel()/np.sqrt(len(P)*12),np.sqrt(w['deviation'])*(x-x0)]
    lower=np.r_[np.full(3,cfg['velocity_scale_bounds'][0]),np.full(3,cfg['acceleration_scale_bounds'][0])];upper=np.r_[np.full(3,cfg['velocity_scale_bounds'][1]),np.full(3,cfg['acceleration_scale_bounds'][1])]
    fit=least_squares(residual,x0,bounds=(lower,upper),max_nfev=cfg['maximum_solver_evaluations'],ftol=1e-6,xtol=1e-6,gtol=1e-6)
    out,c=path(R,fit.x,(a,b),True);c.update(eligible=c['maximum_chart_rad']<cfg['chart_radius_rad'] and c['physical_boundary_jet_residual']<1e-7,parameters=fit.x.tolist(),evaluations=int(fit.nfev),termination=int(fit.status),initial_objective=float(residual(x0)@residual(x0)),final_objective=float(fit.fun@fit.fun),optimized_variables=6,endpoint_edit=False)
    return (out if c['eligible'] else None),c
