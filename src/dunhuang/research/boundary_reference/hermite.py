import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices,sixd

def coefficients(p0,v0,a0,p1,v1,a1,duration):
    if duration<=0:raise ValueError("Positive physical duration required")
    c=np.zeros((6,)+p0.shape);c[0]=p0;c[1]=v0*duration;c[2]=a0*duration**2/2
    rhs=np.stack([p1-c[:3].sum(0),v1*duration-c[1]-2*c[2],a1*duration**2-2*c[2]])
    c[3:]=np.linalg.solve(np.array([[1,1,1],[3,4,5],[6,12,20]],float),rhs.reshape(3,-1)).reshape((3,)+p0.shape)
    return c

def evaluate(c,u,order=0,duration=1.):
    result=np.zeros((len(np.atleast_1d(u)),)+c.shape[1:])
    for k in range(order,6):
        factor=1
        for j in range(order):factor*=k-j
        result+=np.asarray(u).reshape((-1,)+(1,)*(c.ndim-1))**(k-order)*factor*c[k]/duration**order
    return result

def fit(full,a,b):
    # Endpoint frame centers a-1 and b. BDF2 velocities and one-sided accelerations.
    dt=1/30.;D=(b-a+1)*dt
    p=full[a-3:b+3].astype(float)
    v0=(3*p[2]-4*p[1]+p[0])/(2*dt);a0=(p[2]-2*p[1]+p[0])/dt**2
    v1=(-3*p[-3]+4*p[-2]-p[-1])/(2*dt);a1=(p[-3]-2*p[-2]+p[-1])/dt**2
    c=coefficients(p[2],v0,a0,p[-3],v1,a1,D)
    return evaluate(c,np.arange(1,b-a+1)/(b-a+1),duration=D),c,D

def reference(old,a=60,b=90,margin=.05):
    x=old.copy();root,c,D=fit(old[:,4:7],a,b)
    R=matrices(old[:,7:].reshape(-1,24,6)).astype(float);base=R[a-1]
    chart=Rotation.from_matrix((base.swapaxes(-1,-2)[None]@R).reshape(-1,3,3)).as_rotvec().reshape(len(R),24,3)
    yr,rc,_=fit(chart,a,b)
    if np.linalg.norm(chart[a-3:b+3],axis=-1).max()>=np.pi-margin or np.linalg.norm(yr,axis=-1).max()>=np.pi-margin:
        return None,dict(status="SO3_CHART_CUT_REJECTED",continuous_root_c2=False)
    x[a:b,4:7]=root.astype(np.float32)
    new=base[None]@Rotation.from_rotvec(yr.reshape(-1,3)).as_matrix().reshape(b-a,24,3,3)
    x[a:b,7:]=sixd(new).reshape(b-a,144).astype(np.float32)
    # The analytic reference matches the declared endpoint states before FP32 sampling.
    residual=max(float(abs(evaluate(c,[0],0,D)[0]-old[a-1,4:7]).max()),float(abs(evaluate(c,[1],0,D)[0]-old[b,4:7]).max()))
    return x,dict(status="CONTINUOUS_QUINTIC_REFERENCE",continuous_root_c2=True,root_endpoint_position_error_m=residual,
        sampled_discrete_c2_guaranteed=False,source_or_physics_qualified=False,contacts_changed=False)
