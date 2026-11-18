"""Observed-only, reanchored SO(3) quintics with physical body-rate/acceleration jets."""
import numpy as np
from scipy.spatial.transform import Rotation,RotationSpline
from numpy.polynomial.polynomial import polyval,polyder,polyroots
from dunhuang.research.bridge_r38.geometry import MATRIX

def skew(v):
    x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],float)

def jacobian(v):
    v=np.asarray(v);s=float(v@v);K=skew(v)
    if s<1e-8:a=.5-s/24+s*s/720;b=1/6-s/120+s*s/5040
    else:t=np.sqrt(s);a=(1-np.cos(t))/s;b=(t-np.sin(t))/(s*t)
    return np.eye(3)-a*K+b*K@K

def jacobian_direction(v,h):
    v=np.asarray(v);s=float(v@v);K=skew(v);H=skew(h);dot=float(v@h)
    if s<1e-8:a=.5-s/24+s*s/720;b=1/6-s/120+s*s/5040;da=-1/12+s/180-s*s/6720;db=-1/60+s/1260-s*s/60480
    else:t=np.sqrt(s);a=(1-np.cos(t))/s;b=(t-np.sin(t))/(s*t);da=(t*np.sin(t)-2*(1-np.cos(t)))/(s*s);db=(t*(1-np.cos(t))-3*(t-np.sin(t)))/(s*s*t)
    return -da*dot*K-a*H+db*dot*K@K+b*(H@K+K@H)

def polynomial_max(coeff):
    coeff=np.asarray(coeff,float);p=sum((np.convolve(coeff[:,k],coeff[:,k]) for k in range(3)),start=np.zeros(2*len(coeff)-1));roots=polyroots(polyder(p));u=[0.,1.]+[float(x.real) for x in roots if abs(x.imag)<1e-7 and 0<x.real<1]
    return float(np.sqrt(max(0,float(max(polyval(u,p))))))

def coefficients(r,w0,a0,w1,a1,duration):
    J=jacobian(r);v1=np.linalg.solve(J,w1);aa1=np.linalg.solve(J,a1-jacobian_direction(r,v1)@v1)
    return np.linalg.solve(MATRIX,np.stack([np.zeros(3),w0*duration,a0*duration**2,r,v1*duration,aa1*duration**2]))

def jet(c,u,duration):
    v=polyval(u,c);d=polyval(u,polyder(c))/duration;dd=polyval(u,polyder(c,2))/duration**2;J=jacobian(v)
    return J@d,J@dd+jacobian_direction(v,d)@d

def endpoint_jets(R,gap):
    a,b=gap;left=R[a-1];right=R[b];L=Rotation.from_matrix((left.T[None]@R[a-3:a]).reshape(-1,3,3)).as_rotvec();Q=Rotation.from_matrix((right.T[None]@R[b:b+3]).reshape(-1,3,3)).as_rotvec()
    return (3*L[2]-4*L[1]+L[0])/2,L[2]-2*L[1]+L[0],(-3*Q[0]+4*Q[1]-Q[2])/2,Q[2]-2*Q[1]+Q[0]

def segmented(R,gap=(104,152),counts=(4,8,16),radius=np.pi-.05):
    R=np.asarray(R,float);a,b=gap
    if R.ndim!=4 or R.shape[1:]!=(22,3,3) or not 3<=a<b<=len(R)-3:raise ValueError('Observed common22 and three frames per boundary')
    ids=[a-3,a-2,a-1,b,b+1,b+2]
    if not np.isfinite(R[ids]).all():raise ValueError('Missing observed boundary')
    left=R[a-1];right=R[b];delta=Rotation.from_matrix(left.swapaxes(-1,-2)@right).as_rotvec();duration=b-a+1;jets=[endpoint_jets(R[:,j],gap) for j in range(22)];trials=[]
    for count in counts:
        times=np.linspace(a-1,b,count+1);anchors=np.stack([left@Rotation.from_rotvec(delta*u).as_matrix() for u in np.linspace(0,1,count+1)]);segments=[];maximum=0.;jet_error=0.
        for k in range(count):
            dt=times[k+1]-times[k];r=Rotation.from_matrix(anchors[k].swapaxes(-1,-2)@anchors[k+1]).as_rotvec();cc=[]
            for j in range(22):
                w0,aa0,w1,aa1=jets[j];wl=w0 if k==0 else delta[j]/duration;al=aa0 if k==0 else np.zeros(3);wr=w1 if k==count-1 else delta[j]/duration;ar=aa1 if k==count-1 else np.zeros(3);c=coefficients(r[j],wl,al,wr,ar,dt);maximum=max(maximum,polynomial_max(c));q0,q1=jet(c,0,dt),jet(c,1,dt);jet_error=max(jet_error,*[float(np.max(abs(x-y))) for x,y in zip(q0+q1,(wl,al,wr,ar))]);cc.append(c)
            segments.append(np.stack(cc,axis=1))
        trials.append(dict(segments=count,maximum_chart_rad=maximum,jet_residual=jet_error,eligible=bool(maximum<radius and jet_error<1e-7)))
        if trials[-1]['eligible']:
            out=R.copy()
            for t in range(a,b):
                k=min(int(np.searchsorted(times,t,side='right')-1),count-1);u=(t-times[k])/(times[k+1]-times[k]);v=np.einsum('i,ijk->jk',u**np.arange(6),segments[k]);out[t]=anchors[k]@Rotation.from_rotvec(v).as_matrix()
            return out,dict(status='GEOMETRICALLY_ADMITTED_NOT_PHYSICS',selected_segments=count,local_chart_radius_rad=float(radius),trials=trials,hidden_read=False,endpoint_jets='OBSERVED_THREE_FRAME_BODY_RATE_AND_ACCELERATION_WITH_RIGHT_JACOBIAN',internal_anchor='SHORTEST_GEODESIC_NO_HIDDEN_POSE',physical_boundary_jet_residual=jet_error)
    return None,dict(status='REJECT_ALL_REGISTERED_SEGMENT_COUNTS',trials=trials,hidden_read=False)

def spline(R,gap=(104,152),count=4,radius=np.pi-.05):
    R=np.asarray(R,float);a,b=gap;left=R[a-1];right=R[b];r=Rotation.from_matrix(left.swapaxes(-1,-2)@right).as_rotvec();mid=np.linspace(a-1,b,count+1)[1:-1];times=np.r_[a-3,a-2,a-1,mid,b,b+1,b+2];out=R.copy();maximum=0.;errors=[]
    for j in range(22):
        poses=np.concatenate([R[a-3:a,j],left[j]@Rotation.from_rotvec(r[j]*(mid-(a-1))[:,None]/(b-a+1)).as_matrix(),R[b:b+3,j]]);s=RotationSpline(times,Rotation.from_matrix(poses));out[a:b,j]=s(np.arange(a,b)).as_matrix()
        for k,dt in enumerate(np.diff(times)):
            coeff=s.interpolator.c[:,k][::-1]*dt**np.arange(4)[:,None];maximum=max(maximum,polynomial_max(coeff))
        w0,aa0,w1,aa1=endpoint_jets(R[:,j],gap);errors.extend([float(np.max(abs(s(a-1,1)-w0))),float(np.max(abs(s(b,1)-w1))),float(np.max(abs(s(a-1,2)-aa0))),float(np.max(abs(s(b,2)-aa1)))])
    return (out if maximum<radius else None),dict(status='SPLINE_CHART_ELIGIBLE_NOT_BOUNDARY_JET_QUALIFIED' if maximum<radius else 'REJECT_SPLINE_CHART',maximum_chart_rad=maximum,observed_boundary_jet_residual=max(errors),interior='SCIPY_ROTATION_SPLINE_RATE_ACCEL_CONTINUOUS; NOT_RIEMANNIAN_ENERGY_OPTIMUM',hidden_read=False)
