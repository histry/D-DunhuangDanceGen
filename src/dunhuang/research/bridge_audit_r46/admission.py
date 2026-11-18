"""Observed-only continuous chart-domain guard; a local-coordinate gate, not physics."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from numpy.polynomial.polynomial import polyval,polyder,polyroots
from dunhuang.research.bridge_r38.geometry import MATRIX
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.reperformance_r44.archive import Arrays

def certificate(R,gap):
    R=np.asarray(R);a,b=gap;ids=[a-3,a-2,a-1,b,b+1,b+2];anchor=R[a-1];v=Rotation.from_matrix((anchor[None].swapaxes(-1,-2)@R[ids]).reshape(-1,3,3)).as_rotvec().reshape(6,22,3);T=b-a+1
    constraints=np.stack([v[2],(3*v[2]-4*v[1]+v[0])/2*T,(v[2]-2*v[1]+v[0])*T*T,v[3],(-3*v[3]+4*v[4]-v[5])/2*T,(v[5]-2*v[4]+v[3])*T*T]);coeff=np.linalg.solve(MATRIX,constraints.reshape(6,-1)).reshape(6,22,3);maxima=[]
    for j in range(22):
        squared=sum((np.convolve(coeff[:,j,k],coeff[:,j,k]) for k in range(3)),start=np.zeros(11));roots=polyroots(polyder(squared));points=[0.,1.]+[float(r.real) for r in roots if abs(r.imag)<1e-7 and 0<r.real<1];maxima.append(float(np.sqrt(max(0,max(polyval(points,squared))))))
    limit=float(np.pi-.05);maximum=max(maxima);return dict(continuous_chart_maximum_rad=maximum,joint_maximum_rad=maxima,chart_radius_rad=limit,chart_eligible=maximum<limit,decision='LOCAL_CHART_ONLY_ELIGIBLE_NOT_NATIVE' if maximum<limit else 'REJECT_QUINTIC_LOCAL_CHART_EXCURSION',hidden_read=False,semantic_or_physical_qualification=False,policy_basis='EXTEND_EXISTING_R38_ENDPOINT_PI_MINUS_0_05_CHART_ASSUMPTION_TO_WHOLE_POLYNOMIAL; NOT_GLOBAL_MOTION_INFEASIBILITY')

def run(main,out):
    main=Path(main);z=Arrays(main/'arrays.npz');rows=[]
    for r in read(main/'result.json')['records']:rows.append(dict(key=r['key'],source_domain=r['source']['domain'],certificate=certificate(z[r['key']+'_reference_R'],[104,152])))
    write(out,dict(records=rows,input_arrays_sha256=sha(main/'arrays.npz'),default_route_changed=False,optimizer_steps=0));z.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.main,a.out)
