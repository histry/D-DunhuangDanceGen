"""Native-Y-up observed boundary state, expressed in the left Root frame.

All features are deterministic functions of the same 8+8 observations.
They are an inductive bias, not independent information or semantic labels.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices,sixd

FRAMES=np.r_[52:60,90:98]
FAMILIES=("A0","A1","A2","A3","A4")
RAW_DIM=16*147+3
FEATURE_DIM=RAW_DIM*2+48

def extract(native,known=None):
    x=np.asarray(native)
    if x.shape!=(150,151) or x.dtype!=np.float32:raise ValueError("FP32 Native [150,151] required")
    if known is None:
        known=np.zeros_like(x,dtype=bool);known[FRAMES,4:]=True
    known=np.asarray(known)
    if known.shape!=x.shape or known.dtype!=bool :raise ValueError("Boolean observation mask required")
    # Restrict even a caller's broader mask to the actual supplied context.
    k=np.zeros_like(known);k[FRAMES,4:]=known[FRAMES,4:]
    if not np.isfinite(x[k]).all():raise ValueError("Finite actual observations required")
    posok=k[:,4:7].all(-1);rotok=k[:,7:].reshape(150,24,6).all(-1)
    ident=np.array([1,0,0,0,1,0],np.float32)
    safe6=np.where(rotok[:,:,None],x[:,7:].reshape(150,24,6),ident)
    R=matrices(safe6)
    p=np.where(k[:,4:7],x[:,4:7],0.).astype(float)
    basis=R[59,0].astype(float) if rotok[59,0] else np.eye(3)
    anchor=p[59] if posok[59] else np.zeros(3)
    raw=np.zeros((16,147),float);rm=np.zeros_like(raw,dtype=bool)
    for i,t in enumerate(FRAMES):
        if posok[t] and posok[59] and rotok[59,0]:
            raw[i,:3]=(p[t]-anchor)@basis;rm[i,:3]=True
        for j in range(24):
            if rotok[t,j] and (j!=0 or rotok[59,0]):
                r=basis.T@R[t,0] if j==0 else R[t,j]
                raw[i,3+j*6:9+j*6]=sixd(r);rm[i,3+j*6:9+j*6]=True
    state=np.zeros(24,float);valid=np.zeros(24,bool)
    def put(a,values,ok):
        if ok:state[a:a+3]=values;valid[a:a+3]=True
    put(0,(p[90]-p[59])@basis,posok[59] and posok[90] and rotok[59,0])
    put(3,(p[59]-p[58])@basis*30,posok[58:60].all() and rotok[59,0])
    put(6,(p[91]-p[90])@basis*30,posok[90:92].all() and rotok[59,0])
    put(9,(p[59]-2*p[58]+p[57])@basis*900,posok[57:60].all() and rotok[59,0])
    put(12,(p[92]-2*p[91]+p[90])@basis*900,posok[90:93].all() and rotok[59,0])
    put(15,Rotation.from_matrix(basis.T@R[90,0]).as_rotvec(),rotok[59,0] and rotok[90,0])
    put(18,Rotation.from_matrix(R[59,0]@R[58,0].T).as_rotvec()@basis*30,rotok[58:60,0].all())
    put(21,Rotation.from_matrix(R[91,0]@R[90,0].T).as_rotvec()@basis*30,rotok[59,0] and rotok[90:92,0].all())
    # Full left-frame canonicalization retains world-up explicitly, so tilt/gravity
    # information is not discarded by the raw-observation control.
    gravity=np.array([0.,1.,0.])@basis
    return np.r_[raw.ravel(),gravity if rotok[59,0] else np.zeros(3)],np.r_[rm.ravel(),[rotok[59,0]]*3],state,valid

def design(raw,mask,state,valid,family,donor=None):
    if family not in FAMILIES:raise ValueError("Unknown boundary-state family")
    s=np.array(state,copy=True);v=np.array(valid,copy=True)
    if family=="A4":
        if donor is None:raise ValueError("Shuffled control needs an independent same-role donor")
        s,v=map(np.copy,donor)
    active={"A0":0,"A1":3,"A2":15,"A3":24,"A4":24}[family]
    s[active:]=0;v[active:]=False
    return np.concatenate((np.where(mask,raw,0.),mask.astype(float),np.where(v,s,0.),v.astype(float)))

def label(native):
    # Only fitting/scoring calls this. Hidden motion NEVER enters extract/design.
    p=native[:,4:7].astype(float);R=matrices(native[:,7:13]);left=R[59].astype(float)
    u=np.arange(1,31)/31
    root=(p[60:90]-p[59])@left-u[:,None]*((p[90]-p[59])@left)
    turn=Rotation.from_matrix(left.T@R[90]).as_rotvec()
    baseline=Rotation.from_rotvec(u[:,None]*turn).as_matrix()
    relative=left.T[None]@R[60:90]
    return np.concatenate((root.ravel()/.1,(sixd(relative)-sixd(baseline)).ravel()/.3))

def reference(native,prediction,known=None):
    raw,rm,s,v=extract(native,known)
    if not v[[0,1,2,15,16,17]].all():raise ValueError("Root reference requires both observed Root endpoints and orientations")
    left=matrices(native[59,7:13]);u=np.arange(1,31)/31;pred=np.asarray(prediction)
    if pred.shape!=(270,) or not np.isfinite(pred).all():raise ValueError("Finite Root/rotation residual readout required")
    positions=native[59,4:7]+(u[:,None]*s[:3]+pred[:90].reshape(30,3)*.1)@left.T
    baseline=Rotation.from_rotvec(u[:,None]*s[15:18]).as_matrix()
    rotations=left[None]@matrices(sixd(baseline)+pred[90:].reshape(30,6)*.3)
    return positions,rotations
