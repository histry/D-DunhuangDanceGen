"""CPU geometry contracts. No semantic truth, learned routing or gap interpolation."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices

def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def components(intervals):
    out=[]
    for a,b in sorted(intervals):
        if not 0<=a<b:raise ValueError('Nonempty source half-open interval required')
        if out and a<out[-1][1]:raise ValueError('Frozen atomic intervals overlap')
        if out and a==out[-1][1]:out[-1][1]=b
        else:out.append([a,b])
    return out

def rotation_sample(r,positions):
    q=np.asarray(positions,float)
    if q.ndim!=1 or not np.isfinite(q).all() or q.min()<0 or q.max()>len(r)-1:raise ValueError('Cannot invent source observations')
    lo=np.floor(q).astype(int);hi=np.minimum(lo+1,len(r)-1);t=(q-lo)[:,None,None]
    rel=r[lo].swapaxes(-1,-2)@r[hi]
    v=Rotation.from_matrix(rel.reshape(-1,3,3)).as_rotvec().reshape(len(q),22,3)
    return r[lo]@Rotation.from_rotvec((v*t).reshape(-1,3)).as_matrix().reshape(-1,22,3,3)

def relative_fk(r,J,parents):
    pos=[np.zeros((len(r),3))];world=[r[:,0]]
    for j in range(1,22):
        k=int(parents[j]);pos.append(pos[k]+(world[k]@(J[j]-J[k])[None,:,None])[...,0]);world.append(world[k]@r[:,j])
    return np.stack(pos,1)

def multi_warp(interval,music_interval,phases,beats):
    """Ordered multi-anchor DP. delta_music/delta_source always in [0.75,1.33]."""
    a,end=map(int,interval);A,finish=map(int,music_interval);b=end-1;B=finish-1
    if not a<b or not A<B:raise ValueError('At least two actual frames required')
    if not .75-1e-10<=(B-A)/(b-a)<=1.33+1e-10:return None
    # Keep source phase locations; do not relocate a peak to create room.
    selected=[]
    priority={'upper_speed_peak':0,'upper_direction_turn':1,'upper_motion_onset':2,'upper_pose_hold':3}
    for p in sorted(phases,key=lambda p:(priority.get(p['type'],9),p['frame'])):
        x=int(p['frame'])
        if a+12<x<b-12 and all(abs(x-y)>18 for y in selected):selected.append(x)
        if len(selected)==4:break
    source=[a]+sorted(selected)+[b];states={float(A):(0.,[])}
    for k,x in enumerate(source[1:],1):
        linear=A+(x-a)*(B-A)/(b-a)
        if k==len(source)-1:options=[float(B)]
        else:options=sorted(set([float(linear)]+[float(t) for t in beats if A<t<B and abs(t-linear)<=15]))
        current={}
        for y in options:
            choices=[]
            for previous,(cost,path) in states.items():
                rate=(y-previous)/(x-source[k-1])
                if .75-1e-10<=rate<=1.33+1e-10:
                    attraction=0. if k==len(source)-1 else min([abs(y-t) for t in beats if A<t<B] or [0.])/30
                    choices.append((cost+attraction+.03*((y-linear)/30)**2,path+[[x,y]]))
            if choices:current[y]=min(choices,key=lambda z:(z[0],z[1]))
        states=current
    if not states:return None
    cost,path=states[float(B)];warp=[[a,float(A)]]+path
    delta=np.diff(np.array(warp),axis=0);assert (delta>0).all() and (delta[:,1]>=.75*delta[:,0]-1e-8).all() and (delta[:,1]<=1.33*delta[:,0]+1e-8).all()
    return dict(time_warp=warp,anchor_source_frames=source[1:-1],cost=float(cost),rate_limits=[.75,1.33],semantic_truth='UNKNOWN',clock_contract='ACTUAL_FRAME_CENTERS_WITH_HALF_OPEN_SUPPORT')

def smooth_budget(n,edge=24,core=.2,boundary=.6):
    t=np.minimum(np.arange(n),np.arange(n)[::-1])/max(edge,1)
    u=np.clip(t,0,1);u=u*u*(3-2*u)
    return boundary+(core-boundary)*u

def bounded_rotations(reference,proposal,budget):
    """Geodesic residual, not 6D addition. Zero budget exactly preserves reference."""
    if reference.shape!=proposal.shape or reference.shape[-2:]!=(3,3):raise ValueError('Matching rotation matrices required')
    d=reference.swapaxes(-1,-2)@proposal
    rv=Rotation.from_matrix(d.reshape(-1,3,3)).as_rotvec().reshape(reference.shape[:-2]+(3,))
    length=np.linalg.norm(rv,axis=-1);limit=np.broadcast_to(np.asarray(budget),length.shape)
    if not np.isfinite(limit).all() or (limit<0).any():raise ValueError('Finite nonnegative edit budget required')
    rv*=np.minimum(1,limit/np.maximum(length,1e-12))[...,None]
    return reference@Rotation.from_rotvec(rv.reshape(-1,3)).as_matrix().reshape(reference.shape)

class Library:
    def __init__(self,catalog):
        self.path=Path(catalog);self.catalog=read(catalog);self.native={};self.rotations={};self.rows={}
        for e in self.catalog['entries']:self.rows.setdefault(e['source_uid'],[]).append(e)
        for uid,rows in self.rows.items():
            p=Path(rows[0]['reference']['path'])
            if not p.exists():p=self.path.parent/p.name
            if sha(p)!=rows[0]['reference']['sha256']:raise ValueError('Native source hash mismatch')
            x=np.load(p,allow_pickle=False);self.native[uid]=x
            self.rotations[uid]=matrices(x[:,7:].reshape(-1,24,6))[:,:22].astype(float)
    def sample(self,e,times):
        w=np.array(e['time_warp']);q=np.interp(times,w[:,1],w[:,0]);a,b=e['source_interval']
        if (q<a).any() or (q>b-1+1e-8).any():raise ValueError('Time map escapes actual event frame centers')
        return rotation_sample(self.rotations[e['source_uid']],q)
