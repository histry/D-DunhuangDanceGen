"""Event identities/normalization and deterministic schedules. All clocks are label metadata."""
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from dunhuang.io import sha
from dunhuang.research.complete_r36.core import read
from .events import clock_relations,validate_roles

class Population:
    def __init__(self,path):
        self.path=Path(path);self.manifest=read(self.path/'manifest.json');self.rows=self.manifest['rows'];validate_roles(self.rows);normal=self.manifest['normalizer']
        if sha(normal['path'])!=normal['sha256']:raise ValueError('Frozen Train normalization changed')
        self.statistics=read(normal['path'])['statistics'];self.cache={}
    def unit(self,i):
        if i not in self.cache:
            r=self.rows[i];p=Path(r['path'])
            if sha(p)!=r['sha256']:raise ValueError('Event source feature changed')
            z=np.load(p);v={}
            for key in ('music','motion'):
                mask=z[key+'_mask'];n=self.statistics[key];x=(z[key]-np.array(n['mean']))/np.array(n['std']);v[key]=np.where(mask,x,0).astype(np.float32);v[key+'_mask']=mask.copy()
            self.cache[i]=v
        return self.cache[i]
    def batch(self,ids,origins=None,control='actual',donors=None,device='cuda'):
        n=max(self.rows[i]['frames'] for i in ids) if origins is None else 90
        M=np.zeros((len(ids),n,50),np.float32);D=np.zeros((len(ids),n,274),np.float32);MM=np.zeros_like(M,bool);DM=np.zeros_like(D,bool)
        for b,i in enumerate(ids):
            r=self.rows[i];z=self.unit(i);a,d=r['core_slice'][0],r['core_slice'][0]
            length=r['frames']
            if origins is not None:a,d=[int(t)+1 for t in origins[b]];length=90
            other=self.unit(donors[i]) if control=='music_swap' else z
            if max(a,d)+length>len(z['music']) or a+length>len(other['music']):raise ValueError('No padding or circular crop permitted')
            M[b,:length]=other['music'][a:a+length];MM[b,:length]=other['music_mask'][a:a+length];D[b,:length]=z['motion'][d:d+length];DM[b,:length]=z['motion_mask'][d:d+length]
            if origins is not None:
                common=z['music_mask'][1:121].all(0)
                if donors is not None:common &= self.unit(donors[i])['music_mask'][1:121].all(0)
                dm=z['motion_mask'][1:121].all(0)
                if not common.any() or not dm[:-2].any():raise ValueError('No shared real observation support')
                M[b,:,~common]=0;D[b,:,~dm]=0;MM[b]=common;DM[b]=dm
            if control in ('music_zero','both_zero'):M[b]=0
            if control=='both_zero':D[b]=0
        return [torch.tensor(v,device=device) for v in (M,D,DM,MM.any(-1))]
    def relations(self,ids):return clock_relations([self.rows[i] for i in ids])
    def schedule(self,steps,seed,ids=None):
        pool=[i for i,r in enumerate(self.rows) if r['role']=='train' and r['context_frames']>=121] if ids is None else ids
        if any(self.rows[i]['role']!='train' for i in pool):raise ValueError('Train-only schedule')
        groups=defaultdict(list)
        for i in pool:groups[self.rows[i]['audio_sha256']].append(i)
        audios=sorted(groups)
        if len(audios)<2:raise ValueError('Insufficient physical audio groups')
        rng=np.random.default_rng(seed);order=[]
        for step in range(steps):
            batch=[]
            for audio in rng.choice(audios,min(4,len(audios)),replace=False):
                choices=groups[audio];i=int(rng.choice(choices));batch.append(i)
                A=self.rows[i]['audio_frame_interval'];different=[j for j in choices if j!=i];hard=[j for j in different if max(A[0],self.rows[j]['audio_frame_interval'][0])-min(A[1],self.rows[j]['audio_frame_interval'][1])>=15]
                if hard:j=min(hard,key=lambda j:(abs(self.rows[j]['motion_activity']-self.rows[i]['motion_activity']),self.rows[j]['event_uid']))
                elif different:j=int(rng.choice(different))
                else:j=i
                batch.append(j)
            order.append(batch)
        return order
    def fit_population(self):
        groups=defaultdict(list)
        for i,r in enumerate(self.rows):
            if r['role']=='train' and r['context_frames']>=121:groups[r['audio_sha256']].append(i)
        result=[]
        for key in sorted(groups)[:4]:
            options=sorted(groups[key],key=lambda i:self.rows[i]['event_uid']);used=set()
            for i in options:
                uid=self.rows[i]['raw_motion_sha256']
                if uid not in used:result.append(i);used.add(uid)
                if len(used)==2:break
        if len(result)<4:raise ValueError('Genuine fixed Train fit population unavailable')
        return result
