"""Bound global and sync ranking on explicit music/event feature assets."""
import json
import numpy as np
import torch
from .model import RetrievalModel
from .rank import synchronized_scores,authorize
from dunhuang.io import checked

class Reranker:
    def __init__(self,assets,audio_sha256):
        torch.set_num_threads(4)
        self.decision=authorize(checked(assets['global']),checked(assets['qualification']),checked(assets['sync']))
        self.global_model=RetrievalModel().eval()
        payload=torch.load(checked(assets['global']),map_location='cpu',weights_only=True)
        self.global_model.load_state_dict(payload['state']);self.norm=payload['norm']
        self.sync_model=RetrievalModel().eval()
        local=torch.load(checked(assets['sync']),map_location='cpu',weights_only=True)
        if local.get('global_checkpoint_sha256')!=assets['global']['sha256']:
            raise ValueError('Sync model does not bind selected global checkpoint')
        if local.get('norm')!=self.norm:raise ValueError('Global and sync normalization differs')
        self.sync_model.load_state_dict(local['state'])
        receipt=json.loads(checked(assets['music_receipt']).read_text())
        if receipt['source_sha256']!=audio_sha256 or receipt['features_sha256']!=assets['music_features']['sha256']:
            raise ValueError('Alignment query audio differs')
        self.query=np.load(checked(assets['music_features']),allow_pickle=False)
    def scores(self,event,slot,local=True):
        a,b=slot['start'],slot['end'];n=b-a
        missing=max(0,b-len(self.query['music']))
        if missing>1 or a>=len(self.query['music']):raise ValueError('Alignment query clock does not cover slot')
        music=self.query['music'][a:b];mm=self.query['mask'][a:b]
        if missing:
            music=np.pad(music,((0,missing),(0,0)))
            mm=np.pad(mm,((0,missing),(0,0)),constant_values=False)
        with np.load(checked(event['motion_features'])) as data:
            source=data['motion'];source_mask=data['mask']
        # Explicit uniform phase alignment for retrieval scores, not an output-motion edit.
        times=np.linspace(0,len(source)-1,n)
        motion=np.column_stack([np.interp(times,np.arange(len(source)),source[:,j]) for j in range(74)])
        mask=source_mask[np.floor(times).astype(int)]&source_mask[np.ceil(times).astype(int)]
        rate=n/len(source)
        motion[:,:66]/=rate
        motion[:,[66,67]]/=rate
        motion[:,[70,72,73]]/=rate**2
        args=[]
        for name,x,m in [('music',music,mm),('motion',motion,mask)]:
            x=(x-np.array(self.norm[name]['mean']))/np.array(self.norm[name]['std'])*m
            args.extend([torch.as_tensor(x).float()[None],torch.as_tensor(m)[None]])
        if not local:
            with torch.no_grad():
                m=self.global_model.music(args[0],args[1])[0]
                d=self.global_model.motion(args[2],args[3])[0]
            return dict(global_score=float((m*d).sum()),sync_score=None,preference=None)
        # Local confidence is relative to actual noncyclic phase alternatives.
        # Raw logits have an arbitrary additive offset and cannot be mixed with cosine.
        if n<=32:raise ValueError('Too little common context for synchronization controls')
        raw_scores=[]
        trimmed_music=args[0][:,15:-15];trimmed_mask=args[1][:,15:-15]
        for offset in (-15,-8,0,8,15):
            positions=(np.arange(15,n-15)+offset)*(len(source)-1)/(n-1)
            if positions.min()<0 or positions.max()>len(source)-1:raise ValueError('Missing real shift support')
            v=np.column_stack([np.interp(positions,np.arange(len(source)),source[:,j]) for j in range(74)])
            valid=source_mask[np.floor(positions).astype(int)]&source_mask[np.ceil(positions).astype(int)]
            v[:,:66]/=rate;v[:,[66,67]]/=rate;v[:,[70,72,73]]/=rate**2
            v=(v-np.array(self.norm['motion']['mean']))/np.array(self.norm['motion']['std'])*valid
            _,score=synchronized_scores(self.global_model,self.sync_model,trimmed_music,trimmed_mask,
                torch.as_tensor(v).float()[None],torch.as_tensor(valid)[None])
            raw_scores.append(score[0])
        probability=float(torch.softmax(torch.stack(raw_scores),0)[2])
        with torch.no_grad():
            m=self.global_model.music(args[0],args[1])[0]
            d=self.global_model.motion(args[2],args[3])[0]
        return dict(global_score=float((m*d).sum()),sync_score=probability,
            sync_raw_logits=[float(s) for s in raw_scores],sync_reference='relative zero-offset probability on common true support',preference=None)
