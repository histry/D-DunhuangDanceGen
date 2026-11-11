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
        self.decision=authorize(checked(assets['global']),checked(assets['qualification']))
        self.global_model=RetrievalModel().eval()
        payload=torch.load(checked(assets['global']),map_location='cpu',weights_only=True)
        self.global_model.load_state_dict(payload['state']);self.norm=payload['norm']
        self.sync_model=RetrievalModel().eval()
        local=torch.load(checked(assets['sync']),map_location='cpu',weights_only=True)
        if local.get('global_checkpoint_sha256')!=assets['global']['sha256']:
            raise ValueError('Sync model does not bind selected global checkpoint')
        self.sync_model.load_state_dict(local['state'])
        receipt=json.loads(checked(assets['music_receipt']).read_text())
        if receipt['source_sha256']!=audio_sha256 or receipt['features_sha256']!=assets['music_features']['sha256']:
            raise ValueError('Alignment query audio differs')
        self.query=np.load(checked(assets['music_features']),allow_pickle=False)
    def scores(self,event,slot):
        a,b=slot['start'],slot['end'];n=b-a
        if b>len(self.query['music']):raise ValueError('Alignment query clock does not cover slot')
        music=self.query['music'][a:b];mm=self.query['mask'][a:b]
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
        global_score,sync_score=synchronized_scores(self.global_model,self.sync_model,args[0],args[1],args[2],args[3])
        return dict(global_score=float(global_score[0]),sync_score=float(sync_score[0]),preference=None)
