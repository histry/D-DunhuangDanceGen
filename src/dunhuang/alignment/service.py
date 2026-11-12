"""Bound global and sync ranking on explicit music/event feature assets."""
import json
import numpy as np
import torch
from .model import RetrievalModel
from .rank import synchronized_scores,authorize
from dunhuang.io import checked
from dunhuang.planning.time_map import source_positions,motion_features74,SCHEMA
from dunhuang.generation.soft_reference import resample_native

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
        qualification=json.loads(checked(assets['qualification']).read_text())
        if self.decision['enabled'] and qualification.get('execution_time_map_schema')!=SCHEMA:
            self.decision.update(enabled=False,fallback_reason='missing_execution_time_map_qualification')
        self.decision['execution_time_map_schema']=SCHEMA
        self.identity=dict(global_sha256=assets['global']['sha256'],sync_sha256=assets['sync']['sha256'],
            qualification_sha256=assets['qualification']['sha256'],time_map_schema=SCHEMA)
        self.transport=None
        if assets.get('transport'):
            from .transport import ClockNullTransport,SCHEMA as TRANSPORT_SCHEMA
            saved=torch.load(checked(assets['transport']),map_location='cpu',weights_only=True)
            if (saved.get('schema')!=TRANSPORT_SCHEMA or saved.get('mode') not in ('B','C')
                or saved.get('global_checkpoint_sha256')!=assets['global']['sha256']
                or saved.get('norm')!=self.norm or saved.get('execution_time_map_schema')!=SCHEMA
                or saved.get('operational_qualified') is not False):
                raise ValueError('Shadow transport identity, clock or qualification differs')
            cfg=saved['config']
            if receipt.get('extractor',{}).get('librosa_version')!=cfg.get('librosa_version'):
                raise ValueError('Query audio extractor version differs from transport training')
            self.transport=ClockNullTransport(band_seconds=cfg['band_seconds'],epsilon=cfg['epsilon'],iterations=cfg['sinkhorn_iterations']).eval()
            self.transport.load_state_dict(saved['state'],strict=True)
            self.identity['transport_sha256']=assets['transport']['sha256']
            self.decision.update(enabled=False,fallback_reason='clock_transport_shadow_unqualified')
        self.sources={}
        self.query=np.load(checked(assets['music_features']),allow_pickle=False)
    def scores(self,event,slot,local=True):
        a,b=slot['start'],slot['end'];n=b-a
        missing=max(0,b-len(self.query['music']))
        if missing>1 or a>=len(self.query['music']):raise ValueError('Alignment query clock does not cover slot')
        music=self.query['music'][a:b];mm=self.query['mask'][a:b]
        if missing:
            music=np.pad(music,((0,missing),(0,0)))
            mm=np.pad(mm,((0,missing),(0,0)),constant_values=False)
        key=event['reference']['sha256']
        if key not in self.sources:
            self.sources[key]=np.load(checked(event['reference']),allow_pickle=False)
        positions=source_positions(event,np.arange(a,b))
        mapped=resample_native(self.sources[key],positions)
        motion,mask=motion_features74(mapped)
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
            # Same exact mapped trajectory as generation, shifted only on its
            # real music clock; no circular roll or alternative uniform map.
            v=motion[15+offset:n-15+offset]
            valid=mask[15+offset:n-15+offset]
            v=(v-np.array(self.norm['motion']['mean']))/np.array(self.norm['motion']['std'])*valid
            _,score=synchronized_scores(self.global_model,self.sync_model,trimmed_music,trimmed_mask,
                torch.as_tensor(v).float()[None],torch.as_tensor(valid)[None])
            raw_scores.append(score[0])
        probability=float(torch.softmax(torch.stack(raw_scores),0)[2])
        with torch.no_grad():
            m=self.global_model.music(args[0],args[1])[0]
            d=self.global_model.motion(args[2],args[3])[0]
        transport={}
        if self.transport is not None:
            with torch.no_grad():
                _,mt,mv=self.global_model.music(trimmed_music,trimmed_mask)
                _,dt,dv=self.global_model.motion(args[2][:,15:-15],args[3][:,15:-15])
                clock=torch.arange(mt.shape[1])/15
                value=self.transport(mt,dt,mv,dv,clock,clock)
            transport=dict(transport_score=float(value['score'][0]),transport_coverage=float(value['coverage'][0]),
                transport_music_null_fraction=float(value['music_null_fraction'][0]),
                transport_motion_null_fraction=float(value['motion_null_fraction'][0]),
                transport_monotonic_violations=int(value['monotonic_violations'][0]),
                transport_marginal_residual=float(value['marginal_residual'][0]),
                transport_mode='SHADOW_UNQUALIFIED',transport_can_route=False)
        return dict(transport,global_score=float((m*d).sum()),sync_score=probability,
            sync_raw_logits=[float(s) for s in raw_scores],sync_reference='relative zero-offset probability on exact Event Plan time warp',execution_time_map_schema=SCHEMA,preference=None)
