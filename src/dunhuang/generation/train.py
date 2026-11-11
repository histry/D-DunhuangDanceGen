import argparse,json,sys
from pathlib import Path
import numpy as np
from dunhuang.io import checked,pinned,write,sha
from dunhuang.contracts import MotionAdapter
from .adapter import PrimitiveAdapter,masked_reference

def train(assets,destination,steps,task='motion'):
    import torch
    if type(steps) is not int or steps < 1:raise ValueError('Positive training steps required')
    if task not in ('motion','paired'):raise ValueError('Unknown training task')
    destination.mkdir(parents=True,exist_ok=False)
    catalog=json.loads(checked(assets['catalog']).read_text())
    scaler=np.load(checked(assets['scaler']))
    refs={r['source_uid']:r['reference'] for r in catalog['entries']}
    windows=[]
    tested=0;support_counts={'body8':0,'wrists':0,'root':0}
    for uid,ref in refs.items():
        source=np.load(checked(ref),allow_pickle=False)
        for start in range(0,len(source)-149,75):
            clip=source[start:start+150].copy()
            origin=clip[0,4:7].copy();origin[1]=0.
            edge=MotionAdapter.to_edge(clip,origin)
            normalized=edge*scaler['scale']+scaler['bias']
            valid=abs(normalized)<=1+32*np.finfo(np.float32).eps
            tested+=1
            for name,(a,b) in {'body8':(79,127),'wrists':(127,139),'root':(4,7)}.items():
                support_counts[name]+=int(valid[:,a:b].all())
            if np.all(valid):
                windows.append((uid,start,normalized.astype(np.float32)))
    motion_window_count=len(windows)
    paired_roles={}
    if task=='paired':
        if not assets.get('paired_generation_manifest'):
            write(destination/'status.json',dict(status='MISSING_REAL_PAIRED_JUKEBOX',training_steps=0))
            return
        paired=json.loads(checked(assets['paired_generation_manifest']).read_text())
        windows=[]
        for row in paired['rows']:
            role=row['role']
            if role not in ('train','val','test'):raise ValueError('Explicit paired role required')
            group=row['source_group']
            if group in paired_roles and paired_roles[group]!=role:raise ValueError('Paired source group leakage')
            paired_roles[group]=role
            if role=='test':continue
            if row['pairing_evidence']!='real_aligned_audio_motion' or row['fps']!=30 or row['frames']!=150:
                raise ValueError('Real paired generation clock/evidence required')
            native=np.load(checked(row['motion']),allow_pickle=False)
            music=np.load(checked(row['jukebox']),allow_pickle=False)
            if native.shape!=(150,151) or music.shape!=(150,4800) or not np.isfinite(music).all() or not music.any():
                raise ValueError('Actual 151D motion and Jukebox required')
            origin=native[0,4:7].copy();origin[1]=0.
            normalized=MotionAdapter.to_edge(native.copy(),origin)*scaler['scale']+scaler['bias']
            if np.any(abs(normalized)>1+32*np.finfo(np.float32).eps):raise ValueError('Paired target outside frozen support')
            windows.append((row['source_group'],row['start_frame'],normalized.astype(np.float32),music.astype(np.float32)))
    groups=sorted({row[0] for row in windows})
    audit=dict(tested_150_windows=tested,component_supported_windows=support_counts,root_origin='first_observed_context_XZ; height preserved',motion_only_supported_150_windows=motion_window_count,paired_training_rows=len(windows) if task=='paired' else 0,recording_groups=groups,
        source_count=len(refs),paired_jukebox_training_manifest=assets.get('paired_generation_manifest'),
        full_paired_music_training_ready=bool(assets.get('paired_generation_manifest')),
        checkpoint=assets['checkpoint'],scaler=assets['scaler'],
        scope='All 151 target channels in frozen support; no clipping or invented Jukebox features')
    write(destination/'asset_audit.json',audit)
    if len(groups)<2:
        write(destination/'status.json',dict(status='INSUFFICIENT_SUPPORTED_SOURCE_GROUPS',training_steps=0,music_qualified=False))
        return
    validation_group=groups[-1] if task=='motion' else sorted(k for k,v in paired_roles.items() if v=='val')
    fitting=[row for row in windows if (row[0]!=validation_group if task=='motion' else paired_roles[row[0]]=='train')]
    validation=[row for row in windows if (row[0]==validation_group if task=='motion' else paired_roles[row[0]]=='val')]
    if not fitting or not validation:raise ValueError('Separate fitting and validation sources required')
    root=pinned(assets['edge_root'],assets['edge_commit']);sys.path.insert(0,str(root))
    from dataset.preprocess import Normalizer
    from dataset.scaler import MinMaxScaler
    from model.model import DanceDecoder
    from model.diffusion import GaussianDiffusion
    from vis import SMPLSkeleton
    # Exact asset has already been inspected by generation.assets in prepare.
    file=checked(assets['checkpoint'])
    payload=torch.load(file,map_location='cpu',weights_only=False)
    model=DanceDecoder(nfeats=151,seq_len=150,latent_dim=512,ff_size=1024,num_layers=8,num_heads=8,dropout=.1,cond_feature_dim=4800,activation=torch.nn.functional.gelu).cuda().eval()
    model.load_state_dict(payload['ema_state_dict'],strict=True);del payload
    for parameter in model.parameters():parameter.requires_grad_(False)
    diffusion=GaussianDiffusion(model,150,151,SMPLSkeleton('cuda'),schedule='cosine',n_timestep=1000,predict_epsilon=False,loss_type='l2',use_p2=False,cond_drop_prob=.25,guidance_weight=2).cuda().eval()
    torch.manual_seed(20261111);torch.set_num_threads(4)
    adapter=PrimitiveAdapter().cuda()
    state={}
    def condition(module,args,value):return value+adapter(state['value'],state['known'])
    hook=model.cond_encoder.register_forward_hook(condition)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=.0001)
    rng=np.random.default_rng(20261111)
    history=[];best=float('inf')
    def loss(target,music):
        value,known=masked_reference(target)
        state.update(value=value,known=known)
        t=torch.randint(50,950,(len(target),),device='cuda')
        noisy=diffusion.q_sample(target,t)
        predicted=model(noisy,music,t,cond_drop_prob=0.)
        return ((predicted-target)**2)[~known].mean()
    for step in range(1,steps+1):
        ids=rng.integers(0,len(fitting),2)
        target=torch.from_numpy(np.stack([fitting[i][2] for i in ids])).cuda()
        music=torch.from_numpy(np.stack([fitting[i][3] for i in ids])).cuda() if task=='paired' else torch.zeros(len(target),150,4800,device='cuda')
        adapter.train();objective=loss(target,music)
        optimizer.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);optimizer.step()
        if step%50==0 or step==steps:
            adapter.eval()
            with torch.no_grad():
                target=torch.from_numpy(np.stack([row[2] for row in validation[:8]])).cuda()
                music=torch.from_numpy(np.stack([row[3] for row in validation[:8]])).cuda() if task=='paired' else torch.zeros(len(target),150,4800,device='cuda')
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(117)
                    error=float(loss(target,music))
            history.append(dict(step=step,training_loss=float(objective.detach()),validation_loss=error))
            write(destination/'history.json',history)
            if error<best:
                best=error;torch.save(dict(state=adapter.state_dict(),base_sha256=assets['checkpoint']['sha256'],
                    condition_scope='paired_music_completion' if task=='paired' else 'motion_only_masked_completion',music_qualified=False),destination/'adapter.pt')
    hook.remove()
    if any(p.grad is not None for p in model.parameters()):raise ValueError('Frozen model got gradients')
    write(destination/'status.json',dict(status='PAIRED_COMPLETION_TRAINED_UNQUALIFIED' if task=='paired' else 'MOTION_ONLY_TRAINED',training_steps=steps,music_qualified=False,
        validation_source=validation_group,checkpoint_sha256=sha(destination/'adapter.pt')))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=300);ap.add_argument('--task',choices=['motion','paired'],default='motion')
    args=ap.parse_args();train(json.loads(Path(args.assets).read_text()),Path(args.out).resolve(),args.steps,args.task)
