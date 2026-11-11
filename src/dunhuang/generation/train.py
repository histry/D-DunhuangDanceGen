"""Frozen-EDGE completion probes; partial targets never imply 151D qualification."""
import argparse,json,sys,hashlib,subprocess,traceback
from pathlib import Path
import numpy as np
from dunhuang.io import checked,pinned,write,sha
from dunhuang.contracts import MotionAdapter
from .probe import upperbody_probe_target
from .split import split_training_rows
from .soft_reference import resample_native,SCHEMA

SEED=20261111

def phase_target(native,scale,bias,knot=75,partial=False):
    """The same geodesic source-time operator as inference, never linear Rot6D."""
    positions=np.interp(np.arange(150),[0,75,150],[0,knot,150])
    sampled=resample_native(native,np.minimum(positions,149))
    origin=native[0,4:7].copy();origin[1]=0.
    normalized=MotionAdapter.to_edge(sampled,origin)*scale+bias
    valid=np.isfinite(normalized)&(abs(normalized)<=1+32*np.finfo(np.float32).eps)
    target=upperbody_probe_target(normalized,valid) if partial else (normalized if valid.all() else None)
    return target, (positions/150).astype(np.float32)[:,None]

def digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def _train(assets,destination,steps,task='motion',preflight_only=False):
    import torch
    if type(steps) is not int or steps<1:raise ValueError('Positive training steps required')
    if task not in ('motion','paired','upperbody_probe'):raise ValueError('Unknown training task')
    destination.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[3]),'rev-parse','HEAD'],text=True).strip()
    catalog=json.loads(checked(assets['catalog']).read_text()) if task!='paired' else dict(entries=[])
    with np.load(checked(assets['scaler']),allow_pickle=False) as scaler:
        scale,bias=scaler['scale'].copy(),scaler['bias'].copy()
    if scale.shape!=(151,) or bias.shape!=(151,) or not np.isfinite(scale).all() or not np.isfinite(bias).all() or np.any(scale==0):
        raise ValueError('Frozen scaler contract differs')
    refs={};hash_groups={}
    for row in catalog['entries']:
        uid=row['source_uid'];group=row.get('recording_uid',uid)
        ref=row['reference']
        if uid in refs and refs[uid]!=(group,ref):raise ValueError('Inconsistent source identity')
        if ref['sha256'] in hash_groups and hash_groups[ref['sha256']]!=group:
            raise ValueError('Duplicate source bytes assigned to different recording groups')
        refs[uid]=(group,ref);hash_groups[ref['sha256']]=group
    windows=[];probes=[];tested=0;full_count=0
    counts={'body8':0,'wrists':0,'root':0}
    for uid,(group,ref) in refs.items():
        source=np.load(checked(ref),allow_pickle=False)
        for start in range(0,len(source)-149,75):
            native=source[start:start+150].copy()
            origin=native[0,4:7].copy();origin[1]=0.
            normalized=MotionAdapter.to_edge(native,origin)*scale+bias
            valid=np.isfinite(normalized)&(abs(normalized)<=1+32*np.finfo(np.float32).eps)
            tested+=1
            for name,(a,b) in {'body8':(79,127),'wrists':(127,139),'root':(4,7)}.items():
                counts[name]+=int(valid[:,a:b].all())
            if valid.all():
                windows.append((group,start,normalized.astype(np.float32),None,native));full_count+=1
            partial=upperbody_probe_target(normalized,valid)
            if partial is not None:probes.append((group,start,partial,None,native))
    if task=='upperbody_probe':windows=probes
    roles={};paired_audit=None
    if task=='paired':
        windows=[]
        if assets.get('paired_generation_manifest'):
            from dunhuang.data.paired_manifest import inspect
            windows,roles,paired_audit=inspect(assets)
    fitting,validation,validation_group=split_training_rows(windows,task,roles)
    protocol=dict(code_commit=commit,seed=SEED,steps=steps,train_task=task,
        checkpoint=assets['checkpoint'],catalog=assets.get('catalog'),scaler=assets['scaler'],
        paired_manifest=assets.get('paired_generation_manifest'),
        split='whole recording; paired roles explicit; no test fitting',
        train_groups=sorted({r[0] for r in fitting}),validation_groups=sorted({r[0] for r in validation}),
        train_windows=len(fitting),validation_windows=len(validation),
        validation_sampling='up to 8 uniformly spaced validation windows; fixed seed117 timesteps100/500/900',
        selection='lowest fixed validation hidden-channel MSE, checked every50 and final step',
        controls=['zero_adapter','observed_source','condition_removed'],
        phase_training='source positions/150; motion/probe piecewise knot uniform integer69..81; paired identity to preserve actual music clock',
        loss_channels='hidden 48D only' if task=='upperbody_probe' else 'hidden 151D',
        other_target_channels='neutral placeholders NOT ground truth' if task=='upperbody_probe' else 'supported real full motion',
        music='real aligned Jukebox' if task=='paired' else 'all-zero diagnostic input, no music qualification')
    write(destination/'protocol.json',protocol)
    audit=dict(tested_150_windows=tested,component_supported_windows=counts,
        root_origin='first_observed_context_XZ; height preserved',
        motion_only_supported_150_windows=full_count,upperbody_partial_probe_windows=len(probes),
        source_count=len(refs),source_records=[dict(source_uid=u,recording_group=g,reference=r) for u,(g,r) in refs.items()],
        eligible_windows=len(windows),train_task=task,paired_preflight=paired_audit,
        paired_jukebox_training_manifest=assets.get('paired_generation_manifest'),
        full_paired_music_training_ready=task=='paired' and bool(fitting and validation),
        full_motion_qualified=False,music_qualified=False,
        train_groups=protocol['train_groups'],validation_groups=protocol['validation_groups'])
    write(destination/'asset_audit.json',audit)
    ready=bool(fitting and validation)
    if not ready or preflight_only:
        status=('BLOCKED_PAIRED_GENERATION_CONTRACT' if paired_audit and not paired_audit['ready'] else
                'MISSING_REAL_PAIRED_JUKEBOX' if task=='paired' and not assets.get('paired_generation_manifest') else
                'PREFLIGHT_READY_RESEARCH_ONLY' if ready and task=='upperbody_probe' else
                'PREFLIGHT_READY_NOT_QUALIFIED' if ready else 'INSUFFICIENT_DISJOINT_TRAIN_VAL_SOURCES')
        write(destination/'status.json',dict(status=status,training_steps=0,train_task=task,
            music_qualified=False,full_motion_qualified=False,eligible_windows=len(windows)))
        return
    if not torch.cuda.is_available():raise RuntimeError('CUDA required beyond preflight')
    root=pinned(assets['edge_root'],assets['edge_commit']);sys.path.insert(0,str(root))
    from dataset.preprocess import Normalizer
    from dataset.scaler import MinMaxScaler
    from model.model import DanceDecoder
    from model.diffusion import GaussianDiffusion
    from vis import SMPLSkeleton
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    payload=torch.load(checked(assets['checkpoint']),map_location='cpu',weights_only=False)
    # Bind the exported scaler to this exact model payload, not just an arbitrary npz.
    if not np.array_equal(scale,payload['normalizer'].scaler.scale_.cpu().numpy()) or not np.array_equal(bias,payload['normalizer'].scaler.min_.cpu().numpy()):
        raise ValueError('Training scaler differs from frozen checkpoint')
    model=DanceDecoder(nfeats=151,seq_len=150,latent_dim=512,ff_size=1024,num_layers=8,num_heads=8,dropout=.1,cond_feature_dim=4800,activation=torch.nn.functional.gelu).cuda().eval()
    model.load_state_dict(payload['ema_state_dict'],strict=True);del payload
    for parameter in model.parameters():parameter.requires_grad_(False)
    diffusion=GaussianDiffusion(model,150,151,SMPLSkeleton('cuda'),schedule='cosine',n_timestep=1000,predict_epsilon=False,loss_type='l2',use_p2=False,cond_drop_prob=.25,guidance_weight=2).cuda().eval()
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    adapter=__import__('dunhuang.generation.adapter',fromlist=['PrimitiveAdapter']).PrimitiveAdapter().cuda()
    from .adapter import masked_reference
    before=digest(model);state={'enabled':True}
    def condition(module,args,value):
        return value+adapter(state['value'],state['known'],state['phase']) if state['enabled'] else value
    hook=model.cond_encoder.register_forward_hook(condition)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=.0001)
    rng=np.random.default_rng(SEED);augmented=fallbacks=0
    def batch(rows,augment=False):
        nonlocal augmented,fallbacks
        targets=[];phases=[]
        for row in rows:
            knot=int(rng.integers(69,82)) if augment and task!='paired' else 75
            target,phase=phase_target(row[4],scale,bias,knot,task=='upperbody_probe')
            if target is None:
                target,phase=phase_target(row[4],scale,bias,75,task=='upperbody_probe');fallbacks+=1
            if target is None:raise ValueError('Eligible target changed during phase sampling')
            augmented+=int(augment and knot!=75)
            targets.append(target);phases.append(phase)
        target=torch.from_numpy(np.stack(targets)).float().cuda()
        phase=torch.from_numpy(np.stack(phases)).cuda()
        music=torch.from_numpy(np.stack([r[3] for r in rows])).cuda() if task=='paired' else torch.zeros(len(rows),150,4800,device='cuda')
        return target,music,phase
    def objective(target,music,phase,control='observed_source',timestep=None):
        value,known=masked_reference(target,scope='event_soft' if task in ('paired','upperbody_probe') else 'full')
        supervised=~known
        if task=='upperbody_probe':
            loss_mask=torch.zeros_like(supervised);loss_mask[:,:,79:127]=True
            supervised=supervised&loss_mask
        if control=='condition_removed':value=torch.zeros_like(value);known=torch.zeros_like(known)
        state.update(value=value,known=known,phase=phase,enabled=control!='zero_adapter')
        t=torch.randint(50,950,(len(target),),device='cuda') if timestep is None else torch.full((len(target),),timestep,device='cuda',dtype=torch.long)
        predicted=model(diffusion.q_sample(target,t),music,t,cond_drop_prob=0.)
        return ((predicted-target)**2)[supervised].mean()
    val_rows=[validation[i] for i in np.unique(np.linspace(0,len(validation)-1,min(8,len(validation))).astype(int))]
    def evaluate(control):
        adapter.eval();values=[]
        with torch.no_grad(),torch.random.fork_rng(devices=[0]):
            torch.manual_seed(117);torch.cuda.manual_seed_all(117)
            for start in range(0,len(val_rows),2):
                args=batch(val_rows[start:start+2])
                for t in (100,500,900):values.append(float(objective(*args,control=control,timestep=t)))
        return float(np.mean(values))
    initial=evaluate('zero_adapter')
    history=[];best=float('inf')
    try:
        for step in range(1,steps+1):
            rows=[fitting[i] for i in rng.integers(0,len(fitting),2)]
            args=batch(rows,augment=True)
            adapter.train();objective_value=objective(*args)
            if not torch.isfinite(objective_value):raise ValueError('Nonfinite training objective')
            optimizer.zero_grad();objective_value.backward()
            norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite Adapter gradient')
            optimizer.step()
            if step%50==0 or step==steps:
                error=evaluate('observed_source')
                history.append(dict(step=step,training_loss=float(objective_value.detach()),validation_loss=error,gradient_norm=float(norm)))
                write(destination/'history.json',history)
                if error<best:
                    best=error
                    torch.save(dict(state=adapter.state_dict(),base_sha256=assets['checkpoint']['sha256'],
                        condition_schema=SCHEMA if task in ('paired','upperbody_probe') else 'legacy_sparse_reference_v0',
                        condition_scope='paired_music_completion' if task=='paired' else 'upperbody_partial_probe_only' if task=='upperbody_probe' else 'motion_only_masked_completion',
                        training_commit=commit,protocol_sha256=sha(destination/'protocol.json'),
                        full_motion_qualified=False,music_qualified=False),destination/'adapter.pt')
        payload=torch.load(destination/'adapter.pt',map_location='cpu',weights_only=True)
        adapter.load_state_dict(payload['state'])
        controls={c:evaluate(c) for c in ('zero_adapter','observed_source','condition_removed')}
    finally:hook.remove()
    after=digest(model)
    if before!=after or any(p.grad is not None for p in model.parameters()):raise ValueError('Frozen EDGE changed or got gradients')
    write(destination/'validation.json',dict(initial_zero_adapter_mse=initial,controls=controls,
        evaluated_windows=[dict(group=r[0],start=r[1]) for r in val_rows],
        scope='fixed source-held-out partial completion diagnostic, NOT generation, music or Native quality',
        selected_on_this_validation=True))
    write(destination/'status.json',dict(status='UPPERBODY_PARTIAL_PROBE_TRAINED_UNQUALIFIED' if task=='upperbody_probe' else 'PAIRED_COMPLETION_TRAINED_UNQUALIFIED' if task=='paired' else 'MOTION_ONLY_TRAINED',
        training_steps=steps,seed=SEED,code_commit=commit,train_task=task,
        weights_before=before,weights_after=after,frozen_model_unchanged=before==after,
        phase_augmented_examples=augmented,unsupported_warp_fallbacks=fallbacks,
        music_qualified=False,full_motion_qualified=False,
        checkpoint_sha256=sha(destination/'adapter.pt'),protocol_sha256=sha(destination/'protocol.json')))
def train(assets,destination,steps,task='motion',preflight_only=False):
    destination=Path(destination);existed=destination.exists()
    try:
        return _train(assets,destination,steps,task,preflight_only)
    except Exception:
        if not existed and destination.is_dir():
            history=destination/'history.json'
            rows=json.loads(history.read_text()) if history.exists() else []
            write(destination/'status.json',dict(status='FAILED',train_task=task,
                last_recorded_training_step=rows[-1]['step'] if rows else 0,
                music_qualified=False,full_motion_qualified=False,error=traceback.format_exc()))
        raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=300);ap.add_argument('--task',choices=['motion','paired','upperbody_probe'],default='motion')
    ap.add_argument('--preflight-only',action='store_true')
    args=ap.parse_args();train(json.loads(Path(args.assets).read_text()),Path(args.out).resolve(),args.steps,args.task,args.preflight_only)
