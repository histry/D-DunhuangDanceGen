"""Isolated official EDGE worker; never imports the legacy model namespace."""
import argparse
import json
import pickletools
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
from .io import checked, pinned, write
from .music import blend_contacts
from .conditioning import anchor_windows, normalize_constraints, constraint_support
from .sampling import constrained_ddim

def main():
    ap=argparse.ArgumentParser();ap.add_argument('request');a=ap.parse_args()
    q=json.loads(Path(a.request).read_text());out=Path(q['output']);out.mkdir(parents=True,exist_ok=True)
    root=pinned(q['edge_root'],q['edge_commit']);sys.path.insert(0,str(root))
    import torch
    from dataset.preprocess import Normalizer
    from dataset.scaler import MinMaxScaler
    from model.model import DanceDecoder
    from model.diffusion import GaussianDiffusion
    import model.diffusion as dm
    from vis import SMPLSkeleton
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('Real CUDA required for experiment')
    checkpoint=checked(q['checkpoint'])
    with zipfile.ZipFile(checkpoint) as z:
        data=z.read(next(n for n in z.namelist() if n.endswith('data.pkl')))
        names={arg for op,arg,_ in pickletools.genops(data) if op.name=='GLOBAL'}
    allowed={'collections OrderedDict','torch._utils _rebuild_tensor_v2','torch FloatStorage','torch LongStorage','torch DoubleStorage','torch HalfStorage','torch._utils _rebuild_parameter','dataset.preprocess Normalizer','dataset.scaler MinMaxScaler','numpy.core.multiarray _reconstruct','numpy ndarray','numpy dtype','_codecs encode','numpy.core.multiarray scalar'}
    if not names <= allowed:raise ValueError('Unexpected checkpoint pickle globals')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    normalizer=payload['normalizer']
    if not isinstance(normalizer,Normalizer) or not isinstance(normalizer.scaler,MinMaxScaler):raise ValueError('Unexpected normalizer')
    if q.get('save_scaler'):
        np.savez(out/'scaler.npz',scale=normalizer.scaler.scale_.cpu().numpy(),
                 bias=normalizer.scaler.min_.cpu().numpy())
    features=np.load(checked(q['features']),allow_pickle=False)[:q['windows']]
    if features.shape!=(q['windows'],150,4800) or not np.isfinite(features).all() or not features.any():raise ValueError('Real Jukebox windows required')
    values,masks=anchor_windows(q['frames'],q.get('anchors',[]))
    support=constraint_support(normalizer,values,masks)
    soft_audit=None
    soft_phase=None
    if q['mode']=='event_guided':
        if 'event_plan' not in q:raise ValueError('Soft event generation requires an explicit, hashed Event Plan')
        from .planning.event_plan import validate
        from .generation.soft_reference import build_soft_reference
        planned=validate(json.loads(checked(q['event_plan']).read_text()),q['audio']['sha256'],q['frames'])
        prompts,observed,soft_phase,soft_audit=build_soft_reference(
            planned,q['frames'],normalizer.scaler.scale_.cpu().numpy(),normalizer.scaler.min_.cpu().numpy())
        if not observed.any():raise ValueError('No source joint lies within frozen Adapter prompt support')
        normalized=torch.from_numpy(prompts)
        mask=torch.from_numpy(observed)
        write(out/'soft_reference_audit.json',soft_audit)
        write(out/'normalization_preflight.json',dict(supported=True,hard_anchor_support=support,
            classification='SOURCE_EVENT_SOFT_PROMPT_SUPPORT_MASKED',reference_is_not_output_admission=True))
    else:
        write(out/'normalization_preflight.json',support)
        normalized,mask=normalize_constraints(normalizer,values,masks)
    model=DanceDecoder(nfeats=151,seq_len=150,latent_dim=512,ff_size=1024,num_layers=8,num_heads=8,dropout=.1,cond_feature_dim=4800,activation=torch.nn.functional.gelu)
    model.load_state_dict(payload['ema_state_dict'],strict=True);del payload
    skeleton=SMPLSkeleton('cuda')
    diffusion=GaussianDiffusion(model,150,151,skeleton,schedule='cosine',n_timestep=1000,predict_epsilon=False,loss_type='l2',use_p2=False,cond_drop_prob=.25,guidance_weight=2).to('cuda').eval()
    for p in model.parameters():p.requires_grad_(False)
    def digest():
        import hashlib
        h=hashlib.sha256()
        for name,p in sorted(model.named_parameters()):h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    before=digest();torch.manual_seed(q['seed']);torch.cuda.manual_seed_all(q['seed'])
    cond=torch.from_numpy(features).float().cuda();start=time.monotonic()
    adapter_hook=None
    if q['mode']=='event_guided':
        from .generation.adapter import PrimitiveAdapter
        adapter_payload=torch.load(checked(q['adapter']),map_location='cpu',weights_only=True)
        from .generation.adapter import authorize_adapter
        authorize_adapter(adapter_payload,q['checkpoint']['sha256'],q.get('research_allow_motion_only_adapter',False))
        adapter=PrimitiveAdapter().cuda().eval()
        adapter.load_state_dict(adapter_payload['state'],strict=True)
        from .generation.conditions import source_condition
        requested_policy=q.get('research_condition_policy','dense_upper8')
        trained_policy=adapter_payload.get('condition_policy','completion_upper8')
        if not q.get('research_allow_motion_only_adapter') and requested_policy!=trained_policy:
            raise ValueError('Qualified Adapter condition distribution differs')
        condition=source_condition(normalized.cuda(),mask.cuda(),torch.from_numpy(soft_phase).cuda(),
            requested_policy,window_offsets=[75*i for i in range(len(normalized))])
        normalized,mask=condition.value.cpu(),condition.known.cpu()
        soft_audit.update(training_condition_policy=trained_policy,inference_condition_policy=requested_policy,
            visibility_distribution_matches=requested_policy==trained_policy,
            training_music_scope=adapter_payload.get('condition_scope'))
        write(out/'soft_reference_audit.json',soft_audit)
        event_residual=adapter(condition.value,condition.known,condition.phase).detach()
        adapter_hook=model.cond_encoder.register_forward_hook(lambda module,args,value:value+event_residual)

    # Adapter construction must not alter the matched sampler noise stream.
    torch.manual_seed(q['seed']);torch.cuda.manual_seed_all(q['seed'])
    with torch.no_grad():
        if q['mode']=='baseline':
            if masks.any():raise ValueError('Baseline must not contain anchors')
            sample=diffusion.long_ddim_sample((len(features),150,151),cond)
            sampler=dict(method='official_long_ddim',steps=50)
        elif q['mode'] in ('anchored','custom_none','temporal','event_guided'):
            if q['mode']!='custom_none' and not mask.any():raise ValueError('Guided run must have observed reference')
            policy = {'anchored':'hard','custom_none':'none','temporal':'temporal','event_guided':'none'}[q['mode']]
            guide = None
            if policy == 'temporal':
                from .temporal import prepare
                guide = prepare(q,normalizer,out,cond.device)
            sample,sampler=constrained_ddim(diffusion,cond,normalized,mask,guidance=policy,
                temporal=guide,overlap=q.get('overlap_policy','mean'))
        else:raise ValueError('Unknown sampler')
    if adapter_hook is not None:adapter_hook.remove()
    torch.cuda.synchronize()
    raw=normalizer.unnormalize(sample.detach().cpu().clone()).numpy()
    if not np.isfinite(raw).all():raise ValueError('Nonfinite generated motion')
    np.save(out/'windows.npy',raw);np.save(out/'normalized.npy',sample.detach().cpu().numpy())
    # Reuse the exact upstream root interpolation and quaternion SLERP, observing FK.
    captured={};original=skeleton.forward;render=dm.skeleton_render
    def observe(poses,root):
        value=original(poses,root);captured.update(poses=poses.detach().cpu().clone(),root=root.detach().cpu().clone(),fk=value.detach().cpu().clone());return value
    skeleton.forward=observe;dm.skeleton_render=lambda *args,**kwargs:None
    try:
        diffusion.render_sample(sample.detach().cpu().clone(),cond,normalizer,'generation',str(out),sound=False,mode='long',render=False)
    finally:
        skeleton.forward=original;dm.skeleton_render=render
    rotations=matrix_to_rotation_6d(axis_angle_to_matrix(captured['poses'][0])).numpy().reshape(-1,144)
    edge=np.concatenate([blend_contacts(raw[:,:,:4]),captured['root'][0].numpy(),rotations],axis=1).astype(np.float32)
    np.save(out/'edge.npy',edge);np.save(out/'official_fk.npy',captured['fk'][0].numpy())
    after=digest()
    if before!=after:raise RuntimeError('Frozen model changed')
    write(out/'generation.json',dict(mode=q['mode'],seconds=time.monotonic()-start,seed=q['seed'],frames=q['frames'],windows=len(features),
        checkpoint=q['checkpoint'],publisher_provenance_verified=False,edge_commit=q['edge_commit'],
        weights_before=before,weights_after=after,sampler=sampler,
        raw_window_overlap_max_abs=float(abs(raw[:-1,75:]-raw[1:,:75]).max()) if len(raw)>1 else 0.,
        training=False,official_stitch=True,features=q['features'],audio=q['audio'],
        adapter=(q.get('adapter') if q['mode']=='event_guided' else None),
        research_unqualified_adapter=(not (adapter_payload.get('music_qualified') and adapter_payload.get('full_motion_qualified')) if q['mode']=='event_guided' else False),
        soft_reference=soft_audit,adapter_music_qualified=(adapter_payload.get('music_qualified',False) if q['mode']=='event_guided' else None)))
    print('GENERATION_COMPLETE',q['mode'],q['frames'],flush=True)
if __name__=='__main__':main()
