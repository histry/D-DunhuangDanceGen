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
from .conditioning import anchor_windows, normalize_constraints
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
    features=np.load(checked(q['features']),allow_pickle=False)[:q['windows']]
    if features.shape!=(q['windows'],150,4800) or not np.isfinite(features).all() or not features.any():raise ValueError('Real Jukebox windows required')
    values,masks=anchor_windows(q['frames'],q.get('anchors',[]))
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
    with torch.no_grad():
        if q['mode']=='baseline':
            if masks.any():raise ValueError('Baseline must not contain anchors')
            sample=diffusion.long_ddim_sample((len(features),150,151),cond)
            sampler=dict(method='official_long_ddim',steps=50)
        elif q['mode']=='anchored':
            if not masks.any():raise ValueError('Anchored run must have actual anchors')
            sample,sampler=constrained_ddim(diffusion,cond,normalized,mask)
        else:raise ValueError('Unknown sampler')
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
        training=False,official_stitch=True,features=q['features'],audio=q['audio']))
    print('GENERATION_COMPLETE',q['mode'],q['frames'],flush=True)
if __name__=='__main__':main()
