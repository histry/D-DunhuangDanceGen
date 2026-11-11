"""Fresh-process, read-only replay of saved paired completion validation."""
import argparse,sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dunhuang.io import checked,pinned,write,sha
from dunhuang.data.sources import read,ref
from dunhuang.data.paired_manifest import inspect
from dunhuang.generation.train import phase_target,digest
from dunhuang.generation.adapter import PrimitiveAdapter,masked_reference
from dunhuang.generation.split import split_training_rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--training',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    assets=read(a.assets);folder=Path(a.training);protocol=read(folder/'protocol.json');status=read(folder/'status.json')
    if status['status']!='PAIRED_COMPLETION_TRAINED_UNQUALIFIED':raise ValueError('Completed paired training required')
    windows,roles,preflight=inspect(assets)
    if not preflight['ready']:raise ValueError('Input contract changed')
    _,validation,_=split_training_rows(windows,'paired',roles)
    val_rows=[validation[i] for i in np.unique(np.linspace(0,len(validation)-1,min(8,len(validation))).astype(int))]
    with np.load(checked(assets['scaler'])) as s:scale,bias=s['scale'],s['bias']
    edge=pinned(assets['edge_root'],assets['edge_commit']);sys.path.insert(0,str(edge))
    import torch
    from model.model import DanceDecoder
    from model.diffusion import GaussianDiffusion
    from vis import SMPLSkeleton
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    payload=torch.load(checked(assets['checkpoint']),map_location='cpu',weights_only=False)
    model=DanceDecoder(nfeats=151,seq_len=150,latent_dim=512,ff_size=1024,num_layers=8,num_heads=8,
        dropout=.1,cond_feature_dim=4800,activation=torch.nn.functional.gelu).cuda().eval()
    model.load_state_dict(payload['ema_state_dict'],strict=True);del payload
    for parameter in model.parameters():parameter.requires_grad_(False)
    diffusion=GaussianDiffusion(model,150,151,SMPLSkeleton('cuda'),schedule='cosine',n_timestep=1000,
        predict_epsilon=False,loss_type='l2',use_p2=False,cond_drop_prob=.25,guidance_weight=2).cuda().eval()
    adapter=PrimitiveAdapter().cuda().eval()
    checkpoint=checked(dict(path=str(folder/'adapter.pt'),sha256=status['checkpoint_sha256']))
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['base_sha256']==assets['checkpoint']['sha256']
    adapter.load_state_dict(saved['state'],strict=True)
    before=digest(model);adapter_before=digest(adapter);state={}
    def hook(module,args,value):
        return value+adapter(state['value'],state['known'],state['phase']) if state['enabled'] else value
    handle=model.cond_encoder.register_forward_hook(hook)
    controls={}
    try:
        for control in ('zero_adapter','observed_source','condition_removed'):
            values=[]
            with torch.no_grad(),torch.random.fork_rng(devices=[0]):
                torch.manual_seed(117);torch.cuda.manual_seed_all(117)
                for start in range(0,len(val_rows),2):
                    rows=val_rows[start:start+2];targets=[];phases=[]
                    for row in rows:
                        target,phase=phase_target(row[4],scale,bias)
                        if target is None:raise ValueError('Saved validation target lost support')
                        targets.append(target);phases.append(phase)
                    target=torch.from_numpy(np.stack(targets)).float().cuda()
                    phase=torch.from_numpy(np.stack(phases)).cuda()
                    music=torch.from_numpy(np.stack([r[3] for r in rows])).cuda()
                    for timestep in (100,500,900):
                        value,known=masked_reference(target,scope='event_soft');supervised=~known
                        if control=='condition_removed':value=torch.zeros_like(value);known=torch.zeros_like(known)
                        state.update(value=value,known=known,phase=phase,enabled=control!='zero_adapter')
                        t=torch.full((len(target),),timestep,device='cuda',dtype=torch.long)
                        predicted=model(diffusion.q_sample(target,t),music,t,cond_drop_prob=0.)
                        values.append(float(((predicted-target)**2)[supervised].mean()))
            controls[control]=float(np.mean(values))
    finally:handle.remove()
    expected=read(folder/'validation.json')['controls']
    differences={k:abs(controls[k]-expected[k]) for k in controls}
    assert before==digest(model)==status['weights_after']
    assert adapter_before==digest(adapter)
    if max(differences.values())>1e-7:raise ValueError('Saved validation did not replay: '+str(differences))
    write(out,dict(status='READ_ONLY_VALIDATION_REPLAY_VERIFIED',controls=controls,saved_controls=expected,
        max_abs_difference=max(differences.values()),exact=all(v==0 for v in differences.values()),
        input_manifest=assets['paired_generation_manifest'],adapter=ref(checkpoint),
        frozen_model_before=before,frozen_model_after=digest(model),adapter_before=adapter_before,adapter_after=digest(adapter),
        optimizer_calls=0,new_weight_updates=0,validation_rows=[dict(group=r[0],start=r[1]) for r in val_rows],
        scope='full151 hidden-channel completion MSE; not free-running generation or Native quality',
        model_generation_qualified=False))
    print('PAIRED_VALIDATION_REPLAY',controls,'MAX_DIFF',max(differences.values()),flush=True)
if __name__=='__main__':main()
