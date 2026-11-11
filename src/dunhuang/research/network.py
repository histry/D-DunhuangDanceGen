"""Read-only denoiser interventions and randomized VJP diagnosis, not training."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write,sha
from dunhuang.generation.adapter import PrimitiveAdapter,authorize_adapter
from dunhuang.generation.conditions import source_condition,target_supervision
from dunhuang.generation.soft_reference import build_soft_reference
from dunhuang.generation.train import digest,phase_target
from .frozen import denoiser

JOINT_GROUPS=dict(torso=(0,3,6,9,12,15),arms=(13,14,16,17,18,19),wrists=(20,21),
    legs=(1,2,4,5,7,8,10,11),terminal_hands=(22,23))
GROUPS=dict(contact=list(range(4)),root=list(range(4,7)),
    **{name:[7+6*j+k for j in joints for k in range(6)] for name,joints in JOINT_GROUPS.items()})

def stats(x):
    x=x.detach().float()
    result=dict(shape=list(x.shape),mean=float(x.mean()),std=float(x.std(unbiased=False)),
        rms=float(x.square().mean().sqrt()),max_abs=float(x.abs().max()))
    if x.ndim==3 and x.shape[1]>1:
        z=torch.nn.functional.normalize(x,dim=-1);sim=z@z.transpose(-1,-2);n=x.shape[1]
        result.update(token_variance=float(x.var(1,unbiased=False).mean()),
            off_diagonal_token_cosine=float((sim.sum()-sim.diagonal(dim1=-2,dim2=-1).sum())/(len(x)*n*(n-1))))
    return result

def diagnose(assets,plans,probe,prior,out):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,normalizer,diffusion=denoiser(assets)
    scale=normalizer.scaler.scale_.cpu().numpy();bias=normalizer.scaler.min_.cpu().numpy()
    checkpoint=torch.load(checked(probe),map_location='cpu',weights_only=True)
    authorize_adapter(checkpoint,assets['checkpoint']['sha256'],True)
    adapter=PrimitiveAdapter().cuda().eval();adapter.load_state_dict(checkpoint['state'])
    model_before=digest(model);adapter_before=digest(adapter)
    features=np.load(checked(assets['features']),allow_pickle=False)
    if len(features)<3:raise ValueError('Real alternative music window unavailable')
    music=torch.from_numpy(features[:1]).float().cuda()
    changed_music=torch.from_numpy(features[2:3]).float().cuda()
    cached=np.load(Path(prior)/'none150/normalized.npy',allow_pickle=False)
    if cached.shape!=(1,150,151):raise ValueError('Frozen generated-state probe clock differs')
    base=torch.from_numpy(cached).float().cuda()
    conditions={}
    for name,path in plans.items():
        plan=json.loads(Path(path).read_text())
        value,known,phase,_=build_soft_reference(plan,150,scale,bias)
        conditions[name]=source_condition(torch.from_numpy(value).cuda(),torch.from_numpy(known).cuda(),torch.from_numpy(phase).cuda())
    original=conditions['original'];empty=torch.zeros_like(original.value);unknown=torch.zeros_like(original.known)
    make=lambda v,k,p:source_condition(v,k,p)
    conditions.update(source_removed=make(empty,unknown,original.phase),
        phase_zeroed=make(original.value,original.known,torch.zeros_like(original.phase)),
        all_removed=make(empty,unknown,torch.zeros_like(original.phase)),
        completion_visibility=source_condition(original.value,original.known,original.phase,'completion_upper8'))
    state={};activation={};tokens={}
    def inject(module,args,value):
        if not state.get('enabled',True):return value
        c=state['condition'];delta=adapter(c.value,c.known,c.phase)
        if state.get('capture'):
            activation['edge_music_encoder']=stats(value);activation['adapter_residual']=stats(delta)
            activation['residual_to_music_rms']=float(delta.square().mean().sqrt()/value.square().mean().sqrt().clamp_min(1e-12))
            activation['residual_music_cosine']=float(torch.nn.functional.cosine_similarity(delta.flatten()[None],value.flatten()[None]))
        return value+delta
    handles=[model.cond_encoder.register_forward_hook(inject)]
    def capture(name):
        def hook(module,args,value):
            if state.get('capture'):
                activation[name]=stats(value)
                if name=='adapter.input':tokens['input']=value.detach()
        return hook
    handles.extend([adapter.input.register_forward_hook(capture('adapter.input')),
        adapter.encoder.layers[0].register_forward_hook(capture('adapter.transformer')),
        adapter.output.register_forward_hook(capture('adapter.output'))])
    for name,module in model.named_modules():
        if name.startswith('seqTransDecoder') and hasattr(module,'film1'):
            handles.append(module.register_forward_hook(capture('edge.'+name)))
    def predict(c,music_value,noisy,t,enabled=True,capture_layers=False):
        state.update(condition=c,enabled=enabled,capture=capture_layers)
        return model(noisy,music_value,t,cond_drop_prob=0.)
    variants=['zero_adapter','original','replaced','time_warp_changed','source_removed','phase_zeroed',
        'all_removed','completion_visibility','music_changed','zero_adapter_music_changed','music_zero']
    predictions=[];responses=[];activation_sets={}
    try:
        for seed in (20261111,20261112):
            for timestep in (100,500,900):
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                noise=torch.randn_like(base);t=torch.full((1,),timestep,device='cuda',dtype=torch.long)
                noisy=diffusion.q_sample(base,t,noise=noise);row={}
                with torch.no_grad():
                    for variant in variants:
                        c=conditions.get(variant,original)
                        m=changed_music if variant in ('music_changed','zero_adapter_music_changed') else torch.zeros_like(music) if variant=='music_zero' else music
                        value=predict(c,m,noisy,t,enabled=not variant.startswith('zero_adapter'),
                            capture_layers=seed==20261111 and timestep==500 and variant in ('original','completion_visibility'))
                        row[variant]=value.detach()
                        if state['capture']:
                            block=adapter.encoder.layers[0];q=tokens['input']
                            if block.norm_first:q=block.norm1(q)
                            _,attention=block.self_attn(q,q,q,need_weights=True,average_attn_weights=False)
                            entropy=-(attention.clamp_min(1e-12)*attention.clamp_min(1e-12).log()).sum(-1)/np.log(150)
                            activation['attention_normalized_entropy_per_head']=entropy.mean((0,2)).cpu().tolist()
                            activation_sets[variant]=dict(activation);activation.clear()
                zero=row['zero_adapter']
                for variant in variants:
                    baseline=row['zero_adapter_music_changed'] if variant=='music_changed' else zero
                    delta=row[variant]-baseline
                    responses.append(dict(seed=seed,timestep=timestep,condition=variant,
                        noise_sha256=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),
                        group_delta_rms={name:float(delta[:,:,ids].square().mean().sqrt()) for name,ids in GROUPS.items()},
                        prediction_rms=float(row[variant].square().mean().sqrt())))
                interaction=(row['music_changed']-row['zero_adapter_music_changed'])-(row['original']-zero)
                responses.append(dict(seed=seed,timestep=timestep,condition='music_event_interaction',
                    group_delta_rms={name:float(interaction[:,:,ids].square().mean().sqrt()) for name,ids in GROUPS.items()}))
                predictions.append(torch.stack([row[k][0] for k in variants]).cpu().numpy())
        np.savez_compressed(out/'denoiser_predictions.npz',predictions=np.stack(predictions))
        # Random VJPs estimate group sensitivity; NOT a complete physical Jacobian.
        value=original.value.detach().clone().requires_grad_(True)
        phase=original.phase.detach().clone().requires_grad_(True)
        music_grad=music.detach().clone().requires_grad_(True)
        c=source_condition(value,original.known,phase)
        torch.manual_seed(717);t=torch.tensor([500],device='cuda')
        y=predict(c,music_grad,diffusion.q_sample(base,t,noise=torch.randn_like(base)),t)
        jacobian={}
        for name,ids in GROUPS.items():
            entries=[]
            for repeat in range(2):
                torch.manual_seed(900+repeat)
                signs=(torch.randint(0,2,y[:,:,ids].shape,device='cuda')*2-1).float()
                scalar=(y[:,:,ids]*signs).sum()/np.sqrt(len(ids)*150)
                gradients=torch.autograd.grad(scalar,(value,phase,music_grad),retain_graph=True)
                entries.append(dict(reference_observed_rms=float(gradients[0][original.known].square().mean().sqrt()),
                    phase_rms=float(gradients[1].square().mean().sqrt()),music_rms=float(gradients[2].square().mean().sqrt()),
                    hidden_reference_max=float(gradients[0][~original.known].abs().max())))
            jacobian[name]=entries
        del y
        # One actual legacy held-source completion objective, backward only.
        catalog=json.loads(checked(assets['catalog']).read_text())
        protocol=json.loads((Path(prior)/'probe/protocol.json').read_text())
        group=protocol['validation_groups'][0]
        selected=next(r for r in catalog['entries'] if r.get('recording_uid')==group)
        source=np.load(checked(selected['reference']))
        target_np=target_phase=None;start=0
        for start in range(0,len(source)-149,75):
            target_np,target_phase=phase_target(source[start:start+150].copy(),scale,bias,75,True)
            if target_np is not None:break
        if target_np is None:raise ValueError('No valid held-source diagnostic target')
        target=torch.from_numpy(target_np)[None].cuda()
        reference=target.detach().clone().requires_grad_(True)
        p=torch.from_numpy(target_phase)[None].cuda().requires_grad_(True)
        available=torch.zeros_like(target,dtype=torch.bool);available[:,:,79:127]=True
        c=source_condition(reference,available,p,'completion_upper8')
        supervised=target_supervision(target,available,c)
        torch.manual_seed(117);t=torch.tensor([500],device='cuda')
        output=predict(c,torch.zeros_like(music),diffusion.q_sample(target,t,noise=torch.randn_like(target)),t,capture_layers=True)
        loss=(output-target).square()[supervised].mean();loss.backward()
        parameter_gradients={name:dict(l2=float(p.grad.norm()),rms=float(p.grad.square().mean().sqrt()))
            for name,p in adapter.named_parameters() if p.grad is not None}
        completion=dict(loss=float(loss.detach()),source_group=group,start_frame=start,
            reference_gradient_rms=float(reference.grad.square().mean().sqrt()),phase_gradient_rms=float(p.grad.square().mean().sqrt()),
            parameter_gradients=parameter_gradients,scope='legacy hidden48 completion with zero music; other channels are NOT ground truth')
        activation_sets['legacy_completion_zero_music']=dict(activation)
        for parameter in adapter.parameters():parameter.grad=None
        if any(p.grad is not None for p in model.parameters()):raise ValueError('Frozen EDGE acquired gradients')
    finally:
        for handle in handles:handle.remove()
    model_after=digest(model);adapter_after=digest(adapter)
    if model_before!=model_after or adapter_before!=adapter_after:raise ValueError('Read-only diagnosis changed weights')
    result=dict(schema='v07_read_only_condition_diagnostics_v1',responses=responses,activations=activation_sets,
        vjp=jacobian,vjp_method='two Rademacher VJPs per output group at t500; scalar divided by sqrt(output dimensions); raw denoiser, not full-sampler physical controllability',
        completion_gradient=completion,variants=variants,seeds=[20261111,20261112],timesteps=[100,500,900],
        generated_state_source=dict(path=str(Path(prior)/'none150/normalized.npy'),sha256=sha(Path(prior)/'none150/normalized.npy')),
        musical_intervention=dict(kind='different real 5s window of SAME cached song, not independent song or paired training',
            original_window=0,changed_window=2,original_start_frame=0,changed_start_frame=150),
        model_state_before=model_before,model_state_after=model_after,adapter_state_before=adapter_before,adapter_state_after=adapter_after,
        optimizer_steps=0,weight_updates=0,full_motion_qualified=False,music_qualified=False,
        limitation='two noise seeds on one song and few source events; gradients are local normalized-coordinate sensitivities, not causal or physical admission')
    write(out/'network.json',result)
    return result
