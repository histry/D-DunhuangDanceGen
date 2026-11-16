"""Two fixed two-event boundaries, shared Lodge denoising and saved pre/post projection state."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from pytorch3d.transforms import rotation_6d_to_matrix,matrix_to_axis_angle,axis_angle_to_matrix
from dunhuang.research.coverage_r34.common import Runtime,forbid_optimizers,full_packet,sample,same_arrays
from dunhuang.research.coverage_r34.context_contract import context
from dunhuang.research.coverage_r34.context_train import restore_context
from dunhuang.research.complete_r36.core import read,write,sha,Library,rotation_sample,relative_fk,responsibility_budget
from dunhuang.research.state_r27.trace import row_noise
from .clock import source_positions
from .geometry import quintic_bridge,c2_envelope,necessary_jump

def compile_window(plan,boundary,lib):
    start=boundary-128
    if start<0 or start+256>plan['frames']:raise ValueError('Full observed music window required')
    R=[];ids=[];q=[];events=[]
    for e in plan['events']:
        a=max(start,e['music_interval'][0]);b=min(start+256,e['music_interval'][1])
        if a>=b:continue
        positions=source_positions(e,np.arange(a,b));R.append(rotation_sample(lib.rotations[e['source_uid']],positions));q.extend(positions.tolist());ids.extend([e['source_uid']]*(b-a))
        events.append(dict(e,local_interval=[a-start,b-start]))
    reference=np.concatenate(R)
    if reference.shape!=(256,22,3,3):raise ValueError('Do not pad source body')
    return reference,dict(window_start=start,boundary_local=128,boundary_global=boundary,events=events,source_uids=ids,source_positions=q,
                          world_Root='UNKNOWN',contact='UNKNOWN',right_context='ACTUAL_SELECTED_NEIGHBOR_BODY_ONLY')

@torch.no_grad()
def edit(rt,music,source,ctx,base,anchor,budget,seeds,start_t,gate=None):
    rt.c.clear_all();torch.manual_seed(27041);torch.cuda.manual_seed_all(27041)
    rt.c.set_source(source);rt.c.set_context(ctx);old=rt.c.change.inject
    if gate is not None:
        g=torch.as_tensor(gate,device=base.device,dtype=base.dtype)[None,:,None]
        rt.c.change.inject=lambda index,h:h+(old(index,h)-h)*g
    R=torch.tensor(anchor,dtype=base.dtype,device=base.device)[None].repeat(len(base),1,1,1,1)
    limits=torch.tensor(budget,device=base.device,dtype=base.dtype)[None,:,None]
    initial=base.clone();initial[:,:,7:]=R[...,:2,:].reshape(len(base),256,132)
    times=list(reversed(torch.linspace(-1,start_t,51).int().tolist()));weights=np.clip(np.linspace(0,rt.d.guidance_weight*2,50),None,rt.d.guidance_weight)
    traces=[]
    try:
        with row_noise(seeds,device=str(base.device)) as noise:
            eps=torch.randn_like(initial);x=rt.d.q_sample(initial,torch.full((len(base),),start_t,device=base.device,dtype=torch.long),eps)
            for time,nxt,weight in zip(times[:-1],times[1:],weights):
                t=torch.full((len(base),),time,device=base.device,dtype=torch.long)
                _,proposal,*_=rt.d.model_predictions(x,music,None,t,weight=float(weight),clip_x_start=rt.d.clip_denoised)
                P=rotation_6d_to_matrix(proposal[:,:,7:].reshape(len(base),256,22,6));v=matrix_to_axis_angle(R.transpose(-1,-2)@P)
                v=v*torch.minimum(torch.ones_like(v[...,0]),limits/v.norm(dim=-1).clamp_min(1e-8))[...,None]
                corrected=R@axis_angle_to_matrix(v);x0=proposal.clone();x0[:,:,7:]=corrected[...,:2,:].reshape(len(base),256,132)
                # Four identical actual-frame samples, sufficient for one boundary jerk stencil.
                traces.append(torch.stack([proposal[:,126:130],x0[:,126:130]],1).cpu().numpy())
                if nxt<0:x=x0;continue
                alpha=rt.d.alphas_cumprod[time];an=rt.d.alphas_cumprod[nxt]
                pred_noise=(x-alpha.sqrt()*x0)/(1-alpha).sqrt();sigma=((1-alpha/an)*(1-an)/(1-alpha)).clamp_min(0).sqrt()
                coefficient=(1-an-sigma.square()).clamp_min(0).sqrt();x=x0*an.sqrt()+coefficient*pred_noise+sigma*torch.randn_like(x)
        assert noise['draws']==50
    finally:
        rt.c.change.inject=old;rt.c.clear_all()
    return x,noise,np.stack(traces)

def run(cfg,root,out,replay=None):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Fresh generation directory required')
    out.mkdir(parents=True);forbid_optimizers();lib=Library(read('src/dunhuang/research/complete_r36/config.json')['catalog'])
    parent=Path(cfg['r36']);r35=Path(cfg['r35']);old=read(r35/'generation/result.json');songs=read(root/'planning/result.json')['songs']
    failed=next(r for r in old['records'] if r['key']=='song1_Graph')
    old_song=next(s for s in read(r35/'planning/songs.json') if s['id']==failed['song_id'])
    old_plan=read(old_song['plan']['path']);old_boundary=old_plan['events'][failed['selected_slot']]['music_interval'][1]
    new_song=songs[0];new_plan=read(new_song['new_plan']);legal=[e['music_interval'][0] for e in new_plan['events'][1:] if 128<=e['music_interval'][0]<=new_plan['frames']-128]
    if not legal:raise ValueError('No new-plan complete two-event boundary')
    cases=[dict(key='old_failed',song=old_song,plan=old_plan,boundary=old_boundary,selection='R36_SONG1_FIXED_FAILED_EXIT_BOUNDARY; PARENT_JERK4498.4556_VS_ENTRY970.1243'),
           dict(key='new_plan',song=new_song,plan=new_plan,boundary=legal[0],selection='FIRST_INTERNAL_BOUNDARY_WITH_COMPLETE256_MUSIC_STATIC_RULE')]
    rt=Runtime(read('src/dunhuang/research/transition_r35/config.json'),context_training=True);restore_context(rt,parent/'context/context.pt')
    arrays=dict(skeleton_J22=rt.s.J[0,:22].cpu().numpy(),parents=np.array(rt.s.parents[:22]));records=[]
    for c in cases:
        key=c['key'];R,binding=compile_window(c['plan'],c['boundary'],lib);gap=[128-cfg['bridge_halfwidth'],128+cfg['bridge_halfwidth']]
        geometry=quintic_bridge(R,gap);body=R[...,:2,:].reshape(256,132).astype(np.float32)
        wave,sr=sf.read(c['song']['audio']['path'],always_2d=True,dtype='float32');lo=round(binding['window_start']*sr/30);hi=lo+round(256*sr/30)
        if hi>len(wave):raise ValueError('No audio padding')
        sf.write(out/(key+'.wav'),wave[lo:hi],sr,subtype='FLOAT')
        music,_=rt.e(str(out/(key+'.wav')),skip_completed=False);music=np.asarray(music[:256],np.float32)
        if music.shape!=(256,35):raise ValueError('Actual music feature support')
        M=torch.tensor(np.repeat(music[None],2,0),device='cuda');packet=full_packet(np.repeat(body[None],2,0),[key]*2)
        base,_,_=sample(rt,M,seeds=cfg['seeds'])
        values=base.clone();values[:,:,7:]=torch.tensor(body,device='cuda')[None];ctx=context(values,gap,partial_body=True,window_origin=[binding['window_start']]*2)
        arrays[key+'_reference_R']=R;arrays[key+'_reference22']=relative_fk(R,arrays['skeleton_J22'],arrays['parents'])
        arrays[key+'_Geometry_R']=geometry;arrays[key+'_Geometry22']=relative_fk(geometry,arrays['skeleton_J22'],arrays['parents'])
        arrays[key+'_music35']=music;arrays[key+'_Frozen_normalized']=base.cpu().numpy();variants={}
        for arm in ('R36Like','Joint'):
            if arm=='R36Like':
                distance=abs(np.arange(256)-128);u=np.clip(distance/24,0,1);u=u*u*(3-2*u)
                budget=responsibility_budget(.2+.4*(1-u),[64,192],12);anchor=R;gate=None
            else:
                envelope=c2_envelope(256,gap);budget=cfg['bridge_residual_rad']*envelope;anchor=geometry;gate=1-envelope
            pred,noise,trace=edit(rt,M,packet,ctx,base,anchor,budget,cfg['seeds'],cfg['edit_start_t'],gate)
            raw=rt.n.unnormalize(pred.clone());points=rt.f(raw,rt.s)[:,:,:22].detach().cpu().numpy()
            arrays[key+'_'+arm+'_139']=raw.cpu().numpy();arrays[key+'_'+arm+'_22']=points;arrays[key+'_'+arm+'_normalized']=pred.cpu().numpy()
            arrays[key+'_'+arm+'_trace']=trace;arrays[key+'_'+arm+'_budget']=budget
            variants[arm]=dict(noise=noise,initialization='SOURCE_BODY_WITH_MODEL_ROOT_C4',anchor='DISCONTINUOUS_REFERENCE' if arm=='R36Like' else 'QUINTIC_GEOMETRIC_SHARED_TRANSITION',
                               projection='SO3_EACH_X0',source_residual_gate='NONE' if gate is None else 'C2_ZERO_INSIDE_COMMON_TRANSITION',
                               observed_phase_semantics='NOT_TRAINED; NO_SEMANTIC_PHASE_CLAIM')
            print('R37_SAMPLE',key,arm,flush=True)
        records.append(dict(key=key,song=c['song']['name'],binding=binding,gap=gap,seeds=cfg['seeds'],selection=c['selection'],
                            plan_sha256=__import__('hashlib').sha256(__import__('json').dumps(c['plan'],sort_keys=True).encode()).hexdigest(),
                            consumes_new_r37_plan=key=='new_plan',variants=variants,lower_bound=necessary_jump(R,arrays[key+'_R36Like_budget'],128)))
        np.savez_compressed(out/'arrays.npz',**arrays)
    rt.check();rt.close();result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),records=records,
        context_sha256=sha(parent/'context/context.pt'),optimizer_steps=0,paired_training_targets_used=0,whole_song_generations=0,
        native24='NOT_EVALUABLE_22_OF24',operational_qualified=False,method='BOUNDED_LODGE_MODIFICATION_OF_GEOMETRIC_SHARED_BRIDGE_NOT_DOUBLETAKE_REPRODUCTION')
    write(out/'result.json',result)
    if replay:same_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');assert result==read(Path(replay)/'result.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(read('src/dunhuang/research/transition_r37/config.json'),a.root,a.out,a.replay_from)
