"""Real frozen-model 50-step sampling; condition interventions never read gap truth."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,subprocess,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.bridge_r38.study import runtime
from dunhuang.research.bridge_r38.source import packet
from dunhuang.research.bridge_r38.sampler import project
from dunhuang.research.coverage_r34.common import forbid_optimizers,digest
from dunhuang.research.coverage_r34.context_train import restore_context
from dunhuang.research.coverage_r34.context_contract import context
from dunhuang.research.stabilize_r39.projection import Projector
from dunhuang.research.state_r27.trace import row_noise
from dunhuang.research.transition_r37.geometry import c2_envelope
from dunhuang.research.transition_r37.evaluate import decode
from .common import fresh,GAP,SEEDS,regions
from .condition import install,observed_phases


def sample(rt,music,source,ctx,anchor,cfg):
    torch.manual_seed(27041);torch.cuda.manual_seed_all(27041)
    rt.c.clear_all();rt.c.set_source(source);rt.c.set_context(ctx)
    gate=torch.tensor(1-c2_envelope(256,GAP),device='cuda',dtype=anchor.dtype)[None,:,None]
    original=rt.c.inject;traces=[]
    def inject(index,hidden):
        ds=(rt.c.change.inject(index,hidden)-hidden)*gate;dc=rt.c.context_branch.delta(index,hidden)
        traces.append(dict(layer=index,source=float(ds.square().mean().sqrt()),context=float(dc.square().mean().sqrt())))
        return hidden+ds+dc
    rt.c.inject=inject
    projector=Projector(project,'RootContinuous',.03)
    times=list(reversed(torch.linspace(-1,cfg['start_t'],51).int().tolist()));weights=np.clip(np.linspace(0,rt.d.guidance_weight*2,50),None,rt.d.guidance_weight)
    budget=torch.tensor(cfg['budget_rad']*c2_envelope(256,GAP),device='cuda',dtype=anchor.dtype)
    try:
      with torch.no_grad(),row_noise(SEEDS,device='cuda') as noise:
        x=rt.d.q_sample(anchor,torch.full((len(anchor),),cfg['start_t'],device='cuda',dtype=torch.long),torch.randn_like(anchor))
        for t,nxt,w in zip(times[:-1],times[1:],weights):
            ts=torch.full((len(anchor),),t,device='cuda',dtype=torch.long)
            _,proposal,*_=rt.d.model_predictions(x,music,None,ts,weight=float(w),clip_x_start=rt.d.clip_denoised)
            x0=projector(rt,proposal,anchor,budget)
            if nxt<0:x=x0;continue
            alpha=rt.d.alphas_cumprod[t];an=rt.d.alphas_cumprod[nxt];epsilon=(x-alpha.sqrt()*x0)/(1-alpha).sqrt()
            sigma=((1-alpha/an)*(1-an)/(1-alpha)).clamp_min(0).sqrt();x=x0*an.sqrt()+(1-an-sigma.square()).clamp_min(0).sqrt()*epsilon+sigma*torch.randn_like(x)
      assert noise['draws']==50
      return [rt.n.unnormalize(v.clone()).cpu().numpy() for v in (x,proposal)],noise,traces
    finally:rt.c.inject=original;rt.c.clear_all()


def scores(raw,R,J,parents,anchor):
    rows=[];P0=relative_fk(R,J,parents)
    for x,a in zip(raw,anchor):
        r,root=decode(x);P=relative_fk(r,J,parents);rv=Rotation.from_matrix((R.swapaxes(-1,-2)@r).reshape(-1,3,3)).magnitude().reshape(256,22)
        row=regions(r,J,parents,[dict(kind='core',interval=[0,104]),dict(kind='transition',interval=[104,152]),dict(kind='core',interval=[152,256])])
        visible=c2_envelope(256,GAP)==0
        row.update(gap_angle_rad=float(rv[104:152].mean()),gap_fk_rmse_m=float(np.sqrt(np.mean((P[104:152]-P0[104:152])**2))),gap_activity=float(np.sqrt(np.mean((np.diff(P[104:152],axis=0)*30)**2))),protected_raw_exact=bool(np.array_equal(x[visible],a[visible])),root_D3=float(np.quantile(np.sqrt(np.mean((np.diff(root,n=3,axis=0)*27000)**2,axis=1)),.95)))
        rows.append(row)
    return rows


def run(out,controls=False):
    out=fresh(out);cfg=read('src/dunhuang/research/bridge_r38/config.json');data_root=Path('outputs/source_context_r42_01/data_v2');meta=read(data_root/'result.json');data=np.load(data_root/'arrays.npz');origin=read('outputs/reference_r26_01/data/result.json')['records']
    specs=sorted((r for r in meta['records'] if r['domain']=='AIST++' and r['role']=='held'),key=lambda r:r['case'])[:2]
    for uid in ('female_FeiTian','male_pipa_2'):
        candidates=sorted((r for r in meta['records'] if r['recording_uid']==uid),key=lambda r:r['start']);specs.append(candidates[len(candidates)//2])
    assert len(specs)==4
    assert len({next(o for o in origin if o['case']==r['case'])['audio']['sha256'] for r in specs[:2]})==2
    assert sha(data_root/'arrays.npz')==meta['arrays_sha256']
    rt=runtime(cfg);restore_context(rt,'outputs/bridge_r38_01/training/context.pt');rt.c.context_branch.requires_grad_(False);install(rt);before=digest(rt.c.context_branch);forbid_optimizers();rt.d.model.eval();rt.c.eval();arrays=dict(J=data['skeleton_J22'],parents=data['parents']);records=[]
    paired_audio={}
    for index,row in enumerate(specs):
        key=row['key'];truth=data[key+'_truth'];R=decode(truth)[0];anchor=np.repeat(data[key+'_anchor_raw'][None],2,0);a=rt.n.normalize(torch.tensor(anchor,device='cuda',dtype=torch.float32));anchor=rt.n.unnormalize(a.clone()).cpu().numpy()
        m=data[key+'_music'];audio=None
        if row['domain']=='AIST++':
            o=next(q for q in origin if q['case']==row['case']);src=Path('outputs/reference_r26_01/data')/o['crop_audio'];audio=key+'.wav';shutil.copyfile(src,out/audio);paired_audio[key]=dict(path=str(src),sha256=sha(src),clock=o['clock'],source_start=o['source_start'])
        # ChangE zero music retains its lack of pairing; no artistic wrong-music truth.
        music=torch.tensor(np.repeat(m[None],2,0),device='cuda',dtype=torch.float32)
        refR=R.copy();refR[104:152]=np.nan;ref=packet(np.repeat(refR[None],2,0),GAP,[row['recording_uid']]*2)
        rp,qp=observed_phases(ref.values);ref.values['ref_phase']=rp;ref.values['query_phase']=qp
        cp=context(a,GAP);cp.valid[:,:,:4]=False;cp.values[:,:,:4]=float('nan')
        if not row['root_position_observed']:cp.valid[:,:,4:7]=False;cp.values[:,:,4:7]=float('nan')
        cp.validate();arms=['C0Frozen','C2Phase'] if not controls else ['C0Frozen','C1ZeroPhase','C2Phase','C3NoMusic','C4WrongMusic','C5WrongSource','C6NoLeft','C7NoRight','C8WrongPhase']
        methods={};noises={};traces={};arrays[key+'_reference']=relative_fk(R,arrays['J'],arrays['parents']).astype('<f4');arrays[key+'_Geometry']=np.stack([relative_fk(decode(v)[0],arrays['J'],arrays['parents']) for v in anchor]).astype('<f4')
        for arm in arms:
            source=copy.deepcopy(ref);ctx=copy.deepcopy(cp);mm=music
            if arm=='C0Frozen':source.values['valid'][:]=False;source.values['velocity_valid'][:]=False;ctx.valid[:]=False;ctx.values[:]=float('nan')
            if arm=='C1ZeroPhase':source.values['ref_phase'].zero_();source.values['query_phase'].zero_()
            if arm=='C3NoMusic':mm=torch.zeros_like(music)
            if arm=='C4WrongMusic':
                donor=specs[(index+1)%2] if index<2 else specs[0];mm=torch.tensor(np.repeat(data[donor['key']+'_music'][None],2,0),device='cuda',dtype=torch.float32)
            if arm=='C5WrongSource':
                donor=specs[(index+1)%len(specs)];v=decode(data[donor['key']+'_truth'])[0];v[104:152]=np.nan;source=packet(np.repeat(v[None],2,0),GAP,[donor['recording_uid']]*2);source.values['ref_phase'],source.values['query_phase']=observed_phases(source.values)
            if arm=='C6NoLeft':ctx.valid[:,:8]=False;ctx.values[:,:8]=float('nan')
            if arm=='C7NoRight':ctx.valid[:,8:]=False;ctx.values[:,8:]=float('nan')
            if arm=='C8WrongPhase':source.values['ref_phase']=source.values['ref_phase'].roll(32,1);source.values['query_phase']=source.values['query_phase'].roll(32,1)
            (raw,proposal),noise,trace=sample(rt,mm,source,ctx,a,cfg)
            arrays[key+'_'+arm+'_139']=raw;arrays[key+'_'+arm+'_proposal139']=proposal;arrays[key+'_'+arm]=np.stack([relative_fk(decode(v)[0],arrays['J'],arrays['parents']) for v in raw]).astype('<f4')
            methods[arm]=dict(final=scores(raw,R,arrays['J'],arrays['parents'],anchor),raw_proposal=scores(proposal,R,arrays['J'],arrays['parents'],anchor));noises[arm]=noise;traces[arm]=trace
            print('R51_SAMPLE',key,arm,flush=True)
        arrays[key+'_phase_reference']=rp.cpu().numpy();arrays[key+'_phase_query']=qp.cpu().numpy()
        records.append(dict(key=key,domain=row['domain'],role=row['role'],recording=row['recording_uid'],source_interval=row['source_interval'],audio=audio,music_identity=paired_audio.get(key),methods=methods,layer_traces=traces,noise=noises,phase='OBSERVED_KINEMATIC_PROXY_NO_HIDDEN_TARGET',protected_by='EXPLICIT_SAMPLER_PROJECTION_NOT_LEARNED',seeds=list(SEEDS),paired_music=row['domain']=='AIST++'))
    assert digest(rt.c.context_branch)==before;weights=rt.frozen;rt.c.context_branch.zero_grad(set_to_none=True);rt.check();rt.close();np.savez_compressed(out/'arrays.npz',**arrays)
    write(out/'result.json',dict(records=records,weights=weights,input_arrays_sha256=sha(data_root/'arrays.npz'),optimizer_steps=0,new_neural_samples=sum(len(r['methods'])*len(SEEDS) for r in records),whole_song_generations=0,native24='NOT_EVALUABLE_22_OF24',can_route=False))
    from .view import generation
    generation(out)
