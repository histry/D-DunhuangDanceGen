"""Same genuine phase support for every content/control score; clock labels never inputs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,collections
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.transition_r37.content import observations
from dunhuang.research.paired_event_r41.model import EventContent,RADIUS
from dunhuang.research.music_phase_r47.phase_audit import stage_score,rank_zero
from dunhuang.research.phase_r45.audio import extract
from dunhuang.research.sdmr_event_rank.evaluate import group_ci

def time_grid(length,sr,hop):
    if length<1 or sr<=0 or hop<=0:raise ValueError('Physical audio grid')
    return np.arange(length,dtype=float)*hop/sr*30

def sample(values,positions):
    q=np.asarray(positions,float)
    if not len(q) or q.min()<0 or q.max()>len(values)-1:raise ValueError('No circular or padded token support')
    a=np.floor(q).astype(int);b=np.ceil(q).astype(int);v=values[a]*(1-(q-a))[:,None]+values[b]*(q-a)[:,None]
    return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)

def content_scores(mt,dt,frames,offsets):
    d=sample(dt,np.asarray(frames)-RADIUS)
    return [float(np.mean(np.sum(sample(mt,np.asarray(frames)+s-RADIUS)*d,axis=-1))) for s in offsets]

def run(out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh frozen-content comparison')
    if not read('outputs/bridge_dynamics_r48_01/receipts/main_browser_acceptance.json')['passed']:raise ValueError('Watch actual main candidate first')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    cfg=read('src/dunhuang/research/event_feasibility_r48/config.json');oldpath=Path('outputs/music_phase_r47_01/main/result.json');old=read(oldpath);manifestpath=Path('outputs/paired_event_r41_01/data/manifest.json');manifest=read(manifestpath);sources={s['source_uid']:s for s in manifest['sources']};normal=read(checked(manifest['normalizer']))['statistics'];afroot=Path(read('src/dunhuang/research/paired_event_r41/config.json')['source'])/'audio'
    checkpoint=Path('outputs/paired_event_r41_01/main/model.pt');seal=read('outputs/paired_event_r41_01/artifacts_manifest.json');assert sha(checkpoint)==next(f['sha256'] for f in seal['files'] if f['path']=='main/model.pt')
    model=EventContent(96).cuda().eval().requires_grad_(False);model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['state']);sk=np.load('outputs/paired_event_r41_01/data/skeleton.npz');J=sk['J'];parents=sk['parents'];audio={};tokens={};records=[];clock=[];proxy=collections.defaultdict(set)
    for pair in old['phase_pairs']:
        for p in pair['phase_audit']['matches']:
            if p['evidence_state']=='HIGH_CONFIDENCE_UNCALIBRATED_PROXY':proxy[pair['source_pair'][0]].add(int(p['source_frame']))
    for row in old['clock_tasks']:
        s=sources[row['source_uid']];assert s['role']=='train'
        if s['audio_sha256'] not in audio:
            af=afroot/(s['audio_sha256']+'.npz');assert sha(af)==s['features_sha256'];z=np.load(af);sig=extract(checked(s['audio']),read('src/dunhuang/research/phase_r45/config.json'));grid=time_grid(len(sig['onset']),sig['sample_rate'],sig['hop']);assert np.allclose(grid,np.arange(len(grid)),atol=1e-8)
            prior=next(r for r in old['clock_tasks'] if r['audio_sha256']==s['audio_sha256']);audio[s['audio_sha256']]=dict(values=z['values'],mask=z['mask'],signal=sig)
            # Energy is the exact 30Hz value used by frozen R47; its scores stay auditable.
            import soundfile as sf
            y,sr=sf.read(checked(s['audio']),always_2d=True);y=y.mean(1);energy=np.array([np.sqrt(np.mean(y[round(t*sr/30):round((t+1)*sr/30)]**2)) if round(t*sr/30)<len(y) else 0 for t in range(len(grid))]);sig['energy']=(energy/max(float(np.quantile(energy,.95)),1e-10)).clip(0,1).tolist();clock.append(dict(audio_sha256=s['audio_sha256'],sample_rate=sig['sample_rate'],hop=sig['hop'],maximum_grid_drift_frames=float(np.max(abs(grid-np.arange(len(grid))))),onset_center='librosa center=True compensates n_fft/2; nominal output frame n*hop/sr',r41_audio_feature_sha256=sha(af)))
    def get_tokens(source,donor=None,zero=False):
        key=(source['source_uid'],donor,zero)
        if key in tokens:return tokens[key]
        R=np.load(checked(source['source_arrays']))['R'];D,DM=observations(R,J,parents);DM[0,198:272]=False;A=audio[donor or source['audio_sha256']];q=np.arange(len(R))+source['audio_offset_frames'];good=(q>=0)&(q<=len(A['values'])-1);M=np.zeros((len(R),50),np.float32);MM=np.zeros_like(M,bool);ids=np.flatnonzero(good);lo=np.floor(q[ids]).astype(int);hi=np.ceil(q[ids]).astype(int);frac=q[ids]-lo;M[ids]=A['values'][lo]*(1-frac[:,None])+A['values'][hi]*frac[:,None];MM[ids]=A['mask'][lo]&A['mask'][hi]
        D=np.where(DM,(D-np.array(normal['motion']['mean']))/np.array(normal['motion']['std']),0).astype(np.float32);M=np.where(MM,(M-np.array(normal['music']['mean']))/np.array(normal['music']['std']),0).astype(np.float32)
        if zero:M[:]=0;D[:]=0
        with torch.no_grad():mt,dt,mv,dv=model.content(*[torch.tensor(v[None],device='cuda') for v in (M,D,DM,MM.any(-1))])
        result=tuple(v[0].cpu().numpy() for v in (mt,dt,mv,dv));tokens[key]=result;return result
    for num,row in enumerate(old['clock_tasks']):
        s=sources[row['source_uid']];donor=row['audio_donor'];mt,dt,mv,dv=get_tokens(s);zero=get_tokens(s,zero=True);swap=get_tokens(s,donor) if donor else None;frames=np.array(row['audio_times'])-s['audio_offset_frames'];types=np.array(row['types']);keep=[]
        for i,f in enumerate(frames):
            positions=f+np.array(cfg['offsets'])-RADIUS;ix=np.unique(np.r_[np.floor(positions),np.ceil(positions)]).astype(int);di=int(round(f))-RADIUS
            if len(ix) and ix.min()>=0 and ix.max()<len(mv) and 0<=di<len(dv) and mv[ix].all() and dv[di] and (swap is None or swap[2][ix].all()):keep.append(i)
        frames=frames[keep];types=types[keep].tolist();times=frames+s['audio_offset_frames'];signal=audio[s['audio_sha256']]['signal'];variants={};shuffled=np.random.default_rng(cfg['seed']+num).permutation(types).tolist()
        if len(frames):
            for name,channel,kinds,ss in [('B0Energy','energy',types,signal),('B1Rule','mix',types,signal),('B1TypeShuffled','mix',shuffled,signal),('B1MusicSwap','mix',types,audio[donor]['signal'] if donor else None)]:
                if ss is not None:variants[name]=[stage_score(ss,times,kinds,d,channel) for d in cfg['offsets']]
            variants['B2Content']=content_scores(mt,dt,frames,cfg['offsets']);variants['B2BothZero']=content_scores(zero[0],zero[1],frames,cfg['offsets']);variants['B2TypeShuffled']=variants['B2Content'].copy()
            if swap:variants['B2MusicSwap']=content_scores(swap[0],dt,frames,cfg['offsets'])
            proxy_indices=[i for i,f in enumerate(frames) if int(round(f)) in proxy[s['source_uid']]]
            if proxy_indices:
                f=frames[proxy_indices];v=content_scores(mt,dt,f,cfg['offsets']);p=[stage_score(signal,f+s['audio_offset_frames'],[types[i] for i in proxy_indices],d) for d in cfg['offsets']];variants['B3UncalibratedProxyOnly']=(np.array(v)+cfg['stage_weight']*np.array(p)).tolist()
        records.append(dict(source_uid=s['source_uid'],raw_motion_sha256=s['raw_motion_sha256'],audio_sha256=s['audio_sha256'],original_phase_count=row['phase_count'],common_phase_count=len(frames),source_frames=frames.tolist(),audio_offset_frames=s['audio_offset_frames'],variants={k:dict(scores=v,**rank_zero(v)) for k,v in variants.items()},stage_truth='UNKNOWN',B3Verified='NOT_EVALUABLE_NO_VERIFIED_STAGE_LABELS'))
        if (num+1)%50==0:print('R48_FROZEN_CONTENT',num+1,len(old['clock_tasks']),flush=True)
    group=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(list)))
    for r in records:
        for name,v in r['variants'].items():group[name][r['audio_sha256']][r['raw_motion_sha256']].append(v['recall_at1'])
    means={name:{a:float(np.mean([np.mean(v) for v in raws.values()])) for a,raws in music.items()} for name,music in group.items()};comparisons={}
    for baseline in ('B0Energy','B1Rule','B2BothZero','B2MusicSwap'):
        keys=sorted(set(means.get('B2Content',{}))&set(means.get(baseline,{})));values=[means['B2Content'][a]-means[baseline][a] for a in keys];ci=group_ci(values,cfg['seed'],cfg['bootstrap_draws']);comparisons[baseline]=dict(groups=len(keys),mean_improvement=float(np.mean(values)) if values else None,ci95=ci,passed=len(keys)>=cfg['minimum_music_groups'] and ci[0] is not None and ci[0]>0)
    summary={name:dict(groups=len(v),accuracy=float(np.mean(list(v.values()))),ci95=group_ci(list(v.values()),cfg['seed'],cfg['bootstrap_draws'])) for name,v in means.items()};necessary=all(r['passed'] for r in comparisons.values()) and summary.get('B2Content',{}).get('ci95',[0])[0]>.2
    write(out/'result.json',dict(records=records,summary=summary,comparisons=comparisons,audio_clock_audit=clock,checkpoint_sha256=sha(checkpoint),parent_tasks_sha256=sha(oldpath),r41_manifest_sha256=sha(manifestpath),recording_balanced_group_scores=means,necessary_content_screen=necessary,exact_stage_truth=False,B3_verified_status='NOT_EVALUABLE',B3_proxy_scope='DIFFERENT_SPARSE_SUPPORT_DESCRIPTIVE_NOT_FAIR_MAIN_COMPARISON',training_admitted=False,training_stop_reason='NO_VERIFIED_STAGE_SUPERVISION; zero-training audit does not manufacture semantic truth',optimizer_steps=0,test_reads=0,can_route=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);run(p.parse_args().out)
