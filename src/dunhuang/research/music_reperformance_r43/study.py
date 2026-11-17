"""Actual frozen Lodge Core pilot, then fixed-population condition controls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,subprocess,shutil
from pathlib import Path
import numpy as np
import torch
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.bridge_r38.study import runtime
from dunhuang.research.bridge_r38.source import packet
from dunhuang.research.bridge_r38 import sampler
from dunhuang.research.coverage_r34.context_train import restore_context
from dunhuang.research.coverage_r34.context_contract import context
from dunhuang.research.coverage_r34.common import forbid_optimizers
from dunhuang.research.stabilize_r39.projection import Projector
from dunhuang.research.transition_r37.evaluate import decode,world_fk
from dunhuang.research.source_context_r42.diagnose import measure
from dunhuang.research.music_reperformance_r43.replay import exact_arrays
from .contract import permissions
from .projector import sample


def run(out,controls=False,replay=None):
    out=Path(out)
    if out.exists():raise ValueError('Fresh R43 output')
    if controls and not read(out.parent/'receipts/main_browser_acceptance.json').get('accepted'):raise ValueError('Deliver and inspect actual main output before controls')
    out.mkdir(parents=True);cfg=read(Path(__file__).with_name('config.json'));parent=read('src/dunhuang/research/bridge_r38/config.json');meta=read(Path(cfg['data'])/'result.json');data=np.load(Path(cfg['data'])/'arrays.npz');old=np.load('outputs/transition_r37_02/generation/arrays.npz');p0=np.load('outputs/source_context_r42_01/diagnostic/arrays.npz');rmeta=read('outputs/stabilize_r39_01/generation/result.json');origin=read('outputs/reference_r26_01/data/result.json')['records'];spec=[]
    for uid in sorted({r['recording_uid'] for r in meta['records'] if r['domain']=='ChangE'}):
        r=min((r for r in meta['records'] if r['recording_uid']==uid),key=lambda r:r['start']);spec.append(dict(r,evaluation_key=r['key'],key=r['key']+'__music',target_music='old_failed'))
    paired=[r for r in meta['records'] if r['domain']=='AIST++'];groups={}
    for r in paired:
        o=next(q for q in origin if q['case']==r['case'])
        if r['role']=='train':groups.setdefault(o['source_group'],[]).append(r)
    selected=[min(v,key=lambda r:r['case']) for k,v in sorted(groups.items())]+[r for r in paired if r['role']=='held']
    spec += [dict(r,evaluation_key=r['key'],target_music=None) for r in selected]
    for r in rmeta['records']:
        if not r['reference_is_target']:spec.append(dict(key=r['key'],recording_uid='EVENT_PLAN_'+r['key'],domain='ChangE-cross',role='exposed_stress',target_music=r['key'],source_interval=None,binding=r['binding'],root_position_observed=False))
    rt=runtime(parent);restore_context(rt,cfg['main_context']);rt.c.context_branch.requires_grad_(False);forbid_optimizers();arrays=dict(skeleton_J22=data['skeleton_J22'],parents=data['parents']);records=[];cal=read('outputs/stabilize_r39_01/calibration/result.json')['parameters'];noise_seeds=parent['seeds']
    def source(row):
        if row['domain']=='ChangE-cross':return p0[row['key']+'_reference_R']
        return decode(data[row['evaluation_key']+'_truth'])[0]
    for index,row in enumerate(spec):
        key=row['key'];R=source(row);cross=row['domain']=='ChangE-cross'
        if cross:raw=p0[key+'_Geometry_139'].copy();m=old[key+'_music35'];ref_ids=[key]*2
        else:
            raw=np.repeat(data[row['evaluation_key']+'_anchor_raw'][None],2,0);m=data[row['evaluation_key']+'_music'];ref_ids=[row['recording_uid']]*2
            if row['target_music']:m=old[row['target_music']+'_music35']
        a=rt.n.normalize(torch.tensor(raw,device='cuda',dtype=torch.float32));cp=context(a,parent['gap']);cp.valid[:,:,:4]=False;cp.values[:,:,:4]=float('nan')
        if not row['root_position_observed']:cp.valid[:,:,4:7]=False;cp.values[:,:,4:7]=float('nan')
        cp.validate();visible=R.copy();visible[104:152]=np.nan;ref=packet(np.repeat(visible[None],2,0),parent['gap'],ref_ids)
        contract=permissions(visible,cfg,dict(key=key,binding=row.get('binding'),source_interval=row['source_interval'],phase_truth='OBSERVED_KINEMATIC_PROXY_ONLY'));arrays[key+'_body_budget']=contract.body_budget;arrays[key+'_phase_proxy']=contract.phase_proxy;arrays[key+'_root_envelope']=contract.root_envelope;arrays[key+'_reference_R']=R;variants={'Geometry':raw};noise={}
        arms=['Frozen','Core'] if not controls else cfg['controls']
        for arm in arms:
            mm=m.copy();reference=ref
            if arm=='new_plan':mm=old[('new_plan' if row['target_music']!='new_plan' else 'old_failed')+'_music35']
            if arm=='NoMusic':mm=np.zeros_like(mm)
            if arm=='WrongSource':
                donor=spec[(index+1)%len(spec)];v=source(donor).copy();v[104:152]=np.nan;reference=packet(np.repeat(v[None],2,0),parent['gap'],[donor['recording_uid']]*2)
            music=torch.tensor(np.repeat(mm[None],2,0),device='cuda',dtype=torch.float32)
            if arm=='Frozen':
                previous=sampler.project;sampler.project=Projector(previous,'RootContinuous',cal['root_budget_m'])
                try:x,nr,_=sampler.sample_bridge(rt,music,reference,cp,a,parent,noise_seeds)
                finally:sampler.project=previous
            else:x,nr,_=sample(rt,music,reference,cp,a,parent,noise_seeds,contract,cfg['root_bridge_meters'])
            variants[arm]=rt.n.unnormalize(x.clone()).cpu().numpy();noise[arm]=nr;print('R43_SAMPLE',key,arm,flush=True)
        metrics={};curves={}
        for arm,x in variants.items():
            arrays[key+'_'+arm+'_139']=x;arrays[key+'_'+arm+'_22']=np.stack([world_fk(*decode(v),data['skeleton_J22'],data['parents']) for v in x]);metrics[arm],curves[arm]=measure(x,R,data['skeleton_J22'],data['parents'],raw)
        audio=None;music_identity=None
        if row['target_music']:
            audio=row['target_music']+'.wav';src=Path('outputs/transition_r37_02/generation')/audio;shutil.copyfile(src,out/audio);music_identity=dict(parent_file=str(src),sha256=sha(src),paired_to_change=False)
        elif row['domain']=='AIST++':
            o=next(q for q in origin if q['case']==row['case']);src=Path('outputs/reference_r26_01/data')/o['crop_audio'];audio=key+'.wav';shutil.copyfile(src,out/audio);music_identity=dict(source=o['audio'],clock=o['clock'],crop_sha256=sha(src),source_start=o['source_start'],scope='FROZEN_R26_CROP_IDENTITY_NO_NEW_PAIRING')
        records.append(dict(row,label=key+' · '+row['role'],methods=list(variants),seeds=noise_seeds,audio=audio,music_identity=music_identity,metrics=dict(values=metrics,curves=curves),noise=noise,phase_anchors=contract.anchors,scope='R43实际采样；Core上身≤0.1rad，Bridge≤0.06rad，Root≤0.03m；关键代理姿态与下肢Core保护。阶段用于权限，未冒称新神经阶段学习。ChangE音乐非配对。'))
        np.savez_compressed(out/'arrays.npz',**arrays)
    rt.c.clear_all();rt.check();rt.close();result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,records=records,data_sha256=sha(Path(cfg['data'])/'arrays.npz'),source_weights_and_context_frozen=True,phase_neural_input=False,optimizer_steps=0,whole_song_generations=0,operational_qualified=False,can_route=False,native24='NOT_EVALUABLE_22_OF24',contact_truth='UNKNOWN');write(out/'result.json',result)
    if replay:
        exact_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');old_result=read(Path(replay)/'result.json');old_result.pop('code_commit');result.pop('code_commit');assert result==old_result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--controls',action='store_true');p.add_argument('--replay-from');a=p.parse_args();run(a.out,a.controls,a.replay_from)
