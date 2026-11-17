"""Real same-recording windows with independent target and conditioning contracts."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,subprocess
from pathlib import Path
import numpy as np
import torch
from dunhuang.research.complete_r36.core import read,write,sha,Library
from dunhuang.research.bridge_r38.study import runtime
from dunhuang.research.coverage_r34.common import population_data,sample,digest
from dunhuang.research.coverage_r34.context_train import restore_context
from dunhuang.research.transition_r35.train import targets
from dunhuang.research.contract_r40.change_data import rooted_clip
from .contracts import observed_anchor,window_starts,role_check


def build(out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh real continuous dataset')
    out.mkdir(parents=True);cfg=read(Path(__file__).with_name('config.json'));parent=read('src/dunhuang/research/bridge_r38/config.json');rt=runtime(parent);restore_context(rt,Path(cfg['r38'])/'training/context.pt');rt.c.context_branch.requires_grad_(False);before=digest(rt.c.context_branch);_,_,paired,_=population_data();cal=np.load(Path(cfg['r39'])/'calibration/arrays.npz');scale=cal['root_normalizer_scale'];offset=cal['root_normalizer_offset'];lower=(-1-offset)/scale;upper=(1-offset)/scale;roots=np.load(Path(cfg['r39'])/'root_contract/arrays.npz');rootmeta=read(Path(cfg['r39'])/'root_contract/result.json');lib=Library(read('src/dunhuang/research/complete_r36/config.json')['catalog']);records=[];spec=[]
    for uid in cfg['change_train_recordings']+cfg['change_held_recordings']:
        train=uid in cfg['change_train_recordings'];rr=next(r for r in rootmeta['records'] if r['source_uid']==uid)
        for start in window_starts(len(lib.rotations[uid]),train):spec.append(dict(key=uid+'_'+str(start),recording_uid=uid,role='train' if train else 'held',domain='ChangE',start=start,raw_sha256=rr['raw_sha256']))
    ids=read(Path(cfg['r38'])/'training/training.json')['paired_train_cases']
    for c in ids+parent['paired_evaluation_cases']:spec.append(dict(key='paired_'+str(c),recording_uid='paired_'+str(c),case=c,role='train' if c in ids else 'held',domain='AIST++'))
    arrays=dict(skeleton_J22=rt.s.J[0,:22].cpu().numpy(),parents=np.array(rt.s.parents[:22]),root_normalizer_scale=scale,root_normalizer_offset=offset);exclusions=[]
    for k in range(0,len(spec),4):
        batch=spec[k:k+4];music=[];truth=[];raws=[];supported=[]
        for r in batch:
            if r['domain']=='AIST++':
                _,y,m=targets(rt,paired,[r['case']]);raw=rt.n.unnormalize(y.clone())[0].cpu().numpy();music.append(m[0].cpu().numpy());truth.append(raw);raws.append(raw);supported.append(True)
            else:
                R=lib.rotations[r['recording_uid']][r['start']:r['start']+256];root,gauge=rooted_clip(roots[r['recording_uid']+'_displacement'],r['start'],256,lower,upper);raw=np.zeros((256,139),np.float32);raw[:,7:]=R[...,:2,:].reshape(256,132);raw[:,4:7]=root;music.append(np.zeros((256,35),np.float32));truth.append(raw);raws.append(raw);supported.append(gauge['model_box_supported']);r['root_gauge']=gauge
        m=torch.tensor(np.stack(music),device='cuda');seeds=[cfg['seed']+k+j for j in range(len(batch))];base,nr,_=sample(rt,m,seeds=seeds);base_raw=rt.n.unnormalize(base.clone()).cpu().numpy()
        for j,r in enumerate(batch):
            raw=raws[j].copy()
            if r['domain']=='ChangE':raw[:,:4]=base_raw[j,:,:4]
            try:anchor_raw=observed_anchor(raw,base_raw[j],cfg['gap'],supported[j])
            except ValueError as error:
                if str(error)!='Ambiguous SO3 chart':raise
                exclusions.append(dict(r,reason='FROZEN_R39_SO3_CHART_AMBIGUOUS',selection='OBSERVED_BOUNDARY_GEOMETRY_ONLY_NOT_MODEL_QUALITY'));continue
            anchor=rt.n.normalize(torch.tensor(anchor_raw[None],device='cuda',dtype=torch.float32))[0].cpu().numpy();target=rt.n.normalize(torch.tensor(raw[None],device='cuda',dtype=torch.float32))[0].cpu().numpy();a_round=rt.n.unnormalize(torch.tensor(anchor[None],device='cuda'))[0].cpu().numpy();key=r['key'];arrays[key+'_truth']=raw;arrays[key+'_target']=target;arrays[key+'_anchor']=anchor;arrays[key+'_anchor_raw']=a_round;arrays[key+'_music']=music[j]
            records.append(dict(r,root_position_observed=supported[j],root_derivatives_observed=True,contact_observed=False,contact_proxy_available=r['domain']=='AIST++',paired_music=r['domain']=='AIST++',conditioning_contact_observed=False,source_interval=[r.get('start',0),r.get('start',0)+256],gap=cfg['gap'],historical_exposure=True,semantic_event_truth='UNKNOWN',seed=seeds[j]));print('R42_DATA',key,r['role'],supported[j],flush=True)
        np.savez_compressed(out/'arrays.npz',**arrays)
    assert digest(rt.c.context_branch)==before;rt.check();rt.close();isolation=role_check(records);result=dict(records=records,exclusions=exclusions,registered_windows=len(spec),isolation=isolation,arrays_sha256=sha(out/'arrays.npz'),parent_root_contract_sha256=sha(Path(cfg['r39'])/'root_contract/arrays.npz'),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),hidden_gap_read_by_anchor=False,change_music='ZERO35_NO_PAIRED_MUSIC_LOSS',change_contact_truth='UNKNOWN_MASKED',body_observed=True,unsupported_root_positions_excluded_from_position_loss_only=True,optimizer_steps=0,operational_qualified=False);write(out/'result.json',result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();build(a.out)
