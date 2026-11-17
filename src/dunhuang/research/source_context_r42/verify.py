"""Fresh-process payload, checkpoint, lineage and frozen-parent audits."""
import argparse,subprocess
from pathlib import Path
import numpy as np
import torch
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.coverage_r34.common import same_arrays,forbid_optimizers,population_data
from .contracts import exact_arrays


def audit(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Fresh verification')
    forbid_optimizers();checks=[]
    for a,b in [('data_v2/arrays.npz','data_replay/arrays.npz'),('teacher/arrays.npz','teacher_replay/arrays.npz'),('diagnostic/arrays.npz','diagnostic_replay/arrays.npz'),('main_generation/arrays.npz','main_generation_replay/arrays.npz'),('real_generation/arrays.npz','real_generation_replay/arrays.npz'),('aist_generation/arrays.npz','aist_generation_replay/arrays.npz')]:
        exact_arrays(root/a,root/b);checks.append(dict(first=a,replay=b,arrays=len(np.load(root/a).files)))
    updates=[]
    for name in ('main','real','aist'):
        meta=read(root/name/'training.json');assert meta['optimizer_steps']==240 and meta['initial_digest']!=meta['final_digest']
        replay=read(root/(name+'_replay')/'training.json');assert replay['optimizer_steps']==0 and replay['observations']==meta['observations']
        for step in (0,120,239):
            pre=torch.load(root/name/f'step{step}_pre.pt',map_location='cpu',weights_only=True)['state'];post=torch.load(root/name/f'step{step}_post.pt',map_location='cpu',weights_only=True)['state'];norm=float(sum((post[k]-pre[k]).square().sum() for k in pre).sqrt());assert norm>0;updates.append(dict(arm=name,step=step,update_norm=norm,loss_and_gradient_norm_exact=True))
    parents=[]
    for name in ('bridge_r38_01','stabilize_r39_01','contract_r40_01','paired_event_r41_01'):
        p=Path('outputs')/name;m=read(p/'artifacts_manifest.json')
        for r in m['files']:assert sha(p/r['path'])==r['sha256'],(name,r['path'])
        parents.append(dict(root=name,files=len(m['files']),manifest_sha256=sha(p/'artifacts_manifest.json')))
    records=read(root/'data_v2/result.json')['records'];train={r['recording_uid'] for r in records if r['role']=='train' and r['domain']=='ChangE'};held={r['recording_uid'] for r in records if r['role']=='held' and r['domain']=='ChangE'};assert len(train)==len(held)==4 and not train&held
    t=read(root/'teacher/result.json');assert all(next(r for r in records if r['key']==key)['role']=='train' for key in t['training_keys'])
    # Paired case numbers are cache identities; bind the actual recording metadata separately.
    rows,_,_,_=population_data();lineage=[];original=read('outputs/reference_r26_01/data/result.json')['records'];v3={r['recording_uid']:r['role'] for r in read('outputs/paired_event_r41_01/data/manifest.json')['sources']}
    for r in records:
        if r['domain']!='AIST++':continue
        source=[x for x in rows if x['domain']=='aist' and x['case']==r['case']][-1];origin=next(x for x in original if x['case']==r['case']);assert origin['role']=='train' and v3[origin['recording_uid']]=='train';lineage.append(dict(case=r['case'],role=r['role'],recording_uid=origin['recording_uid'],source_group=origin['source_group'],data_v3_role=v3[origin['recording_uid']],raw_motion=origin['raw_motion'],source_clock=origin['clock'],metadata={k:v for k,v in source.items() if k!='data'},identity_scope='FROZEN_PARENT_TRAIN_CACHE_BOUND_TO_DATA_V3_NOT_NEW_BLIND_TEST'))
    for viewer in sorted(root.glob('viewer*')):
        if viewer.is_dir():
            for r in read(viewer/'protocol.json')['files']:assert sha(viewer/r['name'])==r['sha256']
    from dunhuang.research.bridge_r38.study import runtime
    from .train import assemble
    rt=runtime(read('src/dunhuang/research/bridge_r38/config.json'));z=np.load(root/'data_v2/arrays.npz');rr=[r for r in records if r['domain']=='ChangE' and r['role']=='train'][:4];a,m,ref,cp=assemble(rt,z,rr);altered={}
    for r in rr:
        key=r['key']
        for suffix in ('_anchor','_music','_truth'):altered[key+suffix]=z[key+suffix].copy()
        body=altered[key+'_truth'][104:152,7:].reshape(48,22,6).copy();altered[key+'_truth'][104:152,7:]=body[::-1].reshape(48,132);altered[key+'_truth'][104:152,4:7]+=99
    aa,mm,rf,cc=assemble(rt,altered,rr);assert torch.equal(a,aa) and torch.equal(m,mm)
    for key in ref.values:assert torch.equal(ref.values[key],rf.values[key]),key
    assert torch.equal(cp.encoded_input(),cc.encoded_input());parameters=sum(p.numel() for p in rt.c.context_branch.parameters());rt.c.clear_all();rt.check();rt.close()
    assert (root/'receipts/p0_browser_acceptance.json').stat().st_mtime < (root/'main/protocol.json').stat().st_mtime
    assert (root/'receipts/main_browser_acceptance.json').stat().st_mtime < (root/'real/protocol.json').stat().st_mtime
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),arrays=checks,updates=updates,parents=parents,context_parameters=parameters,hidden_body_and_root_intervention_does_not_change_any_condition=True,change_train_recordings=sorted(train),change_held_recordings=sorted(held),aist_case_lineage=lineage,teacher_train_only=True,p0_delivery_preceded_main_training=True,main_delivery_preceded_controls=True,optimizer_steps=0,whole_song_generations=0,operational_qualified=False);write(out,result);print('R42_INDEPENDENT_VERIFIED',len(checks),len(updates),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();audit(a.root,a.out)
