"""Fresh-process selected-gradient replay and immutable-output audit; optimizer construction prohibited."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.coverage_r34.common import forbid_optimizers,same_arrays
from .train import initial,seeded,objective,gradient_readout,digest
from .population import Population


def run(data,root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Fresh independent verification')
    out.mkdir(parents=True);cfg=read('src/dunhuang/research/paired_event_r41/config.json');pop=Population(data);forbid_optimizers();checks=[]
    for folder in ('checks/train_fit','main'):
        directory=root/folder
        if folder=='main' and not read(directory/'result.json')['admitted']:continue
        history=read(directory/'history.json');schedule=read(directory/'schedule.json');tasks=read(directory/'gradient_tasks.json');model=initial(cfg).train()
        for row in tasks:
            step=row['step'];state=torch.load(directory/f'step{step}_pre.pt',map_location='cuda',weights_only=True);model.load_state_dict(state['state']);model.zero_grad(set_to_none=True);seeded(cfg['seed']+step);loss,parts,metadata=objective(model,pop,schedule[step],cfg,step);g=gradient_readout(parts,model);loss.backward();grad=float(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None).sqrt());expected=history[step]
            assert float(loss.detach())==expected['loss'] and grad==expected['gradient'];assert {k:float(v.detach()) for k,v in parts.items()}==expected['parts'];assert g==row['gradient_tasks'];assert all(metadata[k]==expected[k] for k in metadata)
            post=torch.load(directory/f'step{step}_post.pt',map_location='cpu',weights_only=True)['state'];pre={k:v.detach().cpu() for k,v in state['state'].items()};update=float(sum((post[k]-pre[k]).square().sum() for k in pre).sqrt());assert update>0;checks.append(dict(folder=folder,step=step,loss=float(loss.detach()),gradient=grad,gradient_readouts_exact=True,update_norm=update,optimizer_steps=0))
        if folder=='main':
            meta=read(directory/'result.json');model.load_state_dict(torch.load(directory/'initial.pt',map_location='cuda',weights_only=True)['state']);assert digest(model)==meta['initial_digest'];model.load_state_dict(torch.load(directory/'model.pt',map_location='cuda',weights_only=True)['state']);assert digest(model)==meta['final_digest'];assert meta['initial_digest']!=meta['final_digest']
    arrays={}
    for a,b in [('evaluation/arrays.npz','evaluation_v2/arrays.npz'),('evaluation_v2/arrays.npz','evaluation_v2_replay/arrays.npz'),('checks/synthetic_eval.npz','checks_replay/synthetic_eval.npz'),('checks/fit_eval.npz','checks_replay/fit_eval.npz'),('budget/arrays.npz','budget_replay/arrays.npz')]:
        if (root/a).exists() and (root/b).exists():same_arrays(root/a,root/b);arrays[a+'__'+b]=len(np.load(root/a).files)
    parents=[]
    for name in ('bridge_r38_01','stabilize_r39_01','contract_r40_01'):
        p=Path('outputs')/name;manifest=read(p/'artifacts_manifest.json')
        for row in manifest['files']:assert sha(p/row['path'])==row['sha256'],(name,row['path'])
        parents.append(dict(name=name,files=len(manifest['files']),manifest_sha256=sha(p/'artifacts_manifest.json')))
    source=read(Path(data)/'manifest.json');replay=read(root/'data_replay/manifest.json');assert [r['event_uid'] for r in source['rows']]==[r['event_uid'] for r in replay['rows']]
    for a,b in zip(source['rows'],replay['rows']):same_arrays(a['path'],b['path'])
    for a,b in zip(source['sources'],replay['sources']):
        assert a['source_uid']==b['source_uid'];same_arrays(a['source_arrays']['path'],b['source_arrays']['path'])
    for name in ('viewer','viewer_sources','viewer_final'):
        p=root/name
        if p.exists():
            for row in read(p/'protocol.json')['files']:assert sha(p/row['name'])==row['sha256']
    assert (root/'receipts/p0_browser_acceptance.json').stat().st_mtime<(root/'checks/protocol.json').stat().st_mtime
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),gradient_checks=checks,arrays=arrays,parents=parents,source_event_payloads_rebuilt=len(source['rows']),whole_source_recordings_rebuilt=len(source['sources']),all_source_roles_preserved=True,p0_browser_acceptance_preceded_training=True,optimizer_steps=0,operational_qualified=False,can_route=False,whole_song_generations=0);write(out/'result.json',result);print('R41_INDEPENDENT_VERIFICATION',len(checks),'gradients',len(source['rows']),'event payloads',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.data,a.root,a.out)
