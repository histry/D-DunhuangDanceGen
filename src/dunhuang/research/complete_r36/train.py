"""One fixed main candidate: Context task correction, not an ablation sweep."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,itertools,subprocess
from pathlib import Path
import numpy as np
import torch
from dunhuang.research.coverage_r34.common import Runtime,population_data,full_packet,digest,sample
from dunhuang.research.coverage_r34.context_contract import context
from dunhuang.research.coverage_r34.context_train import checkpoint,restore_context
from dunhuang.research.transition_r35.train import targets
from dunhuang.research.control_r33.common import source
from dunhuang.research.content_r32.common import core_body
from .core import read,write,sha

def schedule(cases,seed,steps=160):
    rng=np.random.default_rng(seed);grid=list(itertools.product((999,519,119,19),('actual','q_sample'),('none','general'),(0,1,2),('both','no_left','no_right'),((64,160),(52,172),(80,168))))
    order=rng.permutation(len(grid));rows=[]
    for i in range(steps):
        t,state,src,provenance,mask,gap=grid[int(order[i%len(order)])]
        rows.append(dict(step=i,task='change' if i%4==3 else 'paired',t=t,state=state,source=src,provenance=provenance,mask=mask,gap=list(gap),seed=seed+i,
            cases=[int(x) for x in rng.choice(cases,4,replace=False)]))
    return rows

def run(cfg,root,out,replay=None):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Fresh training output required')
    out.mkdir(parents=True);r35=read('src/dunhuang/research/transition_r35/config.json');rt=Runtime(r35,context_training=True)
    parent=Path(cfg['r35']);restore_context(rt,parent/'context/context.pt');initial=digest(rt.c.context_branch)
    _,_,data,z=population_data();cg=np.load(Path(r35['r34'])/'data/arrays.npz')
    parent_states=np.load(parent/'context/initial_states.npz');cases=parent_states['cases'].tolist();rows=schedule(cases,cfg['seed'])
    states={};cache_noise=[]
    # Actual states use the candidate's planned-reference contract. No old cache is relabelled.
    for group in range(4):
        ids=cases[group*4:group*4+4];_,y,music=targets(rt,data,ids);cp=context(y,[64,160]);cp.provenance[:]=2
        pred,nr,tr=sample(rt,music,context_packet=cp,seeds=[cfg['seed']+c for c in ids],trace=True)
        states['paired'+str(group)]=np.stack([tr['states'][tr['times'].index(t)] for t in (999,519,119,19)])
        states['paired'+str(group)+'_output']=pred.cpu().numpy();cache_noise.append(nr)
    np.savez_compressed(out/'states.npz',**states)
    if replay:
        old=read(Path(replay)/'result.json')
        for name,h in old['files'].items():assert sha(Path(replay)/name)==h
        assert initial==old['initial_digest'] and cache_noise==old['cache_noise']
        saved=np.load(Path(replay)/'states.npz')
        for k,v in states.items():assert np.array_equal(v,saved[k])
    def loss(row):
        torch.manual_seed(row['seed']);torch.cuda.manual_seed_all(row['seed']);rt.c.clear_all()
        batch=4;times=torch.full((batch,),row['t'],device='cuda',dtype=torch.long);ti=(999,519,119,19).index(row['t'])
        if row['task']=='paired':
            selected,y,music=targets(rt,data,row['cases']);table={c:states['paired'+str(i//4)][ti,i%4] for i,c in enumerate(cases)}
            xt=torch.tensor(np.stack([table[c] for c in row['cases']]),device='cuda') if row['state']=='actual' else rt.d.q_sample(y,times,torch.randn_like(y))
            source_packet=source([core_body(r) for r in selected],'aist',uids=['aist_'+str(c) for c in row['cases']]) if row['source']=='general' else None
            ctxvalues=y
            if row['provenance']==1:
                ctxvalues=torch.tensor(np.stack([states['paired'+str(cases.index(c)//4)+'_output'][cases.index(c)%4] for c in row['cases']]),device='cuda')
            partial=False
        else:
            body=torch.tensor(cg['body'][[0,1,0,1]],device='cuda');music=torch.tensor(z['music'][[0,0,1,1]],device='cuda')
            y=torch.full((4,256,139),float('nan'),device='cuda');y[:,:,7:]=body
            xt=torch.tensor(parent_states['change'][ti],device='cuda')
            if row['state']=='q_sample':
                alpha=rt.d.alphas_cumprod[row['t']];xt=xt.clone();xt[:,:,7:]=alpha.sqrt()*body+(1-alpha).sqrt()*torch.randn_like(body)
            source_packet=full_packet(cg['body'][[0,1,0,1]],['female_FeiTian','male_pipa_1']*2)
            ctxvalues=torch.tensor(parent_states['change_output'],device='cuda') if row['provenance']==1 else y
            partial=True
        if source_packet is not None:rt.c.set_source(source_packet)
        with torch.no_grad():teacher=rt.d.model(xt,music,None,times,cond_drop_prob=0)
        cp=context(ctxvalues,row['gap'],row['mask'],row['seed'],partial_body=partial);cp.provenance[:]=row['provenance'];rt.c.set_context(cp)
        pred=rt.d.model(xt,music,None,times,cond_drop_prob=0);rt.c.clear_all()
        # All actually observed body frames, including read/edge regions; no outside-Core body teacher.
        residual=pred[:,:,7:]-y[:,:,7:];parts=dict(body=residual.square().mean(),D1=torch.diff(residual,dim=1).square().mean(),D2=torch.diff(residual,n=2,dim=1).square().mean())
        root=(pred[:,:,4:7]-y[:,:,4:7]).square().mean() if row['task']=='paired' else (pred[:,:,:7]-teacher[:,:,:7]).square().mean()
        parts['root_or_unknown_header_teacher']=root
        return parts['body']+.1*parts['D1']+.02*parts['D2']+.1*root,{k:float(v.detach()) for k,v in parts.items()}
    history=[];observations=[];snapshots=(0,79,159)
    if not replay:
        (out/'snapshots').mkdir();opt=torch.optim.AdamW(rt.c.context_branch.parameters(),lr=1e-4)
        for row in rows:
            if row['step'] in snapshots:checkpoint(rt,out/'snapshots'/f"{row['step']}_pre.pt",cfg,row['step'])
            opt.zero_grad(set_to_none=True);value,parts=loss(row)
            torch.use_deterministic_algorithms(True)
            try:value.backward()
            finally:torch.use_deterministic_algorithms(False)
            gradient=float(torch.sqrt(sum(p.grad.square().sum() for p in rt.c.context_branch.parameters() if p.grad is not None)))
            assert np.isfinite(gradient) and gradient>0 and torch.isfinite(value)
            torch.nn.utils.clip_grad_norm_(rt.c.context_branch.parameters(),1.,error_if_nonfinite=True);opt.step()
            history.append(dict(row,loss=float(value.detach()),parts=parts))
            if row['step'] in snapshots:
                checkpoint(rt,out/'snapshots'/f"{row['step']}_post.pt",cfg,row['step']+1);observations.append(dict(step=row['step'],loss=float(value.detach()),gradient=gradient))
            if (row['step']+1)%40==0:print('R36_CONTEXT',row['step']+1,float(value.detach()),flush=True)
        checkpoint(rt,out/'context.pt',cfg,len(rows));steps=len(rows)
    else:
        from dunhuang.research.state_r27.common import forbid_optimizers
        forbid_optimizers()
        for item in old['observations']:
            restore_context(rt,Path(replay)/'snapshots'/f"{item['step']}_pre.pt");rt.c.context_branch.zero_grad(set_to_none=True)
            value,parts=loss(rows[item['step']]);torch.use_deterministic_algorithms(True)
            try:value.backward()
            finally:torch.use_deterministic_algorithms(False)
            gradient=float(torch.sqrt(sum(p.grad.square().sum() for p in rt.c.context_branch.parameters() if p.grad is not None)))
            assert float(value.detach())==item['loss'] and gradient==item['gradient']
        restore_context(rt,Path(replay)/'context.pt');steps=0;history=old['history'];observations=old['observations']
    final=digest(rt.c.context_branch);rt.c.context_branch.zero_grad(set_to_none=True);rt.check();rt.close()
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),initial_digest=initial,final_digest=final,
        optimizer_steps=steps,registered_steps=160,history=history,observations=observations,cache_noise=cache_noise,
        task_contract='ALL_OBSERVED_BODY_TARGETS; INDEPENDENT_FACTOR_GRID; ACTUAL_GENERATED_CONTEXT; PLANNED_IS_REFERENCE_NOT_TRUTH',
        cache_policy='INITIAL_CANDIDATE_ACTUAL_PLANNED_PAIRED_STATES; FROZEN_PARENT_CHANGE_STATES; NOT_FULLY_ON_POLICY',
        cross_source_planned_truth=False,unknown_ChangE_Root_C4_supervised=False,additional_ablation_calls=0,operational_qualified=False,
        files=old['files'] if replay else {str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.suffix in ('.pt','.npz')})
    if replay:assert final==old['final_digest']
    write(out/'result.json',result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(read('src/dunhuang/research/complete_r36/config.json'),a.root,a.out,a.replay_from)
