"""S1/S2 candidate diagnostics; S3/S4 share actual local residual objective."""
from scipy.ndimage import gaussian_filter1d
from .common import *
from .study import Study,fit

def sample(seed,batch,device,clock=False):
 rng=np.random.default_rng(seed);trans=np.random.default_rng(41100);A=trans.normal(size=(8,50))/np.sqrt(8);B=trans.normal(size=(8,274))/np.sqrt(8);z=gaussian_filter1d(rng.normal(size=(batch,150,8)),.65,axis=1);m=[];d=[];offsets=[]
 for k in range(batch):
  shift=int(rng.choice(CFG['offsets'])) if clock else 0;u=z[k,30:120];v=z[k,30+shift:120+shift]
  m.append(u@A);d.append(v@B);offsets.append(-shift)
 m=torch.tensor(np.stack(m),device=device,dtype=torch.float32);d=torch.tensor(np.stack(d),device=device,dtype=torch.float32)
 # Identity and duration are not side inputs; unknown clock channels are masked by EventContent.
 return (m,d,torch.ones_like(d,dtype=torch.bool),torch.ones(m.shape[:2],device=device,dtype=torch.bool)),offsets

def all_diagonal(tokens):
 m,d,mv,dv=tokens;return torch.stack([diagonal(m[k:k+1],d,mv[k:k+1],dv) for k in range(len(m))])

def scores(net,base,inputs,kind):
 rawtokens=net.content(*inputs)
 if kind in ('S1','S2'):return F.normalize(rawtokens[0].mean(1),dim=-1)@F.normalize(rawtokens[1].mean(1),dim=-1).T
 with torch.no_grad():basetokens=base.content(*inputs)
 if kind=='S3':raw=all_diagonal(rawtokens);frozen=all_diagonal(basetokens)
 else:raw=clock_scores(*rawtokens);frozen=clock_scores(*basetokens)
 return residual_scores(raw,frozen,np.zeros(tuple(raw.shape)))

def loss_for(net,base,inputs,truth,kind):
 if kind in ('S1','S2'):return F.cross_entropy(scores(net,base,inputs,kind)/CFG['temperature'],torch.arange(len(inputs[0]),device=inputs[0].device))
 rawtokens=net.content(*inputs)
 with torch.no_grad():basetokens=base.content(*inputs)
 if kind=='S3':raw=all_diagonal(rawtokens);frozen=all_diagonal(basetokens)
 else:raw=clock_scores(*rawtokens);frozen=clock_scores(*basetokens)
 return torch.stack([ranking_loss(a,b,np.zeros(len(a)),np.arange(len(a))==j) for a,b,j in zip(raw,frozen,truth)]).mean()

def evaluate(s,net,kind):
 arrays={};metrics={};net.eval()
 with torch.no_grad():
  for count in ((8,) if kind=='S1' else (32,) if kind=='S2' else (8,32)):
   for control in ('actual','music_zero','time_reverse'):
    values=[]
    for trial,seed in enumerate(CFG['synthetic_eval_seeds']):
     inputs,off=sample(seed,count,s.device,kind=='S4');truth=[CFG['offsets'].index(x) for x in off] if kind=='S4' else list(range(count))
     if control=='music_zero':inputs=list(inputs);inputs[0]=torch.zeros_like(inputs[0])
     if control=='time_reverse':inputs=list(inputs);inputs[1]=inputs[1].flip(1)
     result=scores(net,s.base,inputs,kind).cpu().numpy();arrays[f'{count}_{control}_{trial}']=result;values.extend(hit(row,np.arange(len(row))==j) for row,j in zip(result,truth))
    metrics[f'{count}_{control}']=float(np.mean(values))
 return metrics,arrays

def run(out):
 out=fresh(out);s=Study();results={}
 for k,kind in enumerate(('S1','S2','S3','S4')):
  folder=fresh(out/kind);net=s.model();opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=CFG['lr']);history=[];batch=32 if kind=='S2' else 6 if kind=='S3' else 8
  for step in range(CFG['synthetic_steps']):
   seeded(CFG['seed']+step);inputs,off=sample(CFG['synthetic_train_seed']+k*10000+step,batch,s.device,kind=='S4');truth=[CFG['offsets'].index(x) for x in off] if kind=='S4' else list(range(batch));opt.zero_grad(set_to_none=True);value=loss_for(net,s.base,inputs,truth,kind);value.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();history.append(float(value.detach()))
  save(net,folder/'model.pt',CFG['synthetic_steps']);metrics,arrays=evaluate(s,net,kind);save_arrays(folder/'scores.npz',arrays);write(folder/'result.json',dict(metrics=metrics,loss=history,updates=CFG['synthetic_steps'],model_sha256=sha(folder/'model.pt'),initial_checkpoint=s.meta['binding']['checkpoint_sha256'],same_objective_as_real=kind in ('S3','S4'),diagnostic_only=kind in ('S1','S2')));results[kind]=metrics;print('R55_SYNTHETIC',kind,metrics,flush=True)
 first=[next(i for i,t in enumerate(s.meta['tasks']) if t['role']=='train' and t['group']==g) for g in s.split['optimization'][:8]];folder=fresh(out/'real_fit');net=s.model();fit(s,net,[first]*CFG['fit_steps'],folder,'Conditional');net.eval()
 with torch.no_grad():acc=float(np.mean([hit(s.evaluate(net,s.meta['tasks'][i])['Conditional'],s.meta['tasks'][i]['positive']) for i in first]))
 content=all(results['S3'][f'{n}_actual']>=CFG['synthetic_content_gate'] and results['S3'][f'{n}_actual']>results['S3'][f'{n}_music_zero'] for n in (8,32));clock=all(results['S4'][f'{n}_actual']>=CFG['synthetic_clock_gate'] and results['S4'][f'{n}_actual']>max(results['S4'][f'{n}_music_zero'],results['S4'][f'{n}_time_reverse']) for n in (8,32))
 gate=dict(S0='R54_23_OF_32_UNCHANGED',synthetic=results,real_fit_R1=acc,fit_task_indices=first,S3_pass=content,S4_pass=clock,admitted=bool(content and clock and acc>=CFG['fit_gate']),S1_S2_do_not_gate=True,code=binding(),commit=fingerprint(),can_route=False);write(out/'gate.json',gate);print('R55_QUALIFICATION',gate['admitted'],flush=True)
