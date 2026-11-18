"""Residual pairwise training with the unchanged R41 local encoder capacity."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch.nn import functional as F
from .common import *
from dunhuang.research.paired_event_r41.model import EventContent
from dunhuang.research.paired_event_r41.train import seeded,digest,save,synthetic

class Study:
 def __init__(self):
  self.meta=read(ROOT/'data/manifest.json');assert self.meta['binding']['code']==code_binding();assert sha(ROOT/'data/arrays.npz')==self.meta['binding']['arrays_sha256'];self.z=np.load(ROOT/'data/arrays.npz');self.device='cuda' if torch.cuda.is_available() else 'cpu';self.base=self.model().eval().requires_grad_(False);self.original={};self.cache={};self.token_cache={};self.frozen_inference=False
 def model(self):
  seeded(CFG['seed']);p=sealed('outputs/paired_event_r41_01','main/model.pt');assert sha(p)==self.meta['binding']['checkpoint_sha256'];net=EventContent().to(self.device);net.load_state_dict(torch.load(p,map_location=self.device,weights_only=True)['state']);assert sum(p.numel() for p in net.parameters())==522632
  for name,p in net.named_parameters():p.requires_grad_(name.startswith(('music.','motion.','local_')))
  return net
 def inputs(self,ids):
  for i in ids:
   if i not in self.cache:self.cache[i]=[self.z[p+str(i)] for p in ('M','D','DM','MV')]
  return [torch.tensor(np.stack([self.cache[i][k] for i in ids]),device=self.device) for k in range(4)]
 def tokens(self,net,ids):
  cacheable=net is self.base or self.frozen_inference
  if not cacheable:return net.content(*self.inputs(ids))
  if torch.is_grad_enabled() and net is not self.base:raise ValueError('Inference cache cannot enter training')
  missing=[i for i in dict.fromkeys(ids) if (id(net),i) not in self.token_cache]
  for i in missing:
   # Fixed batch1 arithmetic makes cached scores independent of access history.
   with torch.no_grad():values=net.content(*self.inputs([i]))
   self.token_cache[id(net),i]=[v[0].detach() for v in values]
  return [torch.stack([self.token_cache[id(net),i][k] for i in ids]) for k in range(4)]
 def raw(self,net,t,mode='conditional',control='actual'):
  ids=t['unconditional_candidates'] if mode=='unconditional' else t['candidates'];q=t['donor'] if control=='wrong_music' else t['query'];m,_,mv,_=self.tokens(net,[q]);_,d,_,dv=self.tokens(net,ids);valid=mv[0,::3][None]&dv[:,::3]
  if not valid.any(-1).all():raise ValueError('Genuine diagonal support absent')
  return (((m[0,::3][None]*d[:,::3]).sum(-1)*valid).sum(-1)/valid.sum(-1))
 def base_scores(self,t,mode='conditional',control='actual'):
  k=(t['uid'],mode,control)
  if k not in self.original:
   with torch.no_grad():self.original[k]=self.raw(self.base,t,mode,control).detach()
  return self.original[k]
 def objective(self,net,task,arm):
  mode='unconditional' if arm=='Unconditional' else 'conditional';h=task['unconditional_heuristic'] if mode=='unconditional' else task['heuristic'];labels=task['unconditional_positive'] if mode=='unconditional' else task['shuffled_positive'] if arm=='Shuffled' else task['positive'];raw=self.raw(net,task,mode);res=raw-self.base_scores(task,mode);score=torch.as_tensor(h,device=self.device)+CFG['lambda']*res;p=torch.tensor(labels,device=self.device);margin=(score[p,None]-score[None,~p])/CFG['temperature'];return F.softplus(-margin).mean()+CFG['residual_penalty']*res.square().mean()
 def evaluate(self,net,t,control='actual'):
  with torch.no_grad():base=self.base_scores(t,control=control).cpu().numpy();raw=self.raw(net,t,control=control).cpu().numpy()
  h=np.array(t['heuristic']);return dict(Heuristic=h,Frozen=base,HybridFrozen=h+CFG['lambda']*base,Residual=raw-base,Conditional=h+CFG['lambda']*(raw-base))

def schedule(tasks,steps,seed):
 groups={}
 for i,t in enumerate(tasks):
  if t['role']=='train':groups.setdefault(t['group'],[]).append(i)
 keys=sorted(groups);rng=np.random.default_rng(seed);order=[]
 for step in range(steps):
  chosen=rng.choice(keys,min(CFG['batch_groups'],len(keys)),replace=False);length=CFG['lengths'][step%3];batch=[]
  for g in chosen:
   options=[i for i in groups[g] if tasks[i]['frames']==length] or groups[g];batch.append(int(rng.choice(options)))
  order.append(batch)
 return order

def gradient(net):
 return np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in net.parameters() if p.grad is not None])

def fit(study,net,order,out,arm,replay=None):
 opt=None if replay else torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=CFG['lr']);history=[];watch=(0,len(order)//2,len(order)-1);initial=digest(net)
 if replay:
  from dunhuang.research.coverage_r34.common import forbid_optimizers
  forbid_optimizers();old=Path(replay);h=read(old/'history.json');checks=[]
  for step in watch:
   net.load_state_dict(torch.load(old/f'step{step}_pre.pt',map_location=study.device,weights_only=True)['state']);net.zero_grad(set_to_none=True);value=torch.stack([study.objective(net,study.meta['tasks'][i],arm) for i in order[step]]).mean();value.backward();g=gradient(net);assert float(value.detach())==h[step]['loss'];assert np.array_equal(g,np.load(old/f'gradient{step}.npy'));post=torch.load(old/f'step{step}_post.pt',map_location=study.device,weights_only=True)['state'];assert any(not torch.equal(v,post[k]) for k,v in net.state_dict().items());checks.append(step)
  write(out/'result.json',dict(exact_loss_gradient_steps=checks,optimizer_calls=0,can_route=False));return
 save(net,out/'initial.pt',0)
 for step,ids in enumerate(order):
  seeded(CFG['seed']+step);net.train();opt.zero_grad(set_to_none=True);value=torch.stack([study.objective(net,study.meta['tasks'][i],arm) for i in ids]).mean();value.backward();g=gradient(net);norm=float(np.linalg.norm(g));assert np.isfinite(norm) and norm>0
  if step in watch:save(net,out/f'step{step}_pre.pt',step);np.save(out/f'gradient{step}.npy',g)
  torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();history.append(dict(step=step,tasks=ids,loss=float(value.detach()),gradient=norm))
  if step in watch:save(net,out/f'step{step}_post.pt',step+1)
  if (step+1)%40==0:print('R54_TRAIN',arm,step+1,history[-1]['loss'],flush=True)
 save(net,out/'model.pt',len(order));write(out/'history.json',history);write(out/'schedule.json',order);write(out/'result.json',dict(arm=arm,updates=len(order),initial_digest=initial,final_digest=digest(net),model_sha256=sha(out/'model.pt'),total_parameters=sum(p.numel() for p in net.parameters()),trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),data_sha256=sha(ROOT/'data/manifest.json'),schedule_sha256=sha(out/'schedule.json'),code=code_binding(),val_used_for_selection=False,can_route=False))

def checks(out):
 out=fresh(out);s=Study();net=s.model();opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=CFG['lr'])
 for step in range(CFG['synthetic_steps']):
  m,d,dm,mv,_=synthetic(54000+step,8);mt,dt,*_=net.content(m,d,dm,mv);scores=F.normalize(mt.mean(1),dim=-1)@F.normalize(dt.mean(1),dim=-1).T;value=F.cross_entropy(scores/CFG['temperature'],torch.arange(8,device=s.device));opt.zero_grad();value.backward();opt.step()
 with torch.no_grad():
  m,d,dm,mv,_=synthetic(65054,32);mt,dt,*_=net.content(m,d,dm,mv);test_scores=F.normalize(mt.mean(1),dim=-1)@F.normalize(dt.mean(1),dim=-1).T;acc=float(test_scores.argmax(1).eq(torch.arange(32,device=s.device)).float().mean())
 save(net,out/'synthetic.pt',CFG['synthetic_steps']);save_arrays(out/'synthetic_eval.npz',dict(scores=test_scores.cpu().numpy()));tasks=s.meta['tasks'];first=[]
 for group in s.meta['groups']['train'][:8]:
  first.append(next(i for i,t in enumerate(tasks) if t['role']=='train' and t['group']==group))
 net=s.model();order=[first for _ in range(CFG['fit_steps'])];fit(s,net,order,out,'Conditional');net.eval()
 from dunhuang.research.sdmr_event_rank.scorer import retrieval_metrics
 hits=[retrieval_metrics(s.evaluate(net,tasks[i])['Conditional'],tasks[i]['positive'])['recall_at1'] for i in first];fit_acc=float(np.mean(hits));admitted=acc>=CFG['synthetic_gate'] and fit_acc>=CFG['fit_gate'];write(out/'gate.json',dict(synthetic_held_realizations_accuracy=acc,real_train_fit_R1=fit_acc,fit_task_indices=first,admitted=admitted,train_only=True,synthetic_semantic_truth=False));print('R54_CHECKS',acc,fit_acc,admitted,flush=True)

def train(out,arm,replay=None):
 out=fresh(out);assert read(ROOT/'checks/gate.json')['admitted']
 if arm!='Conditional' and replay is None:assert read(ROOT/'receipts/main_browser.json')['passed']
 s=Study();order=schedule(s.meta['tasks'],CFG['steps'],CFG['seed']);fit(s,s.model(),order,out,arm,replay)
