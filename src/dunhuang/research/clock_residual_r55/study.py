"""Same R41 capacity and strict R54 population; one local residual loss everywhere."""
from .common import *
from dunhuang.research.residual_sdmr_r54.learn import Study as ParentStudy

class Study(ParentStudy):
 def __init__(self):
  super().__init__();self.split=partition(self.meta)
  from .population import independent_candidates
  self.meta['tasks'],self.population_audit=independent_candidates(self.meta,self.split,CFG['seed'])
 def raw(self,net,t,mode='conditional',control='actual'):
  ids=t['unconditional_candidates'] if mode=='unconditional' else t['candidates'];q=t['donor'] if control=='wrong_music' else t['query'];m,_,mv,_=self.tokens(net,[q]);_,d,_,dv=self.tokens(net,ids)
  if control=='music_zero':
   inputs=self.inputs([q]);inputs[0]=torch.zeros_like(inputs[0]);m,_,mv,_=net.content(*inputs)
  if control=='time_reverse':d=d.flip(1);dv=dv.flip(1)
  return diagonal(m,d,mv,dv)
 def clock(self,net,row,shift,control='actual'):
  inputs=shifted_inputs(self.inputs([row]),shift)
  if control in ('music_zero','both_zero'):
   inputs=list(inputs);inputs[0]=torch.zeros_like(inputs[0])
   if control=='both_zero':inputs[1]=torch.zeros_like(inputs[1])
  if control=='time_reverse':inputs=list(inputs);inputs[1]=inputs[1].flip(1);inputs[2]=inputs[2].flip(1)
  return clock_scores(*net.content(*inputs))[0]
 def clock_base(self,row,shift,control='actual'):
  key=('clock',row,shift,control)
  if key not in self.original:
   with torch.no_grad():self.original[key]=self.clock(self.base,row,shift,control).detach()
  return self.original[key]
 def objective(self,net,task,arm,step=0):
  mode='unconditional' if arm=='Unconditional' else 'conditional';h=task['unconditional_heuristic'] if mode=='unconditional' else task['heuristic'];labels=task['unconditional_positive'] if mode=='unconditional' else task['shuffled_positive'] if arm=='Shuffled' else task['positive']
  raw=self.raw(net,task,mode);loss=ranking_loss(raw,self.base_scores(task,mode),h,labels)
  if arm!='NoClock':
   shift=CFG['offsets'][(step+int(task['query']))%len(CFG['offsets'])];truth=CFG['offsets'].index(-shift)
   if arm=='Shuffled':truth=(truth+1)%len(CFG['offsets'])
   p=np.arange(len(CFG['offsets']))==truth;raw=self.clock(net,task['query'],shift);loss=loss+CFG['clock_weight']*ranking_loss(raw,self.clock_base(task['query'],shift),np.zeros(len(p)),p)
  return loss

def schedule(study,steps):
 rng=np.random.default_rng(CFG['seed']);by={g:[i for i,t in enumerate(study.meta['tasks']) if t['role']=='train' and t['group']==g and t['training_admitted']] for g in study.split['optimization']};by={g:v for g,v in by.items() if v};order=[]
 for step in range(steps):
  batch=[]
  for g in rng.choice(sorted(by),CFG['batch_groups'],replace=False):
   options=[i for i in by[g] if study.meta['tasks'][i]['frames']==[90,150,240][step%3]] or by[g];batch.append(int(rng.choice(options)))
  order.append(batch)
 return order

def gradient(net):return np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in net.parameters() if p.grad is not None])

def fit(s,net,order,out,arm):
 opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=CFG['lr']);initial=digest(net);save(net,out/'initial.pt',0);history=[]
 for step,ids in enumerate(order):
  seeded(CFG['seed']+step);net.train();opt.zero_grad(set_to_none=True);loss=torch.stack([s.objective(net,s.meta['tasks'][i],arm,step) for i in ids]).mean();loss.backward();g=gradient(net);norm=float(np.linalg.norm(g));assert np.isfinite(norm) and norm>0
  if step in (0,len(order)//2,len(order)-1):save(net,out/f'step{step}_pre.pt',step);np.save(out/f'gradient{step}.npy',g)
  torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
  if step in (0,len(order)//2,len(order)-1):save(net,out/f'step{step}_post.pt',step+1)
  history.append(dict(step=step,tasks=ids,loss=float(loss.detach()),gradient=norm))
  if (step+1)%40==0:print('R55_TRAIN',arm,step+1,history[-1]['loss'],flush=True)
 save(net,out/'model.pt',len(order));write(out/'schedule.json',order);write(out/'history.json',history);write(out/'result.json',dict(arm=arm,updates=len(order),initial_digest=initial,final_digest=digest(net),model_sha256=sha(out/'model.pt'),total_parameters=sum(p.numel() for p in net.parameters()),trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),schedule_sha256=sha(out/'schedule.json'),parent_data_sha256=sha(Path(CFG['data'])/'manifest.json'),code=binding(),commit=fingerprint(),partition=s.split,population_audit=s.population_audit,Val_used_for_selection=False,can_route=False))

def train(out,arm):
 assert read(ROOT/'qualification/gate.json')['admitted']
 if arm!='Conditional':assert read(ROOT/'receipts/main_browser.json')['passed']
 out=fresh(out);s=Study();fit(s,s.model(),schedule(s,CFG['steps']),out,arm)
