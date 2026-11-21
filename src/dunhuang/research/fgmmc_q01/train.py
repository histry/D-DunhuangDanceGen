import sys,time
import numpy as np
import torch
from torch.nn import functional as F
from .common import *
from .data import register,Store,windows,statistics,donors
from .network import initial,digest,tensors,clocks

def deterministic():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
def prepare():
 out=fresh(ROOT/'data');reg=register();rows=windows(reg);write(out/'registration.json',reg);write(out/'windows.json',rows);store=Store(reg);normal=statistics(store,rows);save_arrays(out/'normalizer.npz',normal);write(out/'donors.json',donors(store,rows));write(out/'clocks.json',clocks(CFG['frames'])[1]);rng=np.random.default_rng(CFG['seed']);groups=sorted({r['group'] for r in rows if r['split']=='train'});schedule=[]
 for step in range(CFG['steps']):
  selected=[]
  for group in rng.choice(groups,CFG['batch'],replace=False):selected.append(str(rng.choice([r['uid'] for r in rows if r['split']=='train' and r['group']==group])))
  schedule.append(selected)
 write(out/'schedule.json',schedule);seal(out);print('FG_DATA',reg['summary'],len(rows),flush=True)
def fit(name):
 deterministic();verify(ROOT/'data');out=fresh(ROOT/name);reg=read(ROOT/'data/registration.json');rows=read(ROOT/'data/windows.json');byid={r['uid']:r for r in rows};donor=read(ROOT/'data/donors.json');store=Store(reg);normal=dict(np.load(ROOT/'data/normalizer.npz'));net=initial().cuda();before=digest(net);optimizer=torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=.0001);history=[];start=time.monotonic()
 for step,ids in enumerate(read(ROOT/'data/schedule.json')):
  clips=[]
  for uid in ids:
   r=byid[uid];assert r['split']=='train';other=byid[donor[uid]['donor_uid']];assert other['split']=='train'
   if name=='ShuffledPair':
    clips += [store.crop(r,delta=d,donor=other,normal=normal) for d in CFG['clock_offsets']]+[store.crop(r,normal=normal)]
   else:clips += [store.crop(r,delta=d,normal=normal) for d in CFG['clock_offsets']]+[store.crop(r,donor=other,normal=normal)]
  scores=net(*tensors(clips),constrained=name!='OrdinaryPostFilter').reshape(len(ids),6);loss=F.cross_entropy(scores/CFG['temperature'],torch.full((len(ids),),2,device='cuda',dtype=torch.long));optimizer.zero_grad(set_to_none=True);loss.backward();gradient=float(torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True));optimizer.step();history.append(dict(step=step+1,loss=float(loss.detach()),gradient=gradient))
  if (step+1)%25==0:print('FG_TRAIN',name,step+1,float(loss.detach()),flush=True)
 save_arrays(out/'weights.npz',{k:v.detach().cpu().numpy() for k,v in net.state_dict().items()});write(out/'history.json',history);write(out/'training.json',dict(model=name,steps=CFG['steps'],initial_digest=before,final_digest=digest(net),parameters=sum(p.numel() for p in net.parameters()),elapsed_seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),train_music_groups=reg['summary']['train']['music_groups'],data_manifest_sha256=sha(ROOT/'data/manifest.json'),selection='FINAL_STEP_ONLY_NO_DEV_SELECTION',training_domain=CFG['training_domain'],heldout_used_in_optimizer=False,can_route=False));seal(out)
 print('FG_TRAIN_READY',name,flush=True)
def load(name):
 deterministic();verify(ROOT/name);net=initial();z=np.load(ROOT/name/'weights.npz');net.load_state_dict({k:torch.as_tensor(z[k]) for k in z.files});net.cuda().eval().requires_grad_(False);assert digest(net)==read(ROOT/name/'training.json')['final_digest'];return net
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 else:fit(sys.argv[1])
