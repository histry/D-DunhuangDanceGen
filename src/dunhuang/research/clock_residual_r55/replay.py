"""Fresh-process exact score/loss/gradient replay with no optimizer construction."""
from .common import *
from .study import Study,gradient
from .qualify import evaluate as synthetic_evaluate
from .evaluate import evaluate,model

def run(out):
 out=fresh(out)
 from dunhuang.research.coverage_r34.common import forbid_optimizers
 forbid_optimizers();s=Study();checks={}
 for kind in ('S1','S2','S3','S4'):
  folder=ROOT/'qualification'/kind;net=s.model();net.load_state_dict(torch.load(folder/'model.pt',map_location=s.device,weights_only=True)['state']);metrics,arrays=synthetic_evaluate(s,net,kind);old=np.load(folder/'scores.npz');assert sorted(arrays)==sorted(old.files)
  for key,value in arrays.items():assert np.array_equal(value,old[key]),(kind,key)
  assert metrics==read(folder/'result.json')['metrics'];checks[kind]='ALL_SYNTHETIC_SCORES_EXACT'
 for arm in ['qualification/real_fit']+[a for a in CFG['arms'] if (ROOT/a/'result.json').exists()]:
  folder=ROOT/arm;history=read(folder/'history.json');order=read(folder/'schedule.json');net=s.model();real_arm='Conditional' if arm=='qualification/real_fit' else arm;verified=[]
  for step in (0,len(order)//2,len(order)-1):
   seeded(CFG['seed']+step);net.load_state_dict(torch.load(folder/f'step{step}_pre.pt',map_location=s.device,weights_only=True)['state']);net.train();net.zero_grad(set_to_none=True);loss=torch.stack([s.objective(net,s.meta['tasks'][i],real_arm,step) for i in order[step]]).mean();loss.backward();assert float(loss.detach())==history[step]['loss'];assert np.array_equal(gradient(net),np.load(folder/f'gradient{step}.npy'));post=torch.load(folder/f'step{step}_post.pt',map_location=s.device,weights_only=True)['state'];assert any(not torch.equal(v,post[k]) for k,v in net.state_dict().items());verified.append(step)
  checks[arm]=verified
 if (ROOT/'evaluation/result.json').exists():
  evaluate(out/'evaluation');old=np.load(ROOT/'evaluation/scores.npz');new=np.load(out/'evaluation/scores.npz');assert old.files==new.files
  for key in old.files:assert np.array_equal(old[key],new[key]),key
  assert sha(ROOT/'evaluation/scores.npz')==sha(out/'evaluation/scores.npz');assert read(ROOT/'evaluation/result.json')==read(out/'evaluation/result.json');checks['evaluation']='ALL_ARRAYS_AND_SUMMARY_EXACT'
 # Parent source tensors are immutable and rehashed by ParentStudy at construction.
 write(out/'result.json',dict(checks=checks,optimizer_calls=0,parent_manifest_sha256=sha(Path(CFG['data'])/'manifest.json'),parent_arrays_sha256=sha(Path(CFG['data'])/'arrays.npz'),commit=fingerprint(),code=binding(),can_route=False));print('R55_REPLAY_EXACT',checks,flush=True)
