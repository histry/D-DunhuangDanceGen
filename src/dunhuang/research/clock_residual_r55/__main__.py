def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','preview','qualify','train','main_view','evaluate','verify']);p.add_argument('--out',required=True);p.add_argument('--arm',choices=['Conditional','NoClock','Unconditional','Shuffled']);a=p.parse_args()
 if a.mode=='prepare':
  from .study import Study
  from .common import fresh,write,binding,CFG,sha,Path,fingerprint
  s=Study();write(fresh(a.out)/'binding.json',dict(parent_manifest_sha256=sha(Path(CFG['data'])/'manifest.json'),parent_arrays_sha256=sha(Path(CFG['data'])/'arrays.npz'),partition=s.split,code=binding(),commit=fingerprint(),can_route=False))
 elif a.mode in ('preview','main_view'):
  from .view import run
  run(a.out,a.mode=='main_view')
 elif a.mode=='qualify':
  from .qualify import run
  run(a.out)
 elif a.mode=='train':
  from .study import train
  train(a.out,a.arm)
 elif a.mode=='evaluate':
  from .evaluate import evaluate
  evaluate(a.out)
 else:
  from .replay import run
  run(a.out)
if __name__=='__main__':main()
