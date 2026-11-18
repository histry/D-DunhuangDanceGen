def main():
 import argparse
 from .common import ROOT
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','preview','checks','train','main_view','fit_view','evaluate','verify','release']);p.add_argument('--out',required=True);p.add_argument('--arm',choices=['Conditional','Unconditional','Shuffled']);p.add_argument('--replay-from');a=p.parse_args()
 if a.mode=='prepare':
  from .data import build
  build(a.out)
 elif a.mode in ('preview','main_view','fit_view'):
  from .view import run
  run(a.out,a.mode=='main_view',a.mode=='fit_view')
 elif a.mode in ('checks','train'):
  from .learn import checks,train
  if a.mode=='checks':
   from .common import read
   assert read(ROOT/'receipts/preview_browser.json')['passed'];checks(a.out)
  else:train(a.out,a.arm,a.replay_from)
 elif a.mode=='evaluate':
  from .evaluate import run
  run(a.out)
 else:
  from . import release
  getattr(release,a.mode)(a.out)
if __name__=='__main__':main()
