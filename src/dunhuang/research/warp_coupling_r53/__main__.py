def main():
 import argparse
 from .common import ROOT,read
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['preview','main','controls','examples','verify','release','recheck']);p.add_argument('--out',required=True);p.add_argument('--without-controls',action='store_true');a=p.parse_args()
 if a.mode in ('preview','main','controls'):
  from .study import run
  run(a.out,a.mode)
 elif a.mode=='examples':
  from .viewer import examples
  examples(a.out)
 else:
  from . import verify
  if a.mode=='verify':verify.run(a.out,not a.without_controls)
  else:getattr(verify,a.mode)(a.out)
if __name__=='__main__':main()
