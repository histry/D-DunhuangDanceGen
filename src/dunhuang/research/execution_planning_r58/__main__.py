def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['main','controls','verify','render','support','recovery','verify_recovery']);p.add_argument('--out',required=True);p.add_argument('--source');a=p.parse_args()
 if a.mode=='render':
  from .render import run
  run(a.source,a.out)
 elif a.mode in ('recovery','verify_recovery'):
  from .recovery import run,verify
  verify(a.out,a.source) if a.mode=='verify_recovery' else run(a.out)
 else:
  from .study import run,verify,support_audit
  if a.mode=='verify':verify(a.out,a.source)
  elif a.mode=='support':support_audit(a.out)
  else:run(a.out,a.mode)
if __name__=='__main__':main()
