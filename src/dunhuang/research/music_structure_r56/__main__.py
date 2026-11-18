def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['main','controls','verify','render','blind']);p.add_argument('--out',required=True);p.add_argument('--source');a=p.parse_args()
 if a.mode in ('main','controls'):
  from .common import ROOT,read
  if a.mode=='controls':assert read(ROOT/'receipts/main_browser.json')['passed']
  from .study import run
  run(a.out,a.mode)
 elif a.mode=='verify':
  from .study import verify
  verify(a.out,a.source)
 else:
  from .render import run
  run(a.source,a.out,a.mode=='blind')
if __name__=='__main__':main()
