import argparse
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['main','render','controls','verify','sdmr','verify_sdmr']);p.add_argument('--out',required=True);p.add_argument('--source');a=p.parse_args()
 if a.mode=='render':
  from .view import movie
  movie(a.source,a.out)
 elif a.mode in ('sdmr','verify_sdmr'):
  from .sdmr import run,verify
  verify(a.out,a.source) if a.mode=='verify_sdmr' else run(a.out)
 else:
  from .study import main,controls,verify
  if a.mode=='verify':verify(a.out,a.source)
  elif a.mode=='controls':controls(a.out)
  else:main(a.out)
if __name__=='__main__':main()
