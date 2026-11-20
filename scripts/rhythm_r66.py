import argparse
from dunhuang.research.rhythm_r66 import study
from dunhuang.research.rhythm_r66.common import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['main','controls','verify'])
    p.add_argument('--source');p.add_argument('--out');p.add_argument('--replay-mode',choices=['main','controls'],default='main')
    a=p.parse_args()
    if a.mode=='verify':
        if not a.out:p.error('verify requires a fresh --out')
        study.verify(a.source or ROOT/a.replay_mode,a.out,a.replay_mode)
    else:study.compute(a.mode,a.out)

if __name__=='__main__':main()
