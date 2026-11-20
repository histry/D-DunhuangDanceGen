import argparse
from dunhuang.research.sdmr_warp_rank import study
from dunhuang.research.sdmr_warp_rank.common import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['phase_audit','main','controls','verify'])
    p.add_argument('--out');p.add_argument('--replay-mode',choices=['phase_audit','main','controls'])
    a=p.parse_args()
    if a.mode=='verify':
        if not a.out or not a.replay_mode:p.error('verify requires --out and --replay-mode')
        study.verify(a.replay_mode,a.out)
    elif a.mode=='phase_audit':
        from dunhuang.research.sdmr_warp_rank.phase_audit import run
        run(a.out)
    elif a.mode=='main':study.run_main(a.out)
    else:study.run_controls(a.out)

if __name__=='__main__':main()
