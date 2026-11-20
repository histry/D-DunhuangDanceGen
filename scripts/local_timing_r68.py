import argparse
from dunhuang.research.local_timing_r68 import study
def main():
    p=argparse.ArgumentParser(description='R68 fixed local intervention, no training or Planner change')
    p.add_argument('mode',choices=['preview','controls','verify'])
    p.add_argument('--out');p.add_argument('--replay-mode',choices=['preview','controls'])
    a=p.parse_args()
    if a.mode=='verify':
        if not a.out or not a.replay_mode:p.error('verify requires fresh --out and --replay-mode')
        study.verify(a.replay_mode,a.out)
    elif a.mode=='preview':study.run_preview(a.out)
    else:study.run_controls(a.out)
if __name__=='__main__':main()
