"""R65 isolated evaluation sidecar; never invokes Planner or neural training."""
import argparse
from dunhuang.research.quality_r65 import study

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','controls','verify','evaluate'])
    p.add_argument('--reviews');p.add_argument('--roster');p.add_argument('--out')
    a=p.parse_args()
    if a.mode=='prepare':study.prepare()
    elif a.mode=='controls':study.controls()
    elif a.mode=='verify':study.verify()
    else:
        if not a.out:p.error('evaluate requires a fresh --out; --reviews and --roster are optional')
        study.evaluate(a.reviews,a.roster,a.out)

if __name__=='__main__':main()
