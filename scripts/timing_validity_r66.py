"""No Planner, solver or model training entry point exists in R66."""
import argparse
from dunhuang.research.timing_validity_r66 import study
from dunhuang.research.timing_validity_r66.common import ROOT

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['run','verify'])
    p.add_argument('--source',default=str(ROOT));p.add_argument('--out')
    a=p.parse_args()
    if a.mode=='run':study.compute(a.out or ROOT,True)
    else:
        if not a.out:p.error('verify requires fresh --out')
        study.verify(a.source,a.out)

if __name__=='__main__':main()
