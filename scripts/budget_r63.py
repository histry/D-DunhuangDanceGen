"""R63 research entrypoint, independent of the production pipeline."""
import argparse
from dunhuang.research.budget_r63.common import ROOT
from dunhuang.research.budget_r63.study import run, verify

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['main','controls','verify'])
    a=p.parse_args()
    if a.mode=='verify':
        verify()
    else:
        run(ROOT/a.mode,controls=a.mode=='controls')

if __name__=='__main__':
    main()
