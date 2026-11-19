import argparse
from .common import ROOT,fresh

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('main','controls','verify'));a=p.parse_args()
    from .study import run,verify
    if a.action=='main':fresh(ROOT);run(ROOT/'main')
    elif a.action=='controls':run(ROOT/'controls',controls=True)
    else:verify()

if __name__=='__main__':main()
