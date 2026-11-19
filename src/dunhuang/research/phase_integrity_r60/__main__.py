import argparse
from .common import ROOT, fresh, write

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['main','controls','verify']);p.add_argument('--out');p.add_argument('--source',default=str(ROOT/'controls'));p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.action=='main':
        from .population import prepare
        from .view import preview
        fresh(ROOT);assets,result=prepare(ROOT/'population');preview(ROOT/'main',assets,result)
    elif a.action=='controls':
        from .study import controls
        controls(a.out or ROOT/'controls',a.workers)
    else:
        from .study import verify
        verify(a.out or ROOT/'independent',a.source,a.workers)
if __name__=='__main__':main()
