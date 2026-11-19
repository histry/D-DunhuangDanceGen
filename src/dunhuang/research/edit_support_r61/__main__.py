import argparse
from .common import ROOT,fresh
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['main','verify_main','fixed','verify']);a=p.parse_args()
    if a.action=='main':
        from .intervention import main
        fresh(ROOT);main(ROOT/'main')
    elif a.action=='verify_main':
        from .intervention import verify_main
        verify_main(ROOT/'independent_main')
    elif a.action=='fixed':
        from .fixed_gap import run
        run(ROOT/'fixed')
    else:
        from .fixed_gap import verify
        verify(ROOT/'independent_fixed')
if __name__=='__main__':main()
