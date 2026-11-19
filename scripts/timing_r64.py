import argparse
from dunhuang.research.timing_r64.study import run_main,run_controls,verify

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['main','controls','verify'])
    a=p.parse_args();{'main':run_main,'controls':run_controls,'verify':verify}[a.mode]()

if __name__=='__main__':main()
