import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.evaluation_benchmarks.benchmark_report import report
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();assets=json.loads(Path(a.assets).read_text());assets['_file']=str(Path(a.assets).resolve())
    report(assets,a.manifest,a.out)
