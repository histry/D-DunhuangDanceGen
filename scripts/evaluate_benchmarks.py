import argparse,json,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.evaluation_benchmarks.benchmark_report import report
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();assets=json.loads(Path(a.assets).read_text());assets['_file']=str(Path(a.assets).resolve())
    output=Path(a.out);existed=output.exists()
    try:report(assets,a.manifest,output)
    except Exception:
        if not existed and output.is_dir():
            from dunhuang.io import write
            write(output/'status.json',dict(status='FAILED',error=traceback.format_exc(),motion_updates=0,native_results_reclassified=False))
        raise
