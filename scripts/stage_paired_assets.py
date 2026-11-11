import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.data.sources import materialize
ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--population',required=True);ap.add_argument('--out',required=True)
a=ap.parse_args();print(materialize(a.source,a.out,a.population))
