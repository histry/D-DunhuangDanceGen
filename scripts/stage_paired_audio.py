import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.data.sources import materialize_candidate_audio
ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--raw-root',required=True);ap.add_argument('--audit',required=True)
a=ap.parse_args();print(materialize_candidate_audio(a.source,a.raw_root,a.audit))
