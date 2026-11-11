import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.generation.train import train
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=300);ap.add_argument('--task',choices=['motion','paired','upperbody_probe'],default='motion')
    args=ap.parse_args()
    train(json.loads(Path(args.assets).read_text()),Path(args.out).resolve(),args.steps,args.task)
