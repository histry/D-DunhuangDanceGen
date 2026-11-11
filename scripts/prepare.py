import argparse,json,subprocess,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dunhuang.io import write,sha
ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--out',required=True);ap.add_argument('--stage',choices=['all','catalog'],default='all')
a=ap.parse_args();assets=json.loads(Path(a.assets).read_text());out=Path(a.out).resolve()
env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
if a.stage=='all':
    out.mkdir(parents=True,exist_ok=False)
    subprocess.run([assets['edge_python'],'-m','dunhuang.generation.assets',a.assets,str(out/'edge_contract')],env=env,check=True)
    assets['scaler']=json.loads((out/'edge_contract/receipt.json').read_text())['scaler']
    write(out/'assets.json',assets)
    subprocess.run([assets['native_python'],__file__,'--stage','catalog','--assets',str(out/'assets.json'),'--out',str(out)],env=env,check=True)
else:
    from dunhuang.planning.catalog import build
    path=build(assets,out/'catalog')
    assets['catalog']=dict(path=str(path),sha256=sha(path))
    write(out/'assets.json',assets)
