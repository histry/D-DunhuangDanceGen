"""Read-only recomputation from saved meshes and original motion/FK receipts."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dunhuang.io import sha
from dunhuang.evaluation_benchmarks.edge_pfc import pfc
from dunhuang.evaluation_benchmarks.physdiff_metrics import metrics
from dunhuang.evaluation_benchmarks.contact_states import diagnose

def read(p):return json.loads(Path(p).read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root)
    report=read(root/'report.json');protocol=read(root/'protocol.json')
    for p,h in report['input_hashes'].items():assert sha(p)==h,p
    for row in report['records']:
        folder=Path(row['run']);cache=row['mesh_cache'];assert sha(cache['path'])==cache['sha256']
        data=np.load(cache['path']);native=np.load(folder/'native.npy');fk=np.load(folder/'native_fk.npy')
        assert pfc(fk)['value']==row['pfc']['value']
        calculated=metrics(data['minimum_y'],data['joints'],row['floor_y_m'],contact_height=protocol['contact_height_m'])
        assert calculated==row['physdiff'],row['name']
        assert diagnose(data['joints'],row['floor_y_m'],native[:,:4],contact_height=protocol['contact_height_m'])==row['contact_extension']
        assert read(folder/'physical.json')['absolute']==row['historical_native']
    print('VERIFIED',report['cases'],'saved cases /',report['unique_motions'],'unique motions; PFC, mesh metrics, Native identities unchanged.')
if __name__=='__main__':main()
