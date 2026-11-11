"""Independent upstream EDGE/PyTorch3D conversion check, CPU only."""
import argparse,sys,pickle,contextlib,io
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.io import pinned,write,sha
from dunhuang.data.sources import read,ref,raw_path
from dunhuang.data.conversion import decode,contacts
from dunhuang.contracts import MotionAdapter,matrices

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--raw-root',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();assets=read(a.assets);out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    root=pinned(assets['edge_root'],assets['edge_commit']);sys.path.insert(0,str(root))
    import torch
    from vis import SMPLSkeleton
    from dataset.dance_dataset import AISTPPDataset
    from pytorch3d.transforms import rotation_6d_to_matrix
    torch.set_num_threads(4);skeleton=SMPLSkeleton('cpu');results=[]
    pop=read(Path(a.raw_root)/'population.json')
    class Identity:
        def normalize(self,x):return x
    class Reference:
        train=False;normalizer=Identity();data_len=-1
    for dataset in ('aistpp','finedance'):
        row=next(r for r in pop['records'] if r['dataset_id']==dataset and r['purpose_role']=='train')
        path=raw_path(a.raw_root,row['motion_asset']);native,observed=decode(path,dataset);native=native[:150]
        edge=MotionAdapter.to_edge(native);edge[:,:4]=contacts(edge,skeleton)
        if dataset=='aistpp':
            with path.open('rb') as f:raw=pickle.load(f)
            pos=np.asarray(raw['smpl_trans']/raw['smpl_scaling'][0],np.float32)[:300][None]
            poses=np.asarray(raw['smpl_poses'],np.float32)[:300][None]
            with contextlib.redirect_stdout(io.StringIO()):
                expected=AISTPPDataset.process_dataset(Reference(),pos[:,::2],poses[:,::2]).numpy()[0]
            error=float(abs(edge-expected).max());contact_equal=bool(np.array_equal(edge[:,:4],expected[:,:4]))
            if error>1e-5 or not contact_equal:raise ValueError('Official EDGE conversion differs')
            results.append(dict(dataset=dataset,source=ref(path),max_abs=error,contact_equal=contact_equal,
                oracle='pinned AISTPPDataset.process_dataset before normalization; inverse smpl_scaling'))
        else:
            raw=np.load(path)[:150];expected=rotation_6d_to_matrix(torch.from_numpy(raw[:,3:].reshape(-1,52,6)[:,:22])).numpy()
            ours=matrices(native[:,7:].reshape(-1,24,6))[:,:22]
            error=float(abs(expected-ours).max())
            if error>1e-5 or observed[-12:].any():raise ValueError('Body22 reference differs')
            results.append(dict(dataset=dataset,source=ref(path),max_abs=error,missing_joints=[22,23],
                oracle='pinned PyTorch3D row Rot6D interpretation; unknown hands remain masked'))
    write(out,dict(results=results,edge_commit=assets['edge_commit'],optimizer_steps=0))
if __name__=='__main__':main()
