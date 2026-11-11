import json
import sys
from pathlib import Path
import numpy as np
from dunhuang.io import checked,pinned,write,sha

def export(assets,destination):
    root=pinned(assets['edge_root'],assets['edge_commit']);sys.path.insert(0,str(root))
    import torch
    from dataset.preprocess import Normalizer
    from dataset.scaler import MinMaxScaler
    import zipfile,pickletools
    file=checked(assets['checkpoint'])
    with zipfile.ZipFile(file) as archive:
        data=archive.read(next(n for n in archive.namelist() if n.endswith('data.pkl')))
        globals_={arg for op,arg,_ in pickletools.genops(data) if op.name=='GLOBAL'}
    allowed={'collections OrderedDict','torch._utils _rebuild_tensor_v2','torch FloatStorage','torch LongStorage','torch DoubleStorage','torch HalfStorage','torch._utils _rebuild_parameter','dataset.preprocess Normalizer','dataset.scaler MinMaxScaler','numpy.core.multiarray _reconstruct','numpy ndarray','numpy dtype','_codecs encode','numpy.core.multiarray scalar'}
    if not globals_<=allowed:raise ValueError('Unexpected checkpoint globals')
    payload=torch.load(file,map_location='cpu',weights_only=False)
    normalizer=payload['normalizer']
    if not isinstance(normalizer,Normalizer):raise ValueError('Normalizer type differs')
    destination.mkdir(parents=True,exist_ok=False)
    path=destination/'scaler.npz'
    np.savez(path,scale=normalizer.scaler.scale_.cpu().numpy(),bias=normalizer.scaler.min_.cpu().numpy())
    write(destination/'receipt.json',dict(checkpoint=assets['checkpoint'],scaler=dict(path=str(path.resolve()),sha256=sha(path)),
        publisher_provenance_verified=False,normalizer_refitted=False))
if __name__=='__main__':export(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]).resolve())
