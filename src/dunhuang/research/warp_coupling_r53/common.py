from pathlib import Path
import hashlib,json,io,zipfile
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk,rotation_sample
ROOT=Path('outputs/warp_coupling_r53_01')
R52=Path('outputs/strong_baseline_r52_01')
R51=Path('outputs/validation_r51_01')
CFG=read(Path(__file__).with_name('protocol.json'))

def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def array_hash(x):
 x=np.ascontiguousarray(x);return hashlib.sha256(str((x.shape,x.dtype.str)).encode()+x.tobytes()).hexdigest()
def fresh(path):
 p=Path(path)
 if p.exists():raise ValueError('Fresh R53 output required: '+str(p))
 p.mkdir(parents=True);return p
def significant(a,b):return abs(a-b)>max(CFG['meaningful_D3_absolute'],CFG['meaningful_D3_relative']*abs(a))
def save_arrays(path,arrays):
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
  for key in sorted(arrays):
   out=io.BytesIO();np.lib.format.write_array(out,np.asarray(arrays[key]),allow_pickle=False);info=zipfile.ZipInfo(key+'.npy',(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,out.getvalue())
