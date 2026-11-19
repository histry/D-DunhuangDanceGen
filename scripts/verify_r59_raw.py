"""Fresh authenticated raw SMPL/row6D decoding, separate from reuse of sealed features."""
import subprocess
import numpy as np
from dunhuang.io import checked,sha
from dunhuang.contracts import matrices
from dunhuang.data.conversion import decode
from dunhuang.research.paired_benchmark_r59.common import ROOT,CFG,read,write,fresh

def main():
 out=fresh(ROOT/'independent_raw');d=read(ROOT/'data/manifest.json');proofs={r['raw_motion']['sha256']:r for r in read(CFG['a2']+'/source_seals.json')};records=[]
 for number,r in enumerate(d['sources']):
  arrays=np.load(r['arrays']);assert sha(r['arrays'])==r['arrays_sha256'];path=checked(r['raw_motion']);count=r['frames']
  if r['dataset']=='AIST++':
   native,observed=decode(path,'aistpp');old=np.load(checked(proofs[r['raw_motion_sha256']]['native']));assert np.array_equal(native,old);R=matrices(native[:count,7:139].reshape(-1,22,6));root=native[:count,4:7]
  else:
   raw=np.load(path);R=matrices(raw[:count,3:135].reshape(-1,22,6),row=True);root=raw[:count,:3]
  assert np.array_equal(R,arrays['R']);assert np.array_equal(root,arrays['root']);records.append(dict(source_uid=r['source_uid'],dataset=r['dataset'],raw_motion_sha256=r['raw_motion_sha256'],source_arrays_sha256=r['arrays_sha256'],raw_redecode_rotations_exact=True,raw_redecode_translation_exact=True))
  if (number+1)%200==0:print('R59_RAW_REDECODE',number+1,flush=True)
 result=dict(records=records,sources=len(records),all_raw_rotations_and_translation_exact=True,raw_AIST_to_30hz_native_exact=True,script_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha(__file__),decoder_sha256=sha('src/dunhuang/data/conversion.py'),data_manifest_sha256=sha(ROOT/'data/manifest.json'),optimizer_calls=0);write(out/'result.json',result)
if __name__=='__main__':main()
