from pathlib import Path
import numpy as np
from dunhuang.paper1.contracts import read,write,sha,array_hash,save_arrays,fresh,identity,check_seal,source_reads
from dunhuang.research.context_feasibility_p1.common import load as load_p0,boundary_key
from dunhuang.research.complete_r36.core import relative_fk,rotation_sample

ROOT=Path('outputs/paper1_whole_song_witness_p2_01')
P1=Path('outputs/paper1_context_feasibility_p1_01')
CFG=read(Path(__file__).with_name('protocol.json'))

def binding():
    return {p.name:sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.json','.html')}

def seal(folder):
    write(folder/'manifest.json',dict(science=binding(),files={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='manifest.json'}))

def inputs():
    for name,digest in CFG['parent_manifests'].items():
        folder=Path(name).parent
        assert sha(name)==digest,'Frozen parent identity changed'
        check_seal(folder)
    for stage in ('main','controls'):
        assert read(P1/'receipts'/('independent_'+stage+'.json'))['status']=='PASS'
    p,c,z,rows,slots,bodies=load_p0()
    return p,z,rows,slots,bodies,read(P1/'main/result.json'),read(P1/'controls/complete_path_witness.json')

def reconstruct(row,z,source):
    """Re-read genuine recording and clock; array identity is required, not tolerance."""
    uid=row['uid'];A,B=row['target_interval'];w=np.asarray(row['time_warp'],float)
    assert array_hash(source)==row['source_R_sha256']
    q=source_reads(np.interp(np.arange(A,B),w[:,1],w[:,0]),row['source_interval'],[A,B])
    assert array_hash(q)==row['q_sha256'] and np.array_equal(q,z[uid+'_q'])
    delta=np.diff(w,axis=0);rates=delta[:,1]/delta[:,0]
    assert np.all(delta>0) and rates.min()>=.75-1e-10 and rates.max()<=1.33+1e-10
    R=rotation_sample(source,q)
    assert array_hash(R)==row['rotation_sha256'] and np.array_equal(R,z[uid])
    return R,q
