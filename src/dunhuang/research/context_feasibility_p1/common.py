import json,hashlib
from pathlib import Path
import numpy as np
from dunhuang.paper1.contracts import read,write,sha,array_hash,save_arrays,fresh,identity,check_seal
ROOT=Path('outputs/paper1_context_feasibility_p1_01')
P0=Path('outputs/paper1_sdmr_integration_p0_01')
CFG=read(Path(__file__).with_name('protocol.json'))

def binding():return {p.name:sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.json','.html')}
def seal(p):write(p/'manifest.json',dict(files={str(q.relative_to(p)):sha(q) for q in sorted(p.rglob('*')) if q.is_file() and q.name!='manifest.json'},science=binding()))

def load():
    check_seal(P0/'preview');check_seal(P0/'controls')
    if sha('reports/paper1_sdmr_integration_p0_release/manifest.json')!=CFG['p0_public_manifest_sha256']:raise ValueError('P0 public identity changed')
    p=read(P0/'preview/result.json');c=read(P0/'controls/result.json');z=np.load(P0/'preview/arrays.npz',allow_pickle=False)
    rows={r['uid']:r for s in p['songs'] for k in s['slots'] for r in k['candidates']};slots={(s['song'],k['slot']):k for s in p['songs'] for k in s['slots']}
    R={u:z[u] for u in rows}
    if any(array_hash(R[u])!=r['rotation_sha256'] for u,r in rows.items()):raise ValueError('Executed P0 rotations changed')
    return p,c,z,rows,slots,R

def neighbors(slots,s,k):
    if (s,k) not in slots:return []
    layer=slots[s,k];base=layer['selected_baseline_uid'];return sorted([r['uid'] for r in layer['candidates']],key=lambda u:(u!=base,u))

def admissible(a,b):return a['source_uid']!=b['source_uid'] and a['target_interval'][1]+48==b['target_interval'][0]

def boundary_key(L,R,J,parents,reference):
    return identity(dict(boundary=array_hash(np.concatenate((L[-4:],R[:4]))),J=array_hash(J),parents=array_hash(parents),reference=reference,
                         method='FROZEN_R62_G1_48_FOUR_EDGE_SUPPORT',protocol=sha(Path(__file__).parents[1]/'duration_r62/protocol.json')))
