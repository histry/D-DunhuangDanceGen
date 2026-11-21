import base64,gzip,json,shutil
from pathlib import Path
import numpy as np
from .common import *

def encoded(x):return base64.b64encode(np.asarray(x,dtype='<f4').tobytes()).decode()

def build(out,result,metrics,arrays):
    folder=fresh(out/'viewer')
    motions={r['key']:dict(P=encoded(arrays[r['key']+'_P']),meta=r,D3=arrays[r['key']+'_D3'].tolist(),metrics=metrics[r['key']]) for r in result['motions']}
    sources={uid:dict(P=encoded(arrays['source_'+identity(uid)+'_P']),meta=r) for uid,r in result['source_units'].items()}
    payload=dict(tasks=result['tasks'],motions=motions,sources=sources,parents=arrays['parents'].tolist(),counts=result['counts'])
    raw=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    (folder/'data.js').write_bytes(('window.P3="'+base64.b64encode(gzip.compress(raw,mtime=0)).decode()+'";').encode())
    shutil.copyfile(Path(__file__).with_name('viewer.html'),folder/'index.html')
    old=read(P2/'main/viewer/protocol.json')
    write(folder/'protocol.json',dict(task=CFG['task'],main_result_sha256=sha(out/'result.json'),counts=result['counts'],
        required_previously_authorized_local_P2_data=dict(relative_path='../../../paper1_whole_song_witness_p2_01/main/viewer/data.js',sha256=CFG['P2_viewer_data_sha256']),
        existing_audio=old['existing_audio'],view_scope='ACTUAL_BODY22_ROOT_RELATIVE_FK; THREE_SHARED_MUSIC_CLOCK_PANELS; RAW_SOURCE_SEPARATE_CLOCK',
        missing_paths_never_substituted=True,files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in folder.iterdir()},can_route=False,Native24='NOT_EVALUABLE'))
