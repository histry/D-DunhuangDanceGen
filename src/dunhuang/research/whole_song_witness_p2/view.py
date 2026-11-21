import base64,gzip,json,shutil
from pathlib import Path
import numpy as np
from .common import fresh,write,sha,identity

def encoded(P):return base64.b64encode(np.asarray(P,dtype='<f4').tobytes()).decode()

def build(out,result,arrays,parent):
    folder=fresh(out/'viewer');motions={}
    for row in result['motions']+result['baselines']:
        key=row['key'];motions[key]=dict(P=encoded(arrays[key+'_P']),meta=row,D3=arrays[key+'_D3'].tolist())
    sources={uid:dict(P=encoded(arrays['source_'+identity(uid)+'_P']),meta=meta) for uid,meta in result['source_units'].items()}
    payload=dict(tasks=result['tasks'],motions=motions,sources=sources,parents=arrays['parents'].tolist(),counts=result['counts'])
    raw=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    (folder/'data.js').write_bytes(('window.P2="'+base64.b64encode(gzip.compress(raw,mtime=0)).decode()+'";').encode())
    shutil.copyfile(Path(__file__).with_name('viewer.html'),folder/'index.html')
    write(folder/'protocol.json',dict(task=result['task'],main_result_sha256=sha(out/'result.json'),
        counts=result['counts'],view='EXACT_WHOLE_SONG_BODY22_FK; COMMON_AUDIO_CLOCK; RAW_SOURCE_SEPARATE_30FPS_CLOCK',
        no_missing_path_substitution=True,existing_audio=[dict(name=f"song{s['song']}.wav",sha256=s['audio']['audio_sha256']) for s in parent['songs']],
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in folder.iterdir()},Native24='NOT_EVALUABLE',can_route=False))
