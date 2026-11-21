import base64,gzip,json,shutil
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import relative_fk
from .common import fresh,write,sha

def build(out,arrays,result,parent):
    folder=fresh(out/'viewer');items=[]
    for r in result['cases']:
        def encode(name):
            R=arrays.get(name)
            if R is None:return None
            finite=np.isfinite(R).all(axis=(1,2,3));P=np.full((len(R),22,3),np.nan);P[finite]=relative_fk(R[finite],arrays['J'],arrays['parents'])
            return base64.b64encode(P.astype('<f4').tobytes()).decode()
        index=r['index'];s=r['case']['song'];items.append(dict(label=f"{index+1} · {r['case']['role']} · 歌曲{s+2}槽{r['case']['slot']} · {r['case']['source_uid']}",
            frames=r['frames'],origin=r['origin'],audio=f'song{s}.wav',left=encode('case'+str(index)+'_old'),right=encode('case'+str(index)+'_new'),meta=r))
    payload=dict(items=items,parents=arrays['parents'].tolist());raw=json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode()
    (folder/'data.js').write_text('window.CTX="'+base64.b64encode(gzip.compress(raw,mtime=0)).decode()+'";',encoding='utf8');shutil.copyfile(Path(__file__).with_name('viewer.html'),folder/'index.html')
    write(folder/'protocol.json',dict(cases=len(items),failure_cases=14,matched_control_cases=14,coordinate_scope='BODY22_ROOT_RELATIVE_NOT_NATIVE24',
          same_center_clock_in_both_panels=True,failed_constructed_bridges_visible_with_numeric_status=True,missing_witness_right_panel_empty=True,
          existing_audio=[dict(name=f"song{s['song']}.wav",sha256=s['audio']['audio_sha256']) for s in parent['songs']],
          files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in folder.iterdir()},can_route=False))
