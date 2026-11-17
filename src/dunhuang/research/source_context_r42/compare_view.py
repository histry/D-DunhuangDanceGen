"""Append controls to the delivered main cases without selecting new examples."""
import argparse,base64,gzip,json,shutil
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha


def run(viewer,controls,out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh comparison viewer')
    out.mkdir(parents=True);viewer=Path(viewer);payload=json.loads(gzip.decompress(base64.b64decode((viewer/'data.js').read_text(encoding='utf8').split('"')[1])));cases={c['id']:c for c in payload['cases']};tracks=payload['tracks']
    for f in ('index.html','viewer.js','style.css'):shutil.copyfile(viewer/f,out/f)
    for audio in sorted({c['audio'] for c in cases.values() if c.get('audio')}):shutil.copyfile(viewer/audio,out/audio)
    for root in controls:
        root=Path(root);meta=read(root/'result.json');z=np.load(root/'arrays.npz');label=meta['label']
        for r in meta['records']:
            c=cases['new_'+r['key']]
            for name in (label,label+'RTS'):
                ids=[]
                for x in z[r['key']+'_'+name+'_22']:
                    a=np.asarray(x,dtype='<f4');key=str(len(tracks));tracks[key]=dict(shape=list(a.shape),data=base64.b64encode(a.tobytes()).decode());ids.append(key)
                c['panels'].append(dict(name=name,tracks=ids));c['metrics']['values'][name]=r['metrics']['values'][name];c['metrics']['curves'][name]=r['metrics']['curves'][name]
    (out/'data.js').write_text('window.R42_B64="'+base64.b64encode(gzip.compress(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode(),mtime=0)).decode()+'";',encoding='utf8')
    write(out/'protocol.json',dict(cases=len(cases),fps=30,main_selection_preserved=True,files=[dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.iterdir()) if f.name!='protocol.json']));print('R42_COMPARE_VIEW',len(cases),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--viewer',required=True);p.add_argument('--controls',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.viewer,a.controls,a.out)
