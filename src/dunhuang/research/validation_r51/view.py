"""Data-only viewer bound to saved arrays, shared clock and actual frame counts."""
import base64,json,shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from dunhuang.research.complete_r36.core import read,write,sha


def encoded(p):return base64.b64encode(np.asarray(p,dtype='<f4').tobytes()).decode()


def emit(out,cases,parents):
    out.mkdir();(out/'data.js').write_text('window.R51='+json.dumps(dict(cases=cases,parents=parents.tolist()),ensure_ascii=False)+';',encoding='utf8')
    shutil.copyfile(Path(__file__).with_name('viewer.html'),out/'index.html')
    write(out/'protocol.json',dict(cases=len(cases),files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())]))


def plans(root,tracks):
    z=np.load(root/'arrays.npz');result=read(root/'result.json');cases=[]
    # Audio files must exist before protocol creation.
    temp=root/'audio';temp.mkdir()
    for r in result['records']:
        if r['gap']!=48:continue
        a,b=r['music_interval'];name=r['key']+'.wav'
        with sf.SoundFile(tracks[r['song']]['path']) as f:
            sr=f.samplerate;f.seek(round(a*sr/30));sound=f.read(round(b*sr/30)-round(a*sr/30))
        sf.write(temp/name,sound,sr)
        methods=[dict(name=m,data=encoded(z[r['key']+'_'+m])) for m,v in r['selected'].items() if v['status']=='NUMERICAL_SHORT_PLAN']
        cases.append(dict(label='音乐%d · %d事件'%(r['song']+2,r['events']),frames=b-a,origin=a,audio=name,methods=methods,meta=dict(regions=r['regions'],selected=r['selected'],scope='真实ChangE重定时＋冻结G0/G1；不是新神经生成')))
    emit(root/'viewer',cases,z['parents'])
    for p in temp.iterdir():shutil.move(p,root/'viewer'/p.name)
    temp.rmdir();refresh(root/'viewer',len(cases))


def generation(root):
    z=np.load(root/'arrays.npz');result=read(root/'result.json');cases=[]
    for r in result['records']:
        k=r['key'];methods=[dict(name='完整来源（隐藏区仅供评价）',data=encoded(z[k+'_reference'])),dict(name='Geometry',data=encoded(z[k+'_Geometry'][0]))]
        methods += [dict(name=m,data=encoded(z[k+'_'+m][0])) for m in r['methods']]
        cases.append(dict(label=k+' · '+r['domain'],frames=256,origin=0,audio=r['audio'],methods=methods,meta=dict(domain=r['domain'],methods=r['methods'],phase=r['phase'],paired_music=r['paired_music'],seeds=r['seeds'],scope='真实冻结Lodge完整50步；仅缺口[104,152)允许有界修改，外侧由采样投影保留')))
    emit(root/'viewer',cases,z['parents'])
    for r in result['records']:
        if r['audio']:shutil.copyfile(root/r['audio'],root/'viewer'/r['audio'])
    refresh(root/'viewer',len(cases))


def refresh(out,count):
    write(out/'protocol.json',dict(cases=count,files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.name!='protocol.json']))
