"""Offline common-clock viewer. Reference, geometric interpolation and neural samples remain distinct."""
import argparse,base64,gzip,json,shutil
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.transition_r37.evaluate import decode,world_fk


def build(out,extra=None):
    out=Path(out)
    if out.exists():raise ValueError('Fresh viewer directory')
    out.mkdir(parents=True);p=Path('outputs/stabilize_r39_01');z=np.load(p/'generation/arrays.npz');meta=read(p/'generation/result.json');J=z['skeleton_J22'];parents=z['parents'];tracks={};cases=[]
    def track(x):
        key=str(len(tracks));x=np.asarray(x,dtype='<f4');tracks[key]=dict(shape=list(x.shape),data=base64.b64encode(x.tobytes()).decode());return key
    for row in meta['records']:
        key=row['key']
        if row['reference_is_target']:continue
        panels=[dict(name='来源参考 · 局部姿态；异源拼接无连续真值',tracks=[track(z[key+'_reference22'])]*2,reference=True)]
        for title,name in [('几何桥 · 非网络','Geometry'),('冻结R39 · 实际采样','RootContinuous'),('冻结R39 · 固定RTS','ResidualRTS')]:panels.append(dict(name=title,tracks=[track(x) for x in z[key+'_'+name+'_22']]))
        audio=None
        if row.get('audio'):
            audio=key+'.wav';shutil.copyfile(p/'generation'/row['audio'],out/audio)
        cases.append(dict(id=key,name=key+' · 历史曝光异源压力案例',panels=panels,audio=audio,gap=[104,152],seeds=row['seeds'],scope='来源主体显式保护；48帧有界编辑；不是完整事件音乐重演',metrics={}))
    if extra:
        root=Path(extra);e=read(root/'result.json');a=np.load(root/'arrays.npz')
        for row in e['records']:
            key=row['key'];panels=[]
            R=a[key+'_reference_R'];panels.append(dict(name='真实来源／对齐参考',tracks=[track(relative_fk(R,J,parents))]*2,reference=True))
            for name in row['methods']:panels.append(dict(name=name,tracks=[track(x) for x in a[key+'_'+name+'_22']]))
            audio=row.get('audio');target=None
            if audio:
                src=root/audio
                if not src.exists():src=p/'generation'/audio
                shutil.copyfile(src,out/audio)
            cases.append(dict(id='new_'+key,name=row.get('label',key),panels=panels,audio=audio,gap=[104,152],seeds=row['seeds'],scope=row['scope'],metrics=row.get('metrics',{})))
    payload=dict(cases=cases,tracks=tracks,parents=parents.tolist(),fps=30,frames=256,source_truth='UNPAIRED_REFERENCE_OR_SAME_RECORDING_AS_LABELLED',whole_song_generated=False)
    (out/'data.js').write_text('window.R42_B64="'+base64.b64encode(gzip.compress(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode(),mtime=0)).decode()+'";',encoding='utf8')
    for f in ('index.html','viewer.js','style.css'):shutil.copyfile(Path(__file__).parent/f,out/f)
    write(out/'protocol.json',dict(cases=len(cases),fps=30,master_clock='ONE_AUDIO_CLOCK_OR_VISUAL_FRAME_CLOCK_WITHOUT_AUDIO',scope='ACTUAL_SAVED_ARRAYS_NOT_MODEL_CLAIM',files=[dict(name=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.iterdir()) if f.name!='protocol.json']))
    print('R42_VIEWER',len(cases),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--extra');a=p.parse_args();build(a.out,a.extra)
