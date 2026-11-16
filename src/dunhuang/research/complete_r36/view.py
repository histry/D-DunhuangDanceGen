"""Offline XYZ/audio delivery. Plans are labelled as previews, not generated dances."""
import argparse,base64,gzip,shutil
from pathlib import Path
import numpy as np
from .core import Library,read,write,relative_fk,sha

def run(cfg,root,out,generated=None):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Fresh viewer directory required')
    (out/'audio').mkdir(parents=True);tracks={};cases=[];lib=Library(cfg['catalog'])
    z=np.load(Path(cfg['r35'])/'generation/arrays.npz');J=z['skeleton_J22'];parents=z['parents']
    def track(p):
        p=np.ascontiguousarray(p,dtype='<f4');h=__import__('hashlib').sha256(p.tobytes()).hexdigest()
        tracks[h]=dict(shape=list(p.shape),data=base64.b64encode(p.tobytes()).decode());return h
    def oldbody(plan):
        from dunhuang.generation.soft_reference import resample_native
        from dunhuang.contracts import matrices
        from dunhuang.planning.time_map import source_positions
        out=[]
        for e in plan['events']:
            pos=source_positions(e,np.arange(*e['music_interval']))
            r=matrices(resample_native(lib.native[e['source_uid']],pos)[:,7:].reshape(-1,24,6))[:,:22]
            out.append(relative_fk(r,J,parents))
        return np.concatenate(out)
    for song in read(root/'planning/result.json')['songs']:
        plan=read(song['new_plan']);music=read(root/'planning'/(song['id']+'_music.json'))
        old=read(song['old_plan']['path']);a=oldbody(old);b=np.concatenate([relative_fk(lib.sample(e,np.arange(*e['music_interval'])),J,parents) for e in plan['events']])
        audio=out/'audio'/song['name'];shutil.copyfile(song['audio']['path'],audio)
        cases.append(dict(id=song['id'],kind='plan',name=song['name']+' · 整曲编排预演',frames=plan['frames'],audio='audio/'+audio.name,
            panels=[dict(title='R35 短事件基线 · 未生成连接',track=track(a)),dict(title='R36 多尺度动作单元 · 未生成连接',track=track(b))],
            events=plan['events'],old_events=old['events'],beats=music['beats'],phrases=music['phrases'],repetitions=music['repetitions'],
            scope='来源排列预演；边界有切换；未生成过渡；世界Root与接触未知',summary=song))
    catalog=read(root/'planning/catalog.json')
    for uid in lib.rows:
        choices=[e for e in catalog['units'] if e['source_uid']==uid and e['duration_frames']>=180]
        if not choices:continue
        e=choices[0];a,b=e['source_interval'];lo=max(e['support_component'][0],a-45);hi=min(e['support_component'][1],b+45)
        full=relative_fk(lib.rotations[uid][lo:hi],J,parents)
        cases.append(dict(id='source_'+uid,kind='source',name=uid+' · 新旧边界与完整上下文',frames=len(full),audio=None,
            panels=[dict(title='完整连续来源 · 无平滑无生成',track=track(full)),dict(title='相同来源 · 旧切口 / 新动作范围',track=track(full))],
            events=[dict(e,music_interval=[a-lo,b-lo])],old_events=[dict(r,music_interval=[max(0,r['source_interval'][0]-lo),min(len(full),r['source_interval'][1]-lo)]) for r in lib.rows[uid] if max(lo,r['source_interval'][0])<min(hi,r['source_interval'][1])],
            beats=[],phrases=[],repetitions=[],scope='上下文原速30FPS；绿色为新候选，橙色为旧切口；完整性仍是运动学候选',summary=dict(source_start=lo,unit=e)))
    if generated:
        g=Path(generated);a=np.load(g/'arrays.npz');meta=read(g/'result.json')
        for row in meta['records']:
            key=row['key'];wav=out/'audio'/(key+'.wav');shutil.copyfile(g/(key+'.wav'),wav)
            panels=[dict(title='按音乐对齐的身体参考 · Root未知',track=track(a[key+'_reference22']))]
            for arm in ('Frozen','Context','Edit'):
                panels.append(dict(title=arm+' · 实际模型输出',track=track(a[key+'_'+arm+'_22'][0]),variants=[track(x) for x in a[key+'_'+arm+'_22']]))
            cases.append(dict(id=key,kind='generation',name=row['name'],frames=256,audio='audio/'+wav.name,panels=panels,events=[],old_events=[],beats=[],phrases=[],repetitions=[],
                scope='固定同一计划和音乐的短窗；Edit为有界SO(3)来源编辑，非新训练能力或完整物理合格',summary=row,diagnostics=row['diagnostics']))
    import json
    payload=dict(schema='r36_effect_view_v1',cases=cases,tracks=tracks,parents=parents.tolist(),human_reviewed=False,operational_qualified=False)
    raw=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    (out/'data.js').write_text('window.R36_B64="'+base64.b64encode(gzip.compress(raw,mtime=0)).decode()+'";\n',encoding='utf8')
    for f in ('index.html','viewer.js','style.css'):shutil.copyfile(Path(__file__).parent/'web'/f,out/f)
    write(out/'protocol.json',dict(cases=len(cases),fps=30,display='COMMON22_ROOT_RELATIVE; NO_SMOOTHING',whole_song_generated=False,
        files=[dict(name=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
    print('R36_VIEW',len(cases),out,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--generated');a=p.parse_args()
    run(read('src/dunhuang/research/complete_r36/config.json'),a.root,a.out,a.generated)
