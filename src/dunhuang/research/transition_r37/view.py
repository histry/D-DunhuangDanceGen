"""Full-source nonlooping review, exact new-plan preview, actual state comparisons."""
import argparse,base64,gzip,hashlib,json,shutil
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,Library,relative_fk,rotation_sample
from .clock import source_positions
from .evaluate import decode,curves

def run(cfg,root,out,generated=None,source_only=False):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Fresh viewer directory required')
    (out/'audio').mkdir(parents=True);tracks={};cases=[];lib=Library(read('src/dunhuang/research/complete_r36/config.json')['catalog'])
    z=np.load(Path(cfg['r35'])/'generation/arrays.npz');J=z['skeleton_J22'];parents=z['parents']
    def track(p):
        p=np.ascontiguousarray(p,dtype='<f4');h=hashlib.sha256(p.tobytes()).hexdigest();tracks[h]=dict(shape=list(p.shape),data=base64.b64encode(p.tobytes()).decode());return h
    def body(e,ticks):return rotation_sample(lib.rotations[e['source_uid']],source_positions(e,ticks))
    def render_plan(plan):return np.concatenate([relative_fk(body(e,np.arange(*e['music_interval'])),J,parents) for e in plan['events']])
    for s in ([] if source_only else read(root/'planning/result.json')['songs']):
        plan=read(s['new_plan']);old=read(s['old_plan']);music=read(Path(cfg['r36'])/'planning'/(s['id']+'_music.json'));audio=out/'audio'/s['name'];shutil.copyfile(s['audio']['path'],audio)
        cases.append(dict(id=s['id'],kind='plan',name=s['name']+' · 内容记忆规划预演',frames=plan['frames'],audio='audio/'+audio.name,
            panels=[dict(title='R36 · 固定候选基线 · 未生成',track=track(render_plan(old))),dict(title='R37 · 来源区间记忆 · 未生成',track=track(render_plan(plan)))],
            events=plan['events'],old_events=old['events'],beats=music['beats'],phrases=music['phrases'],repetitions=music['repetitions'],summary=s,
            scope='完整音乐编排预演，边界尚未生成；橙色旧计划、绿色新计划。来源区间和新内容/回跳证据随帧显示。SDMR未选路。'))
    catalog=read(Path(cfg['r36'])/'planning/catalog.json')
    for uid in sorted(lib.rotations):
        R=lib.rotations[uid];p=relative_fk(R,J,parents);units=[dict(e,music_interval=e['source_interval']) for e in catalog['units'] if e['source_uid']==uid]
        cases.append(dict(id='source_'+uid,kind='source',name=uid+' · 完整原始录制',frames=len(R),audio=None,
            panels=[dict(title='完整来源原速30FPS · 播完停止',track=track(p)),dict(title='同一来源 · 旧切口与可选长候选',track=track(p))],
            events=units[:1],source_candidates=units,old_events=[dict(e,music_interval=e['source_interval']) for e in lib.rows[uid]],beats=[],phrases=[],repetitions=[],
            scope='完整来源，不是循环裁片；选择候选可跳到真实原始帧。未标注文化语义完整性。世界Root未知。',summary=dict(source_uid=uid,frames=len(R),candidates=len(units))))
    smooth=Path(root/'smoothing');arrays=np.load(smooth/'arrays.npz');meta=read(smooth/'result.json')
    for row in meta['records']:
        key=row['key'];wav=out/'audio'/(key+'.wav');shutil.copyfile(Path(cfg['r36'])/'generation_repair01'/(key+'.wav'),wav)
        methods=('Raw','KF','RTS','ResidualRTS');panels=[dict(title=arm+' · 实际解码后状态',track=track(arrays[key+'_'+arm+'_22'][0]),variants=[track(x) for x in arrays[key+'_'+arm+'_22']]) for arm in methods]
        cases.append(dict(id='smooth_'+key,kind='smoothing',name=key+' · KF / 双向RTS / 参考残差',frames=256,audio='audio/'+wav.name,panels=panels,
            events=[],old_events=[dict(music_interval=i) for i in row['intervals']],beats=[],phrases=[],repetitions=[],summary=row,
            diagnostics_variants=[{arm:row['methods'][arm]['diagnostics'][i] for arm in methods} for i in range(2)],
            scope='按真实动作帧1/30秒处理，保留原始状态；减抖不等于物理合格。参考残差RTS不跨来源事件；KF/RTS未强制保留主体。'))
    if generated:
        g=Path(generated);a=np.load(g/'arrays.npz');meta=read(g/'result.json')
        for row in meta['records']:
            key=row['key'];wav=out/'audio'/('bridge_'+key+'.wav');shutil.copyfile(g/(key+'.wav'),wav)
            ref=a[key+'_reference_R'];panels=[dict(title='原始拼接参考 · 未生成',track=track(a[key+'_reference22'])),dict(title='几何五次补间 · 非学习',track=track(a[key+'_Geometry22']))]
            for arm in ('R36Like','Joint'):panels.append(dict(title=arm+' · 同权重同种子Lodge',track=track(a[key+'_'+arm+'_22'][0]),variants=[track(x) for x in a[key+'_'+arm+'_22']]))
            diag=[]
            for i in range(2):
                values={}
                for arm in ('R36Like','Joint'):
                    R,Root=decode(a[key+'_'+arm+'_139'][i]);values[arm]=curves(R,ref,Root,J,parents)
                values['Geometry']=curves(a[key+'_Geometry_R'],ref,np.zeros((256,3)),J,parents);diag.append(values)
            linked=None
            if row['consumes_new_r37_plan'] and not source_only:
                for song in read(root/'planning/result.json')['songs']:
                    plan=read(song['new_plan']);semantic=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
                    if semantic==row['plan_sha256']:linked=song['id'];break
                if linked is None:raise ValueError('Generation does not bind to the displayed new plan')
            cases.append(dict(id='bridge_'+key,kind='bridge',name=row['song']+' · '+('真实消费新R37计划' if row['consumes_new_r37_plan'] else '保留旧失败边界'),frames=256,audio='audio/'+wav.name,
                linked_plan_case_id=linked,window_start=row['binding']['window_start'],
                panels=panels,events=[dict(e,music_interval=e['local_interval']) for e in row['binding']['events']],old_events=[dict(music_interval=row['gap'])],
                contexts=[[row['gap'][0]-8,row['gap'][0]],[row['gap'][1],row['gap'][1]+8]],beats=[],phrases=[],repetitions=[],summary=row,diagnostics_variants=diag,
                scope='绿色为实际选中事件；橙色为共同过渡区，主体只读。Joint=几何连接底座+有界Lodge修改，不能归为学会衔接或新Context训练收益。Root/接触无来源真值。'))
    payload=dict(schema='r37_watchable_v1',cases=cases,tracks=tracks,parents=parents.tolist(),human_reviewed=False,operational_qualified=False)
    raw=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode();(out/'data.js').write_text('window.R36_B64="'+base64.b64encode(gzip.compress(raw,mtime=0)).decode()+'";\n',encoding='utf8')
    for f in ('index.html','viewer.js','style.css'):shutil.copyfile(Path(__file__).parent/'web'/f,out/f)
    write(out/'protocol.json',dict(cases=len(cases),fps=30,display='ACTUAL_SAVED_STATE_FK; SOURCES_AND_PLANS_UNSMOOTHED',planning_included=not source_only,source_loop_default=False,whole_song_generated=False,
        files=[dict(name=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
    print('R37_VIEW',len(cases),out,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--generated');p.add_argument('--source-and-smoothing-only',action='store_true');a=p.parse_args();run(read('src/dunhuang/research/transition_r37/config.json'),a.root,a.out,a.generated,a.source_and_smoothing_only)
