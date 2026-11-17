"""Reuse the sealed R37 viewer payload; append actual R38 saved trajectories."""
import argparse,base64,gzip,json,shutil,hashlib
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.transition_r37.evaluate import decode,curves

def composition(plan):
    from collections import defaultdict
    occupied=defaultdict(int);runs=[]
    for e in plan['events']:
        a,b=e['music_interval'];uid=e['source_uid'];occupied[uid]+=b-a
        if runs and runs[-1]['source_uid']==uid and runs[-1]['end']==a:runs[-1]['end']=b
        else:runs.append(dict(source_uid=uid,start=a,end=b))
    # Equal time thirds are descriptive sections, not claimed semantic phrases.
    thirds=[]
    for a,b in zip(np.linspace(0,plan['frames'],4,dtype=int)[:-1],np.linspace(0,plan['frames'],4,dtype=int)[1:]):
        counts=defaultdict(int)
        for e in plan['events']:
            u,v=e['music_interval'];counts[e['source_uid']]+=max(0,min(int(b),v)-max(int(a),u))
        thirds.append(dict(interval=[int(a),int(b)],scope='EQUAL_TIME_THIRD_NOT_SEMANTIC_SECTION',source_frames=dict(counts)))
    return dict(source_seconds={k:v/30 for k,v in occupied.items()},source_fraction={k:v/plan['frames'] for k,v in occupied.items()},longest_source_run_seconds=max(r['end']-r['start'] for r in runs)/30,continuous_source_runs=runs,sections=thirds,development_semantics='UNKNOWN',theme_vs_mechanical_repetition='NOT_SEMANTICALLY_LABELED')

def run(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Fresh viewer output required')
    out.mkdir(parents=True);cfg=read('src/dunhuang/research/bridge_r38/config.json');parent=Path(cfg['r37'])/'viewer_final'
    original=(parent/'data.js').read_text(encoding='utf8');encoded=original.split('"')[1];payload=json.loads(gzip.decompress(base64.b64decode(encoded)))
    shutil.copytree(parent/'audio',out/'audio');a=np.load(root/'evaluation/arrays.npz');meta=read(root/'evaluation/evaluation.json');J=a['skeleton_J22'];parents=a['parents'];new=[]
    def track(p):
        p=np.ascontiguousarray(p,dtype='<f4');h=hashlib.sha256(p.tobytes()).hexdigest();payload['tracks'][h]=dict(shape=list(p.shape),data=base64.b64encode(p.tobytes()).decode());return h
    planning=[]
    for case in payload['cases']:
        if case['kind']=='plan':
            song=next(s for s in read(Path(cfg['r37'])/'planning/result.json')['songs'] if s['id']==case['id']);plan=read(song['new_plan']);extra=composition(plan);case['summary']['r38_composition_audit']=extra;planning.append(dict(song=case['id'],**extra));case['scope']+=' R38保留原R37选路，只增加来源占时和最长连续段审查；不强制平均分配。'
            diag=Path(root)/'planning_diagnostics.json'
            if diag.exists():case['summary']['r38_content_diagnostics']=next(s for s in read(diag)['songs'] if s['song']==case['id'])
    for row in meta['records']:
        key=row['key'];panels=[dict(title='来源真值' if row['reference_is_target'] else '硬切参考 · 不是连续真值',track=track(a[key+'_reference22']))]
        diagnostics=[]
        for arm in ('Geometry','FrozenMatched','Trained'):
            variants=[track(p) for p in a[key+'_'+arm+'_22']];panels.append(dict(title={'Geometry':'观测外端几何桥 · 不读取缺口','FrozenMatched':'冻结Context · 相同R38条件合同','Trained':'R38桥接任务训练 · 240步Context'}[arm],track=variants[0],variants=variants))
        if key+'_R37Joint_22' in a:
            variants=[track(p) for p in a[key+'_R37Joint_22']];panels.append(dict(title='历史R37 Joint · 条件合同不同',track=variants[0],variants=variants))
        for i in range(2):
            d={}
            for arm in ('Geometry','FrozenMatched','Trained'):
                R,root_=decode(a[key+'_'+arm+'_139'][i]);d[arm]=curves(R,a[key+'_reference_R'],root_,J,parents)
            diagnostics.append(d)
        audio=None
        if row['audio']:
            file='r38_'+row['audio'];shutil.copyfile(root/'evaluation'/row['audio'],out/'audio'/file);audio='audio/'+file
        events=[]
        if row['binding']:
            for e in row['binding']['events']:events.append(dict(e,music_interval=e['local_interval'],full_target_frames=e['music_interval'][1]-e['music_interval'][0]))
        linked=None
        if key=='new_plan':
            original=next(r for r in read(Path(cfg['r37'])/'generation/result.json')['records'] if r['key']==key)
            for song in read(Path(cfg['r37'])/'planning/result.json')['songs']:
                semantic=hashlib.sha256(json.dumps(read(song['new_plan']),sort_keys=True).encode()).hexdigest()
                if semantic==original['plan_sha256']:linked=song['id'];break
            if linked is None:raise ValueError('New plan generation identity mismatch')
        new.append(dict(id='r38_'+key,kind='bridge',name='R38 · '+key+(' · 真实连续补间' if row['reference_is_target'] else ' · 异源连接压力测试'),frames=256,audio=audio,panels=panels,events=events,linked_plan_case_id=linked,window_start=(row['binding'] or {}).get('window_start'),old_events=[dict(music_interval=row['gap'])],contexts=[[96,104],[152,160]],beats=[],phrases=[],repetitions=[],summary=row,diagnostics_variants=diagnostics,
            scope='橙色为隐藏/编辑域104–151帧；外侧身体硬保护。中间几何桥仅用缺口外三帧运动学，所有Source分支隐藏缺口。'+('此配对案例未参与本次适配；历史曝光，非新独立盲测。未附可核验波形，因此无音频，不评价音乐协调。' if row['reference_is_target'] else 'ChangE异源不存在连续动作真值，误差只表示与原硬切参考的差异。Root来自模型，接触未知。')))
    new.sort(key=lambda c:(0 if c['id']=='r38_new_plan' else 1 if c['id']=='r38_old_failed' else 2,c['id']))
    payload['cases']=new+payload['cases'];payload['schema']='r38_watchable_v1';raw=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode();(out/'data.js').write_text('window.R36_B64="'+base64.b64encode(gzip.compress(raw,mtime=0)).decode()+'";\n',encoding='utf8')
    for file in ('index.html','viewer.js','style.css'):
        text=(parent/file).read_text(encoding='utf8')
        if file=='index.html':text=text.replace('R37','R38').replace('看清内容重用与两侧共同过渡','几何底座与任务学习：看真实增益').replace('<p id="scope"></p>','<p id="scope"></p><p id="composition"></p>')
        if file=='viewer.js':
            text=text.replace("filter(data.cases.some(c=>c.kind==='plan')?'plan':data.cases[0].kind)","filter('bridge')")
            text=text.replace("$('scope').textContent=current.scope;", "$('scope').textContent=current.scope;const audit=current.summary?.r38_composition_audit;$('composition').textContent=audit?('来源占时：'+Object.entries(audit.source_seconds).map(([k,v])=>k+' '+v.toFixed(1)+'秒').join(' / ')+'；最长连续同来源 '+audit.longest_source_run_seconds.toFixed(2)+'秒。连续发展不等于机械重复。'):'';")
        (out/file).write_text(text,encoding='utf8')
    write(root/'planning_composition.json',planning);write(out/'protocol.json',dict(schema='r38_actual_saved_state_view',cases=len(payload['cases']),new_cases=len(new),fps=30,whole_song_generated=False,scope='R37_FROZEN_PLAN_AND_MEDIA_PLUS_R38_ACTUAL_STATE_COMPARISONS',files=[dict(name=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.root,a.out)
