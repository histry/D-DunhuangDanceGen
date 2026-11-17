"""Four saved-state panels with source toggle and actual diagnostic peak navigation."""
import argparse,base64,gzip,json,shutil,hashlib
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.transition_r37.evaluate import decode,world_fk

def run(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Fresh watch directory required')
    out.mkdir(parents=True);cfg=read('src/dunhuang/research/stabilize_r39/config.json');parent=Path(cfg['r38'])/'viewer_final';text=(parent/'data.js').read_text(encoding='utf8');data=json.loads(gzip.decompress(base64.b64decode(text.split('"')[1])))
    shutil.copytree(parent/'audio',out/'audio');old={c['id']:c for c in data['cases']};data['cases']=[c for c in data['cases'] if not c['id'].startswith('r38_')]
    arrays=np.load(root/'generation/arrays.npz');meta=read(root/'generation/result.json');metrics=read(root/'evaluation.json');J=arrays['skeleton_J22'];parents=arrays['parents'];new=[]
    def track(x):
        x=np.ascontiguousarray(x,dtype='<f4');h=hashlib.sha256(x.tobytes()).hexdigest();data['tracks'][h]=dict(shape=list(x.shape),data=base64.b64encode(x.tobytes()).decode());return h
    for row in meta['records']:
        key=row['key'];prior=old['r38_'+key];summary=next(r for r in metrics['records'] if r['key']==key);panels=[]
        if row['audio'] and sha(root/'generation'/row['audio'])!=sha(parent/prior['audio']):raise ValueError('Displayed audio differs from actual sampling case')
        for mode in ('Geometry','Raw','RootContinuous','ResidualRTS'):
            variants=[track(p) for p in arrays[key+'_'+mode+'_22']];panels.append(dict(title={'Geometry':'几何桥 · 无神经修改','Raw':'R38 Raw · 原轨迹逐位复现','RootContinuous':'Root 连续写回 · 实际重新采样','ResidualRTS':'端点约束残差 RTS · 独立后处理'}[mode],track=variants[0],variants=variants))
        if row['reference_is_target']:reference=world_fk(*decode(arrays[key+'_target_139'][0]),J,parents)
        else:reference=arrays[key+'_reference22']
        panels[0]['reference_track']=track(reference)
        case=dict(prior,id='r39_'+key,name='R39 · '+key+(' · 配对补间' if row['reference_is_target'] else ' · 敦煌异源压力测试'),panels=panels,summary=dict(summary,r39=True),diagnostics_variants=summary['diagnostics_variants'])
        case['scope']='固定R38权重、来源、音乐和噪声。橙色104–151帧为编辑域，主体只读；RTS把端点状态和首尾三帧残差零值纳入求解。Root单位米，身体单位rad；C4不做滤波。'+('此案例有连续动作目标，但已历史曝光，无音频，因此不判断音乐协调。' if row['reference_is_target'] else '异源拼接没有唯一连续真值，来源差不是语义错误率；Root和接触没有来源真值。')
        new.append(case)
    new.sort(key=lambda c:(0 if c['id']=='r39_new_plan' else 1 if c['id']=='r39_old_failed' else 2,c['id']));data['cases']=new+data['cases'];data['schema']='r39_actual_state_v1'
    used=set()
    for c in data['cases']:
        for p in c['panels']:
            used.add(p['track']);used.update(p.get('variants',[]))
            if 'reference_track' in p:used.add(p['reference_track'])
    data['tracks']={k:data['tracks'][k] for k in used};raw=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode();(out/'data.js').write_text('window.R36_B64="'+base64.b64encode(gzip.compress(raw,mtime=0)).decode()+'";\n',encoding='utf8')
    for file in ('index.html','viewer.js','style.css'):
        text=(parent/file).read_text(encoding='utf8')
        if file=='index.html':
            text=text.replace('R38','R39').replace('几何底座与任务学习：看真实增益','Root 连续写回与端点约束减抖').replace('固定世界（来源Root未知）','固定世界视角')
            text=text.replace('<option value="root_speed">生成Root速度</option>','<option value="root_speed">生成Root速度</option><option value="root_acceleration">Root加速度</option><option value="root_jerk">Root Jerk</option>')
            text=text.replace('<span id="clock">','<label><input type="checkbox" id="reference">第一栏看来源／真值</label><select id="diagnostic-arm"><option value="Raw">定位 Raw</option><option value="RootContinuous">定位 Root连续</option><option value="ResidualRTS">定位 RTS</option></select><button id="max-root">跳到最大Root加速度</button><button id="max-body">跳到最大参考差</button><span id="clock">')
        if file=='viewer.js':
            text=text.replace("'固定世界 · 来源Root未知 · 帧 '","(current.summary.reference_is_target?'固定世界 · 配对Root参考 · 帧 ':'固定世界 · 来源Root未知 · 帧 ')")
            text=text.replace("const panelTrack=p=>p.variants?p.variants[Number($('seed').value)]:p.track;","const panelTrack=p=>p.reference_track&&$('reference').checked?p.reference_track:(p.variants?p.variants[Number($('seed').value)]:p.track);")
            text=text.replace("function draw(){if(!current)return;", "function draw(){if(!current)return;const first=$('panels').querySelector('h3');if(first&&current.panels[0].reference_track)first.textContent=$('reference').checked?(current.summary.reference_is_target?'配对来源真值':'硬切来源参考 · 不是连续真值'):current.panels[0].title;")
            text=text.replace("$('metric').onchange=draw;", "$('metric').onchange=draw;$('reference').onchange=()=>{bounds();draw();};$('max-root').onclick=()=>{if(!current.summary.r39)return;audio.pause();running=false;const d=current.diagnostics_variants[Number($('seed').value)][$('diagnostic-arm').value].root_acceleration;seek(d.indexOf(Math.max(...d)));$('metric').value='root_acceleration';draw();};$('max-body').onclick=()=>{if(!current.summary.r39)return;audio.pause();running=false;const d=current.diagnostics_variants[Number($('seed').value)][$('diagnostic-arm').value].source_error;seek(d.indexOf(Math.max(...d)));$('metric').value='source_error';draw();};")
            text=text.replace("$('seed').hidden=!['bridge','smoothing'].includes(current.kind);", "$('reference').disabled=!current.panels[0].reference_track;$('max-root').disabled=!current.summary.r39;$('max-body').disabled=!current.summary.r39;$('diagnostic-arm').disabled=!current.summary.r39;$('seed').hidden=!['bridge','smoothing'].includes(current.kind);")
        (out/file).write_text(text,encoding='utf8')
    write(out/'protocol.json',dict(schema='r39_four_panel_v1',cases=len(data['cases']),new_cases=len(new),fps=30,default_case='r39_new_plan',raw_source_preserved=True,whole_song_generated=False,files=[dict(name=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]))
    print('R39_VIEWER',len(data['cases']),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.root,a.out)
