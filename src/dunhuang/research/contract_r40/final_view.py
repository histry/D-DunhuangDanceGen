"""Add fixed-budget Extra to the already delivered R40 viewer without changing results."""
import argparse,base64,gzip,json,hashlib,shutil
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha


def run(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Fresh final viewer')
    out.mkdir(parents=True);source=root/'viewer';data=json.loads(gzip.decompress(base64.b64decode((source/'data.js').read_text(encoding='utf8').split('"')[1])));z=np.load(root/'extra_results/generation/arrays.npz');e=read(root/'extra_results/evaluation.json');cases=[]
    for name in ('index.html','style.css','viewer.js'):shutil.copyfile(source/name,out/name)
    for name in sorted({c['audio'] for c in data['cases'] if c.get('audio')}):
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe audio binding')
        (out/name).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,out/name)
    def track(x):
        x=np.ascontiguousarray(x,dtype='<f4');h=hashlib.sha256(x.tobytes()).hexdigest();data['tracks'][h]=dict(shape=list(x.shape),data=base64.b64encode(x.tobytes()).decode());return h
    for c in data['cases']:
        if c['id'].startswith('r40_comparison_'):continue
        key=c['id'][4:];row=next(r for r in e['records'] if r['key']==key);summary=c['summary'];summary['methods']['Extra']=row['methods']['Matched'];summary['methods']['ExtraRTS']=row['methods']['ResidualRTS']
        for i in range(2):c['diagnostics_variants'][i]['Extra']=row['diagnostics_variants'][i]['Matched'];c['diagnostics_variants'][i]['ExtraRTS']=row['diagnostics_variants'][i]['ResidualRTS']
        c['display_diagnostic_arms']=['Geometry','Frozen','Matched','ResidualRTS'];cases.append(c)
        extra=[track(x) for x in z[key+'_Matched_22']];rts=[track(x) for x in z[key+'_ResidualRTS_22']];frozen=[track(x) for x in z[key+'_Frozen_22']]
        panel=lambda title,vs:dict(title=title,track=vs[0],variants=vs)
        cp=dict(c,id='r40_budget_'+key,name='R40 同预算对照 · '+key,panels=[c['panels'][1],panel('冻结R39 · 同一执行器 · RTS前',frozen),panel('Extra · 旧目标续训240步 · RTS前',extra),c['panels'][2]],display_diagnostic_arms=['Geometry','Frozen','Extra','Matched']);cases.append(cp)
        cases.append(dict(c,id='r40_extra_'+key,name='R40 Extra与RTS · '+key,panels=[c['panels'][0],c['panels'][1],panel('Extra · 旧目标续训 · RTS前',extra),panel('Extra · 固定R39 RTS',rts)],display_diagnostic_arms=['Geometry','Frozen','Extra','ExtraRTS']))
    data['cases']=cases;raw=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode();(out/'data.js').write_text('window.R36_B64="'+base64.b64encode(gzip.compress(raw,mtime=0)).decode()+'";\n',encoding='utf8')
    text=(out/'index.html').read_text(encoding='utf8').replace('R40 完整动作与来源编辑','R40 局部桥任务训练对照').replace('整曲编排预演 / 来源边界 / 实际短窗生成','局部桥任务：真实目标、冻结对照与实际生成；未生成新整曲').replace('<main><nav>','<main><p class="note">本轮未取得稳定优于冻结基线的证据。先比较实际动作，再查看指标；页面中的异源参考不是连续真值。</p><nav>').replace('<option value="ResidualRTS">定位 RTS</option>','<option value="ResidualRTS">定位 Matched RTS</option><option value="Extra">定位 Extra</option><option value="ExtraRTS">定位 Extra RTS</option>')
    (out/'index.html').write_text(text,encoding='utf8');(out/'style.css').write_text((out/'style.css').read_text(encoding='utf8')+'\n[data-kind]:disabled,#stage-link,#reference{display:none}\nlabel:has(#reference){display:none}\n',encoding='utf8')
    text=(out/'viewer.js').read_text(encoding='utf8');old="const diagnostics=current.diagnostics_variants?.[Number($('seed').value)]||current.diagnostics;";new="const rawDiagnostics=current.diagnostics_variants?.[Number($('seed').value)]||current.diagnostics;const diagnostics=rawDiagnostics?Object.fromEntries((current.display_diagnostic_arms||Object.keys(rawDiagnostics)).map(a=>[a,rawDiagnostics[a]])):null;"
    if old not in text:raise ValueError('Pinned diagnostic script changed')
    (out/'viewer.js').write_text(text.replace(old,new),encoding='utf8');write(out/'protocol.json',dict(schema='r40_budget_four_panel_v1',cases=len(cases),fps=30,default_case='r40_new_plan',whole_song_generated=False,files=[dict(name=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='protocol.json']));print('R40_FINAL_VIEWER',len(cases),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.root,a.out)
