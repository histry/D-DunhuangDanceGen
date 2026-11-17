"""Truth/geometry/new executor/fixed RTS, bound to actual saved states."""
import argparse,base64,gzip,json,shutil,hashlib
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.transition_r37.evaluate import decode,world_fk


def run(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Fresh watch directory')
    out.mkdir(parents=True);parent=Path('outputs/stabilize_r39_01/viewer_final02');text=(parent/'data.js').read_text(encoding='utf8');data=json.loads(gzip.decompress(base64.b64decode(text.split('"')[1])));shutil.copytree(parent/'audio',out/'audio');old={c['id']:c for c in data['cases']};arrays=np.load(root/'generation/arrays.npz');meta=read(root/'generation/result.json');ev=read(root/'evaluation.json');new=[]
    def track(x):
        x=np.ascontiguousarray(x,dtype='<f4');h=hashlib.sha256(x.tobytes()).hexdigest();data['tracks'][h]=dict(shape=list(x.shape),data=base64.b64encode(x.tobytes()).decode());return h
    for row in meta['records']:
        key=row['key'];prior=old['r39_'+key];summary=next(r for r in ev['records'] if r['key']==key);reference=world_fk(*decode(arrays[key+'_target_139'][0]),arrays['skeleton_J22'],arrays['parents']) if row['reference_is_target'] else arrays[key+'_reference22'];panels=[]
        panels.append(dict(title='真实连续目标' if row['reference_is_target'] else '异源参考 · 不存在唯一连续真值',track=track(reference)))
        for arm,label in [('Geometry','几何桥 · 不做神经修改'),('Matched','R40执行合同训练 · 固定投影 · RTS前'),('ResidualRTS','同一生成 · 固定R39 RTS')]:
            variants=[track(x) for x in arrays[key+'_'+arm+'_22']];panels.append(dict(title=label,track=variants[0],variants=variants))
        summary['r39']=True;scope='仅48帧桥任务；外侧主体只读。来源模型和Source门控冻结，Context训练240步。Root连续写回和RTS参数沿用R39。'+('有真实连续目标，但历史曝光，不检验音乐协调。' if row['reference_is_target'] else '异源压力测试，无连续目标、来源Root或接触真值。')
        new.append(dict(prior,id='r40_'+key,name='R40 · '+key,panels=panels,summary=summary,diagnostics_variants=summary['diagnostics_variants'],scope=scope))
        # Separate actual frozen-versus-new view without presenting the changed output as truth.
        variants=[track(x) for x in arrays[key+'_Frozen_22']];bp=dict(title='冻结R39 · 同一执行器 · RTS前',track=variants[0],variants=variants)
        new.append(dict(new[-1],id='r40_comparison_'+key,name='R40 冻结对照 · '+key,panels=[panels[1],bp,panels[2],panels[3]]))
    new.sort(key=lambda c:(0 if c['id']=='r40_new_plan' else 1,c['id']));data['cases']=new;used={p['track'] for c in new for p in c['panels']};used.update(v for c in new for p in c['panels'] for v in p.get('variants',[]));data['tracks']={k:data['tracks'][k] for k in used};raw=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode();(out/'data.js').write_text('window.R36_B64="'+base64.b64encode(gzip.compress(raw,mtime=0)).decode()+'";\n',encoding='utf8')
    for file in ('index.html','viewer.js','style.css'):
        text=(parent/file).read_text(encoding='utf8').replace('R39','R40')
        if file=='index.html':text=text.replace('Root 连续写回与端点约束减抖','真实目标与执行合同训练').replace('value="Raw">定位 Raw','value="Frozen">定位 Frozen').replace('value="RootContinuous">定位 Root连续','value="Matched">定位 Matched')
        if file=='viewer.js':text=text.replace("current.panels[0].reference_track", "current.panels[0].reference_track")
        (out/file).write_text(text,encoding='utf8')
    write(out/'protocol.json',dict(schema='r40_four_panel_v1',cases=len(new),fps=30,whole_song_generated=False,files=[dict(name=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]));print('R40_VIEWER',len(new),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.root,a.out)
