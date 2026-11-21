"""Actual source and execution tracks, common target clock; no hidden fallback."""
import json
import numpy as np
from .common import ROOT,CFG,read,write,sha,prepare,fresh
from dunhuang.research.complete_r36.core import relative_fk

def build(root,methods):
    ctx=prepare();reg=read(ROOT/'preview/registration.json');rows=read(ROOT/'preview/candidates.json');dec=read(root/'decisions.json');out=fresh(root/'viewer');tracks={};items=[];by={r['uid']:r for r in rows}
    with np.load(ROOT/'preview/arrays.npz',allow_pickle=False) as z:
        def track(k):
            if k not in z:return None
            if k not in tracks:tracks[k]=np.round(relative_fk(z[k],ctx['J'],ctx['parents']),4).tolist()
            return k
        for c in reg['cases']:
            for target in c['targets']:
                variants=[]
                for method in methods:
                    d=next(v for v in dec if v['case_key']==c['key'] and v['fraction']==target['fraction'] and v['method']==method);uid=d['candidate_uid'];diagnostic=False
                    if uid is None:uid=d.get('diagnostic_uid');diagnostic=uid is not None
                    r=by.get(uid);k=None if uid is None else uid+'__full_R';variants.append(dict(label=method+(' · 未合格近邻诊断，非算法出解' if diagnostic else ''),track=None if k is None else track(k),core_track=None if uid is None else track(uid+'__R'),qualified=d['success'],source=None if r is None else 'S'+str(r['source_order'])+' '+str(r['source_interval']),anchor=None if r is None else r['anchor_target_frame'],q=None if uid is None else np.round(z[uid+'__q'],6).tolist(),source_interval=None if r is None else r['source_interval'],details=d))
                items.append(dict(label=c['source_uid']+' · 时间目标 '+str(target['fraction']),frames=c['pair_frames'],core=c['target_frames'],target=target['target_frame'],raw=False,audio='music_'+str(c['music_song'])+'_'+str(c['target_frames'])+'.wav',variants=variants))
            variants=[dict(label='S'+str(s['order'])+' 原速来源 '+str(s['source_interval']),track=track(c['key']+'__S'+str(s['order'])+'__raw_R'),qualified=None) for s in c['sources']]
            items.append(dict(label=c['source_uid']+' · 两条原速来源',frames=max(s['source_interval'][1]-s['source_interval'][0] for s in c['sources']),core=0,target=None,raw=True,audio=None,variants=variants))
    template=(__import__('pathlib').Path(__file__).with_name('viewer.html')).read_text(encoding='utf8');payload=dict(tracks=tracks,items=items,parents=ctx['parents'].tolist(),intro='音乐只作为已登记时间约束。黄色标注未合格近邻诊断，不替代失败出解。原速来源有独立入口，不拉伸成同步时长。世界Root、接触和艺术质量未验证。')
    (out/'index.html').write_bytes(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':'))).encode());audio=[]
    for c in reg['cases']:
        name='music_'+str(c['music_song'])+'_'+str(c['target_frames'])+'.wav';p=__import__('pathlib').Path('outputs/paper1_auto_eval_01/preview/viewer')/name;assert p.exists();audio.append(dict(name=name,existing_relative_path=str(p),sha256=sha(p),bytes=p.stat().st_size,reuse_only=True))
    write(out/'protocol.json',dict(files={'index.html':dict(bytes=(out/'index.html').stat().st_size,sha256=sha(out/'index.html'))},reused_audio=audio,methods=methods,raw_rotations_included=False,can_route=False));print('SOURCE_TIME_VIEW_READY',flush=True)
