import json
from pathlib import Path
import numpy as np
from .common import ROOT,HERE,CFG,read,write,sha,prepare,fresh
from dunhuang.research.complete_r36.core import relative_fk

def build(phase):
    root=ROOT/phase;out=fresh(root/'viewer');reg=read(ROOT/'preview/registration.json');rows=read(ROOT/'preview/candidates.json');d=read(root/'decisions.json');ctx=prepare();tracks={};items=[];audio={}
    with np.load(ROOT/'preview/arrays.npz',allow_pickle=False) as z:
        def track(k):
            if k not in z:return None
            if k not in tracks:tracks[k]=np.round(relative_fk(z[k],ctx['J'],ctx['parents']),4).tolist()
            return k
        for c in reg['cases']:
            variants=[];T=c['target_frames'];A=c['music_start_frame'];name='music_'+str(c['music_song'])+'_'+str(T)+'.wav';p=Path('outputs/paper1_source_time_01/preview/viewer')/name;assert p.exists();audio[name]=dict(name=name,existing_relative_path=str(p),sha256=sha(p))
            for decision in [v for v in d if v['case_key']==c['key']]:
                r=next((r for r in rows if r['uid']==decision['selected_uid']),None);uid=None if r is None else r['uid'];variants.append(dict(label=decision['method']+' · G1 '+str(decision['G1_pass']),track=None if uid is None else track(uid+'__full_R'),core_track=None if uid is None else track(uid+'__R'),qualified=decision['G1_pass'],source=str(c['source_interval']),q=None if uid is None else np.round(z[uid+'__q'],6).tolist(),source_interval=c['source_interval'],anchor_marks=[] if r is None else r['actual_node_target_frames'],target_marks=[] if r is None or r['targets'] is None else [v['local_frame'] for v in r['targets']],details=decision))
            items.append(dict(label=c['source_uid']+' · '+str(c['requested_nodes'])+'节点',frames=2*T+48,core=T,target=None,raw=False,audio=name,variants=variants,explanation='节点允许缺失、不要求固定阶段类型。粉线是候选中明确登记的音乐目标±3帧，青线是实际映射动作事件。自动角色兼容是假设，未通过独立语义资格。'))
            if c['requested_nodes']==1:
                k=c['key']+'__raw_R';items.append(dict(label=c['source_uid']+' · 原速来源',frames=len(z[k]),core=0,target=None,raw=True,audio=None,variants=[dict(label='原速来源',track=track(k),qualified=None)]))
    t=(HERE.parent/'source_time_q01/viewer.html').read_text(encoding='utf8');t=t.replace("c.fillRect(x.target/x.frames*300,325,2,28);","if(x.target!=null)c.fillRect(x.target/x.frames*300,325,2,28);if(r.target_marks)for(let v of r.target_marks){c.fillStyle='#dd8bbb';c.fillRect((v-3)/x.frames*300,325,6/x.frames*300,20)}if(r.anchor_marks)for(let v of r.anchor_marks){c.fillStyle='#70e1cb';c.fillRect(v/x.frames*300,328,2,22)}")
    t=t.replace('<p id="mode"></p>','<p id="mode"></p><p id="roles"></p>');t=t.replace("$('mode').textContent=x.raw?", "$('roles').textContent=x.explanation||'';$('mode').textContent=x.raw?");payload=dict(items=items,tracks=tracks,parents=ctx['parents'].tolist(),intro='RoleAlignment-Q01：同来源、1–3节点、共同有限候选池的Shadow技术比较。新时钟使用可达证书并重新运行原G1；BAS／SDMR不选路，没有独立艺术质量排名。');(out/'index.html').write_bytes(t.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':'))).encode());write(out/'protocol.json',dict(files={'index.html':dict(bytes=(out/'index.html').stat().st_size,sha256=sha(out/'index.html'))},existing_local_audio=list(audio.values()),methods=CFG['preview_methods'] if phase=='preview' else CFG['execution_methods'],can_route=False));print('ROLE_VIEW_READY',phase,flush=True)

if __name__=='__main__':
    import sys
    build(sys.argv[1])
