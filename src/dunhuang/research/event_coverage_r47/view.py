"""Complete-recording viewer with independently indexed event boundaries, never fixed256 cuts."""
import argparse,base64,gzip,json,shutil
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,Library,relative_fk
from dunhuang.research.reperformance_r44.archive import Arrays

def run(index,out):
    out=Path(out);data=read(Path(index)/'result.json')
    if out.exists():raise ValueError('Fresh catalog viewer')
    out.mkdir(parents=True);lib=Library(read('src/dunhuang/research/complete_r36/config.json')['catalog']);z=Arrays('outputs/bridge_audit_r46_01/main/arrays.npz');J=z['skeleton_J22'];parents=z['parents'];tracks={};cases=[]
    for source in data['sources']:
        uid=source['source_uid'];P=np.asarray(relative_fk(lib.rotations[uid],J,parents),dtype='<f4');tid=uid;tracks[tid]=dict(shape=list(P.shape),data=base64.b64encode(P.tobytes()).decode());units=[dict(r,music_interval=r['source_interval'],family_uid='UNKNOWN_NOT_FITTED') for r in data['units'] if r['source_uid']==uid];old=[dict(r,music_interval=r['source_interval']) for r in lib.rows[uid]];cases.append(dict(id=uid,kind='source',name=uid+' · 完整录制与新旧事件',frames=len(P),audio=None,panels=[dict(title='真实完整来源 · 未补造Root或手指',track=tid),dict(title='相同来源 · 橙旧切分／绿新候选',track=tid)],events=units,old_events=old,beats=[],scope='原速30FPS完整来源，无新生成。新候选合并受保护阶段；旧质量未知间隙保留，不跨越准入。候选语义完整性仍UNKNOWN。上下文与阶段支持见详情。',summary=dict(source=source,units=units)))
    payload=dict(cases=cases,tracks=tracks,parents=parents.tolist(),human_reviewed=False,operational_qualified=False);raw=json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode();(out/'data.js').write_text('window.R36_B64="'+base64.b64encode(gzip.compress(raw,mtime=0)).decode()+'";',encoding='utf8')
    from dunhuang.research import complete_r36
    static=Path(complete_r36.__file__).parent/'web'
    for f in ('index.html','viewer.js','style.css'):shutil.copyfile(static/f,out/f)
    html=(out/'index.html').read_text(encoding='utf8').replace('R36','R47').replace('看见完整动作，再判断生成改变','完整 ChangE 来源与独立 Catalog V2 候选').replace('整曲编排预演 / 来源边界 / 实际短窗生成','完整录制／旧499切分／保护阶段后的候选边界').replace('<select id="case"></select>','<select id="case"></select><label>候选事件 <select id="unit"></select></label>');(out/'index.html').write_text(html,encoding='utf8');js=(out/'viewer.js').read_text(encoding='utf8').replace("filter('plan');","filter('source');")
    target="bounds();draw();}\nfunction filter";assert target in js
    js=js.replace(target,"$('unit').replaceChildren(...current.events.map((e,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1} · 原始帧${e.source_interval.join('–')} · ${e.duration_frames}帧`;return o;}));$('unit').onchange=()=>{running=false;seek(current.events[Number($('unit').value)].source_interval[0]);};bounds();draw();}\nfunction filter")
    js=js.replace('sourcePosition=(sourcePosition+dt*30*Number($(\'rate\').value))%current.frames;',"sourcePosition=Math.min(current.frames-1,sourcePosition+dt*30*Number($('rate').value));if(sourcePosition>=current.frames-1)running=false;");(out/'viewer.js').write_text(js,encoding='utf8');(out/'style.css').write_text((out/'style.css').read_text(encoding='utf8')+'\n[hidden]{display:none!important}\n',encoding='utf8');write(out/'protocol.json',dict(cases=len(cases),source_index_sha256=sha(Path(index)/'result.json'),scope='SOURCE_NOT_GENERATED; SEMANTIC_COMPLETENESS_UNKNOWN',files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())]));z.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.index,a.out)
