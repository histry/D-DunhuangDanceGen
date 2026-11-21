"""Delivery only: reuse the accepted body media, add scalar six-method controls."""
import json,shutil
from pathlib import Path
from dunhuang.paper1.contracts import ROOT,read,write,sha,fresh,check_seal

def build():
    source=ROOT/'preview/viewer';control=ROOT/'controls';check_seal(control);result=read(control/'result.json');out=fresh(control/'viewer')
    compact=[]
    for r in result['slots']:
        compact.append({k:r[k] for k in ('song','slot','candidate_uids','scores','GTWLocal','source_recall_shadow','connections')})
    (out/'shadow.js').write_text('window.SHADOW='+json.dumps(compact,ensure_ascii=False,separators=(',',':'))+';',encoding='utf8')
    page=(source/'index.html').read_text('utf8').replace('<script src="data.js"></script>','<script src="../../preview/viewer/data.js"></script><script src="shadow.js"></script>')
    page=page.replace('<p id="status">','<label>同来源对照 <select id="method"><option>E1W</option><option>GTWLocal</option><option>Single64</option><option>Multi64</option><option>WrongMusic</option><option>MusicZero</option></select></label><p>学习方法仅按当前同一来源的共同候选池显示反事实选择；不改变完整冻结轨迹。G1是与固定历史邻居的局部实测，非替换全曲路径资格。</p><p id="status">')
    page=page.replace("$('audio').src=S.audio", "$('audio').src='../../preview/viewer/'+S.audio")
    script=r'''
const originalCandidate=selectCandidate;selectCandidate=function(){originalCandidate();const r=SHADOW.find(x=>x.song===S.song&&x.slot===slot),j=r.candidate_uids.indexOf(C.uid);$('details').textContent=JSON.stringify({candidate_uid:C.uid,source_uid:C.source_uid,time_warp:C.time_warp,alignment:C.alignment,scores:Object.fromEntries(Object.entries(r.scores).map(([k,v])=>[k,Object.fromEntries(Object.entries(v).map(([n,a])=>[n,a[j]]))])),GTWLocal:r.GTWLocal[j],actual_G1:r.connections[j],source_recall_shadow:r.source_recall_shadow,can_route:false},null,2);$('status').textContent+= '；G1局部 '+(r.connections[j].every(e=>e.passes_original_screen)?'通过':'拒绝')};
$('method').onchange=function(){const r=SHADOW.find(x=>x.song===S.song&&x.slot===slot),name=$('method').value;let ids=K.candidates.map((v,i)=>i).filter(i=>K.candidates[i].event_uid===C.event_uid),values;if(name==='E1W'){ids=ids.filter(i=>K.candidates[i].origin==='E1W')}else{values=name==='GTWLocal'?r.GTWLocal:name==='WrongMusic'?r.scores.wrong.Single64:name==='MusicZero'?r.scores.zero.Single64:r.scores.actual[name];ids=ids.filter(i=>values[i]!==null);ids.sort((a,b)=>(name==='GTWLocal'?values[a]-values[b]:values[b]-values[a])||a-b)}if(!ids.length){$('status').textContent='该方法在此共同有效支持上不可评价；不替换候选';return}$('candidate').value=ids[0];selectCandidate()};
'''
    page=page.replace('(async()=>{',script+'\n(async()=>{');(out/'index.html').write_text(page,encoding='utf8')
    write(out/'protocol.json',dict(uses_accepted_preview_body_data=True,preview_data_sha256=sha(source/'data.js'),existing_audio_reused=True,new_audio_transfer=False,
          methods=['E1W','GTWLocal','Single64','Multi64','WrongMusic','MusicZero'],default_path_unchanged=True,can_route=False,
          files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())}))
    print('CONTROL_VIEW',len(list(out.iterdir())),sum(p.stat().st_size for p in out.iterdir()))

if __name__=='__main__':build()
