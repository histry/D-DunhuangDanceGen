"""Observed source inventory and manual roles; no automatic semantic labels."""
from pathlib import Path
import json
import numpy as np
from dunhuang.research.sdmr_warp_rank import connectivity
from dunhuang.research.complete_r36.core import relative_fk
from dunhuang.research.local_timing_r68.common import write,sha
from .study import ROOT

TEMPLATE='''<!doctype html><meta charset="utf-8"><title>来源原速阶段标注</title><style>body{background:#102132;color:#eee;font:17px sans-serif;margin:20px}button,input,select{padding:8px;margin:5px}canvas{background:#152b3e;width:360px}pre{white-space:pre-wrap}</style><h1>__UID__ · 原速完整来源</h1><p>真实30FPS，身体22关节局部视图；没有自动阶段真值。先看完整前后文，再独立标注。未标注保持UNKNOWN，至少两人一致性未验证前不进入阶段模型。</p><button id="play">原速播放</button><button id="pause">暂停</button><button id="prev">前一帧</button><button id="next">后一帧</button><input id="seek" type="range" min="0" value="0"><span id="clock"></span><br><canvas id="c" width="360" height="400"></canvas><p>标注者编号<input id="person">起始帧<input type="number" id="start" min="0">结束帧（不含）<input type="number" id="end" min="1"><select id="role"><option>UNKNOWN</option><option>PREPARATION</option><option>MAIN</option><option>HOLD</option><option>ENDING</option></select><button id="add">添加本人区间</button><button id="export">导出本人记录</button></p><pre id="rows"></pre><script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=s=>document.getElementById(s);let f=0,on=false,base=0,rows=[];const c=$('c').getContext('2d');$('seek').max=D.P.length-1;function draw(n){f=Math.max(0,Math.min(D.P.length-1,n));$('seek').value=f;$('clock').textContent=f+'/'+(D.P.length-1)+'帧 · '+(f/30).toFixed(3)+'秒';c.clearRect(0,0,360,400);let p=D.P[f].map(x=>[180+115*(x[0]+.35*x[2]),190-115*(x[1]-.15*x[2])]);c.strokeStyle='#8ee0c9';for(let j=1;j<22;j++){c.beginPath();c.moveTo(...p[D.parents[j]]);c.lineTo(...p[j]);c.stroke()}}$('play').onclick=()=>{on=true;base=performance.now()-f/30*1000};$('pause').onclick=()=>on=false;function seek(n){on=false;draw(n)}$('prev').onclick=()=>seek(f-1);$('next').onclick=()=>seek(f+1);$('seek').oninput=e=>seek(+e.target.value);$('add').onclick=()=>{let a=+$('start').value,b=+$('end').value;if(!$('start').value||!$('end').value||!Number.isInteger(a)||!Number.isInteger(b)||a<0||b<=a||b>D.P.length){$('rows').textContent='请填写真实半开帧区间';return}rows.push({start:a,end:b,role:$('role').value});$('rows').textContent=JSON.stringify(rows,null,2)};$('export').onclick=()=>{if(!$('person').value.trim()){ $('rows').textContent='科研导出需要实际标注者编号';return}let x={schema:'SOURCE_PHASE_V1',source_uid:D.uid,source_sha256:D.sha,annotator_id:$('person').value.trim(),intervals:rows};let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(x,null,2)],{type:'application/json'}));a.download=D.uid+'_actual_phases.json';a.click();URL.revokeObjectURL(a.href)};function tick(){if(on){draw(Math.floor((performance.now()-base)/1000*30));if(f===D.P.length-1)on=false}requestAnimationFrame(tick)}draw(0);requestAnimationFrame(tick);
</script>'''

def build():
    ctx=connectivity.prepare();out=ROOT/'source_audit';out.mkdir(exist_ok=False);sources=[];chunks={}
    for uid,R in sorted(ctx['lib'].rotations.items()):
        native=ctx['lib'].rows[uid][0]['reference']['sha256'];n=len(R)
        sources.append(dict(source_uid=uid,frames=n,fps=30,native_sha256=native,source_rotation_support=True,semantic_completeness='UNKNOWN',independent_phase_annotations=0))
        chunks[uid]=[[a,a+240] for a in range(30,n-269,240)]
        data=dict(uid=uid,sha=native,P=np.round(relative_fk(R,ctx['J'],ctx['parents']),4).tolist(),parents=ctx['parents'].tolist())
        (out/(uid+'.html')).write_bytes(TEMPLATE.replace('__UID__',uid).replace('__DATA__',json.dumps(data,separators=(',',':'))).encode())
    selected=[]
    for index in range(max(map(len,chunks.values()))):
        for uid,spans in chunks.items():
            if index<len(spans) and len(selected)<25:selected.append(dict(source_uid=uid,source_interval=spans[index],phase='UNKNOWN',status='SUPPORTED_CANDIDATE_NOT_NEW_INDEPENDENT_RECORDING'))
    write(out/'inventory.json',dict(verified_recordings=sources,total_supported_240frame_intervals=sum(map(len,chunks.values())),proposed_25=selected,
        requested_original72_status='NOT_ASSUMED_AVAILABLE_OR_INDEPENDENT; ONLY_HASH_VERIFIED_LIBRARY_COUNTED',phase_agreement='NOT_EVALUABLE',human_records=0,R74='NOT_STARTED'))
    links=''.join('<li><a href="'+s['source_uid']+'.html">'+s['source_uid']+' · '+str(s['frames'])+'帧</a></li>' for s in sources)
    (out/'index.html').write_bytes(('<meta charset="utf-8"><h1>来源支持盘点与完整原速阶段标注</h1><p>25个支持区间是候选，不是25条独立录制。尚无人工阶段标签或完整性证明。</p><ul>'+links+'</ul>').encode())
    write(ROOT/'source_delivery.json',dict(files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.iterdir()) if p.is_file()]))
    print('SOURCE_AUDIT_READY',len(sources),len(selected),flush=True)

if __name__=='__main__':build()
