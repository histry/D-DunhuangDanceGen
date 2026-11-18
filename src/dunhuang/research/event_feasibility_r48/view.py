"""Inspect actual event/slot/time-map/window contracts without claiming a generated song."""
import argparse,json
from pathlib import Path
from dunhuang.research.complete_r36.core import read,write,sha

def run(main,out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh event contract page')
    out.mkdir(parents=True);data=read(Path(main)/'result.json');rows=[]
    for r in data['events']:
        rows.append({k:r[k] for k in ('event_uid','source_uid','source_interval','legal_duration_range_frames','source_validity','context_left','context_right','stage_evidence_truncated','clock_variants')})
    (out/'data.js').write_text('window.EVENTS='+json.dumps(rows,ensure_ascii=False)+';',encoding='utf8')
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>R48 完整事件与实际音乐槽位</title><style>body{margin:35px auto;max-width:1200px;font:18px system-ui;color:#25423b;background:#f5f3ec}select{padding:8px;max-width:100%}pre{white-space:pre-wrap;font-size:14px}canvas{width:100%;background:white}a{color:#196f66}</style><h1>完整事件、合法槽位与生成窗口</h1><p>139个冻结候选保持完整来源区间；下方列出三首历史敦煌音乐上实际求解的时间图。多槽合并仅为研究候选，不是新整曲规划或神经生成。</p><p><a href="../../event_coverage_r47_01/viewer/index.html">打开完整来源／新旧切分骨架观看页</a></p><label>事件 <select id="event" aria-label="事件"></select></label><p id="info"></p><label>合法音乐槽位 <select id="clock" aria-label="合法音乐槽位"></select></label><canvas id="map" width="1000" height="300"></canvas><pre id="details"></pre><p>横轴：实际目标音乐帧；纵轴：来源帧。每个局部窗口读取同一个连续来源记忆，重叠时钟完全一致。未知来源区不使用；世界Root、接触、艺术匹配仍未知，can_route=false。</p><script src="data.js"></script><script>
const E=window.EVENTS,$=id=>document.getElementById(id);E.forEach((r,i)=>$('event').add(new Option(r.source_uid+' '+r.source_interval+' · '+r.event_uid,i)));
function load(){let r=E[+$('event').value];$('info').textContent='来源['+r.source_interval+')，'+(r.source_interval[1]-r.source_interval[0])+'帧；合法时长'+r.legal_duration_range_frames+'；'+r.clock_variants.length+'个实际音乐可行候选。阶段证据跨界数 '+r.stage_evidence_truncated;$('clock').innerHTML='';r.clock_variants.forEach((v,i)=>$('clock').add(new Option('音乐'+(v.song+2)+'，原槽'+v.slots+'，帧'+v.target_interval,i)));draw()}
function draw(){let r=E[+$('event').value],v=r.clock_variants[+$('clock').value],c=$('map'),x=c.getContext('2d');x.clearRect(0,0,1000,300);if(!v){$('details').textContent='没有合法时长／时间图；不伸展或裁断来源来强配。';return}let w=v.clock.time_warp,A=w[0],B=w[w.length-1];x.strokeStyle='#257b6c';x.lineWidth=3;x.beginPath();w.forEach((p,i)=>{let a=30+(p[1]-A[1])/(B[1]-A[1])*940,b=270-(p[0]-A[0])/(B[0]-A[0])*240;i?x.lineTo(a,b):x.moveTo(a,b)});x.stroke();$('details').textContent=JSON.stringify({...r,clock_variants:undefined,selected_clock:{...v,clock:{...v.clock,windows:v.clock.windows.map(q=>({...q,source_positions:[q.source_positions[0],q.source_positions.at(-1)],position_display:'first/last only; complete arrays retained in result.json'}))}}},null,2)}$('event').onchange=load;$('clock').onchange=draw;load();</script>''',encoding='utf8')
    write(out/'protocol.json',dict(events=139,source_contract_sha256=sha(Path(main)/'result.json'),files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.main,a.out)
