"""Watch three identity-selected paired clips and existing detector evidence.

No detector, lag optimizer, annotation importer, or semantic truth is created.
"""
import json
from pathlib import Path
import subprocess
import numpy as np
from dunhuang.research.complete_r36.core import relative_fk
from .common import write, sha

TYPES = ("speed_peak", "motion_onset", "deceleration", "pose_hold", "root_turn_span", "direction_turn")


def _kind(name):
    for kind in TYPES:
        if name == kind or name.endswith(kind) or kind in name:
            return kind
    return "OTHER"


def build(out, cases, J, parents):
    out = Path(out); watch = out / "viewer"; watch.mkdir()
    parents = list(map(int, parents)); items = []; audio_files = {}
    if len(parents) != 22 or len(cases) != 3:
        raise ValueError("The fixed event audit uses three registered clips and body22")
    for case in cases:
        R = np.asarray(case["R"], dtype=float)
        a, b = map(int, case["source_interval"]); A, B = map(int, case["audio_interval"])
        if R.shape != (240, 22, 3, 3) or not np.isfinite(R).all() or b-a != 240 or B-A != 240 or min(a,A) < 0:
            raise ValueError("A genuine contiguous 240-frame source and once-calibrated audio interval are required")
        audio_path = Path(case["audio_path"])
        if not audio_path.is_file():
            raise ValueError("Verified paired raw audio path is missing")
        raw_sha = sha(audio_path)
        name = raw_sha[:12] + "_" + str(A) + "_" + str(B) + ".wav"
        if name not in audio_files:
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(audio_path),
                            "-ss", str(A/30), "-t", "8", "-vn", "-c:a", "pcm_s16le",
                            "-ac", "1", "-ar", "22050", str(watch/name)], check=True)
            audio_files[name] = dict(raw_audio_sha256=raw_sha, clip_sha256=sha(watch/name),
                                    audio_interval=[A,B], source_offset_applied_again=False)
        P = relative_fk(R, J, parents)
        if P.shape != (240,22,3) or not np.isfinite(P).all():
            raise ValueError("Actual source body22 FK is unavailable")
        events = []
        for event in case["events"]:
            frame = event["frame"]; span = event.get("evidence_span")
            if isinstance(frame, (bool,np.bool_)) or not isinstance(frame,(int,float,np.number)) or not np.isfinite(frame) or not 0 <= frame < 240:
                raise ValueError("Existing detector frame must address the registered local clip")
            if (not isinstance(span,(list,tuple,np.ndarray)) or len(span)!=2
                    or not np.isfinite(span).all() or not span[0] <= frame < span[1]):
                raise ValueError("Keep the real half-open detector evidence, including any clip overhang")
            name_type = str(event["type"])
            events.append(dict(frame=float(frame), type=name_type, track=_kind(name_type),
                               body_region=str(event.get("body_region","UNKNOWN")),
                               evidence_span=[float(span[0]),float(span[1])],
                               evidence_crosses_clip=bool(span[0]<0 or span[1]>240)))
        def event_clock(values):
            values = np.asarray(values,dtype=float)
            if values.ndim!=1 or not np.isfinite(values).all() or np.any((values<0)|(values>=240)):
                raise ValueError("Audio event frames must already be on the local once-calibrated clock")
            return values.tolist()
        items.append(dict(key=case["key"],source=case["source_uid"],group=case["group"],
            source_interval=[a,b],audio_interval=[A,B],audio=name,P=np.round(P,5).tolist(),events=events,
            beats=event_clock(case["beat_frames"]),accents=event_clock(case["accent_frames"])))
    payload=json.dumps(dict(items=items,parents=parents,types=TYPES),ensure_ascii=False,
                       separators=(",",":"),allow_nan=False).replace("<","\\u003c")
    (watch/"index.html").write_bytes(TEMPLATE.replace("__DATA__",payload).encode("utf-8"))
    write(watch/"protocol.json",dict(scope="EXISTING_PAIRED_SOURCE_AND_DETECTOR_EVENT_AUDIT_NOT_GENERATED_DANCE",
        display_cases=3,fps=30,frames=240,selection="FIRST_SOURCE_OF_FIRST_THREE_IDENTITY_ORDERED_REGISTERED_MUSIC_GROUPS_NO_SCORE_SELECTION",
        event_types=list(TYPES),detectors_added=0,best_lag_selected=False,human_reviewed=False,
        detector_frame_scope="EXISTING_PROXY_REPRESENTATIVE; HOLD_OR_TURN_CENTRE_IS_NOT_SEMANTIC_ONSET_OR_OFFSET",
        evidence_scope="ORIGINAL_HALF_OPEN_SPAN_PRESERVED; CLIP_OVERHANG_DISCLOSED_NOT_PADDED",
        audio="ACTUAL_PAIRED_AUDIO_PCM_MONO22050_ONE_MASTER_CLOCK",audio_files=audio_files,
        physical_offset="ALREADY_APPLIED_TO_AUDIO_INTERVAL_BY_CALLER; NEVER_ADDED_AGAIN",
        camera="SHARED_FIXED_ROOT_RELATIVE_BODY22_Y_UP_190PX_PER_METRE",
        semantic_event_truth="UNKNOWN",artistic_quality="NOT_EVALUABLE",Native24="NOT_EVALUABLE",can_route=False))
    write(out/"viewer_manifest.json",dict(files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,
        sha256=sha(p)) for p in sorted(watch.iterdir()) if p.is_file()],
        selection_independent_of_scores=True,raw_rotations=False,trajectories_in_html=True,
        full_audio=False,weights=False,human_labels=False))
    return watch/"index.html"


TEMPLATE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>R67 P0 · 配对来源与事件证据</title><style>
*{box-sizing:border-box}body{padding:22px;margin:0;background:#101b29;color:#e6eef7;font:15px system-ui;line-height:1.6}h1{font-size:25px;margin:0}p{max-width:1250px}.muted{color:#adc0d4}button,select,input{font:inherit;color:#eef7ff;background:#263b51;border:1px solid #647e98;border-radius:5px;padding:7px;margin:4px}button{cursor:pointer}input[type=range]{width:min(70vw,800px)}.layout{display:grid;grid-template-columns:minmax(320px,420px) minmax(500px,1fr);gap:18px;max-width:1450px}canvas{width:100%;background:#192d41;border:1px solid #49657f}.legend{font-size:13px;color:#d5e2ef}pre{white-space:pre-wrap;font-size:12px}.box{border:1px solid #49657f;padding:12px;border-radius:7px}#status{color:#efce96}a{color:#a3d3ff}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #80dbc7}@media(max-width:850px){body{padding:12px}.layout{display:block}.box{margin-top:12px}}
</style><h1>R67 P0 · 配对来源与现有事件代理</h1>
<p>播放真实配对录制中的240帧来源动作与对应8秒音乐。这里没有新生成舞蹈、没有新增事件检测器、没有人工标签，也没有选取“效果最好的滞后”。先直接检查来源运动、检测位置和证据范围。</p>
<p class="muted">彩色标记沿用现有 detector.frame。特别是 hold / turn 的代表帧可能是区间中心，不能当作语义起势或收势。横向色带是原始半开证据区间；超出本片段的部分会标为“跨出片段”，不会补造完整阶段。骨架为 Root-relative body22，不能据此评价真实 Root、接触或敦煌文化正确性。</p>
<select id="sample" aria-label="配对来源案例"></select><button id="play">共同播放</button><button id="pause">共同暂停</button><button id="prev">前一帧</button><button id="next">后一帧</button><label>速度<select id="rate"><option>1</option><option>0.5</option><option>0.25</option><option>1.5</option></select></label><br>
<input type="range" id="seek" min="0" max="239" value="0" aria-label="统一输出帧"><output id="frame"></output><p id="status" role="status"></p>
<div class="layout"><div><canvas id="body" width="420" height="480" aria-label="真实来源骨架"></canvas><p class="legend">固定相机、统一比例；源录制逐帧播放。</p></div><div class="box"><canvas id="timeline" width="1050" height="390" aria-label="六类事件证据时间线，点击跳转"></canvas><p class="legend">蓝短线：音乐拍点；紫短线：重音代理。六行色带：阶段证据范围；彩点：现有检测代表帧。点击时间线可暂停跳转。</p><p id="events"></p></div></div>
<audio id="audio" preload="metadata"></audio><details><summary>原始事件记录与时间身份</summary><pre id="details"></pre></details>
<script>
const data=__DATA__, $=id=>document.getElementById(id), audio=$('audio'), bc=$('body').getContext('2d'),tc=$('timeline').getContext('2d');
const names={speed_peak:'速度峰',motion_onset:'动作启动代理',deceleration:'减速代理',pose_hold:'保持代理',root_turn_span:'Root转向代理',direction_turn:'方向转折代理'},colors=['#65d9c1','#88b8ff','#f0bc7c','#c4a1f2','#f093ae','#c9db83'];let item,at=0,pending=null;
data.items.forEach((v,i)=>$('sample').add(new Option(`${v.group} / ${v.source} / ${v.key}`,i)));
function select(){audio.pause();pending=null;item=data.items[+$('sample').value];audio.src=item.audio;audio.load();audio.playbackRate=+$('rate').value;$('details').textContent=JSON.stringify({case:item.key,source:item.source,group:item.group,source_interval:item.source_interval,audio_interval:item.audio_interval,events:item.events,offset:'已包含在audio_interval，不二次添加',semantic_truth:'UNKNOWN'},null,2);draw(0);$('status').textContent='登记来源已载入；单一真实音乐时钟'}
function draw(f){at=Math.max(0,Math.min(239,Math.trunc(f)));bc.clearRect(0,0,420,480);const P=item.P[at];bc.lineWidth=2.8;bc.strokeStyle='#7bdfca';for(let j=1;j<22;j++){const a=P[j],b=P[data.parents[j]];bc.beginPath();bc.moveTo(210+a[0]*190,245-a[1]*190+a[2]*20);bc.lineTo(210+b[0]*190,245-b[1]*190+b[2]*20);bc.stroke()}bc.font='14px system-ui';bc.fillStyle='#e0ecf7';bc.fillText('实际来源帧 '+(item.source_interval[0]+at),12,24);bc.fillText('Root-relative body22 · 非生成',12,47);tc.clearRect(0,0,1050,390);const x=t=>160+t/240*870;tc.font='15px system-ui';tc.fillStyle='#dfebf6';tc.fillText('音乐拍点 / 重音',8,29);for(const t of item.beats){tc.fillStyle='#7bb5ff';tc.fillRect(x(t),13,2,19)}for(const t of item.accents){tc.fillStyle='#bb98ee';tc.fillRect(x(t),35,2,12)}data.types.forEach((type,i)=>{const y=80+i*46;tc.fillStyle=colors[i];tc.fillText(names[type]||type,8,y+12);tc.strokeStyle='#3c5873';tc.beginPath();tc.moveTo(160,y+16);tc.lineTo(1030,y+16);tc.stroke();for(const e of item.events.filter(e=>e.track===type)){const lo=Math.max(0,e.evidence_span[0]),hi=Math.min(240,e.evidence_span[1]);tc.globalAlpha=.27;tc.fillStyle=colors[i];tc.fillRect(x(lo),y,Math.max(1,x(hi)-x(lo)),26);tc.globalAlpha=1;tc.beginPath();tc.arc(x(e.frame),y+13,4,0,Math.PI*2);tc.fill();if(e.evidence_crosses_clip){tc.font='11px system-ui';tc.fillText(e.evidence_span[0]<0?'◀':'▶',e.evidence_span[0]<0?150:1033,y+15);tc.font='15px system-ui'}}});tc.strokeStyle='#fff';tc.lineWidth=1.5;tc.beginPath();tc.moveTo(x(at),5);tc.lineTo(x(at),365);tc.stroke();tc.fillStyle='#d5e3f1';tc.fillText('0秒',160,385);tc.fillText('8秒',998,385);const active=item.events.filter(e=>Math.abs(e.frame-at)<=1||e.evidence_span[0]<=at&&at<e.evidence_span[1]);$('events').textContent=active.length?active.map(e=>`${names[e.track]||e.type} · ${e.body_region} · 代表帧${e.frame} · 证据[${e.evidence_span.join(',')})${e.evidence_crosses_clip?'（跨出片段）':''}`).join('；'):'当前帧没有覆盖的已登记证据；不代表没有真实动作事件';$('seek').value=at;$('frame').textContent=`本片段帧 ${at}/239 · ${(at/30).toFixed(3)} 秒 · 原始音频帧 ${item.audio_interval[0]+at}`}
function seek(f){audio.pause();draw(f);pending=at/30;if(audio.readyState){audio.currentTime=pending;pending=null}$('status').textContent='已暂停在同一音乐与动作帧 '+at}
$('sample').onchange=select;$('play').onclick=()=>audio.play().catch(e=>$('status').textContent='音频未播放：'+e.message);$('pause').onclick=()=>{audio.pause();draw(Math.floor(audio.currentTime*30+1e-4));$('status').textContent='共同暂停'};$('prev').onclick=()=>seek(at-1);$('next').onclick=()=>seek(at+1);$('seek').oninput=e=>seek(+e.target.value);$('rate').onchange=e=>audio.playbackRate=+e.target.value;$('timeline').onclick=e=>{const rect=$('timeline').getBoundingClientRect(),px=(e.clientX-rect.left)/rect.width*1050;seek(Math.round((px-160)/870*240))};audio.onloadedmetadata=()=>{if(pending!==null){audio.currentTime=pending;pending=null}if(Math.abs(audio.duration-8)>.05)$('status').textContent='音频时长与8秒不一致，请检查'};audio.onseeked=()=>draw(Math.floor(audio.currentTime*30+1e-4));audio.onplay=()=>{$('status').textContent='真实音乐与来源动作共同播放'};audio.onended=()=>{draw(239);$('status').textContent='本配对来源片段播放完成'};audio.onerror=()=>{$('status').textContent='实际音频不可读取，请检查媒体路径；骨架逐帧检查仍保留'};function tick(){if(!audio.paused)draw(Math.floor(audio.currentTime*30+1e-4));requestAnimationFrame(tick)}select();tick();
</script></html>'''
