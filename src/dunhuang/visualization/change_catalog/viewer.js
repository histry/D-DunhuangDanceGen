"use strict";
(async()=>{
 const $=id=>document.getElementById(id),opt=(v,t)=>{const e=document.createElement("option");e.value=v;e.textContent=t;return e;};
 const b=Uint8Array.from(atob(window.CATALOG_B64),x=>x.charCodeAt(0));
 const d=JSON.parse(await new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream("gzip"))).text());delete window.CATALOG_B64;
 const events=new Map(d.events.map(e=>[e.event_uid,e])),cache=new Map();let src,list,selected,frame=0,mode=null,baseFrame=0,baseTime=0,cam;
 function values(){if(!cache.has(src.uid)){const v=Uint8Array.from(atob(src.track.data),x=>x.charCodeAt(0));cache.set(src.uid,new Float32Array(v.buffer));}return cache.get(src.uid);}
 function pos(f,j){const v=values(),i=(f*22+j)*3;return [v[i],v[i+1],v[i+2]];}
 function projection(p){const yaw={front:-.45,side:Math.PI/2,straight:0}[$("camera").value],a=Math.cos(yaw),b=Math.sin(yaw);return [a*p[0]+b*p[2],p[1]-.12*(-b*p[0]+a*p[2])];}
 function camera(){
  const low=[Infinity,Infinity],high=[-Infinity,-Infinity];
  for(let f=0;f<src.frames;f++)for(let j=0;j<22;j++){const p=projection(pos(f,j));for(let k=0;k<2;k++){low[k]=Math.min(low[k],p[k]);high[k]=Math.max(high[k],p[k]);}}
  cam={center:low.map((x,k)=>(x+high[k])/2),span:low.map((x,k)=>Math.max(.5,high[k]-x))};
 }
 function pose(id,visible,label){
  const c=$(id),x=c.getContext("2d"),w=c.width,h=c.height;x.fillStyle="#f2f5ee";x.fillRect(0,0,w,h);x.fillStyle="#60796c";x.font="17px sans-serif";x.fillText(label,15,25);
  if(!visible){x.fillText("当前来源帧不在所选切分事件内",35,h/2);return;}
  const scale=Math.min((w-75)/cam.span[0],(h-80)/cam.span[1]),xy=[];
  for(let j=0;j<22;j++){const p=projection(pos(frame,j));xy.push([w/2+(p[0]-cam.center[0])*scale,h*.53-(p[1]-cam.center[1])*scale]);}
  x.lineWidth=3.2;for(let j=1;j<22;j++){x.strokeStyle=j%2?"#277780":"#b36e40";x.beginPath();x.moveTo(...xy[d.parents[j]]);x.lineTo(...xy[j]);x.stroke();}
  x.fillStyle="#304f4d";for(const p of xy){x.beginPath();x.arc(...p,3,0,Math.PI*2);x.fill();}
  x.fillStyle="#617567";x.fillText("共同22关节 / 局部姿态 / 世界Root未知",15,h-14);
 }
 function timeline(id,lo,hi){
  const c=$(id),x=c.getContext("2d"),w=c.width,h=c.height,xx=t=>(t-lo)/(hi-lo)*w;x.fillStyle="#eceee7";x.fillRect(0,0,w,h);
  list.forEach((e,i)=>{const [a,b]=e.source_interval;if(b<=lo||a>=hi)return;x.fillStyle=e===selected?"#91b9a5":i%2?"#c0ccbb":"#d9c6a9";x.fillRect(xx(Math.max(a,lo)),14,(Math.min(b,hi)-Math.max(a,lo))/(hi-lo)*w,h-30);x.strokeStyle=e===selected?"#245f66":"#fffdf7";x.lineWidth=e===selected?3:1;x.strokeRect(xx(Math.max(a,lo)),14,(Math.min(b,hi)-Math.max(a,lo))/(hi-lo)*w,h-30);});
  if(id==="zoom"){
   const marks=[["#40866e",selected.safe_anchor?.frame],["#bb5540",selected.controlled_peak],["#cc963d",selected.original_peak]];
   for(const [color,t] of marks)if(t!==null&&t!==undefined&&t>=lo&&t<hi){x.strokeStyle=color;x.lineWidth=2;x.beginPath();x.moveTo(xx(t),3);x.lineTo(xx(t),h-15);x.stroke();}
   x.fillStyle="#607669";x.font="14px sans-serif";x.fillText("来源帧 "+lo+" → "+(hi-1),5,h-2);
  }
  x.strokeStyle="#bd553b";x.lineWidth=2;x.beginPath();x.moveTo(xx(frame),0);x.lineTo(xx(frame),h);x.stroke();
 }
 function refresh(){
  frame=Math.max(0,Math.min(src.frames-1,Math.floor(frame)));const [a,b]=selected.source_interval,inside=frame>=a&&frame<b;
  $("clock").textContent="来源 "+frame+" / "+(src.frames-1)+" · "+(frame/30).toFixed(3)+" s";$("frameInput").value=frame;$("scrub").value=frame;
  pose("sourceMotion",true,src.category+" · "+src.uid);pose("cutMotion",inside,"事件 "+(list.indexOf(selected)+1)+" · 完全原帧切片");
  $("sourceFrame").textContent="来源全局帧 "+frame+" · 原速30FPS";
  $("cutFrame").textContent=inside?"事件局部帧 "+(frame-a)+" / "+(b-a-1)+" ↔ 来源帧 "+frame:"所选事件来源区间 ["+a+", "+b+")，当前在片段之外";
  timeline("timeline",0,src.frames);timeline("zoom",Math.max(0,a-60),Math.min(src.frames,b+60));
 }
 const phaseNames={upper_pose_hold:"定格候选",upper_motion_onset:"启动候选",upper_direction_turn:"转折候选",upper_speed_peak:"速度峰候选",upper_deceleration_peak:"减速候选"};
 function select(e,seek=true){
  selected=e;$("event").value=e.event_uid;if(seek){mode=null;frame=e.source_interval[0];}
  const [a,b]=e.source_interval;$("interval").textContent="切分来源 ["+a+", "+b+") · "+(a/30).toFixed(3)+"–"+(b/30).toFixed(3)+" s · "+(b-a)+"帧 / "+((b-a)/30).toFixed(3)+"s。原帧顺序和速度未改动。";
  $("phases").replaceChildren(...e.phases.map(p=>{const button=document.createElement("button");button.textContent=(phaseNames[p.type]||p.type)+" · 来源帧"+p.frame;button.onclick=()=>seekFrame(p.frame);return button;}));
  $("details").textContent=JSON.stringify({source:{uid:src.uid,frames:src.frames,native_sha256:src.native_sha256,source_sha256:src.source_sha256,uncovered_frames:src.uncovered_frames,overlap_frames:src.overlap_frames},event:e,display:d.display,Root:d.Root_truth,audio:d.audio,generation:false},null,2);
  document.querySelectorAll("#rows tr").forEach(r=>r.classList.toggle("selected",r.dataset.uid===e.event_uid));refresh();
 }
 function table(){
  const q=$("filter").value.trim().toLowerCase(),body=$("rows");body.replaceChildren();
  list.forEach((e,i)=>{const [a,b]=e.source_interval;if(q&&!((i+1)+" "+e.event_uid+" "+a+" "+b).toLowerCase().includes(q))return;const row=document.createElement("tr");row.dataset.uid=e.event_uid;row.className=e===selected?"selected":"";
   [String(i+1)+" · "+e.event_uid.slice(-10),"["+a+", "+b+")",((b-a)/30).toFixed(2)+" s",e.anchor_status].forEach(t=>{const td=document.createElement("td");td.textContent=t;row.append(td);});row.onclick=()=>select(e);body.append(row);
  });
 }
 function setSource(){
  mode=null;src=d.sources.find(s=>s.uid===$("source").value);values();list=src.event_uids.map(uid=>events.get(uid));selected=list[0];$("filter").value="";
  $("event").replaceChildren(...list.map((e,i)=>opt(e.event_uid,(i+1)+" · ["+e.source_interval.join(",")+") · "+((e.duration_frames)/30).toFixed(2)+"s")));
  $("scrub").max=src.frames-1;$("frameInput").max=src.frames-1;camera();table();select(selected);
 }
 function seekFrame(f){mode=null;frame=Math.max(0,Math.min(src.frames-1,Math.round(f)));refresh();}
 function start(kind){
  const [a,b]=selected.source_interval;
  if(kind==="cut"&&(frame<a||frame>=b-1))frame=a;
  if(kind==="source"&&frame>=src.frames-1)frame=0;
  mode=kind;baseFrame=frame;baseTime=performance.now();refresh();
 }
 $("source").replaceChildren(...d.sources.map(s=>opt(s.uid,s.category+" · "+s.uid+" · "+s.event_uids.length+"事件")));$("source").value=d.sources[0].uid;
 $("inventory").textContent=d.sources.length+"条连续来源录制 · "+d.events.length+"个实际切分事件。左栏看来源，右栏看所选切片；同一来源帧使用完全相同的姿态。";
 $("source").onchange=setSource;$("event").onchange=()=>select(events.get($("event").value));$("filter").oninput=table;
 $("playSource").onclick=()=>start("source");$("playCut").onclick=()=>start("cut");$("pause").onclick=()=>{mode=null;refresh();};
 $("prevFrame").onclick=()=>seekFrame(frame-1);$("nextFrame").onclick=()=>seekFrame(frame+1);$("startSource").onclick=()=>seekFrame(0);$("startCut").onclick=()=>seekFrame(selected.source_interval[0]);
 $("prevEvent").onclick=()=>select(list[Math.max(0,list.indexOf(selected)-1)]);$("nextEvent").onclick=()=>select(list[Math.min(list.length-1,list.indexOf(selected)+1)]);
 $("jump").onclick=()=>seekFrame(Number($("frameInput").value));$("scrub").oninput=e=>seekFrame(Number(e.target.value));
 $("rate").onchange=()=>{baseFrame=frame;baseTime=performance.now();};$("camera").onchange=()=>{camera();refresh();};
 function clickClock(event,zoom){const r=event.currentTarget.getBoundingClientRect(),lo=zoom?Math.max(0,selected.source_interval[0]-60):0,hi=zoom?Math.min(src.frames,selected.source_interval[1]+60):src.frames;
  const f=Math.min(src.frames-1,Math.max(0,Math.floor(lo+(event.clientX-r.left)/r.width*(hi-lo))));mode=null;
  const e=list.find(e=>e.source_interval[0]<=f&&f<e.source_interval[1]);if(e)select(e,false);frame=f;refresh();
 }
 $("timeline").onclick=e=>clickClock(e,false);$("zoom").onclick=e=>clickClock(e,true);
 function tick(now){
  if(mode){let f=Math.floor(baseFrame+(now-baseTime)/1000*30*Number($("rate").value));const [a,b]=selected.source_interval;
   if(mode==="cut"&&f>=b){if($("loop").checked){f=a+(f-a)%(b-a);baseFrame=f;baseTime=now;}else{f=b-1;mode=null;}}
   if(mode==="source"&&f>=src.frames){f=src.frames-1;mode=null;}
   if(mode==="source"&&$("follow").checked){const e=list.find(e=>e.source_interval[0]<=f&&f<e.source_interval[1]);if(e&&e!==selected)select(e,false);}
   if(f!==frame){frame=f;refresh();}
  }
  requestAnimationFrame(tick);
 }
 setSource();requestAnimationFrame(tick);
})().catch(e=>{document.getElementById("inventory").textContent="读取失败："+e.message;console.error(e);});
