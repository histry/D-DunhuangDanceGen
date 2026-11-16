"use strict";
(async()=>{
 const $=id=>document.getElementById(id),opt=(v,t)=>{const e=document.createElement("option");e.value=v;e.textContent=t;return e;};
 const bytes=Uint8Array.from(atob(window.DANCE_BUNDLE_B64),c=>c.charCodeAt(0));
 const data=JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).text());
 delete window.DANCE_BUNDLE_B64;const cache=new Map();let current,frame=0,selected=1,tab="plan",camera;
 const audio=$("audio");
 function get(id){if(!cache.has(id)){const t=data.tracks[id],v=Uint8Array.from(atob(t.data),c=>c.charCodeAt(0));cache.set(id,{x:new Float32Array(v.buffer),frames:t.shape[0]});}return cache.get(id);}
 function point(id,t,j){const a=get(id),n=(t*22+j)*3;return [a.x[n],a.x[n+1],a.x[n+2]];}
 const project=p=>[Math.cos(-.45)*p[0]+Math.sin(-.45)*p[2],p[1]-.18*(-Math.sin(-.45)*p[0]+Math.cos(-.45)*p[2])];
 function cameras(){let lo=[Infinity,Infinity],hi=[-Infinity,-Infinity];const ids=[current.Old,current.Graph,...Object.values(current.generations).flatMap(g=>[g.Frozen,g.Context])];const local=$("view").value==="local";
  for(const id of ids){const n=get(id).frames;for(let t=0;t<n;t++){const root=local?point(id,t,0):[0,0,0];for(let j=0;j<22;j++){const p=project(point(id,t,j).map((v,k)=>v-root[k]));for(let k=0;k<2;k++){lo[k]=Math.min(lo[k],p[k]);hi[k]=Math.max(hi[k],p[k]);}}}}
  camera={center:lo.map((v,k)=>(v+hi[k])/2),span:lo.map((v,k)=>Math.max(.5,hi[k]-v))};
 }
 function pose(id,t,n,label,forceLocal=false){
  const c=$("motion"+n),x=c.getContext("2d"),w=c.width,h=c.height;x.fillStyle="#f3f6ef";x.fillRect(0,0,w,h);x.fillStyle="#5a7470";x.font="14px sans-serif";x.fillText(label,10,23);
  if(!id||t<0||t>=get(id).frames){x.fillText("该音乐帧尚未生成／无该候选",12,h/2);return;}
  const local=forceLocal||$("view").value==="local",root=local?point(id,t,0):[0,0,0],scale=Math.min((w-65)/camera.span[0],(h-80)/camera.span[1]);
  const xy=Array.from({length:22},(_,j)=>{const p=project(point(id,t,j).map((v,k)=>v-root[k]));return [w/2+(p[0]-camera.center[0])*scale,h*.54-(p[1]-camera.center[1])*scale];});
  x.lineWidth=3;for(let j=1;j<22;j++){x.strokeStyle=j%2?"#27717b":"#b76c3c";x.beginPath();x.moveTo(...xy[data.parents[j]]);x.lineTo(...xy[j]);x.stroke();}
  x.fillStyle="#344b4b";for(const p of xy){x.beginPath();x.arc(...p,2.6,0,Math.PI*2);x.fill();}
  x.fillStyle="#697b6f";x.fillText(local?"局部身体 / Root-relative":"固定世界 / 保留生成 Root",10,h-13);
 }
 function generation(){return current.type==="plan"?(current.generations[$("planChoice").value]?.selected_slot===selected?current.generations[$("planChoice").value]:null):current.generations.study;}
 function timeline(){
  const c=$("timeline"),x=c.getContext("2d"),w=c.width,h=c.height;x.fillStyle="#fafbf6";x.fillRect(0,0,w,h);
  const peaks=current.waveform,max=Math.max(...peaks,.001);x.strokeStyle="#819f8f";x.beginPath();peaks.forEach((v,i)=>{const a=i/peaks.length*w,b=v/max*37;x.moveTo(a,42-b);x.lineTo(a,42+b);});x.stroke();
  x.strokeStyle="#c6a779";for(const t of current.beats){const a=t/current.frames*w;x.beginPath();x.moveTo(a,82);x.lineTo(a,92);x.stroke();}
  for(let i=0;i<current.slots.length;i++){const s=current.slots[i],a=s.start/current.frames*w,b=(s.end-s.start)/current.frames*w;x.fillStyle=i%2?"#c1ceba":"#d6c6ad";x.fillRect(a,100,b,27);x.strokeStyle=i===selected?"#24686e":"white";x.lineWidth=i===selected?3:1;x.strokeRect(a,100,b,27);}
  const a=frame/current.frames*w;x.strokeStyle="#b75335";x.lineWidth=2;x.beginPath();x.moveTo(a,0);x.lineTo(a,h);x.stroke();
 }
 function diagnostic(){
  const c=$("diagnostic"),x=c.getContext("2d"),w=c.width,h=c.height;x.clearRect(0,0,w,h);const g=generation();if(!g)return;
  const v=g.diagnostics[$("metric").value],m=Math.max(.00001,...v.filter(Number.isFinite));x.strokeStyle="#ae6341";x.beginPath();let open=false;
  v.forEach((p,i)=>{if(!Number.isFinite(p)){open=false;return;}const a=30+i/(v.length-1)*(w-45),b=h-25-p/m*(h-50);if(open)x.lineTo(a,b);else x.moveTo(a,b);open=true;});x.stroke();
  const t=frame-g.start;if(t>=0&&t<v.length){x.strokeStyle="#27717b";const a=30+t/(v.length-1)*(w-45);x.beginPath();x.moveTo(a,0);x.lineTo(a,h-20);x.stroke();}
  $("metricValue").textContent=t>=0&&t<v.length&&Number.isFinite(v[t])?" 当前帧 "+v[t].toPrecision(5):" 当前帧无诊断";
 }
 function refresh(){
  if(!current)return;frame=Math.max(0,Math.min(current.frames-1,Math.round(frame)));$("scrub").value=frame;$("frameInput").value=frame;
  $("clock").textContent=(frame/30).toFixed(3)+" s · 音乐帧 "+frame+" / "+(current.frames-1);timeline();
  const plan=current.type==="plan",g=generation(),n=g?frame-g.start:-1;
  if(tab==="plan"&&plan){
   ["旧计划预演 · 未生成连接","Graph 新计划预演 · 未生成连接","替代右事件 · 只读"].forEach((v,i)=>$("title"+i).textContent=v);
   pose(current.Old,frame,0,"原活动启发式计划 / 世界Root未知",true);pose(current.Graph,frame,1,"转场图计划 / 世界Root未知",true);
   const alt=current.alternatives[selected]?.[Number($("alternative").value)||0];pose(alt?.track,alt?frame-alt.start:-1,2,alt?.source_uid||"候选",true);
   $("frame0").textContent="旧计划音乐帧 "+frame;$("frame1").textContent="新计划音乐帧 "+frame;$("frame2").textContent=alt?"候选来源 "+alt.source_uid+" · 音乐帧 "+frame:"无候选";
   $("scope").textContent="整曲来源排列预演，事件切换未生成。点击时间线或选择边界查看端点、动态与可编辑时间。";
  }else{
   ["按音乐对齐的参考","Frozen R33 · 无 Context","冻结来源 + 已训练 Context"].forEach((v,i)=>$("title"+i).textContent=v);
   const id=plan?current[$("planChoice").value]:current.Graph;pose(id,frame,0,plan?"计划来源 / 异源转场非连续真值":"真实来源参考",plan||current.type==="source");
   pose(g?.Frozen,n,1,"固定噪声 · 原始完整采样");pose(g?.Context,n,2,"相同噪声 · Context 主效果");
   $("frame0").textContent="参考音乐帧 "+frame;
   [1,2].forEach(i=>$("frame"+i).textContent=g&&n>=0&&n<256?"实际动作帧 "+n+" · 音乐帧 "+frame:"此事件／音乐帧尚未生成");
   $("scope").textContent=g?(plan?"当前事件主体及编辑边缘写入责任 "+JSON.stringify(g.binding.write_interval)+"；Core保护 "+JSON.stringify(g.binding.protected_core)+"。其余预测不计为完整邻事件。":current.type==="paired"?"真实连续 AIST 缺口 [64,160)，两端各8帧观察；不硬覆盖。":"来源保持 Core [64,160)；上下文来自同录制真实身体，Root/接触未知。"):"未生成该事件。请选择有登记真实短窗的边界；不替代为其他来源输出。";
  }
  diagnostic();window.DANCE_VIEW_STATE={caseId:current.id,frame,selectedSlot:selected,tab,paused:audio.paused,rate:audio.playbackRate,localGenerationFrame:g?frame-g.start:null,planArm:$("planChoice").value};
 }
 function selection(){
  if(current.type==="plan"){
   const b=current.boundaries[selected-1];$("alternative").replaceChildren(...current.alternatives[selected].map((e,i)=>opt(String(i),e.source_uid+" · 活动 "+e.music_score.toFixed(4))));
   const stats=data.selection.songs.find(s=>s.song===current.id);$("costs").textContent="全曲平均转场代理："+stats.old_mean_edge.toFixed(4)+" → "+stats.new_mean_edge.toFixed(4)+"；平均活动匹配退步 "+stats.music_mean_regret.toFixed(5)+"（预算≤0.02）。代理下降不是独立质量证据。";
   const table=document.createElement("table"),head=document.createElement("tr");["边界指标","原计划","新计划"].forEach(t=>{const c=document.createElement("th");c.textContent=t;head.append(c);});table.append(head);
   const names={angle_rad:"全身角差 rad",body_gap_m:"局部FK差 m",wrist_gap_m:"双腕差 m",lower_gap_m:"下肢差 m",velocity_gap_m_s:"速度差 m/s",acceleration_gap_m_s2:"加速度差 m/s²",editable_frames:"可编辑预算 帧",cost:"综合转场代理"};
   for(const [key,name] of Object.entries(names)){const row=document.createElement("tr");[name,b.old[key].toFixed(4),b.new[key].toFixed(4)].forEach(t=>{const c=document.createElement("td");c.textContent=t;row.append(c);});table.append(row);}
   $("boundaryMetrics").replaceChildren(table);$("boundaryDetails").textContent=JSON.stringify({contract:b.contract,current_event:current.slots[selected].candidates[0],alternatives:b.alternative_right},null,2);
  }
  const g=generation();$("details").textContent=JSON.stringify({music:current.audio_provenance,case:current.details,actual_generation:g?.binding||"NOT_GENERATED",context_model:data.context_model,Native24:"NOT_EVALUABLE_22_OF24",additional_ablations:0,whole_song_generations:0},null,2);
  refresh();
 }
 function setTab(value){tab=value;$("planPanel").hidden=value!=="plan"||current.type!=="plan";$("genPanel").hidden=value!=="gen";$("planTab").classList.toggle("active",value==="plan");$("genTab").classList.toggle("active",value==="gen");refresh();}
 function setCase(){
  audio.pause();current=data.cases.find(c=>c.id===$("case").value);frame=0;selected=current.initial_slot;audio.src=current.audio;audio.playbackRate=Number($("rate").value);
  $("scrub").max=current.frames-1;$("frameInput").max=current.frames-1;
  const plan=current.type==="plan";$("boundary").disabled=!plan;$("planTab").disabled=!plan;$("planChoiceLabel").hidden=!plan;
  $("boundary").replaceChildren(...(plan?current.boundaries.map((b,i)=>opt(String(i+1),"边界 "+(i+1)+" · "+(b.contract.boundary_frame/30).toFixed(2)+"s")):[opt("0","连续来源研究窗")]));
  $("boundary").value=String(selected);$("notice").textContent=plan?"原／新计划可全程播放；每计划仅一个登记窗口有真实生成。其他来源未自动取得适配资格。":"正确 Context 对冻结基线的主效果；holdout 是历史曝光开发来源，未执行左右／错邻居消融。";
  cameras();selection();setTab(plan?tab:"gen");
 }
 function seek(t){audio.pause();frame=Math.max(0,Math.min(current.frames-1,Math.round(t)));audio.currentTime=frame/30;refresh();}
 $("case").replaceChildren(...data.cases.map(c=>opt(c.id,c.name)));$("case").value="song0";
 $("case").onchange=setCase;$("planTab").onclick=()=>setTab("plan");$("genTab").onclick=()=>setTab("gen");
 $("play").onclick=()=>audio.paused?audio.play().catch(e=>$("notice").textContent=e.message):audio.pause();
 audio.onplay=()=>{$("play").textContent="暂停音乐";};audio.onpause=()=>{$("play").textContent="播放音乐";if(current){frame=Math.round(audio.currentTime*30);refresh();}};
 audio.onseeked=()=>{frame=audio.paused?Math.round(audio.currentTime*30):Math.floor(audio.currentTime*30+1e-4);refresh();};
 audio.onended=()=>{frame=current.frames-1;refresh();};
 $("prev").onclick=()=>seek(frame-1);$("next").onclick=()=>seek(frame+1);$("jump").onclick=()=>seek(Number($("frameInput").value));
 $("scrub").oninput=e=>seek(Number(e.target.value));$("rate").onchange=e=>audio.playbackRate=Number(e.target.value);
 $("boundary").onchange=e=>{selected=Number(e.target.value);selection();seek(current.slots[selected].start);};
 $("planChoice").onchange=selection;$("view").onchange=()=>{cameras();refresh();};$("alternative").onchange=refresh;$("metric").onchange=refresh;
 $("timeline").onclick=e=>{const r=e.currentTarget.getBoundingClientRect(),t=Math.floor((e.clientX-r.left)/r.width*current.frames);if(current.type==="plan"){const i=current.slots.findIndex(s=>s.start<=t&&t<s.end);if(i>0){selected=i;$("boundary").value=String(i);selection();}}seek(t);};
 $("diagnostic").onclick=e=>{const g=generation();if(!g)return;const r=e.currentTarget.getBoundingClientRect();const f=((e.clientX-r.left)/r.width*1200-30)/1155;seek(g.start+Math.round(Math.max(0,Math.min(1,f))*255));};
 function tick(){if(!audio.paused){const f=Math.floor(audio.currentTime*30+1e-4);if(f!==frame){frame=f;refresh();}}requestAnimationFrame(tick);}
 setCase();requestAnimationFrame(tick);
})().catch(e=>{document.getElementById("notice").textContent="页面加载失败："+e.message;console.error(e);});
