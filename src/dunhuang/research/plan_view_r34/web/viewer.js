"use strict";
(async()=>{
 const $=id=>document.getElementById(id);
 const bytes=Uint8Array.from(atob(window.DANCE_BUNDLE_B64),c=>c.charCodeAt(0));
 const text=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
 const data=JSON.parse(text);window.DANCE_VIEW_DATA=data;delete window.DANCE_BUNDLE_B64;
 const cache=new Map();
 function get(id){if(!cache.has(id)){const t=data.tracks[id],b=Uint8Array.from(atob(t.data),c=>c.charCodeAt(0));cache.set(id,{values:new Float32Array(b.buffer),frames:t.shape[0]});}return cache.get(id);}
 const audio=$("audio");let current,selected=0,candidateIndex=0,frame=0,tab=data.initial_tab,originalIndex=0,origPlaying=false,origStart=0;
 let worldCamera={center:[0,0],span:[3,3]},localCamera=worldCamera;
 const option=(value,label)=>{const x=document.createElement("option");x.value=value;x.textContent=label;return x;};
 function point(track,t,j){const x=get(track),i=(t*22+j)*3;return [x.values[i],x.values[i+1],x.values[i+2]];}
 function project(p){const c=Math.cos(-.45),s=Math.sin(-.45);return [c*p[0]+s*p[2],p[1]-.18*(-s*p[0]+c*p[2])];}
 function camera(ids,local){let lo=[Infinity,Infinity],hi=[-Infinity,-Infinity];for(const id of ids.filter(Boolean)){const a=get(id);for(let t=0;t<a.frames;t++){const r=local?point(id,t,0):[0,0,0];for(let j=0;j<22;j++){const p=point(id,t,j).map((x,k)=>x-r[k]),q=project(p);for(let k=0;k<2;k++){lo[k]=Math.min(lo[k],q[k]);hi[k]=Math.max(hi[k],q[k]);}}}}return {center:lo.map((x,k)=>(x+hi[k])/2),span:lo.map((x,k)=>Math.max(.5,hi[k]-x))};}
 function pose(id,t,canvasId,label,local=false,cameraOverride=null){const canvas=$(canvasId),ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);ctx.fillStyle="#f4f6f0";ctx.fillRect(0,0,w,h);
  ctx.fillStyle="#617575";ctx.font="13px sans-serif";ctx.fillText(label,12,20);if(!id||t<0||t>=get(id).frames){ctx.fillText("该时刻没有真实对应输出",12,h/2);return;}
  const cam=cameraOverride||(local?localCamera:worldCamera),scale=Math.min((w-58)/cam.span[0],(h-80)/cam.span[1]),root=local?point(id,t,0):[0,0,0],xy=[];
  for(let j=0;j<22;j++){const p=project(point(id,t,j).map((x,k)=>x-root[k]));xy.push([w/2+(p[0]-cam.center[0])*scale,h*.54-(p[1]-cam.center[1])*scale]);}
  ctx.lineWidth=3;for(let j=1;j<22;j++){ctx.strokeStyle=j%2?"#246b76":"#b66c3e";ctx.beginPath();ctx.moveTo(...xy[data.parents[j]]);ctx.lineTo(...xy[j]);ctx.stroke();}
  ctx.fillStyle="#293d44";for(const p of xy){ctx.beginPath();ctx.arc(...p,2.7,0,Math.PI*2);ctx.fill();}
  ctx.fillStyle="#6e746d";ctx.fillText(local?"局部姿态 / Root-relative":"固定世界视角 / 不逐帧重居中",12,h-13);
 }
 const slot=()=>current.slots[selected],candidate=()=>slot().candidates[candidateIndex],generation=()=>slot().generation;
 function computeCameras(){const g=generation();const ids=[current.reference_track,g?.baseline,g?.candidate,...Object.values(g?.baseline_variants||{}),candidate()?.aligned_track,data.originals[candidate()?.original]?.track];worldCamera=camera(ids,false);localCamera=camera(ids,true);}
 function frameText(){const g=generation(),n=g?frame-g.window_start:null;$("clock").textContent=(frame/30).toFixed(3)+" s · 音乐帧 "+frame+" / "+(current.frames-1);
  $("refFrame").textContent="来源参考：音乐帧 "+frame+" · 世界Root UNKNOWN";
  $("baseFrame").textContent=g&&n>=0&&n<g.frames?"基线实际动作帧 "+n+" · 音乐帧 "+frame:"基线：该音乐帧尚未生成";
  $("candidateFrame").textContent=g&&n>=0&&n<g.frames?"候选实际动作帧 "+n+" · 音乐帧 "+frame:"候选：该音乐帧尚未生成";
 }
 function timeline(){const c=$("timeline"),ctx=c.getContext("2d"),w=c.width,h=c.height;ctx.clearRect(0,0,w,h);
  ctx.fillStyle="#f3f2ea";ctx.fillRect(0,0,w,h);const peaks=current.waveform,max=Math.max(...peaks,.001);ctx.strokeStyle="#70908b";ctx.beginPath();
  peaks.forEach((v,i)=>{const x=i/(peaks.length-1)*w,y=v/max*40;ctx.moveTo(x,49-y);ctx.lineTo(x,49+y);});ctx.stroke();
  ctx.strokeStyle="#d0a371";for(const b of current.beats){let x=b/current.frames*w;ctx.beginPath();ctx.moveTo(x,90);ctx.lineTo(x,101);ctx.stroke();}
  const palette=["#d7bfa0","#b1c9ba","#bac5d5","#d3b9b7","#c9ccab","#c9bdd6","#b7d0cb","#d8cbad"];
  current.slots.forEach((s,i)=>{const x=s.start/current.frames*w,width=(s.end-s.start)/current.frames*w;ctx.fillStyle=palette[i%8];ctx.fillRect(x,105,width,30);ctx.strokeStyle=i===selected?"#234f57":"#fffdf8";ctx.lineWidth=i===selected?3:1;ctx.strokeRect(x,105,width,30);if(width>20){ctx.fillStyle="#354b4a";ctx.font="11px sans-serif";ctx.fillText(String(i+1),x+3,124);}});
  const x=frame/current.frames*w;ctx.strokeStyle="#b55035";ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();
 }
 function warp(){const c=$("warp"),ctx=c.getContext("2d"),w=c.width,h=c.height,v=candidate().time_warp;ctx.clearRect(0,0,w,h);if(!v?.length)return;
  const xmin=v[0][0],xmax=v[v.length-1][0],ymin=v[0][1],ymax=v[v.length-1][1],px=x=>40+(x-xmin)/Math.max(1,xmax-xmin)*(w-65),py=y=>h-30-(y-ymin)/Math.max(1,ymax-ymin)*(h-55);
  ctx.strokeStyle="#2d7575";ctx.lineWidth=2;ctx.beginPath();v.forEach(([x,y],i)=>i?ctx.lineTo(px(x),py(y)):ctx.moveTo(px(x),py(y)));ctx.stroke();ctx.fillStyle="#98633c";for(const [x,y] of v){ctx.beginPath();ctx.arc(px(x),py(y),v.length>10?1:3,0,Math.PI*2);ctx.fill();}
  ctx.fillStyle="#617575";ctx.font="12px sans-serif";ctx.fillText("来源原始帧 "+xmin.toFixed(1)+" → "+xmax.toFixed(1),40,h-8);ctx.fillText("音乐帧 "+ymin+" → "+ymax,40,14);
  const local=frame-candidate().aligned_global_start;const position=candidate().source_positions[local];if(position!==undefined){ctx.fillStyle="#b54f35";ctx.beginPath();ctx.arc(px(position),py(frame),5,0,Math.PI*2);ctx.fill();}
  $("warpCaption").textContent="来源→音乐时间图；速度=Δ来源帧/Δ音乐帧。"+(position!==undefined?"当前音乐帧 "+frame+" → 来源帧 "+position.toFixed(3):"当前不在该候选时间范围")+"。单调不保证动力学连续。";
 }
 function diagnosis(){const c=$("diagnostic"),ctx=c.getContext("2d"),w=c.width,h=c.height,g=generation();ctx.clearRect(0,0,w,h);if(!g){ctx.fillStyle="#617575";ctx.fillText("该事件没有真实生成诊断",15,90);$("metricValue").textContent="";return;}
  const key=$("metric").value,v=g.diagnostics[key],valid=v.filter(x=>x!==null&&Number.isFinite(x)),max=Math.max(...valid,.00001);ctx.strokeStyle="#b5643c";ctx.lineWidth=1.6;ctx.beginPath();let open=false;
  v.forEach((x,i)=>{if(x===null){open=false;return;}let xx=30+i/(v.length-1)*(w-45),yy=h-25-x/max*(h-48);if(open)ctx.lineTo(xx,yy);else ctx.moveTo(xx,yy);open=true;});ctx.stroke();
  const t=frame-g.window_start;if(t>=0&&t<v.length){const x=30+t/(v.length-1)*(w-45);ctx.strokeStyle="#236972";ctx.beginPath();ctx.moveTo(x,10);ctx.lineTo(x,h-20);ctx.stroke();}
  ctx.fillStyle="#617575";ctx.font="12px sans-serif";ctx.fillText(max.toPrecision(4),5,15);ctx.fillText("音乐帧 "+g.window_start+" — "+(g.window_start+g.frames-1),30,h-5);
  $("metricValue").textContent=t>=0&&t<v.length&&v[t]!==null?"当前帧 "+v[t].toPrecision(5):"该帧无此诊断";
 }
 function refresh(){frame=Math.max(0,Math.min(current.frames-1,Math.floor(frame)));$("scrub").value=String(frame);$("frameInput").value=String(frame);frameText();timeline();warp();
  const active=current.slots.findIndex(s=>s.start<=frame&&frame<s.end);$("eventNow").textContent="当前音乐事件 "+(active<0?"—":active+1)+" · 选中 "+(selected+1);
  const g=generation(),local=$("view").value==="local",n=g?frame-g.window_start:-1;pose(current.reference_track,frame,"reference","来源局部身体 · 世界Root UNKNOWN",true,local?localCamera:worldCamera);
  pose(current.reference_track,frame,"rehearsal","编排计划预演 / 事件切换待生成",true);$("rehearsalCaption").textContent=current.reference_scope;
  let baseline=g?.baseline;if(g&&$("baseline").value!=="Frozen")baseline=g.baseline_variants[$("baseline").value]||g.baseline;
  pose(baseline,n,"baselineMotion","冻结或历史基线 · 原始输出",local);pose(g?.candidate,n,"candidateMotion","R34候选 · 原始输出",local);
  const cd=candidate(),orig=data.originals[cd.original];pose(orig.track,originalIndex,"original","原始来源 / 30FPS / 独立时钟",true);
  $("originalFrame").textContent="原始来源帧 "+(orig.source_start+originalIndex)+" / "+(orig.source_end-1)+" · 主音乐已暂停后可原速播放";
  const ci=frame-cd.aligned_global_start;pose(cd.aligned_track,ci,"aligned","按当前音乐映射的参考 / 世界Root UNKNOWN",true);$("alignedFrame").textContent=ci>=0&&ci<get(cd.aligned_track).frames?"音乐帧 "+frame+" · 来源位置 "+cd.source_positions[ci].toFixed(3):"当前音乐帧不在该候选的对齐范围";
  diagnosis();window.DANCE_VIEW_STATE={caseId:current.id,frame,tab,selectedSlot:selected,generationLocalFrame:g?frame-g.window_start:null,paused:audio.paused,rate:audio.playbackRate,rawSourceIndependent:origPlaying};
 }
 function setTab(value){tab=value;$("planning").hidden=value!=="plan";$("generation").hidden=value!=="generation";$("tabPlan").classList.toggle("active",value==="plan");$("tabGen").classList.toggle("active",value==="generation");refresh();}
 function chooseCandidate(){candidateIndex=Number($("candidate").value)||0;originalIndex=0;origPlaying=false;computeCameras();refresh();}
 function selectedEvent(){const s=slot();candidateIndex=Math.max(0,s.candidates.findIndex(c=>c.selected));$("candidate").replaceChildren(...s.candidates.map((c,i)=>option(String(i),c.source_uid+(c.selected?" · 被选中":""))));$("candidate").value=String(candidateIndex);
  $("selectedTitle").textContent="事件 "+(selected+1)+" · 音乐帧 ["+s.start+", "+s.end+")";const table=document.createElement("table"),head=document.createElement("tr");for(const x of ["候选来源","活动度启发式","SDMR全局 Shadow","同步 Shadow"]){const th=document.createElement("th");th.textContent=x;head.append(th);}table.append(head);
  for(const c of s.candidates){const tr=document.createElement("tr");for(const x of [c.source_uid+(c.selected?" ✓":""),...["heuristic","global_score","sync_score"].map(k=>c.scores[k]===null?"未计算":Number(c.scores[k]).toFixed(4))]){const td=document.createElement("td");td.textContent=x;tr.append(td);}tr.onclick=()=>{$("candidate").value=String(s.candidates.indexOf(c));chooseCandidate();};table.append(tr);}
  $("candidates").replaceChildren(table);const note=document.createElement("p");note.className="hint";note.textContent=s.candidates[0].score_scope;$("candidates").append(note);
  $("rejections").textContent=JSON.stringify({合法候选:s.legal_candidates,不可行原因:s.rejection_counts,示例:s.rejected_examples},null,2);
  const g=generation();$("genStatus").textContent=g?g.qualification+" · "+g.context_status:"尚未生成这个事件。这里不使用其他音乐、时间映射或历史片段代替。";
  $("showGen").textContent=g?"查看同事件真实生成":"该事件尚未生成";$("baseline").replaceChildren(option("Frozen",current.type==="continuous_study"?"R33 Dyn · 原96帧参考":"冻结 R33 · 当前计划条件"),...Object.keys(g?.baseline_variants||{}).map(k=>option(k,k==="FrozenFull"?"不训练 · 256帧参考":"同预算 Extra · 96帧目标")));
  $("baselineTitle").textContent="冻结 / 历史基线";$("scopeBand").replaceChildren();
  if(g){for(const [label,interval,cls] of [["身体观察",g.observation,""],["来源控制",g.source_control,"control"],...(g.contexts||[]).map(x=>["计划身体上下文",x,"ctx"])]){const span=document.createElement("span");span.className=cls;span.textContent=label+" ["+interval[0]+", "+interval[1]+")";$("scopeBand").append(span);}}
  $("details").textContent=JSON.stringify({音乐与计划:current.details,选中候选:s.candidates[candidateIndex],真实生成:g?.details||"NOT_GENERATED",当前资格:g?.qualification||"PLAN_REHEARSAL_ONLY",实际接触:"UNKNOWN",Native24:"NOT_EVALUABLE_22_OF24"},null,2);
  computeCameras();refresh();
 }
 function setCase(){audio.pause();origPlaying=false;current=data.cases.find(c=>c.id===$("case").value);selected=0;frame=0;originalIndex=0;audio.src=current.audio;audio.playbackRate=Number($("rate").value);
  $("scrub").max=String(current.frames-1);$("frameInput").max=String(current.frames-1);$("event").replaceChildren(...current.slots.map((s,i)=>option(String(i),"事件 "+(i+1)+" · "+(s.start/30).toFixed(2)+"–"+(s.end/30).toFixed(2)+"s")));
  $("event").value="0";$("notice").textContent=current.type==="full_song_plan"?"完整音乐计划预演可全程播放；只为带精确绑定的一个内部事件生成了研究短窗，其他事件尚未生成。":"R34连续覆盖主效果：完整身体观察来自同一录制的前后，不是相邻选中事件。Core肩腕门槛未通过。";
  $("clockInfo").textContent="唯一主时钟：音频 currentTime → 30FPS整数音乐帧；三栏直接绘制同一帧。来源原速使用独立时钟，不强制同步变速。";selectedEvent();setTab(tab);
 }
 function seek(f){audio.pause();origPlaying=false;frame=Math.max(0,Math.min(current.frames-1,Math.round(f)));audio.currentTime=frame/30;refresh();}
 $("case").replaceChildren(...data.cases.map(c=>option(c.id,c.name)));$("case").value=data.initial_case;
 $("case").onchange=setCase;$("tabPlan").onclick=()=>setTab("plan");$("tabGen").onclick=()=>setTab("generation");
 $("play").onclick=()=>{origPlaying=false;if(audio.paused)audio.play().catch(e=>{$("notice").textContent="音频播放失败："+e.message;});else audio.pause();};
 audio.onplay=()=>$("play").textContent="暂停音乐";audio.onpause=()=>$("play").textContent="播放音乐";
 audio.onseeked=()=>{frame=Math.floor(audio.currentTime*30+1e-5);refresh();};
 $("prev").onclick=()=>seek(frame-1);$("next").onclick=()=>seek(frame+1);$("jump").onclick=()=>seek(Number($("frameInput").value));$("frameInput").onkeydown=e=>{if(e.key==="Enter")seek(Number(e.target.value));};
 $("scrub").oninput=e=>seek(Number(e.target.value));$("rate").onchange=e=>audio.playbackRate=Number(e.target.value);
 $("event").onchange=e=>{selected=Number(e.target.value);selectedEvent();seek(slot().start);};
 $("candidate").onchange=chooseCandidate;$("view").onchange=refresh;$("metric").onchange=refresh;$("baseline").onchange=refresh;
 $("showGen").onclick=()=>{setTab("generation");if(generation())seek(slot().start);};
 $("originalPlay").onclick=()=>{audio.pause();if(origPlaying){origPlaying=false;return;}origPlaying=true;originalIndex=0;origStart=performance.now();};
 $("timeline").onclick=e=>{const r=e.currentTarget.getBoundingClientRect(),f=Math.floor((e.clientX-r.left)/r.width*current.frames);if(e.clientY-r.top>r.height*.65){const i=current.slots.findIndex(s=>s.start<=f&&f<s.end);if(i>=0){selected=i;$("event").value=String(i);selectedEvent();}}seek(f);};
 $("diagnostic").onclick=e=>{const g=generation();if(!g)return;const r=e.currentTarget.getBoundingClientRect();seek(g.window_start+Math.round((e.clientX-r.left)/r.width*(g.frames-1)));};
 function tick(time){if(!audio.paused){const f=Math.floor(audio.currentTime*30+1e-5);if(f!==frame){frame=f;refresh();}}else if(origPlaying){const n=get(data.originals[candidate().original].track).frames;originalIndex=Math.min(n-1,Math.floor((time-origStart)/1000*30));if(originalIndex===n-1)origPlaying=false;refresh();}requestAnimationFrame(tick);}
 setCase();requestAnimationFrame(tick);
})().catch(e=>{document.getElementById("notice").textContent="本地数据加载失败："+e.message;console.error(e);});
