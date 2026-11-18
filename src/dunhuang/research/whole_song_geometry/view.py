"""Offline full-audio root-relative 3D source/bridge research playback."""
import json,base64
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import write,sha
from dunhuang.research.strong_baseline_r52.view import finish,encode

def run(out,result,P,parents):
 folder=out/'viewer';folder.mkdir();audio_data='data:audio/wav;base64,'+base64.b64encode(Path(result['audio']).read_bytes()).decode();jerk=np.r_[np.zeros(3),np.sqrt(np.mean((np.diff(P.astype(float),n=3,axis=0)*27000)**2,axis=(1,2)))];case=dict(label='dunhuangwu2 · 完整音乐几何整曲',frames=len(P),origin=0,audio=audio_data,methods=[dict(name='Heuristic + E1-W + G0/G1 · 22关节 Root-relative',data=encode(P),jerk=jerk.tolist())],meta=dict(scope='几何整曲研究预览；保留真实来源并合法重定时，过渡为几何构造。世界Root／接触未知，Native24不可评；不是Diffusion重新演绎。',regions=result['regions'],events=result['selected'],connections=result['connections'],quality=result['kinematics']))
 finish(folder,[case],parents);p=folder/'index.html';page=p.read_text(encoding='utf8').replace('R52','Whole Song').replace('保留 Heuristic 强基线；生成比较同一完整来源与几何基线，修复评价时间坐标。来源阶段为观测运动学代理，世界 Root／接触及 Native24 尚未取得完整资格。','dunhuangwu2 完整音乐输入；真实ChangE Core＋合法E1-W＋冻结G0/G1过渡。仅Root-relative身体22关节，世界Root／接触未知，Native24不可评。')
 page=page.replace('<section id="panels">','<p><label>3D观察方位 <input id="yaw" type="range" min="-180" max="180" value="-30"></label></p><div id="eventbar"></div><section id="panels">')
 page=page.replace("160+110*(.85*p[k]-.5*p[k+2])","160+110*(Math.cos(Number(document.getElementById('yaw').value)*Math.PI/180)*p[k]+Math.sin(Number(document.getElementById('yaw').value)*Math.PI/180)*p[k+2])")
 extra="""document.getElementById('yaw').oninput=()=>show(frame);const bar=document.getElementById('eventbar');row.meta.events.forEach((e,i)=>{let b=document.createElement('button');b.textContent=(i+1)+' · '+e.source_uid.slice(0,16)+' ['+(e.target_interval[0]/30).toFixed(1)+','+(e.target_interval[1]/30).toFixed(1)+')s';b.onclick=()=>{audio.pause();running=false;audio.currentTime=e.target_interval[0]/30;show(e.target_interval[0])};bar.append(b)});"""
 page=page.replace('</script></html>',extra+'</script></html>');p.write_text(page,encoding='utf8');write(folder/'protocol.json',dict(frames=len(P),full_audio=True,root_relative=True,audio_embedded_sha256=result['audio_sha256'],physical_qualified=False,files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.iterdir()) if p.name!='protocol.json']))
