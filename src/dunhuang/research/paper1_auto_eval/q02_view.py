"""Actual arrays, direction glyphs and declared write masks, one shared clock."""
import json
import numpy as np
import soundfile as sf
from .q02_common import ROOT,read,write,sha,prepare,fresh
from .view import template
from . import music
from dunhuang.research.complete_r36.core import relative_fk

def build(rows,arrays,out):
    ctx=prepare();cats=music.catalogs();items=[];out=fresh(out)
    for case in sorted({r['case_key'] for r in rows}):
        own=[r for r in rows if r['case_key']==case];base=next(r for r in own if r['label']=='SourceOriginal');variants=[];axes=[];masks=[]
        for r in own:
            if r['status']!='OBSERVED':continue
            R=arrays[r['key']+'__R'];P=relative_fk(R,ctx['J'],ctx['parents']);W=np.empty_like(R);W[:,0]=R[:,0]
            for j in range(1,22):W[:,j]=W[:,ctx['parents'][j]]@R[:,j]
            endpoints=P[:,[20,21],None,:]+.18*W[:,[20,21]].swapaxes(-1,-2);axes.append(np.round(endpoints,5).tolist());masks.append(r['write_frames']);variants.append(dict(label=r['label'],P=np.round(P,4).tolist()))
        cat=cats[base['assigned_music_song']];audio=cat['audio_path'];y,sr=sf.read(audio,always_2d=True);A=int(round(base['music_start_frame']/30*sr));n=int(round(base['frames']/30*sr));assert A+n<=len(y);name=case+'.wav';sf.write(out/name,y[A:A+n],sr,subtype='PCM_16')
        items.append(dict(label=base['source_uid'],classification='PERTURBATION',frames=base['frames'],core=base['frames'],audio=name,status='原速真实来源与前瞻合成控制；音频为登记目标音乐，非ChangE原录配对',variants=variants,axes_by_variant=axes,edit_by_variant=masks,evidence='橙色为全部允许写域；非静止控制首末4帧保持原数组。三色腕轴来自真实父链旋转，统一0.18米，仅为方向示意。静止、合法保持及伪合拍不预设艺术好坏；所有合成控制永不进入编舞候选。'))
    t=template().replace('for(let v of views){','for(let [vi,v] of views.entries()){');mark="c.stroke()}c.fillStyle='#276b89';";assert mark in t
    t=t.replace(mark,"c.stroke()}let ax=D.items[current].axes_by_variant?.[vi]?.[f];if(ax)for(let k=0;k<2;k++)for(let j=0;j<3;j++){let e=ax[k][j];c.strokeStyle=['#ed7d79','#8edb89','#82b8f0'][j];c.beginPath();c.moveTo(...p[20+k]);c.lineTo(150+115*(e[0]+.35*e[2]),180-115*(e[1]-.15*e[2]));c.stroke()}c.fillStyle='#276b89';")
    d=dict(items=items,parents=ctx['parents'].tolist(),anonymous=False,task='music_coordination',variant_select=True,intro='Q02先看实际控制：左栏真实来源原速；右栏可切换静止、末端噪声、合法时间调整与小幅节拍周期振荡。共用音乐、拖动、倍速和逐帧。指标将在观看验收后冻结运行；此页没有自动质量排名。')
    (out/'index.html').write_bytes(t.replace('__TITLE__','Q02 · 自动指标盲区与伪合拍控制').replace('__DATA__',json.dumps(d,ensure_ascii=False,separators=(',',':'))).encode())
    write(out/'protocol.json',dict(files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()},human_records=0,can_route=False))
