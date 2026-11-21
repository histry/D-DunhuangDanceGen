"""Reuse the existing player and review exports; keep automatic overlays out of blind reviews."""
import json,shutil,sys
from pathlib import Path
import numpy as np
from .common import ROOT,HERE,PARENT,read,write,sha,prepare,checked_parent
from .reviews import summarize
from dunhuang.research.complete_r36.core import relative_fk

def template():
    t=(HERE.parent/'gtw_kinematic_baseline/viewer.html').read_text(encoding='utf8')
    replacements={'function draw(frame){':'function frames(){return D.items[current].frames} function draw(frame){$(\'seek\').max=frames()-1;drawEvidence(frame);','Math.min(527,frame)':'Math.min(frames()-1,frame)','${f}/527':'${f}/${frames()-1}','240/528*300':'D.items[current].core/frames()*300','48/528*300':'48/frames()*300','f/528*300':'f/frames()*300','if(f===527)':'if(f===frames()-1)'}
    for a,b in replacements.items():assert a in t;t=t.replace(a,b)
    t=t.replace('<span id="state">','<select id="category"><option value="VALID_DISTINCT">有效不同</option><option value="VALID_IDENTICAL">相同／平局控制</option><option value="UNAVAILABLE">不可评价</option><option value="FLOW_CONTROL">观察流程检查</option><option value="ALL">全部登记</option></select><select id="variantPick" hidden></select><span id="state">')
    t=t.replace('<div id="review"></div>','<div id="evidence"><h3 id="evidenceTitle">只读自动证据 · 未作为选择目标</h3><select id="eventStream"><option value="beat_this:beat">Beat This! Beat</option><option value="beat_this:downbeat">Beat This! Downbeat</option><option value="allin1:beat">All-In-One Beat</option><option value="allin1:downbeat">All-In-One Downbeat</option><option value="acoustic:accent_candidate">声学重音候选</option><option value="allin1:section_boundary">功能段落候选</option></select><canvas id="eventCanvas" width="600" height="150" style="max-width:100%;width:600px"></canvas><pre id="evidenceText"></pre></div><div id="review"></div>')
    helpers=r'''
function drawEvidence(frame){let box=$('evidence');box.hidden=!!D.anonymous;if(D.anonymous)return;let x=D.items[current],g=$('eventCanvas').getContext('2d');g.clearRect(0,0,600,150);g.fillStyle='#24475d';g.fillRect(10,45,580,8);g.fillStyle='#e8af68';let k=+$('variantPick').value||0;if(x.edit_by_variant){for(let t of x.edit_by_variant[k]||[])g.fillRect(10+t/(x.frames-1)*580,45,580/x.frames+1,8)}else{for(let e of x.events||[]){if(e.source+':'+e.kind!==$('eventStream').value)continue;let px=10+e.local_frame_float/(x.frames-1)*580;g.fillRect(px,35,1,30)}g.fillStyle='#7bd4ce';let v=x.motion_minima?.[0]||[];for(let t of v)g.fillRect(10+t*30/(x.frames-1)*580,85,2,12)}g.fillStyle='#fff';g.fillRect(10+Math.min(frame,x.frames-1)/(x.frames-1)*580,20,1,100);$('evidenceText').textContent=x.evidence||'自动事件不是独立标注；动作拍点是实际输出速度低谷，不是速度峰。'}
function filterItems(){let cat=$('category').value;$('pick').replaceChildren();D.items.forEach((x,i)=>{if(cat!=='ALL'&&x.classification!==cat)return;let o=document.createElement('option');o.value=i;o.textContent=x.label;$('pick').append(o)});if(!$('pick').options.length){$('state').textContent='该类没有登记项目';return}load();setupVariants()}
function setupVariants(){if(!D.variant_select)return;let p=$('variantPick'),x=D.items[current];p.hidden=false;p.replaceChildren();x.variants.slice(1).forEach((v,i)=>{let o=document.createElement('option');o.value=i+1;o.textContent=v.label;p.append(o)});p.onchange=()=>{[...$('panels').children].forEach((v,i)=>v.hidden=i!==0&&i!==+p.value);drawEvidence(f)};p.onchange()}
function configureFilter(){for(let o of [...$('category').options]){if(o.value!=='ALL'&&!D.items.some(x=>x.classification===o.value))o.remove()}if(D.variant_select){$('category').hidden=true;$('category').value='ALL';$('eventStream').hidden=true;$('evidenceTitle').textContent='干预写域 · 区外数组不变'}$('category').onchange=filterItems;$('pick').onchange=()=>{load();setupVariants()};$('eventStream').onchange=()=>drawEvidence(f);filterItems()}
'''
    return t.replace('function tick()',helpers+'\nfunction tick()').replace('load();requestAnimationFrame(tick);','configureFilter();requestAnimationFrame(tick);')

def html(out,name,title,data):
    (out/name).write_bytes(template().replace('__TITLE__',title).replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':'))).encode())

def build(phase='preview'):
    root=ROOT/phase;out=root/'viewer';out.mkdir(exist_ok=False);ctx=prepare();reg,parentrows=checked_parent();z=np.load(PARENT/'arrays.npz',allow_pickle=False);by={(r['case_key'],r['method']):r for r in parentrows};cases={c['key']:c for c in reg['cases']};named=[];anonymous=[];administrative=[]
    if phase=='preview':
        comparisons=read(root/'comparisons.json');bindings=read(root/'music_bindings.json');profiles={p['id']:p for p in read(root/'profiles.json')};rows={(r['case_key'],r['method']):r for r in read(root/'executions.json')};oldmap=read(PARENT/'admin/map.json')
        getP=lambda R:np.round(relative_fk(R,ctx['J'],ctx['parents']),4).tolist()
        for i,r in enumerate(comparisons):
            c=cases[r['case_key']];T=c['target_frames'];audio='music_'+str(c['music_song'])+'_'+str(T)+'.wav';src=PARENT/'viewer'/audio
            if not (out/audio).exists():shutil.copyfile(src,out/audio)
            parent_map=next(v for v in oldmap['items'] if v.get('unit')==c['key'] and v['contrast']==r['contrast']);aa=[];nn=[];minima=[]
            for side,m in zip('AB',[parent_map['method_A'],parent_map['method_B']]):
                row=by.get((c['key'],m));name=c['key']+'__'+m+'__full_R';P=None if row is None or not row['audit']['bridge_screen'] or name not in z else getP(z[name]);aa.append(dict(label=side,P=P));nn.append(dict(label=m,P=P))
                pr=rows.get((c['key'],m));minima.append([] if pr is None else [v-c['music_start_frame']/30 for v in profiles[pr['profile_ref']]['motion_minima_seconds']])
            binding=next(v for v in bindings if v['id']==rows[c['key'],'original_Uniform']['music_evidence_ref'])
            common=dict(id=parent_map['id'],classification=r['classification'],frames=2*T+48,core=T,audio=audio,status=r['classification'])
            named.append(dict(common,label=c['unit']['source_uid']+' · '+r['contrast'],variants=nn,events=binding['events'],motion_minima=minima,evidence=json.dumps(dict(delta=r['deltas'],automatic_music_catalog_sha256=binding['catalog_sha256'],music_used_for_selection=False,MotionCritic='NOT_EVALUABLE',quality_winner=None),ensure_ascii=False)))
            anonymous.append(dict(common,label='项目 '+str(i+1),variants=aa))
            administrative.append(dict(parent_map,classification=r['classification'],flow_control=False,control=r['classification']=='VALID_IDENTICAL',different_expected='NO' if r['classification']=='VALID_IDENTICAL' else None,available=r['classification']!='UNAVAILABLE'))
        first=next(x for x in anonymous if x['classification']=='VALID_DISTINCT');P=first['variants'][0]['P']
        for p in [v for v in oldmap['items'] if v['control']]:
            changed=p['different_expected']=='YES';common=dict(id=p['id'],label='观察流程检查 '+('2' if changed else '1'),classification='FLOW_CONTROL',frames=len(P),core=first['core'],audio=first['audio'],status='只检查是否看见运动差异，不预设偏好',variants=[dict(label='A',P=P),dict(label='B',P=[P[0]]*len(P) if changed else P)])
            anonymous.append(common);administrative.append(dict(p,classification='FLOW_CONTROL',flow_control=True))
        pack='paper1_auto_eval_'+sha(root/'manifest.json')[:20];common=dict(parents=ctx['parents'].tolist(),pack_id=pack,anonymous=False,task='music_coordination',intro='先看7项双方通过且数值不同的比较；4项实际相同单列，7项不可评保留。只读Beat This!／All-In-One／R64事件与实际输出速度低谷；它们没有参与GTW或来源选择，不构成艺术评分。')
        html(out,'index.html','自动评价 · 实际有效对照与音乐证据',dict(common,items=named))
        for task in ['naturalness','music_coordination']:
            html(out,task+'.html','静音自然性' if task=='naturalness' else '音乐协调独立验证',dict(common,anonymous=True,task=task,items=anonymous,intro='有效不同、实际相同和不可评分别登记。普通观看无需身份；相同控制不计方法收益。正式独立记录需本人及名册核验，默认MISSING，未填写不算同分。自动音乐事件不显示给独立评价者。'))
        private=root/'admin';private.mkdir();admin=dict(pack_id=pack,parent_pack_id=oldmap['pack_id'],items=administrative);write(private/'map.json',admin);write(private/'roster_template.json',dict(schema='R69_REVIEWER_ROSTER_V1',pack_id=pack,reviewers=[]));write(ROOT/'receipts/human_status.json',summarize([],admin))
    else:
        zz=np.load(root/'arrays.npz',allow_pickle=False);rows=read(root/'perturbations.json');items=[]
        for c in reg['cases']:
            own=[r for r in rows if r['case_key']==c['key']];variants=[]
            for r in own:variants.append(dict(label=r['label'],P=np.round(relative_fk(zz[c['key']+'__'+r['label']+'__R'],ctx['J'],ctx['parents']),4).tolist()))
            items.append(dict(label=c['unit']['source_uid'],classification='PERTURBATION',frames=own[0]['frames'],core=own[0]['frames'],status='真实来源＋冻结合成干预；没有人类赢家',variants=variants,edit_by_variant=[r['write_frame_indices'] for r in own],evidence=json.dumps([{k:r[k] for k in ['label','scope','GTWLocal','clock_rate_legal','changed_frame_count','outside_write_mask_exact','source_hold_mapped_seconds']} for r in own],ensure_ascii=False)))
        html(out,'index.html','受控运动扰动 · 原速静音对照',dict(items=items,parents=ctx['parents'].tolist(),anonymous=False,task='naturalness',variant_select=True,intro='左侧真实来源原速，右侧可选择干预。保持延长和合法速率转折不预设艺术好坏；非法时钟、旋转扰动及30帧重复只检验自动指标敏感性。橙色显示允许写入区，区外数组严格不变。'));zz.close()
    z.close();write(out/'protocol.json',dict(source_manifest_sha256=sha(root/'manifest.json'),phase=phase,ordinary_viewing_requires_identity=False,human_records=0,can_route=False,files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()}));print('AUTO_EVAL_VIEW_READY',phase,flush=True)
    (ROOT/'index.html').write_bytes('''<!doctype html><meta charset="utf-8"><title>Paper 1 自动评价</title><style>body{background:#102132;color:#def;font:18px system-ui;margin:35px}a{color:#8dd}li{margin:18px}</style><h1>Paper 1 · 只读自动评价</h1><p>现有动作保持冻结；按真实差异与资格分组，音乐结构只读接入。</p><ol><li><a href="preview/viewer/index.html">实际有效对照＋自动音乐事件</a></li><li><a href="preview/viewer/naturalness.html">静音自然性独立验证</a></li><li><a href="preview/viewer/music_coordination.html">有音乐独立验证</a></li><li><a href="controls/viewer/index.html">受控运动扰动（后续阶段）</a></li></ol><p>没有综合艺术质量分数。MotionCritic、BAP、PFC与Native24不满足合同的保持不可评；真实独立记录为0。</p>'''.encode())

if __name__=='__main__':build(sys.argv[1] if len(sys.argv)>1 else 'preview')
