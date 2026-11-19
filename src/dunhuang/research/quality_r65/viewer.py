"""Strictly anonymous, local-only R65 review pages.

No methods, source identities, numerical scores or admin maps are serialized.
Only a reviewer click can create ratings/annotations or export a review.
"""
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np


HERE = Path(__file__).parent
DIMENSIONS = ('music_coordination', 'transition_timing', 'continuity', 'richness')


def _keys(value, allowed, name):
    if not isinstance(value, dict) or set(value)-set(allowed):
        raise ValueError(name+' contains fields outside the reviewer whitelist')


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', value):
        raise ValueError('Opaque reviewer identifier required')
    return value


def _audio(value):
    _keys(value, ('path', 'media_sha256'), 'audio')
    p = PurePosixPath(value['path'])
    if (p.is_absolute() or len(p.parts) != 2 or p.parts[0] != 'audio'
            or p.suffix.lower() not in ('.m4a', '.wav', '.mp3', '.ogg', '.flac', '.aac')
            or not re.fullmatch(r'[A-Za-z0-9_.-]+', p.name)):
        raise ValueError('Only opaque local audio paths are permitted')
    digest = value['media_sha256']
    if not isinstance(digest, str) or not re.fullmatch(r'[a-fA-F0-9]{64}', digest):
        raise ValueError('Audio media SHA256 required')
    return dict(path=str(p), media_sha256=digest)


def project(payload):
    """Validate exact public schema and make two independent minimal payloads."""
    _keys(payload, ('pack_id', 'protocol_sha256', 'fps', 'frames', 'parents',
                    'dimensions', 'forms', 'cases', 'music'), 'payload')
    pack = _id(payload['pack_id'])
    protocol = payload['protocol_sha256']
    if not re.fullmatch(r'[a-fA-F0-9]{64}', protocol):
        raise ValueError('Protocol SHA256 required')
    if payload['fps'] != 30 or payload['frames'] != 528:
        raise ValueError('Actual528-frame/30FPS contract required')
    if list(payload['dimensions']) != list(DIMENSIONS):
        raise ValueError('Four preregistered quality dimensions required')
    parents = payload['parents']
    if (len(parents) != 22 or any(isinstance(p, bool) or not isinstance(p, int)
                                or p < -1 or p >= 22 for p in parents)):
        raise ValueError('Body22 parent indices required')
    cases = []
    for c in payload['cases']:
        _keys(c, ('blind_item_id', 'audio', 'variants'), 'case')
        variants = []
        if len(c['variants']) != 4:
            raise ValueError('All four anonymous variants must be retained')
        for v in c['variants']:
            _keys(v, ('blind_variant_id', 'data', 'bounds', 'available'), 'variant')
            if not isinstance(v['available'], bool):
                raise ValueError('Variant availability must be a boolean')
            data, bounds = v['data'], v['bounds']
            if v['available']:
                try:
                    raw = base64.b64decode(data, validate=True)
                except (ValueError, TypeError):
                    raise ValueError('Actual body22 Float32 coordinates required')
                if len(raw) != 528*22*3*4 or not np.isfinite(np.frombuffer(raw, dtype='<f4')).all():
                    raise ValueError('Coordinate payload violates actual528/body22 contract')
                b = np.asarray(bounds, float)
                if b.shape != (2, 3) or not np.isfinite(b).all() or (b[1] < b[0]).any():
                    raise ValueError('Finite shared-camera bounds required')
                bounds = b.tolist()
            elif data is not None or bounds is not None:
                raise ValueError('Unavailable variant must stay blank, without replacement coordinates')
            variants.append(dict(blind_variant_id=_id(v['blind_variant_id']),
                                 data=data, bounds=bounds, available=v['available']))
        vids = [v['blind_variant_id'] for v in variants]
        if len(set(vids)) != 4:
            raise ValueError('Anonymous variant IDs must be unique within each item')
        cases.append(dict(blind_item_id=_id(c['blind_item_id']), audio=_audio(c['audio']), variants=variants))
    ids = [c['blind_item_id'] for c in cases]
    if len(cases) != 29 or len(set(ids)) != len(ids):
        raise ValueError('All29 unique frozen cases are required')
    music = []
    for m in payload['music']:
        _keys(m, ('blind_item_id', 'audio', 'duration_seconds'), 'music item')
        duration = m['duration_seconds']
        if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not np.isfinite(duration) or duration <= 0:
            raise ValueError('Actual full-audio duration required')
        music.append(dict(blind_item_id=_id(m['blind_item_id']), audio=_audio(m['audio']), duration_seconds=float(duration)))
    mids = [m['blind_item_id'] for m in music]
    if len(music) != 3 or len(set(mids)) != 3:
        raise ValueError('All three full anonymous audio recordings are required')
    forms, music_forms = [], []
    for f in payload['forms']:
        _keys(f, ('form_id', 'case_order', 'panel_orders', 'music_order'), 'form')
        fid = _id(f['form_id'])
        order = f['case_order']
        if len(order) != len(ids) or set(order) != set(ids) or set(f['panel_orders']) != set(ids):
            raise ValueError('Every form must retain each frozen case exactly once')
        for c in cases:
            vid = [v['blind_variant_id'] for v in c['variants']]
            ordered = f['panel_orders'][c['blind_item_id']]
            if len(ordered) != 4 or set(ordered) != set(vid):
                raise ValueError('Frozen four-panel permutation required for each case')
        if len(f['music_order']) != 3 or set(f['music_order']) != set(mids):
            raise ValueError('Every form must retain all three full music items')
        forms.append(dict(form_id=fid, case_order=list(order),
                          panel_orders={k: list(v) for k, v in f['panel_orders'].items()}, music_order=list(f['music_order'])))
        music_forms.append(dict(form_id=fid, music_order=list(f['music_order'])))
    if len(forms) != 2 or {f['form_id'] for f in forms} != {'F01', 'F02'}:
        raise ValueError('Two fixed review forms F01/F02 required')
    base = dict(pack_id=pack, protocol_sha256=protocol)
    quality = dict(base, fps=30, frames=528, parents=list(parents),
                   dimensions=list(DIMENSIONS), forms=forms, cases=cases)
    audio = dict(base, forms=music_forms, music=music)
    return quality, audio


COMMON_JS = r'''
function r65Uuid(){if(globalThis.crypto&&crypto.randomUUID)return crypto.randomUUID();const b=new Uint8Array(16);if(globalThis.crypto&&crypto.getRandomValues)crypto.getRandomValues(b);else for(let i=0;i<16;i++)b[i]=Math.floor(Math.random()*256);return Array.from(b,x=>x.toString(16).padStart(2,'0')).join('');}
function r65Score(value){if(value===undefined||value===null||value==='')return{value:null,reason:'UNRATED'};if(value==='UNSURE'||(value&&value.reason==='UNSURE'))return{value:null,reason:'UNSURE'};if(value&&value.reason==='UNRATED')return{value:null,reason:'UNRATED'};const n=Number(value&&typeof value==='object'?value.value:value);if(!Number.isInteger(n)||n<1||n>5)throw Error('评分必须为1至5或无法判断');return{value:n,reason:null};}
function r65QualityItems(D,form,state){return form.case_order.filter(id=>state.items[id]&&state.items[id].touched).map(id=>{const item=state.items[id];return{blind_item_id:id,variant_order:form.panel_orders[id].slice(),ratings:form.panel_orders[id].map(vid=>({blind_variant_id:vid,scores:Object.fromEntries(D.dimensions.map(dim=>[dim,r65Score(item.ratings&&item.ratings[vid]?item.ratings[vid][dim]:undefined)]))}))};});}
function r65MusicItems(D,form,state){return form.music_order.filter(id=>state.items[id]&&state.items[id].touched).map(id=>{const m=D.music.find(m=>m.blind_item_id===id),item=state.items[id],tracks={};for(const kind of ['beat','downbeat','accent','phrase']){const t=item.tracks[kind]||{status:'PARTIAL',events:[]};if(!['COMPLETE','PARTIAL','UNSURE','NO_CLEAR_EVENTS'].includes(t.status))throw Error('标注状态无效');const events=t.events.map(event=>{const seconds=Number(event.seconds);if(!Number.isFinite(seconds)||seconds<0||seconds>m.duration_seconds||!['HIGH','MEDIUM','LOW'].includes(event.confidence))throw Error('事件时间或置信度无效');return{seconds,confidence:event.confidence};}).sort((a,b)=>a.seconds-b.seconds);if(t.status==='NO_CLEAR_EVENTS'&&events.length)throw Error('无清晰事件状态要求事件列表为空');tracks[kind]={status:t.status,events};}const unsure_intervals=(item.unsure_intervals||[]).map(v=>{const a=Number(v.start),b=Number(v.end),reason=String(v.reason||'').trim();if(!Number.isFinite(a)||!Number.isFinite(b)||a<0||b<=a||b>m.duration_seconds||!reason)throw Error('无法判断区间须有有效起止与原因');return{start:a,end:b,reason};});return{blind_item_id:id,media_sha256:m.audio.media_sha256,coverage_seconds:[0,m.duration_seconds],tracks,unsure_intervals};});}
function r65BuildDocument(task,D,form,state){if(!state.review_id||!String(state.reviewer_id||'').trim()||!String(state.reviewer_roster_ref||'').trim())throw Error('请填写评价者编号和身份登记引用；身份仍须管理员独立核验');const i=state.independence||{};if(!i.own_judgment||!i.no_other_reviews||!i.no_method_identity)throw Error('请如实确认独立判断声明；不能满足时请保存草稿而非提交正式记录');const items=task==='quality'?r65QualityItems(D,form,state):r65MusicItems(D,form,state);if(!items.length)throw Error('尚无实际评分或标注；不会导出空白人工记录');return{schema:task==='quality'?'R65_QUALITY_REVIEW_V1':'R65_MUSIC_REVIEW_V1',pack_id:D.pack_id,protocol_sha256:D.protocol_sha256,review_id:state.review_id,reviewer_id:String(state.reviewer_id).trim(),reviewer_roster_ref:String(state.reviewer_roster_ref).trim(),form_id:form.form_id,reviewed_at:new Date().toISOString(),independence:{own_judgment:true,other_reviews_seen:false,method_identity_seen:false,prior_r64_exposure:!!i.prior_r64_exposure},items};}
function r65NewState(){return{review_id:r65Uuid(),reviewer_id:'',reviewer_roster_ref:'',independence:{own_judgment:false,no_other_reviews:false,no_method_identity:false,prior_r64_exposure:false},items:{}};}
function r65Download(value,name){const blob=new Blob([JSON.stringify(value,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),500);}
function r65StoreKey(D,task){return'r65-review-draft:'+D.pack_id+':'+task;}
function r65ReadStore(D,task){try{const s=JSON.parse(localStorage.getItem(r65StoreKey(D,task)));if(s&&s.pack_id===D.pack_id&&s.protocol_sha256===D.protocol_sha256&&s.drafts)return s;}catch(e){}return{pack_id:D.pack_id,protocol_sha256:D.protocol_sha256,active_form_id:D.forms[0].form_id,drafts:{}};}
function r65Message(value,error=false){const el=document.getElementById('message');el.textContent=value;el.className=error?'message error':'message';}
function r65SaveStore(D,task,store){try{localStorage.setItem(r65StoreKey(D,task),JSON.stringify(store));return true;}catch(e){r65Message('浏览器无法保存本地草稿，请使用“下载草稿”保留记录。',true);return false;}}
function r65BindIdentity(D,task,store,getState,getForm,onReset){const $=id=>document.getElementById(id);function hydrate(){const state=getState();$('reviewer').value=state.reviewer_id;$('roster').value=state.reviewer_roster_ref;for(const[k,id]of [['own_judgment','own'],['no_other_reviews','others'],['no_method_identity','methods'],['prior_r64_exposure','prior']])$(id).checked=!!state.independence[k];const locked=Object.values(state.items).some(x=>x.touched);$('reviewer').readOnly=locked&&!!state.reviewer_id;$('roster').readOnly=locked&&!!state.reviewer_roster_ref;$('reviewId').textContent='本地记录编号：'+state.review_id+' · 身份自报，尚未核验';}for(const[id,key]of [['reviewer','reviewer_id'],['roster','reviewer_roster_ref']])$(id).addEventListener('input',()=>{getState()[key]=$(id).value;r65SaveStore(D,task,store);});for(const[k,id]of [['own_judgment','own'],['no_other_reviews','others'],['no_method_identity','methods'],['prior_r64_exposure','prior']])$(id).addEventListener('change',()=>{getState().independence[k]=$(id).checked;r65SaveStore(D,task,store);});$('save').onclick=()=>{r65SaveStore(D,task,store);r65Download({schema:task==='quality'?'R65_QUALITY_DRAFT_V1':'R65_MUSIC_DRAFT_V1',pack_id:D.pack_id,protocol_sha256:D.protocol_sha256,form_id:getForm().form_id,draft:getState()},D.pack_id+'_'+task+'_'+getForm().form_id+'_draft.json');r65Message('草稿已下载；不是已核验的人工真值。');};$('export').onclick=()=>{try{const doc=r65BuildDocument(task,D,getForm(),getState());r65Download(doc,D.pack_id+'_'+task+'_'+getForm().form_id+'_'+doc.review_id+'.json');r65Message('已导出实际填写的记录。身份与独立性仍需管理员核验，未填写项目未被补成标签。');}catch(e){r65Message(e.message,true);}};$('newReview').onclick=()=>{if(!confirm('新建独立草稿会清空当前表单在本浏览器中的评分／标注。请先下载已有草稿。继续吗？'))return;store.drafts[getForm().form_id]=r65NewState();r65SaveStore(D,task,store);onReset();hydrate();r65Message('已新建独立记录编号。');};return hydrate;}
'''


def write_viewers(out, payload):
    """Write exactly two HTMLs and return path/size/SHA records, no review files."""
    quality, music = project(payload)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for name, public in (('index.html', quality), ('music.html', music)):
        source = (HERE/name.replace('index.html', 'viewer.html')).read_text(encoding='utf8')
        encoded = json.dumps(public, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        encoded = encoded.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
        content = source.replace('__R65_PAYLOAD__', encoded).replace('__R65_COMMON_JS__', COMMON_JS)
        path = out/name
        path.write_text(content, encoding='utf8')
        files.append(dict(path=name, bytes=path.stat().st_size,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return files
