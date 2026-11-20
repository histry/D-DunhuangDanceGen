"""Actual body22 playback plus separate silent/music preference projections."""
import json,secrets,hmac,hashlib,shutil,sys
from pathlib import Path
import numpy as np
from .study import ROOT
from dunhuang.research.local_timing_r68.common import read,write,sha
from dunhuang.research.complete_r36.core import relative_fk
from dunhuang.research.sdmr_warp_rank import connectivity

def build(mode='preview'):
    root=ROOT/mode;out=root/'viewer';out.mkdir(exist_ok=False);(out/'audio').mkdir();ctx=connectivity.prepare();cases=read(root/'registration.json')['cases'];rows=read(root/'records.json');sel=read(root/'selections.json');z=np.load(root/'arrays.npz');payload=[];byunit={}
    audiofiles=read('outputs/retiming_baselines_r69_01/main/viewer/protocol.json')['audio_files']
    for r in rows:byunit.setdefault(r['unit'],[]).append(r)
    for unit,rr in byunit.items():
        case=cases[rr[0]['case_index']];delta=rr[0]['delta'];variants=[]
        for r in rr:
            prefix=unit+'__'+r['method']+'__';R=z[prefix+'full_R'] if prefix+'full_R' in z else None
            label=r['method']+' · '+('G1原门通过' if r.get('connection',{}).get('passes_original_screen') else 'G1未通过／无执行')
            variants.append(dict(label=label,method=r['method'],P=None if R is None else np.round(relative_fk(R,ctx['J'],ctx['parents']),4).tolist()))
        # Missing methods are placeholders, never silently replaced.
        for method in [p+'_'+m for p in (['A','B'] if mode=='controls' else ['A']) for m in ['GTWLocal','GTWExtended','GTWContext']]:
            if not any(v['method']==method for v in variants):variants.append(dict(label=method,method=method,P=None))
        audio=next(a for a in audiofiles.values() if a['raw_audio_sha256']==case['audio_sha256'] and a['pair_music_interval']==case['pair_music_interval'])
        dest=out/audio['path'];dest.parent.mkdir(exist_ok=True);source=Path('outputs/retiming_baselines_r69_01/main/viewer')/audio['path'];assert sha(source)==audio['sha256'];shutil.copyfile(source,dest)
        status='共同候选实际峰='+str(next(s['actual_peak'] for s in sel if s['unit']==unit))
        payload.append(dict(label=f'{case["source_uid"]} · {delta:+d}帧',status=status,unit=unit,source_uid=case['source_uid'],audio=audio['path'],variants=variants))
    template=Path(__file__).with_name('viewer.html').read_text(encoding='utf8')
    def html(name,title,data):
        (out/name).write_bytes(template.replace('__TITLE__',title).replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':'))).encode('utf8'))
    common=dict(parents=ctx['parents'].tolist(),pack_id='gtw_'+sha(root/'manifest.json')[:20],anonymous=False,task='music_coordination',intro='GTW思想适配，非原算法完整复现。相同来源／240帧Core／48帧G1；主比较三评分器共享实际峰与候选池。E1W和Uniform是不同对齐目标的参考。没有人工偏好，未宣称质量胜出。')
    html('index.html','GTW 强运动学基线 · 实际对照',dict(common,items=payload))
    if mode=='controls':
        write(out/'protocol.json',dict(source_manifest_sha256=sha(root/'manifest.json'),human_pack='FROZEN_PREVIEW_PACK_ONLY',human_records=0,R74='NOT_STARTED'))
        write(root/'viewer_delivery.json',dict(files=[dict(path=p.relative_to(root).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()]))
        print('GTW_VIEW_READY',mode,len(payload),flush=True);return
    seed=secrets.token_bytes(32);admin=[];anonymous=[]
    for item in payload:
        by={v['method']:v for v in item['variants']}
        for left,right in [('A_GTWLocal','E1W'),('A_GTWExtended','A_GTWLocal'),('A_GTWContext','A_GTWExtended')]:
            uid=hmac.new(seed,(item['unit']+left+right).encode(),hashlib.sha256).hexdigest();swap=int(uid[0],16)%2;vv=[by[left],by[right]]
            if swap:vv.reverse()
            anonymous.append(dict(id=uid,label='项目 '+str(len(anonymous)+1),status='' if all(v['P'] is not None for v in vv) else '缺失执行，不替换',audio=item['audio'],variants=[dict(label='AB'[i],P=v['P']) for i,v in enumerate(vv)]))
            admin.append(dict(id=uid,source_uid=item['source_uid'],unit=item['unit'],contrast=left+'_vs_'+right,positive_side='B' if swap else 'A',control=False,available=all(v['P'] is not None for v in vv)))
    # Only explicit difference detection has a known answer; no fabricated naturalness winner.
    valid=next(v for p in payload for v in p['variants'] if v['P'] is not None)
    for changed in [False,True]:
        uid=secrets.token_hex(32);P=valid['P'];Q=[P[0]]*528 if changed else P
        anonymous.insert(0,dict(id=uid,label='观察流程检查 '+str(int(changed)+1),status='只检查运动差异观察，不预设自然性赢家',audio=payload[0]['audio'],variants=[dict(label='A',P=P),dict(label='B',P=Q)]))
        admin.append(dict(id=uid,source_uid='CONTROL',contrast='DIFFERENCE_ONLY',positive_side=None,control=True,available=True,different_expected='YES' if changed else 'NO'))
    for task in ['naturalness','music_coordination']:
        items=[{k:v for k,v in p.items() if task!='naturalness' or k!='audio'} for p in anonymous]
        html(task+'.html','静音自然性独立评价' if task=='naturalness' else '有音乐协调独立评价',dict(common,anonymous=True,task=task,items=items,intro='匿名A/B固定顺序；允许TIE／UNSURE／MISSING。静音自然性与音乐协调分开记录；未填写不是同分。普通观看无需身份，科研记录必须另行核验独立性。'))
    private=root/'admin';private.mkdir();(private/'seed.private').write_bytes(seed);write(private/'map.json',dict(pack_id=common['pack_id'],items=admin))
    write(out/'protocol.json',dict(pack_id=common['pack_id'],source_manifest_sha256=sha(root/'manifest.json'),task_modes=['naturalness','music_coordination'],real_records=0,phase_labels=0,R74='NOT_STARTED'))
    write(root/'viewer_delivery.json',dict(files=[dict(path=p.relative_to(root).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()]))
    print('GTW_VIEW_READY',mode,len(payload),flush=True)

if __name__=='__main__':build('controls' if '--controls' in sys.argv else 'preview')
