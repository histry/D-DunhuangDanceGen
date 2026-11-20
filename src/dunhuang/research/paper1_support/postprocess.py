"""Scalar attribution and adapters to the existing independent-review contract."""
import json,secrets,hmac,hashlib,sys
from pathlib import Path
import numpy as np
from .study import ROOT,HERE,read,write,sha,prepare
from dunhuang.research.complete_r36.core import relative_fk
from dunhuang.research.gtw_kinematic_baseline.reviews import summarize

def attribution():
    root=ROOT/'controls';reg=read(root/'registration.json');rows=read(root/'records.json');pools=read(root/'pools.json');by={(r['case_key'],r['method']):r for r in rows};effects=[]
    for c in reg['cases']:
        key=c['key'];u=by[key,'original_Uniform'];g=by[key,'original_GTWLocal'];a=by.get((key,'alternative_Uniform'));ag=by.get((key,'alternative_GTWLocal'))
        def delta(x,y):return dict(longest_low_seconds=x['core']['longest_low_seconds']-y['core']['longest_low_seconds'],mapped_source_low_seconds=x['mapped_low_seconds']-y['mapped_low_seconds'],FK_D3_q95=x['core']['FK_D3_q95']-y['core']['FK_D3_q95'],FK_activity_path_mean_m=x['core']['FK_activity_path_mean_m']-y['core']['FK_activity_path_mean_m'])
        effects.append(dict(case_key=key,source_uid=c['unit']['source_uid'],target_frames=c['target_frames'],timing_only=delta(g,u),timing_only_G1_identical=u['connection']['bridge_sha256']==g['connection']['bridge_sha256'],source_only_uniform=None if a is None else delta(a,u),source_and_clock=None if ag is None else delta(ag,u),alternative_status='AVAILABLE' if a else 'MISSING_MATCHED_SOURCE',source_confound='SAME_RECORDING_DIFFERENT_SOURCE_CONTENT_AND_UP_TO_10_PERCENT_NATURAL_LENGTH_DIFFERENCE; FINITE_CLOCK_POOLS_DIFFER; NOT_PURE_WARP_EFFECT',quality='NOT_EVALUABLE'))
    ranges=[dict(case_key=p['case_key'],source=p['source'],candidates=len(p['rows']),minimum_mapped_source_low_seconds=min(r['mapped_low_seconds'] for r in p['rows']),maximum_mapped_source_low_seconds=max(r['mapped_low_seconds'] for r in p['rows'])) for p in pools]
    result=dict(effects=effects,capacity_ranges=ranges,original_GTW_changed=sum(by[c['key'],'original_Uniform']['candidate_uid']!=by[c['key'],'original_GTWLocal']['candidate_uid'] for c in reg['cases']),original_GTW_longer_low=sum(v['timing_only']['longest_low_seconds']>1e-8 for v in effects),source_alternatives=sum(v['source_only_uniform'] is not None for v in effects),fixed_source_bridge_invariance=all(v['timing_only_G1_identical'] for v in effects),independent_music_quality='NOT_EVALUABLE',human_records=0,R74='NOT_STARTED',can_route=False)
    out=ROOT/'attribution';out.mkdir(exist_ok=False);write(out/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['effects','capacity_ranges']}))

def review_pack():
    root=ROOT/'controls';out=root/'viewer';ctx=prepare();reg=read(root/'registration.json');rows=read(root/'records.json');by={(r['case_key'],r['method']):r for r in rows};z=np.load(root/'arrays.npz',allow_pickle=False);seed=secrets.token_bytes(32);items=[];admin=[]
    def points(c,method):
        row=by.get((c['key'],method));name=c['key']+'__'+method+'__full_R'
        # Failed numerical screens remain unavailable for quality claims.
        if row is None or not row['audit']['bridge_screen'] or name not in z:return None
        return np.round(relative_fk(z[name],ctx['J'],ctx['parents']),4).tolist()
    for c in reg['cases']:
        contrasts=[('original_GTWLocal','original_Uniform','TIMING_ONLY'),('alternative_Uniform','original_Uniform','SOURCE_ONLY_UNIFORM'),('alternative_GTWLocal','alternative_Uniform','ALTERNATIVE_TIMING_ONLY')]
        for left,right,label in contrasts:
            token=hmac.new(seed,(c['key']+label).encode(),hashlib.sha256).hexdigest();swap=int(token[0],16)%2;pair=[points(c,left),points(c,right)]
            if swap:pair.reverse()
            audio='music_'+str(c['music_song'])+'_'+str(c['target_frames'])+'.wav';available=all(p is not None for p in pair)
            items.append(dict(id=token,label='项目 '+str(len(items)+1),audio=audio,frames=2*c['target_frames']+48,core=c['target_frames'],status='' if available else '缺失或未通过数值门，保留不可评价',variants=[dict(label='AB'[i],P=p) for i,p in enumerate(pair)]))
            admin.append(dict(id=token,source_uid=c['unit']['source_uid'],unit=c['key'],contrast=label,positive_side='B' if swap else 'A',control=False,available=available,method_A=right if swap else left,method_B=left if swap else right))
    valid=next(p for c in reg['cases'] for m in ['original_Uniform'] for p in [points(c,m)] if p is not None)
    for changed in [False,True]:
        token=secrets.token_hex(32);Q=[valid[0]]*len(valid) if changed else valid;c=reg['cases'][0]
        items.append(dict(id=token,label='观察流程检查 '+str(int(changed)+1),audio='music_'+str(c['music_song'])+'_'+str(c['target_frames'])+'.wav',frames=len(valid),core=c['target_frames'],status='只检查是否看到运动差异，不预设自然性赢家',variants=[dict(label='A',P=valid),dict(label='B',P=Q)]))
        admin.append(dict(id=token,source_uid='CONTROL',contrast='DIFFERENCE_ONLY',positive_side=None,control=True,available=True,different_expected='YES' if changed else 'NO'))
    template=(HERE.parent/'gtw_kinematic_baseline/viewer.html').read_text(encoding='utf8')
    # Thin adapter: preserve the existing review/export UI and importer; only make its actual frame budget dynamic.
    replacements={'function draw(frame){':'function frames(){return D.items[current].frames} function draw(frame){$(\'seek\').max=frames()-1;','Math.min(527,frame)':'Math.min(frames()-1,frame)','${f}/527':'${f}/${frames()-1}','240/528*300':'D.items[current].core/frames()*300','48/528*300':'48/frames()*300','f/528*300':'f/frames()*300','if(f===527)':'if(f===frames()-1)'}
    for a,b in replacements.items():
        assert a in template,a;template=template.replace(a,b)
    pack='paper1_support_'+sha(root/'manifest.json')[:20];private=root/'admin';private.mkdir(exist_ok=False);(private/'seed.private').write_bytes(seed);mapping=dict(pack_id=pack,items=admin);write(private/'map.json',mapping)
    for task in ['naturalness','music_coordination']:
        payload=dict(pack_id=pack,anonymous=True,task=task,parents=ctx['parents'].tolist(),items=items,intro='匿名候选；静音自然性与音乐协调分开评价。普通观看无需身份，科研记录须独立名册核验。历史具名结果已曝光者属于开发反馈。缺失保留，不预设赢家。')
        title='静音自然性' if task=='naturalness' else '音乐协调性';(out/(task+'.html')).write_bytes(template.replace('__TITLE__',title).replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':'))).encode())
    write(ROOT/'receipts/human_status.json',summarize([],mapping));write(private/'roster_template.json',dict(schema='R69_REVIEWER_ROSTER_V1',pack_id=pack,reviewers=[]))
    delivery=read(out/'protocol.json');delivery['review_contract']='RETIME_PREF_V1; gtw_kinematic_baseline.reviews; separate tasks; at least two independently eligible people per item';delivery['human_records']=0
    delivery['files']={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file() and p.name!='protocol.json'};write(out/'protocol.json',delivery);z.close()
    print('PAPER1_REVIEW_PACK',len(items),'REAL_RECORDS=0',flush=True)

def release():
    assert read(ROOT/'receipts/independent.json')['status']=='PASS';out=Path('reports/paper1_support_release');out.mkdir(exist_ok=False)
    for phase in ['main','controls']:
        for name in ['registration','summary','pools','decisions','manifest']:
            write(out/(phase+'_'+name+'.json'),read(ROOT/phase/(name+'.json')))
        rows=read(ROOT/phase/'records.json')
        for r in rows:r['core']={k:v for k,v in r['core'].items() if k not in ['low_cells','cell_angular','cell_linear']}
        write(out/(phase+'_records.json'),rows)
    write(out/'attribution.json',read(ROOT/'attribution/result.json'))
    for name in ['independent','roundoff_correction','preview_browser_acceptance','human_status','asset_copy']:write(out/(name+'.json'),read(ROOT/'receipts'/(name+'.json')))
    write(out/'protocol.json',read(HERE/'protocol.json'));write(out/'manifest.json',dict(files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
if __name__=='__main__':{'attribution':attribution,'reviews':review_pack,'release':release}[sys.argv[1]]()
