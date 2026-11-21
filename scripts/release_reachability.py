"""Publish frozen scalar certificates; annotate existing playback without new solves."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.research.reachability_q01.common import ROOT,CFG,HERE,read,write,sha,fresh,verify

def inspect_view(replay=False):
    base=ROOT/'controls/viewer';html=(base/'index.html').read_text(encoding='utf8');raw=html.split('<script id="data" type="application/json">',1)[1].split('</script>',1)[0];d=json.loads(raw);tracks_hash=hashlib.sha256(json.dumps(d['tracks'],sort_keys=True,separators=(',',':')).encode()).hexdigest();pop=read(ROOT/'controls/population.json');results=read(ROOT/'controls/results.json')
    for item in d['items']:
        source=item['label'].split(' · ')[0];display=next(v for v in pop['display'] if v['source_uid']==source);u=next(v for v in pop['units'] if v['unit_uid']==display['unit_uid']);a,b=u['source_interval'];r=next(v for v in results if v['unit_uid']==u['unit_uid'] and v['hypothesis']=='BeatTriple');labels=['Onset','Peak','Low-run end'];rows=[]
        for k,(label,e) in enumerate(zip(labels,u['source_events'])):
            bound=[e['source_frame']-a]*2 if item['raw'] else r['individual_intervals'][k];target=None if item['raw'] else u['targets']['BeatTriple'][k]['local_target_frame'];rows.append(dict(label=label,source_frame=e['source_frame'],interval=bound,target=target))
        item['inspection_rows']=rows;item['explanation']='来源阶段位置与固定音乐目标的只读事后解释。条形为各节点边际可达范围，粉线是BeatTriple指定目标±3帧；并非新的联合解或人工语义标注。'+('原速模式显示来源节点相对实际起点的位置。' if item['raw'] else '本轮无可执行三锚点时钟，不用匀速动作冒充新方法；Uniform仅作Core参考，Bridge区留空。')
        if not item['raw']:
            v=item['variants'][0];T=u['target_frames'];v['q']=[a+(b-a-1)*i/(T-1) for i in range(T)];v['source_interval']=[a,b];v['target_marks']=[e['local_target_frame'] for e in u['targets']['BeatTriple']];v['anchor_marks']=[(e['source_frame']-a)*(T-1)/(b-a-1) for e in u['source_events']]
    d['intro']='B/C完整人口诊断：139单位、417登记位置；本轮58个可比较三锚点假设全部被速率排除，359项缺失／不可比较，未产生新对齐动作。展示来源在证书计算前确定。男琵琶2与男彩带分合无完整代理三元组，未补样。视图解释为事后只读，冻结证书与动作不变。'
    assert hashlib.sha256(json.dumps(d['tracks'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==tracks_hash
    html=html.replace(raw,json.dumps(d,ensure_ascii=False,separators=(',',':')),1).replace('function drawCertificate(x){','function originalCertificate(x){',1)
    helper="""function drawCertificate(x){let rows=x.inspection_rows||[],g=$('reach').getContext('2d');g.clearRect(0,0,600,180);$('reach').hidden=!rows.length;rows.forEach((r,k)=>{let y=30+k*55,den=x.raw?x.frames-1:x.core-1,b=r.interval;g.fillStyle='#9eb8ca';g.fillText(r.label+' s='+r.source_frame,8,y);if(b){let ok=r.target==null||(r.target+3>=b[0]&&r.target-3<=b[1]);g.fillStyle=ok?'#50ad99':'#b7784f';g.fillRect(175+b[0]/den*400,y-12,Math.max(3,(b[1]-b[0])/den*400),15)}if(r.target!=null){g.fillStyle='#dc8bb9';g.fillRect(175+(r.target-3)/den*400,y-16,6/den*400,23)}});$('certificate').textContent=x.explanation||''} """
    html=html.replace('function stop(){',helper+'function stop(){',1);out=fresh(ROOT/('inspection_replay' if replay else 'inspection_viewer'));(out/'index.html').write_bytes(html.encode());old=read(base/'protocol.json');write(out/'protocol.json',dict(files={'index.html':dict(bytes=(out/'index.html').stat().st_size,sha256=sha(out/'index.html'))},existing_local_audio=old['existing_local_audio'],posthoc_readonly_view=True,source_controls_manifest_sha256=sha(ROOT/'controls/manifest.json'),source_original_view_sha256=sha(base/'index.html'),body_tracks_unchanged_sha256=tracks_hash,no_new_rotation_arrays=True,can_route=False))
    if replay:
        diff=[n for n in ['index.html','protocol.json'] if sha(out/n)!=sha(ROOT/'inspection_viewer'/n)];assert not diff,diff;write(ROOT/'receipts/view_replay.json',dict(status='PASS',differences=diff,body_tracks_unchanged=True,scope='POSTHOC_READONLY_PRESENTATION_ONLY'))
    print('INSPECTION_VIEW',len(d['items']),sum(p.stat().st_size for p in out.iterdir()))

def export():
    assert read(ROOT/'receipts/independent.json')['status']=='PASS';verify(ROOT/'preview');verify(ROOT/'controls');out=fresh('reports/paper1_reachability_release')
    for phase,names in [('preview',['registration','certificates','summary','manifest']),('controls',['population','results','summary','manifest'])]:
        for n in names:write(out/(phase+'_'+n+'.json'),read(ROOT/phase/(n+'.json')))
    for n in ['browser_acceptance','independent','view_replay']:write(out/(n+'.json'),read(ROOT/'receipts'/(n+'.json')))
    write(out/'protocol.json',CFG);launch=Path('outputs/paper1_reachability_launch');exits={n:int((launch/(n+'.exit')).read_text()) for n in ['preview','controls','view','replay']};assert all(v==0 for v in exits.values());write(out/'runtime.json',dict(status='PASS',scientific_commit='78a59eb0558dfa7d0ca84fb346b877495502316b',contract_tests=25,independent_LP_cases=160,LP_optimization_calls_in_tests=240,exit_status=exits,baseline_producers_unchanged=True,training_steps=0,new_planner_runs=0,new_G1_calls=read(ROOT/'controls/summary.json')['actual_G1_solves'],new_aligned_motion_executions=read(ROOT/'controls/summary.json')['rate_feasible'],whole_song='NOT_STARTED_NO_CORRESPONDENCE_VALIDITY_EVIDENCE',can_route=False))
    pop=read(out/'controls_population.json');rows=read(out/'controls_results.json');write(out/'attribution.json',dict(metadata_missing_reason_counts={s:sum(s in u['missing'] for u in pop['units']) for s in ['NO_ORDERED_SUPPORTED_TRIPLE','NO_LEGAL_REGISTERED_BUDGET']},missing_reason_counts={s:sum(s in r.get('reason',[]) for r in rows) for s in ['NO_ORDERED_SUPPORTED_TRIPLE','NO_LEGAL_REGISTERED_BUDGET','MISSING_MUSIC_EVENT_ID','NONORDERED_AUTOMATIC_TARGET_EVENTS']},individual_node_exclusion_counts={str(i):sum(r.get('individual_rate_status',[None]*3)[i]=='EXCLUDED_BY_RATE' for r in rows) for i in range(3)},note='OVERLAPPING_MISSING_REASONS_MUST_NOT_BE_SUMMED; FIXED_SELECTION_RULE_NOT_ALL_POSSIBLE_CORRESPONDENCES'))
    write(out/'manifest.json',dict(files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}));print('PUBLIC_JSON',len(list(out.iterdir())),sum(p.stat().st_size for p in out.iterdir()))

if __name__=='__main__':
    if sys.argv[1]=='view':inspect_view()
    elif sys.argv[1]=='view_replay':inspect_view(True)
    elif sys.argv[1]=='export':export()
    else:raise ValueError(sys.argv[1])
