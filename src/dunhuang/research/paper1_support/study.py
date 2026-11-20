"""Audited source capacity and bounded clock opportunity, isolated from baselines."""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from .contracts import MusicEvidence,SourcePrimitive,ExecutionCandidate,ExecutionAudit,SelectionDecision,boundary_unchanged
from dunhuang.research.complete_r36.core import Library,rotation_sample,relative_fk
from dunhuang.research.warp_coupling_r53.common import array_hash,save_arrays
from dunhuang.research.warp_coupling_r53.warp_lattice import suffix,unrank,validate
from dunhuang.research.gtw_kinematic_baseline.features import ReferenceCost
from dunhuang.research.hold_timing import metrics
from dunhuang.research.duration_r62.cross_source import make_observed,solve_boundary,measure

ROOT=Path('outputs/paper1_support_01');HERE=Path(__file__).parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def uid(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:24]
CFG=read(HERE/'protocol.json')
def seal(out):
    deps=['complete_r36/core.py','warp_coupling_r53/warp_lattice.py','hold_timing/metrics.py','gtw_kinematic_baseline/features.py','duration_r62/cross_source.py','manifold_bridge_r47/geometry.py']
    write(out/'manifest.json',dict(code={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ['.py','.json','.html']},dependencies={p:sha(HERE.parent/p) for p in deps},files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'}))
def prepare():
    audit=Path('outputs/hold_timing_01/audit');assert sha(audit/'manifest.json')==CFG['hold_audit_manifest']
    frozen=read(audit/'manifest.json')['files']
    for n in ['natural_sources.json','calibration.json']:assert sha(audit/n)==frozen[n]
    config=read(HERE.parent/'complete_r36/config.json');assert sha(config['catalog'])==config['catalog_sha256']
    lib=Library(config['catalog']);natural=read(audit/'natural_sources.json');cal=read(audit/'calibration.json')
    for s,R in lib.rotations.items():assert array_hash(R)==natural[s]['rotation_sha256']
    z=np.load('outputs/duration_r62_01/main/cross_source/arrays.npz',allow_pickle=False);J=z['J'];parents=z['parents'];z.close()
    gate=read('outputs/timing_r64_01/main/result.json')['population']['original_numerical_gate']
    units=read('outputs/event_coverage_r47_01/index/result.json')['units']
    for u in units:assert u['native_sha256']==lib.rows[u['source_uid']][0]['reference']['sha256']
    return dict(lib=lib,natural=natural,cal=cal,J=J,parents=parents,gate=gate,units=units)

def inventory(ctx):
    base=ROOT/'assets/change_official_14';assert sha(base/'sources.json')==CFG['source_manifest_sha256'];m=read(base/'sources.json');rows=[]
    for s in m['sources']:
        p=base/s['file'];assert sha(p)==s['sha256'];z=np.load(p,allow_pickle=False);poses=z['poses'];trans=z['trans'];fps=float(z['mocap_framerate']);assert poses.shape==(s['frames'],165) and trans.shape==(len(poses),3) and fps==30 and np.isfinite(poses).all() and np.isfinite(trans).all()
        R=Rotation.from_rotvec(poses[:,:66].reshape(-1,3)).as_matrix().reshape(-1,22,3,3)
        parity=None
        if s['source_id'] in ctx['lib'].rotations:
            old=ctx['lib'].rotations[s['source_id']];assert R.shape==old.shape
            parity=float(np.max(np.abs(R-old)));assert parity<2e-6
        rows.append(dict(source_uid=s['source_id'],recording_uid=s['recording_uid'],track=s['performer_track_id'],frames=len(R),source_fps=30,bytes=p.stat().st_size,sha256=sha(p),solo_eligible=s['solo_compatible'],admission='SOLO_BODY22_VERIFIED' if s['solo_compatible'] else 'MULTI_TRACK_MANUAL_REVIEW_REQUIRED',native_body22_max_matrix_difference=parity,world_root_contact='UNKNOWN',semantic_completeness='UNKNOWN',dancer_identity='UNVERIFIED'))
        z.close()
    return dict(rows=rows,tracks=len(rows),recording_groups=len({r['recording_uid'] for r in rows}),solo_tracks=sum(r['solo_eligible'] for r in rows),raw_unique_hashes=len({r['sha256'] for r in rows}),claim_72='NOT_VERIFIED_IN_INSPECTED_PROJECT_AND_LEGACY_ASSET_ROOTS',raw_originals_modified=False)

def roster(ctx):
    cases=[];status=[];units=ctx['units'];natural=ctx['natural']
    for s in sorted(ctx['lib'].rotations):
        found=[];mask=natural[s]['regions']['body']['low_cells']
        for u in sorted([u for u in units if u['source_uid']==s],key=lambda u:(u['source_interval'],u['unit_uid'])):
            a,b=u['source_interval'];T=next((t for t in CFG['target_frames'] if .75<=(t-1)/(b-a-1)<=1.33),None)
            if T is None:continue
            for h in natural[s]['regions']['body']['low_runs']:
                l,r=h['cells_half_open']
                if a+4<l<r<b-5 and sum(not v for v in mask[a:l])>=30 and sum(not v for v in mask[r:b-1])>=30:found.append((u,h,T))
        if not found:status.append(dict(source_uid=s,status='NO_COMPLETE_MIXED_SUPPORT_AT_FROZEN_BUDGETS'));continue
        u,h,T=found[0];a,b=u['source_interval'];options=[]
        for alt in units:
            x,y=alt['source_interval']
            if alt['source_uid']==s and alt['unit_uid']!=u['unit_uid'] and abs((y-x)/(b-a)-1)<=.1 and .75<=(T-1)/(y-x-1)<=1.33:
                options.append((float(np.mean(mask[x:y-1])),abs((y-x)-(b-a)),x,alt))
        alternative=min(options,key=lambda x:x[:3])[3] if options else None
        if alternative and np.mean(mask[alternative['source_interval'][0]:alternative['source_interval'][1]-1])>=np.mean(mask[a:b-1]):alternative=None
        # Neighbor chosen by UID/length before costs or G1: no success-driven replacement.
        neighbor=next(v for v in sorted(units,key=lambda v:(v['source_uid'],v['source_interval'])) if v['source_uid']!=s and .75<=(T-1)/(v['duration_frames']-1)<=1.33)
        cases.append(dict(key=uid([u['unit_uid'],h['cells_half_open'],T]),unit=u,hold=h,target_frames=T,alternative=alternative,neighbor=neighbor,music_song=len(cases)%3,music_start_frame=900,source_role='AUTOMATIC_KINEMATIC_MIXED_NOT_SEMANTIC_COMPLETE'))
        status.append(dict(source_uid=s,status='REGISTERED',eligible_opportunities=len(found),alternative='AVAILABLE' if alternative else 'MISSING_MATCHED_LESS_LOW_SOURCE'))
    return cases,status

def clocks(u,hold,T):
    a,b=u['source_interval'];end=b-1;scale=(T-1)/(end-a)
    if hold is None:return [np.array([[a,0],[end,T-1]],float)]
    l,r=hold['cells_half_open'];xs=[float(a),a+3/scale,float(l),float(r),end-3/scale,float(end)]
    if not np.all(np.diff(xs)>0):raise ValueError('Protected edge overlaps selected hold')
    options=[[0.],[3.]]+[[scale*(x-a)+d for d in CFG['offset_grid']] for x in [l,r]]+[[float(T-4)],[float(T-1)]]
    g=dict(source=xs,options=options);counts=suffix(g);paths=[np.asarray(unrank(g,i,counts),float) for i in range(counts[0][0])]
    # Exactly retain the unmodified clock even if floating knot evaluation differs.
    uniform=np.array([[a,0],[end,T-1]],float);return [uniform]+paths

def evaluate_pool(ctx,u,h,T,music):
    s=u['source_uid'];a,b=u['source_interval'];R=ctx['lib'].rotations[s];baseq=np.linspace(a,b-1,T);baseR=rotation_sample(R,baseq);scorer=ReferenceCost(R[max(0,a-30):min(len(R),b+30)],ctx['J'],ctx['parents']);rows=[];arrays={}
    for w in clocks(u,h,T):
        validate(w,[a,b],[0,T]);q=np.interp(np.arange(T),w[:,1],w[:,0]);q[:4]=baseq[:4];q[-4:]=baseq[-4:]
        # Guard restoration is roundoff only; sampled slopes still satisfy inverse rate bounds.
        if not ((np.diff(q)>=1/1.33-1e-8)&(np.diff(q)<=1/.75+1e-8)).all():raise ValueError('Sampled inverse clock rate failed')
        k=array_hash(q)
        if k in arrays:continue
        V=rotation_sample(R,q);assert boundary_unchanged(q,baseq,V,baseR)
        primitive=SourcePrimitive(u['unit_uid'],s,tuple([a,b]),u['source_sha256']);candidate=ExecutionCandidate(primitive,music,T,tuple(map(tuple,w.tolist()))).validate()
        cost=scorer.score(V,only='GTWLocal')[0]['GTWLocal'];holds=metrics.occupancy(q,ctx['natural'][s]['regions']['body']['low_runs'],[a,b]);hold_seconds=sum(x['target_seconds'] for x in holds)
        rows.append(dict(key=k,candidate=asdict(candidate),GTWLocal=cost,mapped_low_seconds=hold_seconds,max_clock_shift_frames=float(np.max(np.abs(q-baseq))),protected_four_frames_exact=True))
        arrays[k]=(q,V)
    return rows,arrays,array_hash(baseq)

def summarize(R,ctx):
    sig,P=metrics.signals(R,ctx['J'],ctx['parents']);d=metrics.describe(R,P,sig['body'],ctx['cal']['thresholds']['0.2']['body']);return d

def connect(ctx,R,neighbor,T):
    s=neighbor['source_uid'];a,b=neighbor['source_interval'];right=rotation_sample(ctx['lib'].rotations[s],np.linspace(a,b-1,T));obs,gap=make_observed(R,right,48);curves,proof=solve_boundary(obs,gap);small=curves['G1']
    if small is None:return None,dict(constructible=False,passes_original_screen=False,proof=proof,bridge_sha256=None,inherited=False)
    m=measure(small,obs,ctx['J'],ctx['parents'],gap,ctx['gate']);full=np.concatenate([R,small[gap[0]:gap[1]],right]);return full,dict(m,proof=proof,bridge_sha256=array_hash(small[gap[0]:gap[1]]),inherited=False)

def run(phase='main',dest=None):
    if phase=='controls' and not dest:
        receipt=read(ROOT/'receipts/preview_browser_acceptance.json');assert receipt['status']=='PASS'
    out=Path(dest) if dest else ROOT/phase;out.mkdir(parents=True,exist_ok=False);ctx=prepare();inv=inventory(ctx);cases,status=roster(ctx)
    # This registration exists before the first GTW score or G1 solve.
    write(out/'registration.json',dict(protocol=CFG,cases=cases,status=status,source_inventory=inv,input_files={p:sha(p) for p in ['outputs/event_coverage_r47_01/index/result.json','outputs/hold_timing_01/audit/manifest.json','outputs/duration_r62_01/main/cross_source/arrays.npz','outputs/timing_r64_01/main/result.json']},original_numerical_gate=ctx['gate'],exposure='HISTORICALLY_EXPOSED_RECORDINGS_AND_MUSIC_NOT_INDEPENDENT_TEST'))
    records=[];pools=[];saved={};decisions=[]
    for c in cases:
        T=c['target_frames'];audio=Path('outputs/complete_r36_01/viewer/audio')/('dunhuangwu%d.wav'%(c['music_song']+2));music=MusicEvidence(sha(audio),(c['music_start_frame'],c['music_start_frame']+T));sources=[('original',c['unit'],c['hold'])]
        if phase=='controls' and c['alternative']:sources.append(('alternative',c['alternative'],None))
        for origin,u,h in sources:
            # Alternative receives the same finite timing mechanism, using its first fully internal low run if present.
            if origin=='alternative':
                a,b=u['source_interval'];h=next((v for v in ctx['natural'][u['source_uid']]['regions']['body']['low_runs'] if a+5<v['cells_half_open'][0]<v['cells_half_open'][1]<b-5),None)
            rows,arrays,base=evaluate_pool(ctx,u,h,T,music);pools.append(dict(case_key=c['key'],source=origin,rows=rows));best=min(rows,key=lambda v:(v['GTWLocal'],v['max_clock_shift_frames'],v['key']));chosen=[('Uniform',next(v for v in rows if v['key']==base)),('GTWLocal',best)]
            if phase=='controls':chosen.append(('MinHoldDiagnostic',min(rows,key=lambda v:(v['mapped_low_seconds'],v['GTWLocal'],v['key']))))
            for label,v in chosen:
                q,R=arrays[v['key']];name=c['key']+'__'+origin+'_'+label;full,connection=connect(ctx,R,c['neighbor'],T);saved[name+'__q']=q;saved[name+'__R']=R
                if full is not None:saved[name+'__full_R']=full
                audit=ExecutionAudit(v['key'],array_hash(R),connection['passes_original_screen']);decision=SelectionDecision(v['key'],label);decisions.append(dict(case_key=c['key'],source=origin,decision=asdict(decision)))
                records.append(dict(case_key=c['key'],method=origin+'_'+label,source_uid=u['source_uid'],source_interval=u['source_interval'],candidate_uid=v['key'],audit=asdict(audit),GTWLocal=v['GTWLocal'],mapped_low_seconds=v['mapped_low_seconds'],core=summarize(R,ctx),connection=connection,full_rotation_sha256=None if full is None else array_hash(full),target_frames=T,protected_four_frames_exact=v['protected_four_frames_exact']))
        print('SUPPORT_CASE',phase,c['key'],T,flush=True)
    save_arrays(out/'arrays.npz',saved);write(out/'pools.json',pools);write(out/'records.json',records);write(out/'decisions.json',decisions)
    write(out/'summary.json',dict(cases=len(cases),source_status=status,selected_positions=len(records),distinct_selected_rotations=len({r['audit']['rotation_sha256'] for r in records}),G1_screen_passes=sum(r['audit']['bridge_screen'] for r in records),candidate_clocks=sum(len(p['rows']) for p in pools),varying_hold_pools=sum(max(r['mapped_low_seconds'] for r in p['rows'])-min(r['mapped_low_seconds'] for r in p['rows'])>1e-7 for p in pools),human_records=0,music_quality='NOT_EVALUABLE',native24='NOT_EVALUABLE',can_route=False,new_training_steps=0,R74='NOT_STARTED'))
    seal(out)

if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'main')
