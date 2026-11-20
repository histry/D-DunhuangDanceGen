"""R67 isolated shared-candidate experiments; no training or Planner upgrade."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import rotation_sample,multi_warp
from dunhuang.research.warp_coupling_r53.common import array_hash,save_arrays
from dunhuang.research.paired_benchmark_r59.data import Data
from dunhuang.research.rhythm_r66 import metrics as body_metrics
from dunhuang.research.timing_r64.music import _reward
from .common import CFG,ROOT,fresh,read,write,sha,binding
from . import candidates,encoder,readout,metrics,population,connectivity

def seal(out):
    out=Path(out)
    write(out/'manifest.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},binding=binding()))

def deformation(warp):
    w=np.asarray(warp,float);delta=np.diff(w,axis=0);rho=(w[-1,1]-w[0,1])/(w[-1,0]-w[0,0])
    return float(np.average(np.log(delta[:,1]/delta[:,0]/rho)**2,weights=delta[:,0]))

def rank_indices(values):
    values=list(values)
    if any(v is not None and (isinstance(v,(bool,str)) or not isinstance(v,(int,float,np.number)) or not np.isfinite(v)) for v in values):
        raise ValueError('Scores must be finite numeric evidence or None')
    good=[(i,v) for i,v in enumerate(values) if v is not None]
    if not good:return []
    best=max(v for i,v in good)
    return [i for i,v in good if abs(v-best)<=1e-12]


def screened_rank_indices(values,connections):
    """Filter the shared score vector by actual G1 status, retaining bank indices."""
    values=list(values);connections=list(connections)
    if len(values)!=len(connections):raise ValueError('Scores and G1 results must use the same candidate bank')
    rank_indices(values)  # Invalid scores cannot be concealed by a rejected connection.
    return rank_indices([v if row.get('passes_original_screen') is True else None for v,row in zip(values,connections)])

def encode_bank(net,rotations,music):
    motions=[net.encode_rotations(R) for R in rotations]
    support=np.logical_and.reduce([v[1] for v in music.values()]+[v[1] for v in motions])
    results={};score={}
    for name,(tokens,valid,proof) in music.items():
        result=readout.score_bank(tokens,[x[0] for x in motions],valid,[x[1] for x in motions],centres=np.arange(6,234),fixed_support=support)
        results[name]=result;score[name]=result['scores']
        score[name]['RawDiagonal']=[float(np.mean(np.sum(tokens[support]*d[0][support],axis=1))) if support.any() else None for d in motions]
    return motions,support,results,score

def _energy(a,b,support):
    a=np.asarray(a)[6:234][support];b=np.asarray(b)[6:234][support]
    if len(a)<2:return None
    a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-12 else None

def run_main(out=None):
    target=fresh(out or ROOT/'main');ctx=connectivity.prepare();pop=dict(cases=ctx['change_cases'],source_intervals=8,music_tracks=3,scope='FROZEN_R66B_TWELVE_DISPLAY_POSITIONS_EIGHT_SOURCE_INTERVALS')
    banks=[dict(case_key=c['key'],bank=candidates.bank(c,16,CFG['seed'])) for c in pop['cases']]
    write(target/'banks.json',banks)
    net=encoder.FrozenEncoder()
    # R66's frozen raw50 source features already have a verified real audio clock.
    p=Path('outputs/exposed_full_song_01/music/arrays.npz');inventory=read('outputs/exposed_full_song_01/music/inventory.json')
    if not np.array_equal(net.joints[:22],ctx['J'][:22]) or not np.array_equal(net.parents[:22],ctx['parents'][:22]):raise ValueError('R41 and geometry skeletons differ')
    if sha(p)!=CFG['change_raw50_feature_arrays_sha256']:raise ValueError('Frozen raw music feature arrays changed')
    cp=Path('outputs/timing_r64_01/main/music_clocks.json')
    if sha(cp)!=CFG['r64_music_clocks_sha256']:raise ValueError('Frozen R64 clocks changed')
    music_arrays=np.load(p);clocks=read(cp)
    records=[];arrays={};display=[]
    for case,bank_record in zip(pop['cases'],banks):
        A,B=case['target_interval'];song=int(case['songidx']);clock=clocks[str(song)]
        if clock['audio_sha256']!=case['audio_sha256']:raise ValueError('ChangE target audio differs')
        asset=next(v for v in inventory if int(v['case'])==song)
        if asset['audio']['sha256']!=case['audio_sha256'] or B>asset['clock']['frames']:raise ValueError('Raw music features leave actual identified audio support')
        M=music_arrays[str(song)+'__music50'][A:B];MM=music_arrays[str(song)+'__music_mask'][A:B]
        mp=net.encode_music_raw(M,MM);bank=bank_record['bank']
        source=ctx['lib'].rotations[case['source_uid']];R=[rotation_sample(source,c['q']) for c in bank['candidates']]
        if array_hash(R[0])!=case['baseline_core_R_sha256']:raise ValueError('Frozen E1W baseline changed')
        motions,support,details,scores=encode_bank(net,R,{'actual':mp})
        own=[];connections=[]
        baseline_body=R[0]
        for j,(candidate,body) in enumerate(zip(bank['candidates'],R)):
            connection,geometry=connectivity.check(case,body,ctx);connections.append(connection)
            key=case['key']+'_'+candidate['uid'][:16];arrays[key]=body;arrays[key+'_q']=np.asarray(candidate['q'])
            for geom_name,geom_value in geometry.items():arrays[key+'_G1_'+geom_name]=geom_value
            # Full graph, every ranker and every selected display read this same executed table.
            records.append(dict(case_key=case['key'],source_uid=case['source_uid'],candidate_uid=candidate['uid'],
                rotation_sha256=array_hash(body),q_sha256=candidate['q_sha256'],scores={m:v[j] for m,v in scores['actual'].items()},
                connectivity=connection,encoding=motions[j][2],deformation=deformation(candidate['time_warp']),
                source_interval=case['source_interval'],target_interval=case['target_interval'],semantic_adaptation='NOT_EVALUABLE',
                actual_body_mean_angle_from_baseline_rad=float(np.linalg.norm(Rotation.from_matrix((baseline_body.swapaxes(-1,-2)@body).reshape(-1,3,3)).as_rotvec(),axis=-1).mean()),
                dynamics=body_metrics.measure(body,source,ctx['J'],ctx['parents'],candidate['q'],case['source_interval'],include_curves=False)))
        q=np.asarray([c['q'] for c in bank['candidates']]);diverse=int(np.argmax(np.mean(abs(q-q[:1]),axis=1)))
        selections=[('E1W',0),('Diverse',diverse)]
        qualified={name:screened_rank_indices(values,connections) for name,values in scores['actual'].items()}
        write(target/'selections'/(case['key']+'.json'),dict(raw_score_ties={n:rank_indices(v) for n,v in scores['actual'].items()},G1_screened_score_ties=qualified,abstain_when_empty=True,selection_display='RAW_SCORE_FIRST_REGISTERED_TIE; ACTUAL_G1_STATUS_SHOWN_SEPARATELY; NOT_QUALIFIED_ROUTING'))
        for name in ('Single64','Multi64'):
            tied=rank_indices(scores['actual'][name]);selections.append((name,tied[0] if tied else None))
        for name,j in selections:
            if j is None:
                own.append(dict(method=name,R=None,q=None,row=dict(status='NOT_EVALUABLE',reason='READOUT_HAS_NO_FINITE_SCORE')));continue
            c=bank['candidates'][j];body=R[j]
            measurement=body_metrics.measure(body,source,ctx['J'],ctx['parents'],c['q'],case['source_interval'],include_curves=True)
            own.append(dict(method=name,R=body,q=np.asarray(c['q']),row=dict(status='LEGAL_TIME_MAP',metrics=measurement,
                candidate_uid=c['uid'],score=scores['actual'].get(name,[None]*len(R))[j] if name in scores['actual'] else None,
                connectivity=connections[j],actual_candidate_count=len(R),semantic_adaptation='NOT_EVALUABLE')))
        display.append((case,own));write(target/'readouts'/(case['key']+'.json'),details)
        print('R67_CHANGE',len(display),len(pop['cases']),len(R),flush=True)
    net.check();save_arrays(target/'arrays.npz',arrays)
    write(target/'banks.json',banks);write(target/'records.json',records);write(target/'population.json',pop)
    write(target/'registration.json',dict(binding=binding(),protocol=CFG,encoder=net.binding,weights_before=net.before,weights_after=net.weight_digest(),
        connectivity_counts=ctx['counters'],feature_arrays_sha256=sha(p),inventory_sha256=sha('outputs/exposed_full_song_01/music/inventory.json')))
    from .viewer import build
    build(target,display,ctx['J'],ctx['parents'],clocks)
    seal(target);print('R67_MAIN_FINISHED',len(records),flush=True)

def paired_case(source_uid):
    phase=[dict(type='synthetic_grid_anchor',frame=f) for f in (48,96,144,192)]
    beats=list(range(6,239,6));base=multi_warp([0,240],[0,240],phase,beats)
    return dict(source_uid=source_uid,event_uid='REGISTERED_NEUTRAL_SYNTHETIC_GRID',source_interval=[0,240],target_interval=[0,240],
        support_component=[0,240],phase_inputs=phase,beats=beats,time_warp=base['time_warp'])

def run_controls(out=None):
    target=fresh(out or ROOT/'controls');pop=population.paired(CFG['r59_manifest']);data=Data(Path(CFG['r59_manifest']).parent)
    banks=[]
    for item in pop['cases']:
        case=paired_case(item['source_uid']);bank=candidates.bank(case,16,CFG['seed']);injection=candidates.inject(case,bank,item['key'])
        banks.append(dict(key=item['key'],bank=bank,injection=injection))
    write(target/'banks.json',banks)
    net=encoder.FrozenEncoder();rows=[];arrays={};proofs={}
    from dunhuang.research.paired_event_r41.events import profiles
    for item,bank_record in zip(pop['cases'],banks):
        if not item['eligible']:raise ValueError('Registered paired crop has no support; no replacement')
        z=data.source(item['source_uid']);a,b=item['source_interval'];original=z['R'][a:b]
        donor=item['donor'];dz=data.source(donor['source_uid']);da,db=donor['source_interval']
        music={'actual':net.encode_music_normalized(z['M'][a:b],z['MV'][a:b]),
            'zero':net.encode_music_normalized(np.zeros_like(z['M'][a:b]),z['MV'][a:b]),
            'swap':net.encode_music_normalized(dz['M'][da:db],dz['MV'][da:db])}
        energy={'actual':z['E'][a:b],'zero':np.zeros(240),'swap':dz['E'][da:db]}
        case=paired_case(item['source_uid']);bank=bank_record['bank'];injection=bank_record['injection'];q0=bank['candidates'][0]['q']
        for state in CFG['paired_states']:
            if state=='injected' and injection['status']!='CONSTRUCTED':
                raise ValueError('Preregistered mathematical positive control not constructible; no easier replacement')
            source=original if state=='untouched' else rotation_sample(original,injection['q_injected'])
            h=[[0,0],[239,239]] if state=='untouched' else injection['h']
            bodies=[rotation_sample(source,c['q']) for c in bank['candidates']]
            motions,support,details,scores=encode_bank(net,bodies,music)
            mask=np.zeros(240,bool);mask[6:234]=support
            errors=candidates.clock_errors(bank,h,q0,mask);oracle=candidates.domain_oracle(bank,h,q0,mask)
            costs=[deformation(c['time_warp']) for c in bank['candidates']]
            for control in music:
                scores[control]['Energy']=[_energy(energy[control],profiles(R)['global'],support) for R in bodies]
            uid=item['key']+'_'+state;indices={c['uid']:j for j,c in enumerate(bank['candidates'])}
            prefixes={k:[indices[v] for v in values] for k,values in bank['prefixes'].items()}
            hashes=[array_hash(R) for R in bodies]
            teacher=rotation_sample(original,q0)
            body_errors=[float(np.linalg.norm(Rotation.from_matrix((teacher[mask].swapaxes(-1,-2)@R[mask]).reshape(-1,3,3)).as_rotvec(),axis=-1).mean()) for R in bodies]
            rows.append(dict(uid=uid,source_uid=item['source_uid'],raw_motion_sha256=item['raw_motion_sha256'],group=item['group'],state=state,
                source_interval=item['source_interval'],candidate_errors=[r['mae_frames'] for r in errors],prefixes=prefixes,
                scores_by_control=scores,actual_candidate_count=len(bodies),deformation_costs=costs,
                bank_oracle_min=min(r['mae_frames'] for r in errors),full_domain_oracle_error=oracle['minimum_mae_frames'],
                rotation_hashes=hashes,distinct_actual_body_count=len(set(hashes)),body_reconstruction_mean_angles_rad=body_errors,body_error_scope='DOUBLE_SO3_RESAMPLING_FLOOR_NOT_NEURAL_RECONSTRUCTION',injection_status='NOT_INJECTED' if state=='untouched' else injection['status'],
                physical_clock_truth='CONTROLLED_COMPOSITION_NOT_ARTISTIC_OR_SEMANTIC_TRUTH'))
            for j,R in enumerate(bodies):arrays[uid+'_'+str(j)]=R
            proofs[uid]=dict(music={k:v[2] for k,v in music.items()},motion=[v[2] for v in motions],readouts=details,
                oracle=oracle,composed_errors=errors,common_source_target_token_mask=support.tolist())
            print('R67_PAIRED',len(rows),24,state,flush=True)
    save_arrays(target/'arrays.npz',arrays);write(target/'banks.json',banks);write(target/'rows.json',rows)
    write(target/'proofs.json',proofs);write(target/'population.json',pop)
    write(target/'summary.json',metrics.summarize(rows,seed=CFG['seed'],draws=CFG['bootstrap_draws']))
    write(target/'registration.json',dict(binding=binding(),protocol=CFG,encoder=net.binding,weights_before=net.before,weights_after=net.weight_digest()))
    seal(target);print('R67_CONTROLS_FINISHED',len(rows),flush=True)

def verify(mode,out):
    source=ROOT/mode;old=read(source/'manifest.json');now=binding()
    if old['binding']['code']!=now['code'] or old['binding']['dependencies']!=now['dependencies']:raise ValueError('Scientific code changed')
    for name,digest in old['files'].items():
        if sha(source/name)!=digest:raise ValueError('Frozen source output changed')
    if mode=='phase_audit':
        from .phase_audit import run
        run(out)
    elif mode=='main':run_main(out)
    else:run_controls(out)
    differences=[n for n,v in old['files'].items() if sha(Path(out)/n)!=v]
    write(Path(out)/'independent.json',dict(mode=mode,scientific_outputs_exact=not differences,differences=differences,
        rebuilt_files=len(old['files']),source_manifest_sha256=sha(source/'manifest.json'),binding=now,training_steps=0))
    if differences:raise ValueError('Independent replay differs')
