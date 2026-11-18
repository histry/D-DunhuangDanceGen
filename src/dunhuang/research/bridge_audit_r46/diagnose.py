"""Train-only candidate oracle upper bounds; never an inference or training target."""
import argparse,copy,collections,subprocess,shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.reperformance_r44.archive import Arrays
from dunhuang.research.reperformance_r44.motion import retime,world,kinematics
from dunhuang.research.phase_r45.clock import feasible_bank,choose
from dunhuang.research.phase_r45.audio import extract,timing
from dunhuang.research.sdmr_event_rank.evaluate import group_ci

METRICS=('rms_timing_seconds','wrist_acceleration_timing_seconds')

def shifted(signal,offset):
    result=copy.deepcopy(signal);n=len(signal['onset']);idx=np.arange(n)-offset;valid=(idx>=0)&(idx<n)
    for key in ('onset','beat_strength'):result[key]=np.interp(idx,np.arange(n),signal[key],left=0,right=0).tolist()
    for key in ('beats','rms_accent_frames','waveform_transient_frames'):result[key]=[int(t+offset) for t in signal[key] if 0<=t+offset<n]
    result['observed']=valid.tolist();result['shift_frames']=offset;result['shift_scope']='KNOWN_AUDIO_CLOCK_PERTURBATION_NOT_ARTISTIC_ERROR';return result

def selection(bank,phases,signal,cfg):
    current=copy.deepcopy(bank)
    if signal is not None and 'observed' in signal:
        for p in current:
            p['eligible']=p['eligible'] and all(signal['observed'][int(t)] for t in p['target'][1:-1])
    if not any(p['eligible'] for p in current):return None
    return choose(current,phases,signal,cfg)

def best_oracle(bank,errors):
    usable=[(sum(e[k] for k in METRICS),p) for p,e in zip(bank,errors) if p['eligible'] and all(e[k] is not None for k in METRICS)]
    return min(usable,key=lambda v:(v[0],v[1]['pattern_id']))[1] if usable else None

def run(out):
    out=Path(out);cfg=read(Path(__file__).with_name('config.json'));root=Path('outputs/bridge_audit_r46_01')
    if not read(root/'receipts/main_browser_acceptance.json')['passed']:raise ValueError('Deliver and inspect actual bridge effects before oracle controls')
    if out.exists():raise ValueError('Fresh R46 oracle output')
    out.mkdir(parents=True);parent=Path(cfg['parent'])/'main';meta=read(parent/'result.json');pcfg=meta['config'];z=Arrays(parent/'arrays.npz');J=z['skeleton_J22'];parents=z['parents'];rows=[r for r in meta['records'] if r['source']['domain']=='AIST++' and r['source']['role']=='train'];assert len(rows)==8
    signals={r['key']:extract(parent/r['audio'],pcfg) for r in rows};rms={}
    for r in rows:
        y,sr=sf.read(parent/r['audio']);rms[r['key']]=float(np.sqrt(np.mean(y*y)))
    records=[];values=dict(skeleton_J22=J,parents=parents);groups=collections.defaultdict(dict)
    for i,r in enumerate(rows):
        key=r['key'];raw=z[key+'_Geometry_139'][0];A=signals[key];donors=[x for x in rows if x['key']!=key and signals[x['key']]['audio_sha256']!=A['audio_sha256']];energy_donors=[x for x in donors if float(np.dot(A['onset'][:256],signals[x['key']]['onset'][:256])/(np.linalg.norm(A['onset'][:256])*np.linalg.norm(signals[x['key']]['onset'][:256])+1e-12))<.9 and A['beats']!=signals[x['key']]['beats']];energy=min(energy_donors,key=lambda x:(abs(np.log(max(rms[x['key']],1e-12)/max(rms[key],1e-12))),x['key'])) if energy_donors else None;wrong=next(x for x in rows[i+1:]+rows[:i] if x in donors);conditions=dict(Correct=A,Wrong=signals[wrong['key']],NoMusic=None);conditions.update({'EnergyMatched':signals[energy['key']]} if energy else {});conditions.update({'Shift'+str(s):shifted(A,s) for s in cfg['time_shifts']});clocks={name:np.arange(256,dtype=float) for name in list(conditions)+['OracleCorrect','OracleWrong']};bank_records=[];available={k:True for k in clocks}
        for a,b in pcfg['core_intervals']:
            bank,phases,evidence=feasible_bank(raw,J,parents,a,b,pcfg);small=dict(pcfg,core_intervals=[[a,b]]);correct_errors=[];wrong_errors=[];candidate_rows=[]
            for p in bank:
                motion=retime(raw,p['q']);ea=timing(motion,A,J,parents,small);eb=timing(motion,conditions['Wrong'],J,parents,small);correct_errors.append(ea);wrong_errors.append(eb);candidate_rows.append(dict(pattern_id=p['pattern_id'],target=p['target'],eligible=p['eligible'],identity=p['identity'],correct_music_error=ea,wrong_music_error=eb,source_feasibility=p['ratios']))
            choices={name:selection(bank,phases,signal,pcfg) for name,signal in conditions.items()};choices['OracleCorrect']=best_oracle(bank,correct_errors);choices['OracleWrong']=best_oracle(bank,wrong_errors)
            for name,p in choices.items():
                if p is None:available[name]=False
                else:clocks[name][a:b]=p['q'][a:b]
            bank_records.append(dict(interval=[a,b],candidate_errors=candidate_rows,chosen={k:None if p is None else p['pattern_id'] for k,p in choices.items()},selection_scope='OracleCorrect uses both registered evaluation proxies on Train only; OracleWrong optimizes wrong music and is evaluated against correct audio; no inference access'))
        methods={name:retime(raw,q) for name,q in clocks.items() if available[name]};methods['Geometry']=raw.copy();metrics={};curves={}
        for name,x in methods.items():
            values[key+'_'+name+'_139']=np.stack([x,x]);xyz=world(x,J,parents);values[key+'_'+name+'_22']=np.stack([xyz,xyz]);values[key+'_'+name+'_clock']=clocks.get(name,np.arange(256,dtype=float));metrics[name]=dict(timing(x,A,J,parents,pcfg),**kinematics(x,J,parents));curves[name]=np.r_[np.zeros(3),np.sqrt(np.mean(np.diff(xyz,n=3,axis=0)**2,axis=(1,2)))*27000].tolist()
        group=r['source']['music_identity']['source']['sha256'];assert group not in groups['registered'];groups['registered'][group]=key
        for k in METRICS:
            for control in ('Wrong','EnergyMatched','NoMusic','OracleWrong'):
                p=metrics['OracleCorrect'][k] if 'OracleCorrect' in metrics else None;c=metrics[control][k] if control in metrics else None
                if p is not None and c is not None:groups['OracleCorrect/'+k+'/'+control][group]=c-p
            for control in ('Wrong','EnergyMatched','NoMusic'):
                p=metrics['Correct'][k] if 'Correct' in metrics else None;c=metrics[control][k] if control in metrics else None
                if p is not None and c is not None:groups['Correct/'+k+'/'+control][group]=c-p
        assert np.array_equal(methods['NoMusic'],raw)
        if 'Correct' in methods:assert np.array_equal(methods['Correct'],z[key+'_Phase_139'][0])
        values[key+'_reference_R']=z[key+'_reference_R'];shutil.copyfile(parent/r['audio'],out/r['audio']);records.append(dict(key=key,source={k:v for k,v in r['source'].items() if k in ('recording_uid','domain','role','music_identity','source_interval')},audio=r['audio'],methods=list(methods),music=A,seeds=[0,0],phase_maps=[],metrics=dict(methods=metrics,curves=curves),banks=bank_records,available=available,donors=dict(Wrong=wrong['key'],EnergyMatched=energy['key'] if energy else None),rms_by_query=rms,scope='TRAIN_ONLY_ORACLE_UPPER_BOUND_NOT_GENERATED_REPERFORMANCE; all methods evaluated on original paired audio; no held source or Test tensors; duplicated playback tracks not independent seeds.'))
        print('R46_ORACLE',key,'methods',len(methods),flush=True)
    summary={}
    for k,g in groups.items():
        if k=='registered':continue
        ci=group_ci(list(g.values()),cfg['bootstrap_seed'],cfg['bootstrap_draws']);summary[k]=dict(group_improvements_seconds=g,groups=len(g),mean=float(np.mean(list(g.values()))) if g else None,ci95=ci,passed=len(g)>=cfg['minimum_evaluable_groups'] and ci[0] is not None and ci[0]>0)
    oracle_pass=all(summary.get('OracleCorrect/'+k+'/'+c,{}).get('passed',False) for k in METRICS for c in ('Wrong','EnergyMatched','NoMusic','OracleWrong'));fixed_pass=all(summary.get('Correct/'+k+'/'+c,{}).get('passed',False) for k in METRICS for c in ('Wrong','EnergyMatched','NoMusic'))
    np.savez_compressed(out/'arrays.npz',**values);write(out/'result.json',dict(records=records,config=cfg,summary=summary,oracle_capacity_screen=oracle_pass,fixed_score_screen=fixed_pass,neural_training_admitted=False,reason='ORACLE_IS_IN_SAMPLE_DIAGNOSTIC_NOT_VALIDATED_TRAINING_CONTRACT',parent_arrays_sha256=sha(parent/'arrays.npz'),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),optimizer_steps=0,new_neural_samples=0,can_route=False));z.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
