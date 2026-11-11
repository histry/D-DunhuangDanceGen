"""Independent read-only verification of saved loop receipts and fresh replays."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dunhuang.io import checked,sha,write
from dunhuang.planning.event_plan import semantic_plan_id
from dunhuang.planning.replanner import verify_cache

def read(path):return json.loads(Path(path).read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();root=Path(a.root).resolve();output=Path(a.out).resolve()
    if output.exists():raise FileExistsError(output)
    assert read(root/'status.json')['exit_status']==0
    protocol=read(root/'protocol.json');assets=read(checked(protocol['assets']))
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
    result=dict(protocol=protocol,runs=[],replays=[],all_native_pass=False,full_song_qualified=False,
                optimizer_steps=0,event_global_blacklist=[],causal_claim='NONE')
    for name,frames,limits in protocol['specs']:
        folder=root/name;state=read(folder/'state.json');b=state['budget']
        assert state['terminal'] and state['input_identity'] and state['learned_risk'] is None
        assert state['generations_started']<=b['max_generations']
        assert len(state['plans'])<=b['max_plans'] and not state['event_global_blacklist']
        assert not state['full_generation_qualified'] and not state['full_song_native_pass']
        assert state['diagnostic_replans']<=b['max_diagnostic_replans']
        receipts=[p for p in folder.glob('resume_*.json') if not p.name.startswith('resume_contract_')]
        assert len(receipts)>=1
        for resume in receipts:assert read(resume)['new_generations']==0
        rows=[]
        for entry in state['ledger']:
            cache=read(checked(entry['cache']));feedback=verify_cache(cache,entry['key'])
            case=Path(cache['case']);physical=read(case/'physical.json');g=read(case/'generation.json')
            assert g['weights_before']==g['weights_after'] and not g['training']
            assert feedback['native_replay_exact'] and feedback['frames']==frames and not feedback['full_song_scope']
            assert not feedback['event_globally_infeasible'] and feedback['learned_physical_risk'] is None
            assert feedback['native_result']==('PASS' if physical['absolute']['ok'] else 'FAIL')
            assert feedback['native_contract_id']==read(folder/'native_contract.json')['id']
            assert feedback['generated_motion']==sha(case/'native.npy') and physical['fk_max_abs_m']<1e-5
            p=read(case/'event_plan.json')
            assert all(e['feasibility']['native']=='UNKNOWN' for e in p['events'])
            row=dict(case=str(case),key=entry['key'],action=entry['action'],variant=entry['variant'],
                native_result=entry['native_result'],native_reasons=physical['absolute']['reasons'],
                native_audit=physical['audit'],native_limits=physical['limits'],
                route=entry['route'],motion_sha256=sha(case/'native.npy'),fk_error_m=physical['fk_max_abs_m'],
                jerk={k:v for k,v in feedback['jerk'].items() if k!='rows'},
                broad_failure=feedback['broad_failure'],violating_frame_fraction=feedback['violating_frame_fraction'],
                event_response_rms=entry.get('matched_event_response_rms'))
            rows.append(row)
        for index,plan in enumerate(state['plans']):
            value=read(checked(plan['plan']))
            assert plan['native_result']=='UNKNOWN'
            if index:
                parent=read(checked(plan['parent']));slots=value['lineage']['changed_slots']
                assert value['lineage']['parent_semantic_id']==semantic_plan_id(parent)
                checked(value['lineage']['feedback'])
                assert len(slots)==1
                for slot,(old,new) in enumerate(zip(parent['events'],value['events'])):
                    if slot not in slots:assert old==new
                    else:assert old['event_uid']!=new['event_uid']
        result['runs'].append(dict(name=name,stop_reason=state['stop_reason'],
            generations_started=state['generations_started'],plans=len(state['plans']),
            diagnostic_replans=state['diagnostic_replans'],active_seconds=state['active_seconds'],cases=rows))
    for index in (0,2):
        old=root/'diagnostic150'/('candidate_'+str(index));fresh=root/('fresh_replay_'+str(index))
        equal={f:sha(old/f)==sha(fresh/f) for f in ('normalized.npy','windows.npy','edge.npy','native.npy','native_fk.npy')}
        assert all(equal.values())
        destination=root/('feedback_fresh_'+str(index)+'.json')
        subprocess.run([assets['native_python'],'-m','dunhuang.planning.feedback','--request',old/'request.json',
            '--case',old,'--contract-file',root/'diagnostic150/native_contract.json','--out',destination],env=env,check=True)
        assert read(destination)==read(old/'feedback.json')
        result['replays'].append(dict(candidate=index,arrays_equal=equal,native_feedback_exact=True))
    assert sha(root/'diagnostic150/candidate_0/native.npy')==sha(ROOT/'outputs/v07_01/temporal150/native.npy')
    result['default_temporal_exact']=True
    result['native_pass_count']=sum(c['native_result']=='PASS' for r in result['runs'] for c in r['cases'])
    result['candidate_count']=sum(len(r['cases']) for r in result['runs'])
    result['unique_motion_count']=len({c['motion_sha256'] for r in result['runs'] for c in r['cases']})
    write(output,result);print(json.dumps({k:result[k] for k in ('candidate_count','unique_motion_count','native_pass_count')}))
if __name__=='__main__':main()
