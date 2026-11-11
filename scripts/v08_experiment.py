"""Pre-registered small diagnostics, not a search for the best Native score."""
import argparse,json,os,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dunhuang.io import sha,write
from dunhuang.pipeline import run
from dunhuang.planning.replanner import route,budget,verify_cache

def read(path):return json.loads(Path(path).read_text())
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    assets=read(a.assets)
    if subprocess.check_output(['git','diff','HEAD','--'],cwd=ROOT,text=True):raise ValueError('Tracked code dirty')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    specs=[
        ('diagnostic150',150,dict(max_plans=2,max_generations=4,attempts_per_plan=2,
            diagnostic_continue_on_broad_failure=True,max_diagnostic_replans=1,inner_order='condition_first')),
        ('overlap225',225,dict(max_plans=1,max_generations=2,attempts_per_plan=2,
            diagnostic_continue_on_broad_failure=True,max_diagnostic_replans=0,inner_order='seed_first')),
        ('default150',150,dict(max_plans=3,max_generations=6,attempts_per_plan=2))]
    write(out/'protocol.json',dict(code_commit=commit,assets=ref(a.assets),specs=specs,seed=20261111,
        optimizer_steps=0,training=False,short_clip_only=True,native_thresholds_unchanged=True,
        primary_question='Can exact Native feedback drive bounded regeneration and scoped replanning?',
        quality_claim='None. Overrides exercise interfaces despite known widespread failures.'))
    try:
        subprocess.run([assets['edge_python'],'-m','pytest','-q'],cwd=ROOT,check=True)
        results=[]
        for name,frames,limits in specs:
            q=dict(assets,profile='dunhuang',generator='temporal',retrieval_policy='heuristic',
                seed=20261111,frames=frames,output=str(out/name),
                closed_loop=dict(limits,enabled=True,max_active_seconds=1200,candidate_timeout_seconds=300))
            write(out/(name+'.json'),q);run(q)
            state=read(out/name/'state.json')
            if state['stop_reason']=='STOP_EXECUTION_UNKNOWN':raise ValueError(state.get('error'))
            before={str(p):sha(p) for p in (out/name).rglob('*.npy')}
            count=state['generations_started'];run(dict(q,closed_loop=dict(q['closed_loop'],resume=True)))
            assert read(out/name/'state.json')['generations_started']==count
            assert before=={str(p):sha(p) for p in (out/name).rglob('*.npy')}
            for entry in state['ledger']:verify_cache(read(entry['cache']['path']),entry['key'])
            results.append(dict(name=name,state=ref(out/name/'state.json'),resumed_no_new_generation=True))
        initial=out/'diagnostic150/candidate_0'
        prior=ROOT/'outputs/v07_01/temporal150/native.npy'
        assert sha(initial/'native.npy')==sha(prior),'Default Temporal numerical path changed'
        for name in ('diagnostic150','overlap225'):
            state=read(out/name/'state.json')
            assert not state['full_song_native_pass'] and not state['full_generation_qualified']
        state=read(out/'diagnostic150/state.json')
        assert len(state['plans'])==2,'Bounded local replan did not execute'
        for index in (0,2):
            case=out/'diagnostic150'/('candidate_'+str(index))
            q=read(case/'request.json');q['output']=str(out/('fresh_replay_'+str(index)))
            run(q)
            for filename in ('normalized.npy','windows.npy','edge.npy','native.npy','native_fk.npy'):
                assert sha(case/filename)==sha(Path(q['output'])/filename),'Fresh replay differs: '+filename
        assert read(out/'default150/state.json')['generations_started']==1
        write(out/'results.json',dict(cases=results,default_temporal_parity=True,
            fresh_inference_replay_exact=[0,2],resume_no_new_generations=True,optimizer_steps=0))
        write(out/'status.json',dict(status='COMPLETED_DIAGNOSTIC_ONLY',exit_status=0))
    except Exception:
        write(out/'status.json',dict(status='FAILED',traceback=traceback.format_exc()))
        raise
if __name__=='__main__':main()
