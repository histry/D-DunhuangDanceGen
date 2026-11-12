"""Published, bounded single-GPU sequence: preflight -> best method -> ablation."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.io import write,sha,checked
from dunhuang.pipeline import run


def read(path):return json.loads(Path(path).read_text())
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--paired-assets',required=True);ap.add_argument('--generation-assets',required=True)
    ap.add_argument('--plans',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();paired=read(a.paired_assets);generation=read(a.generation_assets);plans=read(a.plans)
    env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','diff','HEAD','--'],text=True):raise ValueError('Tracked checkout is dirty')
    protocol=dict(schema='event_transfer_comparison_v1',code_commit=commit,steps_per_model=1200,
        modes=['cross','additive'],training_seed=20261115,comparison='same parameters, initialization, data, steps, objective, Val selection and execution budget',
        paired_assets=ref(a.paired_assets),generation_assets=ref(a.generation_assets),plans=ref(a.plans),
        test_fitting=False,reference_pose_exact_claim=False,native_thresholds_unchanged=True,
        maximum_factorial_generations_per_model=12,maximum_dunhuang_generations_per_model=2,
        no_225_or_full_song_promotion=True)
    write(root/'protocol.json',protocol)
    def command(argv,name):
        started=time.monotonic()
        with (root/(name+'.log')).open('w') as f:
            result=subprocess.run(list(map(str,argv)),env=env,stdout=f,stderr=subprocess.STDOUT)
        write(root/(name+'.exit.json'),dict(exit_status=result.returncode,seconds=time.monotonic()-started,
             code_commit=commit))
        if result.returncode:raise RuntimeError('Stage failed: '+name)
    try:
        python=paired['edge_python']
        command([python,'-m','pytest','-q'],'tests')
        common=[python,'-m','dunhuang.research.legacy.event_transfer','--assets',a.paired_assets]
        command(common+['--preflight-only','--out',root/'preflight'],'preflight')
        summaries={}
        for mode in protocol['modes']:
            folder=root/mode;folder.mkdir()
            train=folder/'train';mechanism=folder/'mechanism'
            command(common+['--mode',mode,'--steps','1200','--out',train],mode+'_training')
            command([python,'-m','dunhuang.research.mechanism','--assets',a.paired_assets,
                     '--adapter',train/'adapter.pt','--out',mechanism],mode+'_mechanism')
            # Reuse the frozen A65 requests and two real ChangE plans. No paired label is invented.
            crossdomain=[]
            for case in ('original','replaced'):
                q=dict(generation,frames=150,seed=20261115,profile='dunhuang',generator='adapter',
                    retrieval_policy='heuristic',output=str(folder/('dunhuang_'+case)),
                    input_event_plan=ref(plans[case]),adapter=ref(train/'adapter.pt'),
                    research_native_event_adapter=True,research_allow_motion_only_adapter=True,
                    research_condition_policy='completion_upper8')
                run(q)
                out=Path(q['output'])
                crossdomain.append(dict(case=case,motion=ref(out/'native.npy'),physical=read(out/'physical.json'),
                    generation=read(out/'generation.json')))
            command([python,'scripts/verify_event_transfer.py','--assets',a.paired_assets,'--training',train,
                     '--out',folder/'validation_replay.json'],mode+'_validation_replay')
            summaries[mode]=dict(training=read(train/'status.json'),initialization=read(train/'initialization.json'),
                validation=read(train/'validation.json'),factorial=read(mechanism/'generation.json'),
                dunhuang=crossdomain,replay=read(folder/'validation_replay.json'))
            write(root/'progress.json',summaries)
        if summaries['cross']['initialization']!=summaries['additive']['initialization']:
            raise ValueError('Parameter count/initialization/frozen base did not match')
        write(root/'results.json',summaries)
        write(root/'status.json',dict(status='COMPLETED_BOUNDED_EVENT_TRANSFER_RESEARCH',
            training_steps=2400,models=2,free_generations=28,
            seconds=time.monotonic()-start,music_qualified=False,full_motion_qualified=False,
            stop_reason='END_REGISTERED_150_FRAME_MECHANISM_COMPARISON_NO_PRODUCTION_PROMOTION'))
        print('TRANSFER_EXPERIMENT_COMPLETED',flush=True)
    except Exception:
        write(root/'status.json',dict(status='FAILED',error=traceback.format_exc(),music_qualified=False,full_motion_qualified=False))
        raise

if __name__=='__main__':main()
