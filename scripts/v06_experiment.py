"""Bounded V0.6 research: asset preflight, one probe, matched-noise controls.
No qualified model or downstream long-generation promotion is inferred.
"""
import argparse,json,os,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dunhuang.io import write,sha
from dunhuang.pipeline import run

def read(p):return json.loads(Path(p).read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--prior-root',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=300)
    args=ap.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    assets=read(args.assets);prior=Path(args.prior_root).resolve()
    if subprocess.check_output(['git','-C',str(ROOT),'diff','HEAD','--'],text=True):raise ValueError('Tracked source changes')
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
    def command(argv):
        print('RUN',argv,flush=True);subprocess.run(argv,env=env,cwd=ROOT,check=True)
    def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
    write(out/'protocol.json',dict(commit=commit,assets=ref(args.assets),prior_root=str(prior),
        seed=20261111,training_steps=args.steps,training='one source-held-out upperbody probe only',
        cases=['official150','temporal150','none150','zero150','probe150','none225','zero225','probe225'],
        overlap225='engineering diagnostic even when150 Native fails; never quality promotion',
        stop_after225=True,no450_or_whole_quality_promotion=True))
    try:
        command([assets['edge_python'],'-m','pytest','-q'])
        # Preflight actual published Event Plans at three lengths.
        for n in (150,450,3601):
            command([assets['edge_python'],str(ROOT/'scripts/inspect_event_reference.py'),'--plan',str(prior/f'dunhuang_{n}/event_plan.json'),
                '--scaler',assets['scaler']['path'],'--out',str(out/f'reference_{n}')])
        for task in ('motion','paired','upperbody_probe'):
            command([assets['edge_python'],str(ROOT/'scripts/train.py'),'--assets',args.assets,'--task',task,'--preflight-only','--out',str(out/f'preflight_{task}')])
        if read(out/'preflight_upperbody_probe/status.json')['status']!='PREFLIGHT_READY_RESEARCH_ONLY':
            write(out/'status.json',dict(status='BLOCKED_PROBE_ASSETS',training_steps=0));return
        command([assets['edge_python'],str(ROOT/'scripts/train.py'),'--assets',args.assets,'--task','upperbody_probe','--steps',str(args.steps),'--out',str(out/'probe')])
        status=read(out/'probe/status.json')
        if status.get('training_steps')!=args.steps:raise ValueError('Probe did not complete')
        import torch,numpy as np
        payload=torch.load(out/'probe/adapter.pt',map_location='cpu',weights_only=True)
        payload['state']['output.weight'].zero_();payload['state']['output.bias'].zero_()
        payload['condition_scope']='zero_residual_control';torch.save(payload,out/'zero_adapter.pt')
        plans={150:prior/'dunhuang_150/event_plan.json'}
        planning=out/'planning225';planning.mkdir()
        request=dict(assets,profile='dunhuang',generator='adapter',retrieval_policy='heuristic',frames=225,output=str(planning))
        write(planning/'request.json',request)
        command([assets['native_python'],'-m','dunhuang.planning.worker',str(planning/'request.json')])
        plans[225]=planning/'event_plan.json'
        # First replay both preserved paths to verify actual numerical invariance.
        for mode,profile in [('baseline','edge_baseline'),('temporal','dunhuang')]:
            name='official150' if mode=='baseline' else 'temporal150'
            q=dict(assets,mode=mode,profile=profile,seed=20261111,frames=150,output=str(out/name),anchors=[])
            if profile=='dunhuang':q['input_event_plan']=ref(plans[150])
            run(q)
            old=prior/('edge_baseline_150' if mode=='baseline' else 'dunhuang_150')/'native.npy'
            if sha(out/name/'native.npy')!=sha(old):raise ValueError('Preserved path changed: '+name)
        results=[]
        for n in (150,225):
            for kind in ('none','zero','probe'):
                name=kind+str(n)
                q=dict(assets,mode='custom_none',seed=20261111,frames=n,output=str(out/name),anchors=[])
                if kind!='none':
                    q.update(profile='dunhuang',generator='adapter',retrieval_policy='heuristic',
                        input_event_plan=ref(plans[n]),research_allow_motion_only_adapter=True,
                        adapter=ref(out/('zero_adapter.pt' if kind=='zero' else 'probe/adapter.pt')))
                run(q)
                command([assets['edge_python'],str(ROOT/'scripts/evaluate.py'),'--run',str(out/name)])
                g=read(out/name/'generation.json');p=read(out/name/'physical.json')
                results.append(dict(case=name,motion_sha256=sha(out/name/'native.npy'),generation=g,
                    native=p['absolute'],metrics=p['audit'],fk_error_m=p['fk_max_abs_m']))
            none,zero,probe=[out/(k+str(n)) for k in ('none','zero','probe')]
            assert sha(none/'native.npy')==sha(zero/'native.npy'),'Zero Adapter changed baseline'
            noise=[read(p/'generation.json')['sampler']['noise_sha256'] for p in (none,zero,probe)]
            assert len(set(noise))==1,'Matched sampler noise differs'
            write(out/f'matched_{n}.json',dict(zero_residual_exact=True,same_noise=True,noise_sha256=noise[0],
                trained_changes_output=sha(probe/'native.npy')!=sha(zero/'native.npy'),
                normalized_rms_change=float(np.sqrt(np.mean((np.load(probe/'normalized.npy')-np.load(zero/'normalized.npy'))**2))),
                native_zero=read(zero/'physical.json')['absolute'],native_probe=read(probe/'physical.json')['absolute']))
        write(out/'results.json',results)
        write(out/'status.json',dict(status='COMPLETED_RESEARCH_UNQUALIFIED',training_steps=args.steps,
            generation_cases=8,native_qualified=sum(r['native']['ok'] for r in results),
            preserved_official_and_temporal_exact=True,music_qualified=False,full_motion_qualified=False,
            stop_reason='No supported full151D paired training; no quality promotion beyond225 diagnostic'))
    except Exception:
        write(out/'status.json',dict(status='FAILED',traceback=traceback.format_exc(),music_qualified=False,full_motion_qualified=False))
        raise
if __name__=='__main__':main()
