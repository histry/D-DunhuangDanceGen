"""V0.7 bounded diagnostics: no optimizer, no gate relaxation, no new training."""
import argparse,json,os,subprocess,sys,traceback
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write,sha,pinned

def read(p):return json.loads(Path(p).read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--prior',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--alignment-request')
    ap.add_argument('--stage',choices=['all','support','network','geometry'],default='all');ap.add_argument('--plans')
    a=ap.parse_args();assets=read(a.assets);prior=Path(a.prior).resolve();out=Path(a.out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[3];env=os.environ.copy();env['PYTHONPATH']=str(root/'src')
    def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
    def command(args):
        print('RUN',args,flush=True);subprocess.run(args,cwd=root,env=env,check=True)
    if a.stage=='support':
        from .support import audit
        audit(assets,out);return
    if a.stage=='network':
        from .network import diagnose
        diagnose(assets,read(a.plans),ref(prior/'probe/adapter.pt'),prior,out);return
    if a.stage=='geometry':
        legacy=pinned(assets['legacy_root'],assets['legacy_commit']);sys.path.insert(0,str(legacy))
        from evaluation.paired_alignment_observations_v3 import motion_from_edge151
        from dunhuang.planning.time_map import motion_features74,source_positions
        from dunhuang.generation.soft_reference import resample_native
        catalog=read(checked(assets['catalog']));refs={r['source_uid']:r['reference'] for r in catalog['entries']}
        comparisons=[]
        for uid,r in refs.items():
            native=np.load(checked(r))[:150];v,m=motion_features74(native);old,mask,_=motion_from_edge151(native)
            comparisons.append(dict(source=uid,max_abs=float(abs(v-old).max()),mask_equal=bool(np.array_equal(m,mask))))
            if not np.allclose(v,old,rtol=1e-3,atol=1e-4) or not np.array_equal(m,mask):raise ValueError('74D extractor parity differs')
        write(out/'geometry.json',dict(legacy_commit=assets['legacy_commit'],comparisons=comparisons))
        return
    if subprocess.check_output(['git','-C',str(root),'diff','HEAD','--'],text=True):raise ValueError('Tracked code changes')
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    write(out/'protocol.json',dict(commit=commit,assets=ref(a.assets),prior=str(prior),adapter=ref(prior/'probe/adapter.pt'),
        training_steps=0,optimizer_steps=0,seeds=[20261111,20261112],timesteps=[100,500,900],
        interpretation='raw-denoiser local sensitivities plus bounded 150-frame Native comparisons; not V1 training',
        interventions=['source replacement','legal time-warp perturbation','source removed','phase zeroed','all removed','same-song other real Jukebox window','matched completion visibility'],
        no_full_song_promotion=True))
    try:
        command([assets['edge_python'],'-m','pytest','-q'])
        common=['--assets',a.assets,'--prior',str(prior)]
        for stage,python in [('support',assets['edge_python']),('geometry',assets['native_python'])]:
            command([python,'-m','dunhuang.research.v07']+common+['--stage',stage,'--out',str(out/stage)])
        plans_folder=out/'plans';plans_folder.mkdir()
        from .plans import counterfactual_plans
        plans=counterfactual_plans(assets,prior,plans_folder);write(plans_folder/'index.json',plans)
        command([assets['edge_python'],'-m','dunhuang.research.v07']+common+
            ['--stage','network','--plans',str(plans_folder/'index.json'),'--out',str(out/'network')])
        if a.alignment_request:
            request=read(a.alignment_request);folder=out/'shadow';folder.mkdir()
            request.update(output=str(folder));write(folder/'request.json',request)
            command([assets['native_python'],'-m','dunhuang.planning.worker',str(folder/'request.json')])
            plan=read(folder/'event_plan.json')
            assert not plan['retrieval_qualified']
            assert [e['event_uid'] for e in plan['events']]==[e['event_uid'] for e in read(plans['original'])['events']]
        from dunhuang.pipeline import run
        cases=[]
        for name,profile,mode,planfile,seed,policy in [
            ('official150','edge_baseline','baseline',None,20261111,None),
            ('temporal150','dunhuang','temporal',plans['original'],20261111,None),
            ('dense_s1','dunhuang','event_guided',plans['original'],20261111,'dense_upper8'),
            ('matched_s1','dunhuang','event_guided',plans['original'],20261111,'completion_upper8'),
            ('replaced_s1','dunhuang','event_guided',plans['replaced'],20261111,'dense_upper8'),
            ('warp_s1','dunhuang','event_guided',plans['time_warp_changed'],20261111,'dense_upper8'),
            ('dense_s2','dunhuang','event_guided',plans['original'],20261112,'dense_upper8'),
            ('replaced_s2','dunhuang','event_guided',plans['replaced'],20261112,'dense_upper8')]:
            q=dict(assets,frames=150,seed=seed,profile=profile,mode=mode,anchors=[],output=str(out/name))
            if planfile:q['input_event_plan']=ref(planfile)
            if mode=='event_guided':
                q.update(generator='adapter',retrieval_policy='heuristic',adapter=ref(prior/'probe/adapter.pt'),
                    research_allow_motion_only_adapter=True,research_condition_policy=policy)
            run(q)
            p=read(out/name/'physical.json');g=read(out/name/'generation.json')
            cases.append(dict(case=name,seed=seed,motion_sha256=sha(out/name/'native.npy'),native=p['absolute'],
                metrics=p['audit'],fk_error_m=p['fk_max_abs_m'],sampler=g['sampler'],soft_reference=g.get('soft_reference')))
        prior_checks=dict(official150=prior/'official150/native.npy',temporal150=prior/'temporal150/native.npy',dense_s1=prior/'probe150/native.npy')
        for name,old in prior_checks.items():
            if sha(out/name/'native.npy')!=sha(old):raise ValueError('Frozen generation parity failed: '+name)
        for seed in (20261111,20261112):
            rows=[r for r in cases if r['seed']==seed and r['case'] not in ('official150','temporal150')]
            if len({r['sampler']['noise_sha256'] for r in rows})!=1:raise ValueError('Counterfactual sampling noise differs')
        write(out/'generation.json',dict(cases=cases,prior_exact_hashes={k:sha(v) for k,v in prior_checks.items()},
            main_native_passes=sum(c['native']['ok'] for c in cases),optimizer_steps=0))
        write(out/'status.json',dict(status='COMPLETED_DIAGNOSTIC_ONLY',training_steps=0,optimizer_steps=0,generation_cases=len(cases),
            full_motion_qualified=False,music_qualified=False,source_support_diagnosis=True,read_only_network_diagnosis=True))
    except Exception:
        write(out/'status.json',dict(status='FAILED',error=traceback.format_exc(),training_steps=0))
        raise

if __name__=='__main__':main()
