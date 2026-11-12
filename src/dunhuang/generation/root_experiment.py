"""Matched soft Root/contact branch and one masked-condition ablation on frozen EDGE."""
import argparse,json,subprocess,time,copy,os
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha,write
from .event_transfer import TransferRuntime,prepare,read,SEED
from .root_contact import RootContactAdapter,root_contact_features,SCHEMA
from .train import digest

CONTROLS=('original','root_removed','zero_adapter','wrong_source','reference_removed')

def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))

class RootRuntime(TransferRuntime):
    def __init__(self,assets,index,initial,arm):
        super().__init__(assets,index,'cross');self.arm=arm
        self.adapter=RootContactAdapter('cross').cuda()
        payload=torch.load(checked(initial),map_location='cuda',weights_only=True)
        if payload['condition_schema']!='native_event_music_v1' or payload['base_sha256']!=assets['checkpoint']['sha256']:
            raise ValueError('Root branch base differs')
        missing=self.adapter.load_state_dict(payload['state'],strict=False)
        if missing.missing_keys!=['root_input.weight'] or missing.unexpected_keys:raise ValueError('Unexpected Root branch migration')
    def inject(self,module,args,music):
        if not self.state.get('enabled',True):return music
        delta,reconstruction=self.adapter(self.state['geometry'],music,True,root=self.state['root'])
        self.state['reconstruction']=reconstruction
        return music+delta
    def batch(self,pairs,control='original',wrong=None):
        target,music,geometry=super().batch(pairs,control,wrong)
        chosen=wrong if control=='wrong_source' else pairs
        data=[self.data(p) for p in chosen]
        native=torch.from_numpy(np.stack([r[2] for r in data])).cuda()
        observed=torch.from_numpy(np.stack([r[3] for r in data])).cuda()
        proxy=torch.ones(len(native),150,4,device='cuda',dtype=torch.bool)
        if control=='reference_removed':observed=torch.zeros_like(observed);proxy=torch.zeros_like(proxy)
        features=root_contact_features(native,observed,proxy)
        if self.arm=='masked' or control=='root_removed':features=torch.zeros_like(features)
        self.state['root']=features
        return target,music,geometry


def generate(assets,index,pairs,root,out,seed):
    import soundfile as sf
    from dunhuang.planning.event_plan import validate
    from dunhuang.evaluation_benchmarks.native_shadow import LegacyBackend
    from dunhuang.research.root_diagnosis import decompose
    root=Path(root);out=Path(out);out.mkdir();results=[]
    distinct={}
    for p in pairs:distinct.setdefault(index[p['target_uid']]['music_uid'],p)
    chosen=list(distinct.values())[:2]
    for arm,control in (('full','original'),('full','root_removed'),('masked','original')):
        for i,pair in enumerate(chosen):
            folder=out/(arm+'_'+control+'_'+str(i));folder.mkdir()
            row=index[pair['target_uid']];source=index[pair['source_uid']]
            source_native=np.load(checked(source['motion']))
            source_mask=np.load(checked(source['observed']))
            derived=np.load(checked(source['derived']))[:,:4]
            mask_file=folder/'root_observed.npy';proxy_file=folder/'contact_observed.npy'
            np.save(mask_file,source_mask);np.save(proxy_file,derived)
            evidence=dict(schema='source_root_observation_v1',source_sha256=source['motion']['sha256'],
                fps=30,units='m',up_axis='y',provenance='official fitted AIST source capture; full strict paired preflight',
                observed=ref(mask_file),contact_observed=ref(proxy_file),contact_semantics='UNVERIFIED_GEOMETRIC_PROXY')
            write(folder/'root_observation.json',evidence)
            audio,sr=sf.read(checked(row['audio']),always_2d=True)
            start=round((row['start_frame']/30+row['audio_offset_seconds'])*sr)
            crop=audio[start:start+5*sr]
            if len(crop)!=5*sr:raise ValueError('Paired audio crop lacks actual samples')
            sf.write(folder/'audio.wav',crop,sr,subtype='FLOAT')
            features=np.load(checked(row['jukebox']))
            np.save(folder/'features.npy',features[None])
            event=dict(event_uid='root_probe_source',source_uid=source['recording_uid'],reference=source['motion'],
                source_interval=[0,150],music_interval=[0,150],time_warp=[[0,0],[150,150]],
                primitive=dict(source_frame=75,music_frame=75,joints=list(range(12,20)),radius=12),
                feasibility=dict(native='UNKNOWN'))
            if control!='root_removed':event['root_observation']=ref(folder/'root_observation.json')
            plan=dict(schema='event_plan_v1',fps=30,frames=150,audio_sha256=sha(folder/'audio.wav'),
                time_scale_limits=[.75,1.33],events=[event])
            validate(plan,plan['audio_sha256'],150);write(folder/'event_plan.json',plan)
            q=dict(assets,mode='event_guided',output=str(folder),frames=150,windows=1,seed=seed,
                features=ref(folder/'features.npy'),audio=ref(folder/'audio.wav'),event_plan=ref(folder/'event_plan.json'),
                adapter=ref(root/arm/'adapter.pt'),anchors=[],beat_frames=[],overlap_policy='mean',
                research_native_event_adapter=True,research_allow_motion_only_adapter=True,
                research_condition_policy='completion_upper8',paired_jukebox_receipt=row['jukebox_receipt'],
                source_task=pair,source_pose_pasted=False)
            write(folder/'request.json',q)
            env=dict(os.environ);env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])+os.pathsep+env.get('PYTHONPATH','')
            for name,python,module in [('edge',assets['edge_python'],'dunhuang.edge_backend'),('native',assets['native_python'],'dunhuang.physical')]:
                with (folder/(name+'.log')).open('w') as log:
                    subprocess.run([python,'-m',module,str(folder/'request.json')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            native=np.load(folder/'native.npy');fk=np.load(folder/'native_fk.npy')
            motion=ref(folder/'native.npy')
            results.append(dict(arm=arm,control=control,music_index=i,seed=seed,motion=motion,
                legacy=read(folder/'physical.json')['absolute'],root_diagnostics=decompose(native,fk),
                root_reference_audit=read(folder/'root_reference_audit.json'),
                source_task=pair,qualified=False))
            print('ROOT_GENERATED',arm,control,i,flush=True)
    write(out/'results.json',results)
    return results

def run(assets,initial,out,steps,generation):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    index,pairs,gate=prepare(assets);train=[p for p in pairs if p['role']=='train'];val=[p for p in pairs if p['role']=='val']
    registered=read('outputs/event_transfer_01/cross/train/protocol.json')['validation_pairs']
    if any(p not in val for p in registered):raise ValueError('Historical fixed validation population changed')
    protocol=dict(schema=SCHEMA,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        base=assets['checkpoint'],initial_adapter=initial,paired_manifest=assets['paired_generation_manifest'],
        steps=steps,seed=SEED,learning_rate=.0001,batch=2,selection='fixed final step; no Val selection',
        train_pairs=len(train),validation_pairs=registered,contact_semantics='UNVERIFIED_GEOMETRIC_PROXY',
        independent_static_contact_confidence=0,output_root_replaced=False,rootnet_added=False,qualified=False)
    write(out/'protocol.json',protocol)
    rng=np.random.default_rng(SEED);schedule=rng.integers(0,len(train),(steps,2));np.save(out/'schedule.npy',schedule)
    results={};initials={}
    for arm in ('full','masked'):
        folder=out/arm;folder.mkdir();torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        rt=RootRuntime(assets,index,initial,arm);rt.validation=val
        initials[arm]=dict(adapter_digest=digest(rt.adapter),parameters=sum(p.numel() for p in rt.adapter.parameters()))
        optimizer=torch.optim.AdamW(rt.adapter.parameters(),lr=.0001);history=[]
        try:
            for step,ids in enumerate(schedule,1):
                target,music,geometry=rt.batch([train[i] for i in ids])
                rt.adapter.train();t=torch.randint(50,950,(len(ids),),device='cuda')
                predicted=rt.model(rt.diffusion.q_sample(target,t),music,t,cond_drop_prob=0.)
                loss,components=rt.loss(target,predicted,geometry)
                if not torch.isfinite(loss):raise ValueError('Nonfinite Root condition loss')
                optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(rt.adapter.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('Nonfinite Root condition gradient')
                optimizer.step()
                if step%100==0 or step==steps:
                    history.append(dict(step=step,loss=components,gradient_norm=float(norm)))
                    write(folder/'history.json',history);print('ROOT_TRAIN',arm,step,components['mse'],flush=True)
            torch.save(dict(state=rt.adapter.state_dict(),mode='cross',condition_schema=SCHEMA,
                condition_policy='completion_upper8',base_sha256=assets['checkpoint']['sha256'],
                protocol_sha256=sha(out/'protocol.json'),training_commit=protocol['code_commit'],
                condition_scope='cross-recording source Root/body soft condition; no exact Root retiming labels',
                step=steps,arm=arm,qualified=False,music_qualified=False,full_motion_qualified=False),folder/'adapter.pt')
            validation=rt.evaluate(registered,CONTROLS);write(folder/'validation.json',validation)
            if digest(rt.model)!=rt.before:raise ValueError('Frozen EDGE weights changed')
            results[arm]=dict(validation=validation,weights_before=rt.before,weights_after=digest(rt.model),
                checkpoint=ref(folder/'adapter.pt'))
        finally:rt.close()
        del rt,optimizer
        torch.cuda.empty_cache()
    assert initials['full']==initials['masked']
    write(out/'initializations.json',initials)
    generations=generate(assets,index,registered,out,out/'generation',20261117) if generation else []
    write(out/'result.json',dict(status='COMPLETED_ROOT_CONTACT_SOFT_CONDITION',results=results,
        initializations=initials,training_steps_per_arm=steps,generation_executions=len(generations),
        test_evaluations=0,rootnet_added=False,output_root_replaced=False,operational_qualified=False,
        seconds=time.monotonic()-start))
    print('ROOT_CONDITION_COMPLETED',flush=True)

def main():
    p=argparse.ArgumentParser()
    for k in ('assets','initial-adapter','out'):p.add_argument('--'+k,required=True)
    p.add_argument('--steps',type=int,default=400);p.add_argument('--generate',action='store_true')
    a=p.parse_args();run(read(a.assets),ref(a.initial_adapter),a.out,a.steps,a.generate)
if __name__=='__main__':main()
