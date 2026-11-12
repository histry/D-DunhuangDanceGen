"""Frozen event information-flow and matched factorial execution experiments."""
import argparse, hashlib, json, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
from dunhuang.io import checked, write, sha
from dunhuang.contracts import MotionAdapter, Q
from dunhuang.generation.targets import digest, phase_target
from dunhuang.generation.source_pairs import source_pairs
from .event_metrics import descriptor, representation_stats, identity_probe, dynamics, event_margin, event_timing

SEEDS = (20261113, 20261114)
TIMES = (100, 500, 900)
CONTROLS = ('original', 'reference_removed', 'mask_only', 'phase_reversed',
            'source_replaced', 'music_replaced', 'music_shifted', 'zero_adapter')
GROUPS = dict(contact=list(range(4)), root=list(range(4,7)),
              body0_11_rotation=list(range(7,79)), upper8=list(range(79,127)),
              wrists_hands=list(range(127,151)), legs=[7+6*j+k for j in (1,2,4,5,7,8,10,11) for k in range(6)])


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def population(assets):
    from dunhuang.data.paired_manifest import inspect
    _, _, gate = inspect(assets)
    if not gate['ready']:
        raise ValueError('Strict full-feature preflight failed')
    manifest = read(checked(assets['paired_generation_manifest']))
    unique = {}
    for r in sorted(manifest['rows'], key=lambda r:r['window_uid']):
        if r['role']=='val':
            unique.setdefault(r['recording_uid'], r)
    candidates=list(unique.values())
    if len(candidates)<8:
        raise ValueError('Eight distinct validation captures required')
    rows=[candidates[i] for i in np.linspace(0,len(candidates)-1,8).astype(int)]
    return rows, gate, source_pairs(manifest)


class Runtime:
    def __init__(self, assets, adapter_ref):
        import torch
        from .frozen import denoiser
        from dunhuang.generation.adapter import PrimitiveAdapter, authorize_adapter
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        self.model,self.normalizer,self.diffusion=denoiser(assets)
        with np.load(checked(assets['scaler'])) as s:
            self.scale=s['scale'].copy();self.bias=s['bias'].copy()
        if (not np.array_equal(self.scale,self.normalizer.scaler.scale_.cpu().numpy()) or
                not np.array_equal(self.bias,self.normalizer.scaler.min_.cpu().numpy())):
            raise ValueError('Frozen normalizer binding differs')
        saved=torch.load(checked(adapter_ref),map_location='cpu',weights_only=True)
        self.native_geometry=saved.get('condition_schema')=='native_event_music_v1'
        if self.native_geometry:
            from dunhuang.generation.event_music_adapter import EventMusicAdapter,authorize_event_adapter
            authorize_event_adapter(saved,assets['checkpoint']['sha256'],True)
            self.adapter=EventMusicAdapter(saved['mode']).cuda().eval()
        else:
            authorize_adapter(saved,assets['checkpoint']['sha256'],True)
            self.adapter=PrimitiveAdapter().cuda().eval()
        self.adapter.load_state_dict(saved['state'],strict=True)
        self.before=(digest(self.model),digest(self.adapter))
        self.state={};self.layers={}
        self.handles=[self.model.cond_encoder.register_forward_hook(self.inject)]
        for name, module in [('adapter_128',self.adapter.encoder),('adapter_512',self.adapter.output)]:
            self.handles.append(module.register_forward_hook(self.capture(name)))
        for i, module in enumerate(self.model.seqTransDecoder.stack):
            self.handles.append(module.register_forward_hook(self.capture('decoder_'+str(i))))

    def capture(self,name):
        def hook(module,args,value):
            if self.state.get('capture'):
                self.layers[name]=value.detach().cpu().numpy()[0]
        return hook

    def inject(self,module,args,music):
        if self.state.get('enabled',True):
            c=self.state['condition']
            if self.native_geometry:
                from dunhuang.generation.event_geometry import event_geometry
                if self.state.get('mask_only',False):
                    # A mask-hint intervention is not a fabricated zero SO(3) pose.
                    import torch
                    geometry=torch.zeros(len(c.value),150,153,device=c.value.device,dtype=c.value.dtype)
                    geometry[:,:,144:152]=c.known[:,:,79:127].reshape(-1,150,8,6).all(-1).to(c.value.dtype)
                    geometry[:,:,152:]=c.phase
                else:
                    geometry=event_geometry(c.value,c.known,c.phase)
                delta=self.adapter(geometry,music)
            else:
                delta=self.adapter(c.value,c.known,c.phase)
            result=music+delta
        else:
            result=music
        if self.state.get('capture'):
            self.layers['edge_condition']=result.detach().cpu().numpy()[0]
            self.layers['edge_music']=music.detach().cpu().numpy()[0]
        return result

    def condition(self,target):
        import torch
        from dunhuang.generation.conditions import source_condition
        phase=torch.arange(150,device='cuda').float()[None,:,None]/150
        if self.native_geometry:
            raw=((target.detach().cpu().numpy()-self.bias)/self.scale)
            target=torch.from_numpy(np.stack([MotionAdapter.to_native(x) for x in raw])).cuda()
        return source_condition(target,torch.ones_like(target,dtype=torch.bool),phase,'completion_upper8')

    def tensors(self,row):
        import torch
        native=np.load(checked(row['motion']),allow_pickle=False)
        target,_=phase_target(native,self.scale,self.bias)
        if target is None:
            raise ValueError('Validated target no longer representable')
        return (torch.from_numpy(target)[None].cuda(),
                torch.from_numpy(np.load(checked(row['jukebox'])))[None].cuda(), native)

    def close(self):
        for h in self.handles:
            h.remove()
        if self.before!=(digest(self.model),digest(self.adapter)):
            raise ValueError('Read-only mechanism run changed weights')
        if any(p.grad is not None for p in self.model.parameters()):
            raise ValueError('Frozen EDGE acquired gradients')


def diagnose(runtime,rows,out):
    import torch
    from dunhuang.generation.conditions import ReferenceCondition
    rt=runtime;data=[rt.tensors(r) for r in rows]
    targets=[r[0] for r in data];musics=[r[1] for r in data]
    conditions=[rt.condition(x) for x in targets]
    # Every event uses the SAME noisy state and music, preventing target identity leakage.
    torch.manual_seed(909);torch.cuda.manual_seed_all(909)
    t=torch.tensor([500],device='cuda')
    noisy=rt.diffusion.q_sample(targets[0],t,noise=torch.randn_like(targets[0]))
    features={};labels=[];views=[]
    with torch.no_grad():
        for identity,c in enumerate(conditions):
            for view in range(3):
                known=c.known & ((torch.arange(150,device='cuda')%3!=view)[None,:,None])
                value=torch.where(known,c.value,torch.zeros_like(c.value))
                current=ReferenceCondition(value,known,c.phase,'held_visibility_probe')
                rt.state.update(condition=current,enabled=True,capture=True)
                rt.model(noisy,musics[0],t,cond_drop_prob=0.)
                rt.layers['input_reference']=value.cpu().numpy()[0]
                for name,token in rt.layers.items():
                    features.setdefault(name,[]).append(descriptor(token))
                labels.append(identity);views.append(view)
    rt.state['capture']=False
    probes={k:dict(identity_probe(v,labels,views),**representation_stats(v)) for k,v in features.items()}
    np.savez_compressed(out/'identity_features.npz',**{k:np.stack(v) for k,v in features.items()},
                        labels=np.array(labels),views=np.array(views))
    records=[];predictions=[]
    with torch.no_grad():
        for seed in SEEDS:
            for i,(target,music,native) in enumerate(data):
                c=conditions[i];other=conditions[(i+1)%len(conditions)]
                alternative=next(musics[j] for j,r in enumerate(rows) if r['music_uid']!=rows[i]['music_uid'])
                for timestep in TIMES:
                    torch.manual_seed(seed+i);torch.cuda.manual_seed_all(seed+i)
                    t=torch.tensor([timestep],device='cuda');noise=torch.randn_like(target)
                    noisy=rt.diffusion.q_sample(target,t,noise=noise)
                    noise_sha=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
                    for control in CONTROLS:
                        current=c;m=music
                        if control=='reference_removed':
                            current=ReferenceCondition(torch.zeros_like(c.value),torch.zeros_like(c.known),c.phase,'removed')
                        elif control=='mask_only':
                            current=ReferenceCondition(torch.zeros_like(c.value),c.known,c.phase,'mask_only')
                        elif control=='phase_reversed':
                            current=ReferenceCondition(c.value,c.known,1-c.phase,'phase_reversed')
                        elif control=='source_replaced':
                            # Preserve the original availability map while changing only pose identity.
                            current=ReferenceCondition(torch.where(c.known,other.value,torch.zeros_like(other.value)),c.known,c.phase,'replaced')
                        elif control=='music_replaced':m=alternative
                        elif control=='music_shifted':m=torch.roll(music,37,1)
                        rt.state.update(condition=current,enabled=control!='zero_adapter',capture=False,mask_only=control=='mask_only')
                        y=rt.model(noisy,m,t,cond_drop_prob=0.)
                        row=dict(seed=seed,timestep=timestep,window_uid=rows[i]['window_uid'],control=control,noise_sha256=noise_sha,
                                 group_mse={k:float((y-target)[:,:,ids].square().mean()) for k,ids in GROUPS.items()},
                                 full_mse=float((y-target).square().mean()),
                                 hidden_mse=float((y-target).square()[~c.known].mean()),
                                 normalized_velocity_rms=float((y[:,1:]-y[:,:-1]).square().mean().sqrt()),
                                 target_normalized_velocity_rms=float((target[:,1:]-target[:,:-1]).square().mean().sqrt()))
                        records.append(row);predictions.append(y.cpu().numpy()[0])
    np.savez_compressed(out/'denoiser_predictions.npz',predictions=np.stack(predictions))
    write(out/'diagnostics.json',dict(identity_probes=probes,denoiser=records,
        scope='read-only closed-set visibility probes and target-noised completion; not free-running quality',
        selection='eight uniformly spaced distinct historical Val recordings, first eligible window per capture',
        music_shift_is='deliberately invalid circular-feature negative, not a paired training sample',
        zero_optimizer_calls=True))


def assembled(runtime,sample,music,out):
    import torch
    import model.diffusion as dm
    from pytorch3d.transforms import matrix_to_rotation_6d,axis_angle_to_matrix
    from scipy.spatial.transform import Rotation
    from dunhuang.contracts import matrices
    rt=runtime;skeleton=rt.diffusion.smpl;captured={}
    # Upstream implementations call this member smpl (verified by the exact pinned loader).
    original=skeleton.forward;render=dm.skeleton_render
    def observe(poses,root):
        value=original(poses,root)
        captured.update(poses=poses.detach().cpu().clone(),root=root.detach().cpu().clone(),fk=value.detach().cpu().clone())
        return value
    skeleton.forward=observe;dm.skeleton_render=lambda *args,**kwargs:None
    try:
        rt.diffusion.render_sample(sample.detach().cpu().clone(),music,rt.normalizer,'mechanism',str(out),
                                   sound=False,mode='long',render=False)
    finally:
        skeleton.forward=original;dm.skeleton_render=render
    raw=rt.normalizer.unnormalize(sample.detach().cpu().clone()).numpy()[0]
    rotations=matrix_to_rotation_6d(axis_angle_to_matrix(captured['poses'][0])).numpy().reshape(-1,144)
    edge=np.concatenate([raw[:,:4],captured['root'][0].numpy(),rotations],axis=1).astype(np.float32)
    np.save(out/'edge.npy',edge);np.save(out/'official_fk.npy',captured['fk'][0].numpy())
    np.save(out/'normalized.npy',sample.cpu().numpy())
    np.save(out/'raw_native.npy',MotionAdapter.to_native(raw))
    aa=Rotation.from_matrix(matrices(raw[:,7:].reshape(-1,24,6),row=True).reshape(-1,3,3)).as_rotvec().reshape(1,150,24,3).astype(np.float32)
    with torch.no_grad():
        fk=original(torch.from_numpy(aa).cuda(),torch.from_numpy(raw[None,:,4:7]).cuda())[0].cpu().numpy()@Q
    np.save(out/'raw_fk.npy',fk)


def generate(rt,rows,assets,out):
    import torch
    from dunhuang.sampling import constrained_ddim
    second=next(i for i,r in enumerate(rows) if r['music_uid']!=rows[0]['music_uid'])
    chosen=[rows[0],rows[second]];data=[rt.tensors(r) for r in chosen]
    cases=[]
    for seed in SEEDS:
        for music_id in range(2):
            for event_id in (-1,0,1):
                name=f's{seed}_m{music_id}_e{event_id}'
                folder=out/name;folder.mkdir()
                target,music,native=data[music_id]
                c=rt.condition(data[max(event_id,0)][0])
                rt.state.update(condition=c,enabled=event_id>=0,capture=False,mask_only=False)
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                snapshots={}
                def observe(step,t,predicted,clean):
                    if step in (0,24,49):
                        snapshots[str(step)]=predicted.cpu().numpy()
                with torch.no_grad():
                    sample,sampler=constrained_ddim(rt.diffusion,music,torch.zeros_like(target),
                                                   torch.zeros_like(target,dtype=torch.bool),guidance='none',observer=observe)
                    assembled(rt,sample,music,folder)
                np.savez_compressed(folder/'denoising_stages.npz',**snapshots)
                request=dict(assets,frames=150,output=str(folder),anchors=[],beat_frames=[])
                write(folder/'physical_request.json',request)
                subprocess.run([assets['native_python'],'-m','dunhuang.physical',str(folder/'physical_request.json')],check=True)
                native_out=np.load(folder/'native.npy');fk=np.load(folder/'native_fk.npy')
                metrics=dynamics(native_out,fk)
                raw_metrics=dynamics(np.load(folder/'raw_native.npy'),np.load(folder/'raw_fk.npy'))
                # Compare all identical four-frame templates before and after assembly.
                raw_jerk=np.diff(np.load(folder/'raw_fk.npy'),n=3,axis=0)*30**3
                final_jerk=np.diff(fk,n=3,axis=0)*30**3
                cases.append(dict(case=name,seed=seed,music_index=music_id,event_index=event_id,
                    music_window_uid=chosen[music_id]['window_uid'],
                    source_window_uid=chosen[max(event_id,0)]['window_uid'] if event_id>=0 else None,
                    sampler=sampler,motion=ref(folder/'native.npy'),fk=ref(folder/'native_fk.npy'),
                    physical=read(folder/'physical.json'),metrics=metrics,raw_window_metrics=raw_metrics,
                    event_timing=event_timing(native_out,data[max(event_id,0)][2]),
                    identical_template_jerk_delta_max_mps3=float(abs(final_jerk-raw_jerk).max())))
                write(out/'generation_progress.json',cases)
    margins=[]
    for seed in SEEDS:
        for m in range(2):
            pair=[next(c for c in cases if c['seed']==seed and c['music_index']==m and c['event_index']==e) for e in (0,1)]
            if pair[0]['sampler']['noise_sha256']!=pair[1]['sampler']['noise_sha256']:
                raise ValueError('Factorial sampler noise differs')
            margins.append(dict(seed=seed,music_index=m,
                **event_margin(*(np.load(checked(c['motion'])) for c in pair),data[0][2],data[1][2])))
    write(out/'generation.json',dict(cases=cases,event_preference=margins,
          budgets=dict(generations=12,steps_each=50,seeds=list(SEEDS),music=2,events=2,zero_adapter_cases=4),
          native_passes=sum(bool(c['physical']['absolute']['ok']) for c in cases),
          scope='historical Val factorial diagnosis, not a held-out benchmark or production promotion'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--assets',required=True);ap.add_argument('--adapter',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--diagnose-only',action='store_true')
    a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();assets=read(a.assets)
    try:
        rows,gate,pairs=population(assets)
        write(out/'protocol.json',dict(schema='event_mechanism_v1',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            assets=ref(a.assets),adapter=ref(a.adapter),preflight=gate,selected_windows=rows,
            source_pair_counts={role:sum(p['role']==role for p in pairs) for role in ('train','val')},
            source_relation='shared official basic routine/music/time, independent capture; NOT pose-exact truth',
            seeds=list(SEEDS),timesteps=list(TIMES),controls=list(CONTROLS),optimizer_calls=0,
            maximum_generations=0 if a.diagnose_only else 12,native_thresholds_unchanged=True))
        write(out/'source_pairs.json',pairs)
        rt=Runtime(assets,ref(a.adapter))
        try:
            diagnose(rt,rows,out)
            if not a.diagnose_only:
                generate(rt,rows,assets,out)
        finally:
            rt.close()
        write(out/'status.json',dict(status='COMPLETED_MECHANISM_DIAGNOSTIC_UNQUALIFIED',optimizer_calls=0,
            model_before=rt.before[0],model_after=digest(rt.model),
            adapter_before=rt.before[1],adapter_after=digest(rt.adapter),
            seconds=time.monotonic()-start,full_motion_qualified=False,music_qualified=False))
        print('MECHANISM_COMPLETED',flush=True)
    except Exception:
        write(out/'status.json',dict(status='FAILED',error=traceback.format_exc(),optimizer_calls=0))
        raise

if __name__=='__main__':
    main()
