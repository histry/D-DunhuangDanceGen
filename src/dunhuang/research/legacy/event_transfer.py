"""Bounded cross-recording event-conditioned denoising, not retiming truth."""
import argparse, json, subprocess, time, traceback
from pathlib import Path
from collections import Counter
import numpy as np
from dunhuang.io import checked,sha,write
from dunhuang.generation.source_pairs import source_pairs,validate_pair
from dunhuang.generation.event_geometry import event_geometry,_matrix
from dunhuang.generation.event_music_adapter import EventMusicAdapter
from dunhuang.generation.targets import phase_target,digest

SEED=20261115
CONTROLS=('original','zero_adapter','reference_removed','wrong_source',
          'phase_reversed','music_replaced','music_shifted')
WEIGHTS=dict(velocity=.1,rotation=.02,source_reconstruction=.05)


def read(path):return json.loads(Path(path).read_text())


def prepare(assets):
    from dunhuang.data.paired_manifest import inspect
    _,_,gate=inspect(assets)
    if not gate['ready']:raise ValueError('Strict paired assets not ready')
    manifest=read(checked(assets['paired_generation_manifest']))
    index={r['window_uid']:r for r in manifest['rows']}
    pairs=source_pairs(manifest)
    counts=Counter(p['role'] for p in pairs)
    if not counts['train'] or not counts['val']:raise ValueError('Disjoint cross-recording Train/Val tasks required')
    for p in pairs:validate_pair(p,index)
    validation=[index[p['target_uid']] for p in pairs if p['role']=='val']
    if len({r['music_uid'] for r in validation})<2 or len({r['choreography_uid'] for r in validation})<2:
        raise ValueError('Distinct Val music/routines required for registered controls')
    return index,pairs,gate


class TransferRuntime:
    def __init__(self,assets,index,mode):
        import torch
        from dunhuang.research.frozen import denoiser
        self.torch=torch;self.assets=assets;self.index=index;self.cache={}
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.model,self.normalizer,self.diffusion=denoiser(assets)
        with np.load(checked(assets['scaler'])) as s:self.scale=s['scale'].copy();self.bias=s['bias'].copy()
        if (not np.array_equal(self.scale,self.normalizer.scaler.scale_.cpu().numpy()) or
                not np.array_equal(self.bias,self.normalizer.scaler.min_.cpu().numpy())):
            raise ValueError('Training scaler/checkpoint mismatch')
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        self.adapter=EventMusicAdapter(mode).cuda()
        self.before=digest(self.model);self.state={}
        self.handle=self.model.cond_encoder.register_forward_hook(self.inject)

    def inject(self,module,args,music):
        if not self.state.get('enabled',True):return music
        delta,source=self.adapter(self.state['geometry'],music,return_aux=True)
        self.state['reconstruction']=source
        return music+delta

    def data(self,pair):
        if pair['target_uid'] not in self.cache:
            row=self.index[pair['target_uid']]
            native=np.load(checked(row['motion']),allow_pickle=False)
            target,_=phase_target(native,self.scale,self.bias)
            if target is None:raise ValueError('Source task target changed support')
            music=np.load(checked(row['jukebox']),allow_pickle=False)
            self.cache[pair['target_uid']]=(target,music)
        source_key='source:'+pair['source_uid']
        if source_key not in self.cache:
            row=self.index[pair['source_uid']]
            native=np.load(checked(row['motion']),allow_pickle=False)
            available=np.load(checked(row['observed']),allow_pickle=False)
            self.cache[source_key]=(native,available)
        return (*self.cache[pair['target_uid']],*self.cache[source_key])

    def batch(self,pairs,control='original',wrong=None):
        import torch
        from dunhuang.generation.conditions import source_condition
        data=[self.data(p) for p in pairs]
        target=torch.from_numpy(np.stack([r[0] for r in data])).cuda()
        music=torch.from_numpy(np.stack([r[1] for r in data])).cuda()
        source_data=[self.data(p) for p in wrong] if control=='wrong_source' else data
        source=torch.from_numpy(np.stack([r[2] for r in source_data])).cuda()
        available=torch.from_numpy(np.stack([r[3] for r in source_data])).cuda()
        phase=torch.arange(150,device='cuda').float()[None,:,None].expand(len(pairs),-1,-1)/150
        if control=='reference_removed':available=torch.zeros_like(available)
        if control=='phase_reversed':phase=1-phase
        condition=source_condition(source,available,phase,'completion_upper8')
        geometry=event_geometry(condition.value,condition.known,condition.phase)
        if control=='music_replaced':
            candidates=[p for p in self.validation if self.index[p['target_uid']]['music_uid']!=self.index[pairs[0]['target_uid']]['music_uid']]
            music=torch.from_numpy(np.stack([self.data(candidates[i%len(candidates)])[1] for i in range(len(pairs))])).cuda()
        elif control=='music_shifted':music=torch.roll(music,37,1)
        self.state.update(geometry=geometry,enabled=control!='zero_adapter')
        return target,music,geometry

    def loss(self,target,predicted,geometry):
        import torch
        mse=(predicted-target).square().mean()
        velocity=((predicted[:,1:]-predicted[:,:-1])-(target[:,1:]-target[:,:-1])).square().mean()
        scale=torch.as_tensor(self.scale,device='cuda');bias=torch.as_tensor(self.bias,device='cuda')
        # Both rotations use EDGE rows; transposing both preserves geodesic distance.
        pr=_matrix(((predicted-bias)/scale)[:,:,7:].reshape(-1,150,24,6))
        tr=_matrix(((target-bias)/scale)[:,:,7:].reshape(-1,150,24,6))
        cosine=(((pr.transpose(-1,-2)@tr).diagonal(dim1=-2,dim2=-1).sum(-1))-1)/2
        rotation=(1-cosine.clamp(-1,1)).mean()
        observed=(geometry[:,:,144:152]>0).repeat_interleave(6,dim=-1)
        reconstruction=(self.state['reconstruction']-geometry[:,:,:48]).square()[observed].mean()
        total=mse+WEIGHTS['velocity']*velocity+WEIGHTS['rotation']*rotation+WEIGHTS['source_reconstruction']*reconstruction
        return total,dict(mse=float(mse.detach()),velocity=float(velocity.detach()),
                          rotation=float(rotation.detach()),source_reconstruction=float(reconstruction.detach()))

    def evaluate(self,pairs,controls=CONTROLS):
        import torch
        self.adapter.eval();result={}
        for control in controls:
            metrics=[]
            with torch.no_grad(),torch.random.fork_rng(devices=[0]):
                torch.manual_seed(117);torch.cuda.manual_seed_all(117)
                for pair in pairs:
                    row=self.index[pair['target_uid']]
                    wrong=next(p for p in self.validation if self.index[p['source_uid']]['choreography_uid']!=row['choreography_uid'])
                    target,music,geometry=self.batch([pair],control,[wrong])
                    for timestep in (100,500,900):
                        t=torch.tensor([timestep],device='cuda')
                        y=self.model(self.diffusion.q_sample(target,t),music,t,cond_drop_prob=0.)
                        metrics.append(dict(mse=float((y-target).square().mean()),
                            group_mse={name:float((y-target)[:,:,ids].square().mean()) for name,ids in {
                                'contact':list(range(4)),'root':list(range(4,7)),
                                'upper8':list(range(79,127)),'other_rotation':list(range(7,79))+list(range(127,151))}.items()},
                            normalized_velocity_rms=float((y[:,1:]-y[:,:-1]).square().mean().sqrt()),
                            target_normalized_velocity_rms=float((target[:,1:]-target[:,:-1]).square().mean().sqrt())))
            result[control]=dict(mse=float(np.mean([r['mse'] for r in metrics])),
                normalized_velocity_rms=float(np.mean([r['normalized_velocity_rms'] for r in metrics])),
                target_normalized_velocity_rms=float(np.mean([r['target_normalized_velocity_rms'] for r in metrics])),
                group_mse={k:float(np.mean([r['group_mse'][k] for r in metrics])) for k in metrics[0]['group_mse']})
        return result

    def close(self):
        self.handle.remove()
        if digest(self.model)!=self.before or any(p.grad is not None for p in self.model.parameters()):
            raise ValueError('Frozen EDGE weights/gradients changed')


def run(assets,destination,mode='cross',steps=1200,preflight_only=False):
    import torch
    if mode not in ('cross','additive') or type(steps) is not int or not 1<=steps<=1200:
        raise ValueError('Registered fusion mode and bounded steps1..1200 required')
    out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:
        index,pairs,gate=prepare(assets)
        train=[p for p in pairs if p['role']=='train'];val=[p for p in pairs if p['role']=='val']
        selected=[val[i] for i in np.linspace(0,len(val)-1,8).astype(int)]
        protocol=dict(schema='cross_recording_event_training_v1',mode=mode,steps=steps,seed=SEED,
            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            paired_manifest=assets['paired_generation_manifest'],checkpoint=assets['checkpoint'],scaler=assets['scaler'],
            preflight=gate,source_tasks=len(pairs),train_tasks=len(train),validation_tasks=len(val),
            validation_pairs=selected,selection='lowest full151 fixed8 Val MSE each100/final; no Test fitting',
            batch=2,learning_rate=1e-4,gradient_clip=1,loss_weights=WEIGHTS,
            source_condition='independent capture of shared official basic routine/song/time; not pose-exact labels',
            geometry='native local upper8 SO(3) absolute/relative/adjacent + observed mask + source phase; no reference Scaler clipping',
            full_target_supervision=True,music_qualified=False,full_motion_qualified=False)
        write(out/'protocol.json',protocol);write(out/'source_pairs.json',pairs)
        if preflight_only:
            write(out/'status.json',dict(status='SOURCE_TASK_PREFLIGHT_READY',training_steps=0,music_qualified=False,full_motion_qualified=False));return
        rt=TransferRuntime(assets,index,mode);rt.validation=val
        write(out/'initialization.json',dict(parameter_count=sum(p.numel() for p in rt.adapter.parameters()),
            initial_adapter_digest=digest(rt.adapter),frozen_base_digest=rt.before))
        optimizer=torch.optim.AdamW(rt.adapter.parameters(),lr=1e-4);rng=np.random.default_rng(SEED)
        best=float('inf');history=[];initial=rt.evaluate(selected,('zero_adapter',))
        try:
            for step in range(1,steps+1):
                batch=[train[i] for i in rng.integers(0,len(train),2)]
                target,music,geometry=rt.batch(batch)
                rt.adapter.train()
                t=torch.randint(50,950,(len(batch),),device='cuda')
                y=rt.model(rt.diffusion.q_sample(target,t),music,t,cond_drop_prob=0.)
                loss,components=rt.loss(target,y,geometry)
                if not torch.isfinite(loss):raise ValueError('Nonfinite event transfer loss')
                optimizer.zero_grad();loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(rt.adapter.parameters(),1.)
                if not torch.isfinite(norm):raise ValueError('Nonfinite event transfer gradients')
                optimizer.step()
                if step%100==0 or step==steps:
                    val_result=rt.evaluate(selected,('original',))
                    score=val_result['original']['mse']
                    history.append(dict(step=step,training_components=components,gradient_norm=float(norm),
                        validation=val_result['original'],seconds=time.monotonic()-start))
                    write(out/'history.json',history)
                    print('EVENT_TRANSFER',mode,step,score,flush=True)
                    if score<best:
                        best=score
                        torch.save(dict(state=rt.adapter.state_dict(),mode=mode,
                            condition_schema='native_event_music_v1',base_sha256=assets['checkpoint']['sha256'],
                            condition_policy='completion_upper8',
                            condition_scope='official_cross_recording_event_conditioned_denoising',
                            music_source='real_aligned_jukebox_layer66_30hz',
                            target_channels='fitted147_plus_derived_contact4',
                            training_commit=protocol['code_commit'],protocol_sha256=sha(out/'protocol.json'),
                            selected_step=step,music_qualified=False,full_motion_qualified=False),out/'adapter.pt')
            saved=torch.load(out/'adapter.pt',map_location='cpu',weights_only=True)
            rt.adapter.load_state_dict(saved['state'])
            controls=rt.evaluate(selected)
            write(out/'validation.json',dict(controls=controls,initial_zero_adapter=initial,
                validation_pairs=selected,scope='cross-recording full151 denoising; not free-running/Native qualification'))
        finally:
            rt.close()
        write(out/'status.json',dict(status='CROSS_RECORDING_TRAINED_UNQUALIFIED',mode=mode,training_steps=steps,
            selected_step=saved['selected_step'],weights_before=rt.before,weights_after=digest(rt.model),
            checkpoint_sha256=sha(out/'adapter.pt'),protocol_sha256=sha(out/'protocol.json'),
            seconds=time.monotonic()-start,music_qualified=False,full_motion_qualified=False))
    except Exception:
        write(out/'status.json',dict(status='FAILED',error=traceback.format_exc(),music_qualified=False,full_motion_qualified=False))
        raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--mode',choices=['cross','additive'],default='cross');ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--preflight-only',action='store_true')
    a=ap.parse_args();run(read(a.assets),a.out,a.mode,a.steps,a.preflight_only)

if __name__=='__main__':main()
