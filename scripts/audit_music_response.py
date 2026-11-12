"""Read-only attribution of mismatch sensitivity to frozen EDGE vs Adapter."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.research.legacy.event_transfer import prepare,TransferRuntime,read
from dunhuang.generation.event_music_adapter import authorize_event_adapter
from dunhuang.generation.targets import digest
from dunhuang.io import sha,write


def main():
    import torch
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--training',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    assets=read(a.assets);folder=Path(a.training);protocol=read(folder/'protocol.json');status=read(folder/'status.json')
    if status['checkpoint_sha256']!=sha(folder/'adapter.pt'):raise ValueError('Checkpoint changed')
    index,pairs,gate=prepare(assets)
    rt=TransferRuntime(assets,index,protocol['mode']);rt.validation=[p for p in pairs if p['role']=='val']
    payload=torch.load(folder/'adapter.pt',map_location='cpu',weights_only=True)
    authorize_event_adapter(payload,assets['checkpoint']['sha256'],True)
    rt.adapter.load_state_dict(payload['state']);adapter_before=digest(rt.adapter)
    # All noises/targets/pairs are matched, only injection and music are varied.
    rt.handle.remove();results={}
    try:
        for enabled in (False,True):
            hook=rt.model.cond_encoder.register_forward_hook(
                lambda module,args,value:rt.inject(module,args,value) if enabled else value)
            try:
                results['adapter' if enabled else 'frozen_edge']=rt.evaluate(
                    protocol['validation_pairs'],('original','music_replaced','music_shifted'))
            finally:hook.remove()
    finally:rt.close()
    if digest(rt.adapter)!=adapter_before:raise ValueError('Attribution audit changed Adapter')
    saved=read(folder/'validation.json')['controls']
    if any(abs(results['adapter'][k]['mse']-saved[k]['mse'])>1e-7 for k in results['adapter']):
        raise ValueError('Attribution does not replay saved controls')
    penalties={name:{control:v[control]['mse']-v['original']['mse']
               for control in ('music_replaced','music_shifted')} for name,v in results.items()}
    write(out,dict(status='READ_ONLY_MUSIC_ATTRIBUTION_COMPLETED',mode=protocol['mode'],
        controls=results,mismatch_penalties=penalties,
        adapter_minus_base_mismatch_penalty={k:penalties['adapter'][k]-penalties['frozen_edge'][k] for k in penalties['adapter']},
        optimizer_calls=0,new_weight_updates=0,adapter_before=adapter_before,adapter_after=digest(rt.adapter),
        base_before=rt.before,base_after=digest(rt.model),
        scope='matched target-noised completion sensitivity; inherited frozen music response is not evidence of new music-event learning',
        music_qualified=False,full_motion_qualified=False))
    print('MUSIC_ATTRIBUTION_COMPLETED',protocol['mode'],penalties,flush=True)

if __name__=='__main__':main()
