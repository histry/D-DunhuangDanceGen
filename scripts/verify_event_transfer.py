"""Fresh-process neural validation replay, no optimizer and no source rewrite."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.io import checked,sha,write
from dunhuang.research.legacy.event_transfer import prepare,TransferRuntime,read,CONTROLS
from dunhuang.generation.event_music_adapter import authorize_event_adapter
from dunhuang.generation.targets import digest


def main():
    import torch
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--training',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    assets=read(a.assets);root=Path(a.training);status=read(root/'status.json');protocol=read(root/'protocol.json')
    if status['status']!='CROSS_RECORDING_TRAINED_UNQUALIFIED':raise ValueError('Completed cross-recording training required')
    if sha(root/'protocol.json')!=status['protocol_sha256'] or sha(root/'adapter.pt')!=status['checkpoint_sha256']:
        raise ValueError('Training artifact identity changed')
    if protocol['paired_manifest']!=assets['paired_generation_manifest'] or protocol['checkpoint']!=assets['checkpoint']:
        raise ValueError('Replay scientific inputs differ')
    history=read(root/'history.json')
    expected=list(range(100,protocol['steps']+1,100))
    if not expected or expected[-1]!=protocol['steps']:expected.append(protocol['steps'])
    if [r['step'] for r in history]!=expected or status['training_steps']!=protocol['steps']:
        raise ValueError('Incomplete registered training history')
    selected=min(history,key=lambda r:r['validation']['mse'])['step']
    if selected!=status['selected_step']:raise ValueError('Checkpoint selection rule changed')
    index,pairs,gate=prepare(assets)
    if pairs!=read(root/'source_pairs.json'):raise ValueError('Source correspondence changed')
    rt=TransferRuntime(assets,index,protocol['mode']);rt.validation=[p for p in pairs if p['role']=='val']
    payload=torch.load(root/'adapter.pt',map_location='cpu',weights_only=True)
    authorize_event_adapter(payload,assets['checkpoint']['sha256'],True)
    rt.adapter.load_state_dict(payload['state']);before=digest(rt.adapter)
    try:controls=rt.evaluate(protocol['validation_pairs'])
    finally:rt.close()
    saved=read(root/'validation.json')['controls']
    differences={k:abs(controls[k]['mse']-saved[k]['mse']) for k in CONTROLS}
    if max(differences.values())>1e-7:raise ValueError('Saved neural controls did not replay')
    if digest(rt.adapter)!=before or rt.before!=status['weights_before'] or rt.before!=status['weights_after']:
        raise ValueError('Read-only replay changed parameters')
    write(out,dict(status='READ_ONLY_EVENT_TRANSFER_REPLAY_VERIFIED',mode=protocol['mode'],
        controls=controls,saved_controls=saved,max_abs_difference=max(differences.values()),
        exact=all(v==0 for v in differences.values()),optimizer_calls=0,new_weight_updates=0,
        base_before=rt.before,base_after=digest(rt.model),adapter_before=before,adapter_after=digest(rt.adapter),
        checkpoint_sha256=sha(root/'adapter.pt'),protocol_sha256=sha(root/'protocol.json'),
        validation_pairs=protocol['validation_pairs'],music_qualified=False,full_motion_qualified=False))
    print('EVENT_TRANSFER_REPLAY_VERIFIED',protocol['mode'],max(differences.values()),flush=True)

if __name__=='__main__':main()
