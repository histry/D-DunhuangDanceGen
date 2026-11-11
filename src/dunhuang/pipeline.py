"""The single whole-song entry. Neural and Native namespaces stay in subprocesses."""
import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from .io import checked, write
from .music import clock_plan

def run(q):
    if q.get('learned_retrieval'):
        from .alignment.rank import authorize
        request=q['learned_retrieval']
        decision=authorize(checked(request['checkpoint']),checked(request['qualification']))
        q=dict(q,retrieval_decision=decision)
        if decision['enabled']:
            selection=json.loads(checked(request['selection']).read_text())
            if selection['checkpoint_sha256']!=decision['checkpoint_sha256']:
                raise ValueError('Selection checkpoint differs')
            q['anchors']=selection['anchors']
    import soundfile as sf
    audio=checked(q['audio']);info=sf.info(audio)
    clock=clock_plan(info.frames,info.samplerate)
    if 'feature_receipt' in q:
        receipt=json.loads(checked(q['feature_receipt']).read_text())
        binding=receipt.get('original_audio',receipt.get('audio'))
        if binding['sha256']!=q['audio']['sha256'] or receipt['clock']!=clock or receipt['layer']!=66:
            raise ValueError('Cached features are not bound to this audio clock')
        import numpy as np, hashlib
        features=np.load(checked(q['features']),allow_pickle=False)
        if features.shape!=(clock['windows'],150,4800):raise ValueError('Feature clock mismatch')
        if 'slices' not in receipt and receipt.get('features',{}).get('sha256')!=q['features']['sha256']:
            raise ValueError('Extracted features differ from their producer receipt')
        if 'slices' in receipt:
            if len(receipt['slices'])!=len(features):raise ValueError('Incomplete feature receipt')
            for index,(row,record) in enumerate(zip(features,receipt['slices'])):
                if record['start_frame']!=index*75 or hashlib.sha256(row.tobytes()).hexdigest()!=record['features_sha256']:
                    raise ValueError('Feature window provenance mismatch')
    else:
        raise ValueError('Hashed feature extraction receipt required')
    frames=q.get('frames',clock['frames'])
    if type(frames) is not int or not 1<=frames<=clock['frames']:raise ValueError('Requested duration exceeds audio')
    q=dict(q,code_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[2]),'rev-parse','HEAD'],text=True).strip(),frames=frames,windows=max(1,(max(0,frames-150)+74)//75+1))
    out=Path(q['output']).resolve();out.mkdir(parents=True,exist_ok=False);q['output']=str(out)
    request=out/'request.json';write(request,q);write(out/'clock.json',clock)
    env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]);env.pop('PYTHONHOME',None)
    try:
        if q.get('profile')=='dunhuang':
            with (out/'planning.log').open('w') as log:
                subprocess.run([q['native_python'],'-m','dunhuang.planning.worker',str(request)],env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
            from .planning.event_plan import validate,anchors
            from .io import sha
            plan=validate(json.loads((out/'event_plan.json').read_text()),q['audio']['sha256'],frames)
            q['anchors']=anchors(plan)
            q['event_plan']=dict(path=str(out/'event_plan.json'),sha256=sha(out/'event_plan.json'))
            q['mode']='event_guided' if q.get('generator')=='adapter' else 'temporal';q['trajectory_radius']=12;q['trajectory_strength']=.55
            q['generation_stage']='v1_event_adapter_research' if q['mode']=='event_guided' else 'v0.5_event_plan_temporal'
            write(request,q)
        for python,module in [(q['edge_python'],'dunhuang.edge_backend'),(q['native_python'],'dunhuang.physical')]:
            with (out/(module.split('.')[-1]+'.log')).open('w') as log:
                subprocess.run([python,'-m',module,str(request)],env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
        gate=json.loads((out/'physical.json').read_text())['absolute']
        write(out/'status.json',dict(status='COMPLETED',physical_ok=gate['ok'],full_generation_qualified=False))
    except Exception:
        preflight=out/'normalization_preflight.json'
        rejected=preflight.exists() and not json.loads(preflight.read_text())['supported']
        write(out/'status.json',dict(status='REJECTED_UNSUPPORTED_ANCHORS' if rejected else 'FAILED',error=traceback.format_exc(),full_generation_qualified=False))
        raise
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True)
    ap.add_argument('--assets');ap.add_argument('--audio');ap.add_argument('--out');ap.add_argument('--frames',type=int);ap.add_argument('--plan')
    args=ap.parse_args()
    q=json.loads(Path(args.config).read_text())
    if args.assets:
        q=dict(json.loads(Path(args.assets).read_text()),**q)
    if args.audio:
        from .io import sha
        q['audio']=dict(path=str(Path(args.audio).resolve()),sha256=sha(args.audio))
    if args.plan:
        from .io import sha
        q['input_event_plan']=dict(path=str(Path(args.plan).resolve()),sha256=sha(args.plan))
    if args.out:q['output']=args.out
    if args.frames:q['frames']=args.frames
    if q.get('profile')=='edge_baseline':
        q['mode']='baseline';q['anchors']=[]
    print(run(q),flush=True)
if __name__=='__main__':main()
