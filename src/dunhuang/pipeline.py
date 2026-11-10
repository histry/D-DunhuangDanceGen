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
    import soundfile as sf
    audio=checked(q['audio']);info=sf.info(audio)
    clock=clock_plan(info.frames,info.samplerate)
    frames=q.get('frames',clock['frames'])
    if type(frames) is not int or not 1<=frames<=clock['frames']:raise ValueError('Requested duration exceeds audio')
    q=dict(q,frames=frames,windows=max(1,(max(0,frames-150)+74)//75+1))
    out=Path(q['output']).resolve();out.mkdir(parents=True,exist_ok=False);q['output']=str(out)
    request=out/'request.json';write(request,q);write(out/'clock.json',clock)
    env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]);env.pop('PYTHONHOME',None)
    try:
        for python,module in [(q['edge_python'],'dunhuang.edge_backend'),(q['native_python'],'dunhuang.physical')]:
            with (out/(module.split('.')[-1]+'.log')).open('w') as log:
                subprocess.run([python,'-m',module,str(request)],env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
        gate=json.loads((out/'physical.json').read_text())['absolute']
        write(out/'status.json',dict(status='COMPLETED',physical_ok=gate['ok'],full_generation_qualified=False))
    except Exception:
        write(out/'status.json',dict(status='FAILED',error=traceback.format_exc(),full_generation_qualified=False))
        raise
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
    print(run(json.loads(Path(args.config).read_text())),flush=True)
if __name__=='__main__':main()
