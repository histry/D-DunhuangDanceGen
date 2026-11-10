"""Extract new WAV Jukebox features using the pinned official environment."""
import argparse
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.io import pinned,sha,write
from dunhuang.music import clock_plan
ap=argparse.ArgumentParser();ap.add_argument('--edge-root',required=True);ap.add_argument('--audio',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
root=pinned(a.edge_root,'17c3428669ed6733edd9d8c66f7dc62060b8e46d');sys.path.insert(0,str(root))
from data.audio_extraction.jukebox_features import extract
x,sr=sf.read(a.audio,dtype='float32',always_2d=True);clock=clock_plan(len(x),sr)
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
x=np.pad(x.mean(axis=1),(0,clock['silence_samples']));features=[]
for i in range(clock['windows']):
    begin=round(i*2.5*sr);file=out/('slice_%03d.wav'%i);sf.write(file,x[begin:begin+5*sr],sr,subtype='FLOAT')
    value,_=extract(str(file),skip_completed=False,dest_dir=str(out/'cache'))
    value=np.asarray(value,np.float32)
    if value.shape!=(150,4800) or not np.isfinite(value).all() or not value.any():raise ValueError('Invalid extracted Jukebox features')
    features.append(value)
np.save(out/'features.npy',np.stack(features));write(out/'receipt.json',dict(clock=clock,layer=66,fps=30,audio=dict(path=str(Path(a.audio).resolve()),sha256=sha(a.audio)),features=dict(path=str((out/'features.npy').resolve()),sha256=sha(out/'features.npy'))))
