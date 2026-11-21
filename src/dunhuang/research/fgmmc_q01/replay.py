"""Fresh-process inference replay with frozen final checkpoints, zero optimizer calls."""
import numpy as np
import torch
from .common import *
from .data import register,Store,windows,statistics,donors
from .evaluate import run as evaluate
from .preview import run as preview

def run():
 def forbidden(*a,**k):raise RuntimeError('Replay cannot create optimizer')
 torch.optim.AdamW=forbidden;torch.optim.Adam=forbidden;torch.optim.SGD=forbidden
 verify(ROOT/'data');reg=register();assert reg==read(ROOT/'data/registration.json');rows=windows(reg);assert rows==read(ROOT/'data/windows.json');store=Store(reg);normal=statistics(store,rows);old=np.load(ROOT/'data/normalizer.npz');assert all(np.array_equal(normal[k],old[k]) for k in normal);assert donors(store,rows)==read(ROOT/'data/donors.json');evaluate(ROOT/'controls_replay');manifest=verify(ROOT/'controls');diff=[n for n,h in manifest['files'].items() if sha(ROOT/'controls_replay'/n)!=h];assert not diff,diff
 preview(ROOT/'view_replay',CFG['models']);view_diff=[n for n in ['selections.json','viewer/index.html','viewer/protocol.json'] if sha(ROOT/'view_replay'/n)!=sha(ROOT/'controls_view'/n)];assert not view_diff,view_diff
 initials=[read(ROOT/n/'training.json')['initial_digest'] for n in CFG['models']];assert len(set(initials))==1
 write(ROOT/'receipts/independent.json',dict(status='PASS',fresh_process=True,raw_features_reconstructed=True,new_train_normalization_exact=True,donors_exact=True,scalar_differences=diff,view_differences=view_diff,initial_model_digest_same_all_arms=True,checkpoint_sha256={n:sha(ROOT/n/'weights.npz') for n in CFG['models']},optimizer_calls=0,retraining_performed=False,scope='INDEPENDENT_FROZEN_CHECKPOINT_INFERENCE_NOT_TRAINING_REPRODUCTION',can_route=False));print('FG_REPLAY_PASS',flush=True)
if __name__=='__main__':run()
