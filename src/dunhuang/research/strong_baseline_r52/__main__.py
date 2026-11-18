import argparse
from .common import ROOT,read
p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','prepare','train','generate','music-controls','verify','release','recheck']);p.add_argument('--out',required=True);p.add_argument('--checkpoint');p.add_argument('--controls',action='store_true');p.add_argument('--no-phase',action='store_true');p.add_argument('--replay-from');a=p.parse_args()
if a.controls and not read(ROOT/'receipts/trained_browser.json')['passed']:raise ValueError('Deliver actual main effect before controls')
if a.mode=='train' and not a.replay_from and not read(ROOT/('receipts/trained_browser.json' if a.no_phase else 'receipts/main_browser.json'))['passed']:raise ValueError('Inspect delivered actual generation before training/comparison')
if a.mode=='music-controls' and not read(ROOT/'receipts/main_browser.json')['passed']:raise ValueError('Inspect frozen main effect before acoustic controls')
if a.mode=='plan':
 from .planning import run
 run(a.out,a.controls)
elif a.mode=='music-controls':
 from .music_controls import run
 run(a.out)
elif a.mode=='prepare':
 from .prepare import run
 run(a.out)
elif a.mode=='train':
 from .train import run
 run(a.out,not a.no_phase,a.replay_from)
elif a.mode=='generate':
 from .generate import run
 run(a.out,a.checkpoint,a.controls,not a.no_phase)
else:
 from . import publication
 getattr(publication,a.mode)(a.out)
