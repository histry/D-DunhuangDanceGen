import argparse
from .common import ROOT
p=argparse.ArgumentParser();p.add_argument('mode',choices=['plans','sample','paired','verify','release']);p.add_argument('--out',required=True);p.add_argument('--controls',action='store_true');a=p.parse_args()
if a.controls:
    from dunhuang.research.complete_r36.core import read
    if not read(ROOT/'browser_receipt.json')['passed']:raise ValueError('Watch actual primary output before controls')
if a.mode=='plans':
    from .funnel import run
    run(a.out,a.controls)
elif a.mode=='sample':
    from .pilot import run
    run(a.out,a.controls)
elif a.mode=='paired':
    from .paired import run
    run(a.out)
else:
    from . import publication
    getattr(publication,a.mode)(a.out)
