import argparse
from .common import ROOT
p=argparse.ArgumentParser();p.add_argument('stage',choices=['data','main','controls','verify','verify_data','diagnostics','render']);p.add_argument('--out',required=True);p.add_argument('--source');a=p.parse_args()
if a.stage=='data':
 from .data import prepare
 prepare(a.out)
elif a.stage=='render':
 from .view import render
 render(a.source,a.out)
elif a.stage=='verify_data':
 from .data import verify
 verify(a.out,a.source)
elif a.stage=='diagnostics':
 from .diagnostics import run
 run(a.out)
elif a.stage=='verify':
 from .scoring import verify
 verify(a.out,a.source)
else:
 from .scoring import run
 run(a.out,a.stage)
