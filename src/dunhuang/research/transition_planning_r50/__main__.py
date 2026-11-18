import argparse
from . import planning

p = argparse.ArgumentParser()
p.add_argument("mode",choices=["primary","audit","verify"])
p.add_argument("--out",required=True)
a = p.parse_args()
if a.mode=="primary":
    planning.run(a.out)
else:
    from . import audit
    getattr(audit,a.mode)(a.out)
