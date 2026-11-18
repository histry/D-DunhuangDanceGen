import argparse
from . import planning

p = argparse.ArgumentParser()
p.add_argument("mode",choices=["primary","audit","verify","release","recheck"])
p.add_argument("--out")
a = p.parse_args()
if a.mode=="primary":
    planning.run(a.out)
elif a.mode in ("release","recheck"):
    from . import publication
    publication.run() if a.mode=="release" else publication.recheck()
else:
    from . import audit
    getattr(audit,a.mode)(a.out)
