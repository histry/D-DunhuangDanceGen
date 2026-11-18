import argparse
from . import rank, evaluate, qualify, release

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("primary","controls","val","qualify","verify","release","recheck"))
parser.add_argument("--out")
args = parser.parse_args()
if args.mode == "val":
    evaluate.run(args.out)
elif args.mode == "qualify":
    qualify.run(args.out)
elif args.mode in ("verify","release","recheck"):
    getattr(release, args.mode)()
else:
    rank.run(args.out, controls=args.mode == "controls")
