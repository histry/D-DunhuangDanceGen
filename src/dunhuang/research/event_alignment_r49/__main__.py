import argparse
from . import rank, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("primary","controls","val"))
parser.add_argument("--out", required=True)
args = parser.parse_args()
if args.mode == "val":
    evaluate.run(args.out)
else:
    rank.run(args.out, controls=args.mode == "controls")
