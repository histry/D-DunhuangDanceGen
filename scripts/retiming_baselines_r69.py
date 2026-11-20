"""R69 frozen-source timing baselines, no training or production upgrade."""
import argparse
from dunhuang.research.retiming_baselines_r69 import study
from dunhuang.research.retiming_baselines_r69.common import read,write
from dunhuang.research.retiming_baselines_r69.reviews import load_documents,summarize_reviews
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 run=sub.add_parser('run');run.add_argument('--out')
 verify=sub.add_parser('verify');verify.add_argument('--out',required=True)
 review=sub.add_parser('review');review.add_argument('--admin',required=True);review.add_argument('--files',nargs='*',default=[]);review.add_argument('--roster');review.add_argument('--out',required=True)
 args=p.parse_args()
 if args.command=='run':study.run(args.out)
 elif args.command=='verify':study.verify(args.out)
 else:
  if Path(args.out).exists():raise ValueError('Fresh review analysis required; raw records are never overwritten')
  admin=read(args.admin);documents=load_documents(args.files,admin)
  write(args.out,summarize_reviews(admin,documents,read(args.roster) if args.roster else None))
if __name__=='__main__':main()
