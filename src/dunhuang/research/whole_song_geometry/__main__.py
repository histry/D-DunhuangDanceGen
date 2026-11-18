def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','verify']);p.add_argument('--out',required=True);p.add_argument('--source',default='outputs/whole_song_geometry_02/main');a=p.parse_args()
 from .study import run,verify
 if a.mode=='run':run(a.out)
 else:verify(a.out,a.source)
if __name__=='__main__':main()
