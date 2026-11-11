import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.generation.train import *
if __name__=='__main__':
    import runpy
    runpy.run_module('dunhuang.generation.train',run_name='__main__')
