import sys
from dunhuang.research.source_capacity.common import read
def main():
    q=read(sys.argv[1])
    if q.get("research_r9") is not True or q.get("adapter"):raise ValueError("Explicit R9 frozen model request required")
    import dunhuang.temporal
    from .guide import prepare
    dunhuang.temporal.prepare=prepare
    from dunhuang.edge_backend import main as backend
    backend()
if __name__=="__main__":main()
