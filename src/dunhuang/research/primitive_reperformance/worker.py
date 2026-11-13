"""Isolated reference-control worker; no change to the frozen backend defaults."""
import sys
from pathlib import Path
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read
def main():
    q=read(sys.argv[1])
    if q.get("research_reperformance_v1") is not True or q["mode"]!="temporal" or q.get("adapter"):raise ValueError("Explicit frozen-EDGE reperformance request required")
    checked(q["event_plan"])
    import dunhuang.temporal
    from .guide import prepare
    dunhuang.temporal.prepare=prepare
    from dunhuang.edge_backend import main as backend
    backend()
    file=Path(q["output"])/"generation.json";g=read(file)
    g["research_reperformance"]=dict(group=q["condition_group"],complete_E1W_reference=True,neural_weight_training=False,source_root_translation=False,source_contacts=False)
    write(file,g)
if __name__=="__main__":main()
