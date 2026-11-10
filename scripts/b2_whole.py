"""Continue from completed 5s and 15s receipts without overwriting either run."""
import sys
from pathlib import Path
from b2 import ROOT, compare, pinned, read

previous=Path(sys.argv[1]).resolve()
if not read(previous/'5s_comparison.json')['advance']:
    raise ValueError('Five-second comparison did not qualify for extension')
read(previous/'15s_comparison.json')
out=Path(sys.argv[2]).resolve()
out.mkdir(parents=True,exist_ok=False)
q=read(ROOT/'outputs/supported_upper8/whole/request.json')
sys.path.insert(0,str(pinned(q['legacy_root'],q['legacy_commit'])))
from data_pipeline.prepare_domains import _baseline_environment
_baseline_environment()
compare(out,'whole',q,read(previous/'saved_diagnostics.json')['whole_hard']['motion_sha256'])
