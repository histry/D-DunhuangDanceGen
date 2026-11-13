from pathlib import Path
from dunhuang.io import sha,checked
from dunhuang.research.source_capacity.common import read,commit,fresh,frozen,weight,same_arrays,forbid_optimizers
def prior(config):
    root=Path(config["source_expression"]);public=read("reports/source_expression_summary.json");receipt=read("reports/source_expression_receipt.json")
    if receipt["exit_status"]!=0 or sha("reports/source_expression_summary.json")!=receipt["summary_sha256"]:raise ValueError("R6 successful receipt differs")
    for stage in ("prepared","diagnosis","probe","music","review"):
        for name in ("protocol.json","result.json"):
            if sha(root/stage/name)!=public["evidence"][stage+"/"+name]:raise ValueError("Frozen R6 phase differs")
    if sha(root/"probe/arrays.npz")!=public["private_arrays"]["probe/arrays.npz"]:raise ValueError("R6 candidate arrays differ")
    r6=read(config["frozen_config"]);prep=read(root/"prepared/protocol.json")
    if prep["config"]!=r6 or config["times"]!=r6["times"] or config["arms"]!=["A0","A1","A2","A3","A4"] or config["cohorts"]!=["calibration","disjoint_probe"]:raise ValueError("Registered populations and budgets differ")
    if config["training_allowed"] or config["whole_generation"]["learned_adapter_enabled"] or config["whole_generation"]["paired_R7_guidance_enabled"] or config["independent_population"] is not None:raise ValueError("Unregistered next-stage request")
    return root,r6,prep
