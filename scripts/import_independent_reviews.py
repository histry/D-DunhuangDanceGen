"""Private administrator import: preserve originals; never infer missing human votes."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import checked,write,sha
from dunhuang.research.source_expression.common import read,fresh
from dunhuang.research.source_expression.review_contract import summarize
from dunhuang.research.source_expression.review_statistics import statistics
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--protocol",required=True);p.add_argument("--roster",required=True);p.add_argument("--records",nargs="+",required=True);p.add_argument("--adjudications");p.add_argument("--out",required=True);a=p.parse_args()
    protocol=read(a.protocol);bindings=read(checked(protocol["bindings"]));roster=read(a.roster)
    if roster.get("coverage_attested") is not True:raise ValueError("Private administrator must attest evidence reconstruction, sealed selection and identity/role coverage")
    for b in bindings:
        for k in ("raw","native","media"):
            checked(b[k])
        if b.get("audio"):checked(b["audio"])
    records=[r for f in a.records for r in read(f)]
    summary,decisions,accepted=summarize(bindings,records,roster["reviewers"],sha(a.roster),read(a.adjudications) if a.adjudications else ())
    out=fresh(a.out);write(out/"original_records.json",records);write(out/"bound_records.json",accepted);write(out/"decisions.json",decisions);write(out/"summary.json",summary);write(out/"statistics.json",statistics(bindings,accepted,decisions))
    write(out/"receipt.json",dict(protocol_sha256=sha(a.protocol),roster_sha256=sha(a.roster),original_files={str(i):sha(f) for i,f in enumerate(a.records)},adjudications_sha256=sha(a.adjudications) if a.adjudications else None,human_reviewed=summary["data_review_verified"],operational_qualified=False))
