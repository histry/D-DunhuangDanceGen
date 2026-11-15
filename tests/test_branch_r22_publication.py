import importlib.util
from pathlib import Path
import pytest

def module():
    spec=importlib.util.spec_from_file_location("r22collector",Path("scripts/collect_r22.py"))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_partial_publication_is_never_deleted_or_overwritten(tmp_path):
    m=module();out=tmp_path/"published";out.mkdir();old=out/"r22_summary.json";old.write_text("preserve")
    with pytest.raises(AssertionError,match="Preserve"):m.run(tmp_path/"missing",out,tmp_path/"missing_receipt")
    assert old.read_text()=="preserve"

def test_q_summary_preserves_raw_clipped_branch_scope_and_absolute_errors():
    rows=[]
    for t,v in zip((999,519,119,19),(1.,2.,3.,4.)):
        rows.append(dict(case=24,variant=2,t=t,unknown=v,branches={name:{stage:dict(unknown=v*factor) for stage,factor in (("raw",3),("clipped",2))} for name in ("conditional_music","unconditional_music")}))
    r=module().qsummary(rows)[0]
    assert r["points"]==4 and r["grid_mean"]==2.5 and r["t19"]==4.
    assert r["branches"]["conditional_music"]==dict(raw=7.5,clipped=5.)
