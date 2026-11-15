import importlib.util
from pathlib import Path
import pytest

def collector():
    spec=importlib.util.spec_from_file_location("r21_collector",Path("scripts/collect_r21.py"))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_publishing_never_overwrites_partial_reports_before_reading_inputs(tmp_path):
    m=collector();p=tmp_path/"published";p.mkdir();old=p/"r21_summary.json";old.write_text("preserve original",encoding="utf8")
    with pytest.raises(AssertionError,match="fresh"):m.run(tmp_path/"not_a_study",p)
    assert old.read_text(encoding="utf8")=="preserve original"

def test_q_summary_keeps_point_count_old_times_absolute_errors_and_branch_identity():
    m=collector();rows=[]
    for t,value in zip((999,519,119,19),(1.,2.,3.,4.)):
        rows.append(dict(case=7,variant=2,t=t,unknown=value,
            branches={"unconditional_music":dict(unknown=value*2),"conditional_music":dict(unknown=value/2)},
            controls={n:value*3 for n in ("left_wrong","right_wrong","both_absent")}))
    result=m.q_summary(rows)[0]
    assert result["grid_points"]==4 and result["grid_mean"]==2.5 and result["old4_mean"]==2.5 and result["t19"]==4.
    assert result["branch_old4"]==dict(unconditional_music=5.,conditional_music=1.25)
