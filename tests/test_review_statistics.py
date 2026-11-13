from dunhuang.research.source_expression.review_statistics import statistics
def test_review_accounting_keeps_unsure_missing_and_source_units():
    bindings=[dict(blind_item_id="a",source_uid="s1",task_type="POSE",music_identity=None),dict(blind_item_id="b",source_uid="s2",task_type="EVENT",music_identity=None)]
    records=[dict(blind_item_id="a",reviewer_id="r1",decision="UNSURE",reason_code="VISIBILITY_INSUFFICIENT"),dict(blind_item_id="a",reviewer_id="r2",decision="VALID")]
    decisions=[dict(blind_item_id="a",state="PARTIAL_REVIEW"),dict(blind_item_id="b",state="UNREVIEWED")]
    s=statistics(bindings,records,decisions)
    assert s["missing_original_reviews"]==2 and len(s["by_source"])==2
    assert s["by_reviewer"]["r1"]["UNSURE"]==1
    assert s["pairwise_agreement"][0]["agreement"]==0
    assert not s["model_performance_qualification"]
