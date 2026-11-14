"""Mechanism gate, separate from full Native and culture/style qualification."""
def qualification(records,cfg,jerk_limit):
    by={(r["case"],r["arm"]):r for r in records};cases=[]
    if len(by)!=len(records) or len(by)!=cfg["cases"]*len(cfg["arms"]):raise ValueError("Complete fixed population required")
    for i in range(cfg["cases"]):
        if any((i,a) not in by for a in cfg["arms"]):raise ValueError("Missing fixed case")
        r=by[i,"C1_correct"];base=by[i,"C0"];g=cfg["gate"]
        source_wins=all(r["event_pose_error_rad"]<by[i,a]["event_pose_error_rad"] for a in ("C1_removed","C1_wrong_event"))
        context_win=r["edge_root_rmse_m"]<by[i,"C1_wrong_context"]["edge_root_rmse_m"]
        inner=r["center_jerk_max_mps3"]<=base["center_jerk_max_mps3"]*(1-g["center_jerk_gain"])
        external=all(s["root_step_m"]<=g["external_root_step_m"] and s["foot_step_max_m"]<=g["external_foot_step_m"] and s["crossing_jerk_max_mps3"]<=jerk_limit for s in r["external_joins"])
        source=r["common_generated_span_pose_rad"]<=base["common_generated_span_pose_rad"]+g["source_nonregression_rad"]
        activity=g["activity_ratio"][0]<=r["activity_ratio"]<=g["activity_ratio"][1]
        checks=dict(source_condition_advantage=source_wins,context_condition_advantage=context_win,center_continuity=inner,
            external_continuity=external,common_span_source_protected=source,activity_protected=activity,edge_root_compatible=r["edge_root_rmse_m"]<=g["edge_root_rmse_m"])
        cases.append(dict(case=i,checks=checks,qualified=all(checks.values())))
    passed=sum(c["qualified"] for c in cases)>=cfg["gate"]["minimum_cases"]
    return dict(cases=cases,qualified_cases=sum(c["qualified"] for c in cases),cross_window_training_admitted=bool(passed),
        native_gate_separate=True,operational_qualified=False)
