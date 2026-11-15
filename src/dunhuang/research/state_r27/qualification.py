"""Train-only routing; source control and physical quality are distinct gates."""
def fitting_gate(records,initial):
    rows=[];teachers=set()
    for time in (999,519):
        good=[x for x in records if x["t"]==time and x["mode"]=="correct"]
        def find(source,mode,items):return next(x for x in items if (x["teacher"],x["mode"],x["t"])==(source,mode,time))
        counts=dict(fit=0,wrong_body=0,wrong_local=0,none_gain=0)
        for x in good:
            i=x["teacher"];old=find(i,"correct",initial);wrong=find(i,"wrong",records);none=find(i,"none",records)
            counts["fit"]+=x["clipped_mse"]<=.5*old["clipped_mse"]
            counts["wrong_body"]+=x["metrics"]["body22_rad"]<wrong["metrics"]["body22_rad"]
            counts["wrong_local"]+=x["metrics"]["pelvis_wrist_rmse_m"]<wrong["metrics"]["pelvis_wrist_rmse_m"]
            counts["none_gain"]+=x["metrics"]["body22_rad"]<=.95*none["metrics"]["body22_rad"]
            if x["metrics"]["body22_rad"]<wrong["metrics"]["body22_rad"]:teachers.add(i)
        rows.append(dict(t=time,counts=counts,passed=all(v>=3 for v in counts.values())))
    return dict(per_time=rows,all_teacher_ids_respond=len(teachers)==4,passed=all(x["passed"] for x in rows) and len(teachers)==4)

def sampling_gate(records,arm):
    rows=[]
    for music in (0,1):
        selected=[r for r in records if r["arm"]==arm and r["mode"]=="correct" and r["teacher"]//2==music]
        tests=[]
        for r in selected:
            teacher,seed=r["teacher"],r["seed_index"]
            wrong=next(x for x in records if (x["arm"],x["mode"],x["teacher"],x["seed_index"])==(arm,"wrong",teacher,seed))
            none=next(x for x in records if x["mode"]=="none" and (x["teacher"],x["seed_index"])==(teacher,seed))
            a,b,c=r["metrics"],wrong["metrics"],none["metrics"]
            values=dict(body_gain=a["body22_rad"]<=.75*c["body22_rad"],body_reference=a["body22_rad"]<b["body22_rad"],
                local_reference=a["pelvis_wrist_rmse_m"]<b["pelvis_wrist_rmse_m"],
                activity=a["body_activity_ratio"] is not None and .6<=a["body_activity_ratio"]<=1.6,
                extent=a["arm_range_ratio"] is not None and .6<=a["arm_range_ratio"]<=1.6)
            tests.append(dict(teacher=teacher,seed_index=seed,tests=values,passed=all(values.values())))
        source_counts={i:sum(x["passed"] for x in tests if x["teacher"]==i) for i in (2*music,2*music+1)}
        rows.append(dict(music=music,conditions=tests,passed=sum(x["passed"] for x in tests)>=3 and min(source_counts.values())>=1))
    return dict(groups=rows,passed=all(x["passed"] for x in rows),scope="MODEL_GENERATED_SOURCE_CONTROL_ONLY_NOT_NATIVE_OR_CULTURAL_QUALITY")
