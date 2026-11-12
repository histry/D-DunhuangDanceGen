"""New task splits consume explicit recording roles; historical heuristics live in research."""
def split_training_rows(windows,task,roles):
    if task not in ("paired","motion","upperbody_probe"):raise ValueError("Unknown training task")
    groups={row[0] for row in windows}
    if not groups:return [],[],[]
    if any(g not in roles or roles[g] not in ("train","val","test") for g in groups):
        raise ValueError("Split V3 requires explicit recording roles")
    return ([r for r in windows if roles[r[0]]=="train"],[r for r in windows if roles[r[0]]=="val"],
        sorted(g for g in groups if roles[g]=="val"))
