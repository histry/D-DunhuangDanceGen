"""Integer lexicographic DAG costs, with complete Bellman labels and UID tie breaks."""
from dunhuang.paper1.contracts import identity

def unit(row):
    return row['source_uid'],row['event_uid'],tuple(row['source_interval'])

def node_cost(row,baseline,objective):
    if objective not in ('source_warp','source_only'):raise ValueError('Unregistered objective')
    changed=unit(row)!=unit(baseline)
    values=(int(changed),int(not changed and row['q_sha256']!=baseline['q_sha256']))
    return values if objective=='source_warp' else values[:1]

def constrained(layers,slot,allowed):
    if not 0<=slot<len(layers):raise ValueError('Invalid registered center slot')
    result=[list(layer) for layer in layers];result[slot]=[u for u in result[slot] if u in set(allowed)]
    if not result[slot]:raise ValueError('Registered center source has no candidate')
    return result

def change_counts(ids,baseline,rows,center):
    if ids is None:return None
    per_slot=[]
    for k,(u,v) in enumerate(zip(ids,baseline)):
        a,b=rows[u],rows[v];c=node_cost(a,b,'source_warp')
        per_slot.append(dict(slot=k,uid=u,baseline_uid=v,source_unit_changed=bool(c[0]),same_unit_actual_clock_changed=bool(c[1]),
            recording_changed=a['source_uid']!=b['source_uid'],rotation_changed=a['rotation_sha256']!=b['rotation_sha256'],
            left_neighbor_changed=bool(k and ids[k-1]!=baseline[k-1]),right_neighbor_changed=bool(k+1<len(ids) and ids[k+1]!=baseline[k+1])))
    return dict(source_units=sum(v['source_unit_changed'] for v in per_slot),same_unit_clocks=sum(v['same_unit_actual_clock_changed'] for v in per_slot),
        context_source_units_excluding_center=sum(v['source_unit_changed'] for v in per_slot if v['slot']!=center),
        recordings=sum(v['recording_changed'] for v in per_slot),actual_core_arrays=sum(v['rotation_changed'] for v in per_slot),per_slot=per_slot)

def signature(layers,edges,rows,baseline,objective):
    where={u:k for k,layer in enumerate(layers) for u in layer}
    if len(where)!=sum(map(len,layers)) or len(layers)!=len(baseline):raise ValueError('Unique layer identities and reference slots required')
    if any(where.get(v)!=where.get(u,-2)+1 for u,v in edges):raise ValueError('Only adjacent-layer measured edges admitted')
    ids=set(where)|set(baseline)
    return identity(dict(layers=layers,edges=sorted(edges),baseline=baseline,objective=objective,
        cost_inputs={u:dict(unit=unit(rows[u]),q_sha256=rows[u]['q_sha256']) for u in sorted(ids)}))

def _candidates(k,u,labels,rows,baseline,edges,objective):
    weight=node_cost(rows[u],rows[baseline[k]],objective)
    if not k:return [(weight,(u,),None)]
    return [(tuple(a+b for a,b in zip(value['cost'],weight)),tuple(value['path'])+(u,),v)
            for v,value in labels[k-1].items() if value is not None and (v,u) in edges]

def solve(layers,edges,rows,baseline,objective='source_warp'):
    edges=set(edges);digest=signature(layers,edges,rows,baseline,objective);labels=[]
    for k,layer in enumerate(layers):
        current={}
        for u in sorted(layer):
            candidates=_candidates(k,u,labels,rows,baseline,edges,objective)
            if candidates:
                cost,path,parent=min(candidates);current[u]=dict(cost=list(cost),path=list(path),predecessor=parent)
            else:current[u]=None
        labels.append(current)
    last=[(tuple(r['cost']),tuple(r['path'])) for r in labels[-1].values() if r is not None]
    best=min(last) if last else None
    certificate=dict(input_sha256=digest,objective=objective,labels=labels,ids=None if best is None else list(best[1]),cost=None if best is None else list(best[0]),
        status='NO_COMPLETE_PATH_IN_FROZEN_MEASURED_GRAPH' if best is None else 'EXACT_MEASURED_GRAPH_MINIMUM',tie_break='LEXICOGRAPHIC_FULL_UID_PATH',
        scope='ONLY_REGISTERED_CENTER_SOURCE_AND_FROZEN_MEASURED_ELIGIBLE_GRAPH; NOT_CONTINUOUS_OR_UNKNOWN_EDGE_DOMAIN')
    verify(certificate,layers,edges,rows,baseline);return certificate

def verify(cert,layers,edges,rows,baseline):
    """Check every Bellman equality and predecessor, not just the selected path."""
    edges=set(edges);objective=cert['objective']
    if cert['input_sha256']!=signature(layers,edges,rows,baseline,objective):raise ValueError('Certificate input binding differs')
    labels=cert['labels']
    if len(labels)!=len(layers):raise ValueError('Missing layer labels')
    for k,layer in enumerate(layers):
        if set(labels[k])!=set(layer):raise ValueError('Missing node labels')
        for u in layer:
            candidates=_candidates(k,u,labels,rows,baseline,edges,objective);actual=labels[k][u]
            expected=min(candidates) if candidates else None
            if expected is None:
                if actual is not None:raise ValueError('Unreachable node has a label')
            elif actual is None or (tuple(actual['cost']),tuple(actual['path']),actual['predecessor'])!=expected:raise ValueError('Bellman optimality or predecessor mismatch')
    last=[(tuple(v['cost']),tuple(v['path'])) for v in labels[-1].values() if v is not None];best=min(last) if last else None
    if (None if best is None else list(best[0]))!=cert['cost'] or (None if best is None else list(best[1]))!=cert['ids']:raise ValueError('Global terminal minimum differs')
    expected_status='NO_COMPLETE_PATH_IN_FROZEN_MEASURED_GRAPH' if best is None else 'EXACT_MEASURED_GRAPH_MINIMUM'
    if cert['status']!=expected_status:raise ValueError('Certificate status differs')
    return True
