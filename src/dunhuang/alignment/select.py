"""Pure candidate selection. Generation is owned exclusively by pipeline.py."""

def select_candidates(candidates,policy='heuristic',qualification=None,top_k=5):
    if policy not in ('heuristic','learned_shadow','learned_qualified'):
        raise ValueError('Unknown retrieval policy')
    if policy=='learned_qualified' and not (qualification or {}).get('enabled'):
        raise ValueError('Learned retrieval lacks validation qualification')
    qualified=[c for c in candidates if c['feasibility']['source']=='PASS']
    def score(candidate):
        if policy!='learned_qualified':
            return candidate['scores']['heuristic']
        values=candidate['scores']
        if values['global_score'] is None or values['sync_score'] is None:
            raise ValueError('Missing global or synchronization score')
        return values['global_score']+.25*values['sync_score']
    return sorted(qualified,key=lambda c:(-score(c),c['event_uid']))[:top_k]
