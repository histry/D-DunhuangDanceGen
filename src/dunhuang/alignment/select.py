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

def select(*args,**kwargs):
    """Compatibility route for historical research scripts, outside formal inference."""
    import importlib.util
    from pathlib import Path
    file=Path(__file__).resolve().parents[3]/'scripts/legacy_sdmr_select.py'
    spec=importlib.util.spec_from_file_location('legacy_sdmr_select',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.select(*args,**kwargs)
