"""Coarse recording-level labels explicitly derived from SMPL filenames."""
RULES=(('feitian','飞天'),('lotus','莲花'),('ribbon','飘带'),('pipa','琵琶'),('meditation','冥想'),('36pose','三十六式'),('drum','鼓舞'))
def annotate(name):
    key=name.lower()
    found=[label for token,label in RULES if token in key]
    return dict(category=found[0] if len(found)==1 else 'UNKNOWN',source_filename=name,
        provenance='user_authorized_filename_annotation',scope='recording_level_coarse_action',
        event_specific_semantics='UNKNOWN',music_semantic_pair_label=False)
