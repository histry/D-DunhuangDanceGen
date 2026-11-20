"""R65 isolated evaluation sidecar; never invokes Planner or neural training."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from dunhuang.research.quality_r65 import study


def filter_independent_music_comparisons(result, out):
    """CLI-only correction; never change the sealed R65 scientific package.

    The package already selects distinct authenticated persons in its stats.
    Use that exact selection for automatic-vs-human rows as well. Excluded
    comparisons remain private evidence, and the unfiltered result is preserved
    byte-for-byte. Empty/already-valid results are not rewritten.
    """
    qualifications = result.get('stats', {}).get('qualifications', [])
    allowed = {q['reviewer_id'] for q in qualifications
               if q.get('schema') == 'R65_MUSIC_REVIEW_V1'
               and q.get('contract_qualified') is True
               and q.get('counted_as_independent') is True
               and not q.get('duplicate_person_document', False)}
    kept, excluded = [], []
    for comparison in result.get('automatic_vs_human', []):
        reviewer = comparison.get('reviewer_id')
        if reviewer in allowed:
            kept.append(comparison)
            continue
        matches = [q for q in qualifications if q.get('schema') == 'R65_MUSIC_REVIEW_V1'
                   and q.get('reviewer_id') == reviewer]
        reason = ('DUPLICATE_PERSON_DOCUMENT' if any(q.get('duplicate_person_document') for q in matches)
                  else 'NOT_A_COUNTED_QUALIFIED_INDEPENDENT_MUSIC_PERSON')
        excluded.append(dict(reviewer_id=reviewer, reason=reason,
            scope='PRESERVED_NONINDEPENDENT_COMPARISON; EXCLUDED_FROM_INDEPENDENT_HUMAN_EVIDENCE',
            comparison=deepcopy(comparison)))
    if not excluded:
        return result
    out = Path(out)
    path = out/'result.json'
    original = path.read_bytes()
    if json.loads(original.decode('utf-8-sig')) != result:
        raise ValueError('CLI filter requires the exact result just written by study.evaluate')
    backup = out/'private_unfiltered_study_result.json'
    # Never overwrite an earlier original. The normal evaluate output is fresh.
    with backup.open('xb') as stream:
        stream.write(original)
    updated = deepcopy(result)
    updated['automatic_vs_human'] = kept
    updated['automatic_vs_human_nonindependent'] = excluded
    updated['independent_music_comparison_filter'] = dict(
        version='R65_CLI_DISTINCT_MUSIC_PERSON_FILTER_V1',
        selected_by='SEALED_PACKAGE_STATS_QUALIFICATIONS; NO_NEW_IDENTITY_OR_LABEL_INFERENCE',
        independent_comparisons_retained=len(kept), nonindependent_comparisons_preserved=len(excluded),
        original_result_private_file=backup.name,
        original_result_sha256=hashlib.sha256(original).hexdigest(),
        original_review_records_preserved=True, metrics_recomputed=False,
        scope='PRIVATE_EVALUATION_SIDECAR_ONLY; NOT_A_NEW_HUMAN_QUALITY_QUALIFICATION')
    path.write_text(json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    return updated


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','controls','verify','evaluate'])
    p.add_argument('--reviews');p.add_argument('--roster');p.add_argument('--out')
    a=p.parse_args()
    if a.mode=='prepare':study.prepare()
    elif a.mode=='controls':study.controls()
    elif a.mode=='verify':study.verify()
    else:
        if not a.out:p.error('evaluate requires a fresh --out; --reviews and --roster are optional')
        result=study.evaluate(a.reviews,a.roster,a.out)
        filter_independent_music_comparisons(result,a.out)

if __name__=='__main__':main()
