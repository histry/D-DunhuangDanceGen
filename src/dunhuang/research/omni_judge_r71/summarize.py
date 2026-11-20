"""Describe automatic order/control responses without selecting a quality winner."""
from collections import Counter
from .prepare import ROOT, read, write
from .judge import DIMS


def normalize(preference, order):
    if preference in ('TIE', 'UNSURE'):
        return preference
    return 'baseline' if (preference == 'A') == (order == 'AB') else 'candidate'


def run():
    result = dict(human_reviewed=False, human_calibrated=False,
                  human_agreement='NOT_EVALUABLE', artistic_quality='NOT_EVALUABLE',
                  Native24='NOT_EVALUABLE', quality_winner=None, stages={})
    for stage in ['pilot', 'controls', 'full']:
        folder = ROOT / stage
        rows = [read(p) for p in sorted(folder.glob('*.json')) if p.stem.isdigit()]
        if not rows:
            continue
        groups = {}
        for row in rows:
            job = row['job']
            groups.setdefault((job['pair_id'], job['condition']), {})[job['order']] = row
        dimensions = {}
        for d in DIMS:
            counts = Counter()
            agreements = Counter()
            for (_, condition), orders in groups.items():
                normalized = {}
                for order, row in orders.items():
                    value = normalize(row['parsed'][d]['preference'], order) if row['parsed'] else 'INVALID_SCHEMA'
                    counts[(condition, value)] += 1
                    normalized[order] = value
                if set(normalized) == {'AB', 'BA'}:
                    value = 'INVALID_SCHEMA' if 'INVALID_SCHEMA' in normalized.values() else 'AGREE' if normalized['AB'] == normalized['BA'] else 'DISAGREE'
                    agreements[(condition, value)] += 1
            dimensions[d] = dict(normalized_response_counts=[dict(condition=c, response=v, count=n) for (c,v),n in sorted(counts.items())],
                                 order_consistency=[dict(condition=c, status=v, count=n) for (c,v),n in sorted(agreements.items())])
        result['stages'][stage] = dict(inferences=len(rows), registered_pairs=len(groups), dimensions=dimensions,
            interpretation='DESCRIPTIVE_AUTOMATIC_RESPONSES_NOT_INDEPENDENT_HUMAN_OR_MUSIC_VALIDITY')
    write(ROOT / 'automatic_summary.json', result)
    print('R71_AUTOMATIC_SUMMARY', {k:v['inferences'] for k,v in result['stages'].items()}, flush=True)


if __name__ == '__main__':
    run()
