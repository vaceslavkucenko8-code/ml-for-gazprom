"""Auditable lexical subject matching; abstains when scope is not explicit."""
from itertools import combinations
from .evidence_overlap import tokens, independent_groups
from datetime import date

# Stage and generic research vocabulary must not act as a technology identity.
GENERIC = set('a an the and or of to in on for with from by at is are was were be been being has have had this that these those it its their our we they not no only still yet remains remain limited laboratory lab use used using prototype prototypes experimental experiment research study studies trial trials testing tested test development developed technology technologies method methods system systems device devices new early stage small controlled commercial industrial deployment adoption widespread mass application applications approach approaches field demonstrated demonstrates demonstrate paper results shows show introduced presents proposed proposal'.split())


def subject_terms(claim, source):
    text = claim['quote']
    context = claim.get('scope_context')
    if isinstance(context, dict):
        quote = context.get('quote', '')
        if quote and quote in source.get('text', ''):
            text += ' ' + quote
    return sorted({t for t in tokens(text) if len(t) > 2 and not t.isdigit() and t not in GENERIC})


def scoped_confirmation(claims, sources, overlaps):
    buckets = {}
    for c in claims:
        s = sources[c['source_id']]
        if s.get('independence_reviewed') is not True or not s.get('independence_group'):
            continue
        for anchor in combinations(subject_terms(c, s), 2):
            buckets.setdefault(anchor, {}).setdefault(c['feature'], set()).add(c['source_id'])
    results = []
    for anchor, features in sorted(buckets.items()):
        groups = {f: independent_groups(ids, sources, overlaps, f) for f, ids in features.items()}
        counts = {f: len(g) for f, g in groups.items()}
        confirmed = sorted(f for f, n in counts.items() if n >= 2)
        if not confirmed:
            continue
        chronology = {}
        for f in ('early_stage', 'limited_adoption'):
            # Later republications of an existing evidence group add no time evidence.
            first_dates = sorted(min(sources[sid]['published_at'] for sid in group)
                                 for group in groups.get(f, []))
            span = (date.fromisoformat(first_dates[-1])-date.fromisoformat(first_dates[0])).days if len(first_dates)>1 else 0
            chronology[f] = {'first_publication_dates': first_dates, 'span_days': span,
                             'source_groups': groups.get(f, [])}
        dates = sorted({dt for value in chronology.values() for dt in value['first_publication_dates']})
        results.append({'subject_terms': list(anchor), 'features': confirmed,
                        'independent_counts': counts, 'publication_dates': dates,
                        'chronology_by_feature': chronology,
                        'source_ids': sorted(set().union(*(features[f] for f in confirmed)))})
    return results
