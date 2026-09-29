"""Expose pairs for review, never equate stage differences with disinformation."""
from .claim_scope import subject_terms


def compare_stages(claims, sources):
    early = [c for c in claims if c['feature'] in ('early_stage', 'limited_adoption')]
    mature = [c for c in claims if c['feature'] in ('mass_adoption', 'industry_standard')]
    pairs = []
    for left in early:
        for right in mature:
            common = sorted(set(subject_terms(left, sources[left['source_id']])) &
                            set(subject_terms(right, sources[right['source_id']])))
            pairs.append({'early_claim': left, 'mature_claim': right,
                          'shared_subject_terms': common,
                          'status': 'potential_conflict' if len(common) >= 2 else 'scope_unresolved',
                          'publication_dates': [sources[c['source_id']]['published_at'] for c in (left, right)],
                          'meaning': 'Различие стадий требует проверки метода, масштаба и времени; логическое противоречие не установлено.'})
    return pairs
