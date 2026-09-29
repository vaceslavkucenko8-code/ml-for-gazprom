"""Fictional cases solely for software verification, not ML validation data."""
from copy import deepcopy


def candidate(identifier='example', extra_feature=None):
    text = 'Лабораторный прототип испытан на двух установках. Внедрение ограничено пилотом. Новое исследование опубликовано. Тема: промышленные датчики.'
    features = [('early_stage','Лабораторный прототип испытан на двух установках.'),
                ('limited_adoption','Внедрение ограничено пилотом.'),
                ('emerging_activity','Новое исследование опубликовано.'),
                ('relevance','Тема: промышленные датчики.')]
    if extra_feature:
        text += ' Дополнительное проверенное утверждение.'
        features.append((extra_feature,'Дополнительное проверенное утверждение.'))
    result = {'candidate_id': identifier, 'name_ru': 'ВЫМЫШЛЕННЫЙ тестовый датчик',
              'query': 'промышленные датчики', 'technology_group_id': identifier,
              'sources': [{'source_id':'s1','url':'https://example.org/fictional-lab',
                           'title_original':'ВЫМЫШЛЕННЫЙ документ', 'text':text,
                           'published_at':'2026-01-01','language':'ru','source_type':'research',
                           'trust_level':'high','trust_reason':'Условие теста, не реальная оценка источника',
                           'independence_group':'fictional-lab','independence_reviewed':True}],
              'evidence': []}
    for i,(feature,quote) in enumerate(features):
        result['evidence'].append({'evidence_id':f'e{i}', 'source_id':'s1', 'feature':feature,
                                  'quote':quote,'reviewed':True,'scope_matches_candidate':True,
                                  'observed_at':'2026-01-01','event_status':'observed',
                                  'query': result['query']})
    return result


def examples():
    good = candidate('synthetic-early')
    mature = candidate('synthetic-mature', 'mass_adoption')
    pending = candidate('synthetic-pending')
    pending['evidence'] = []
    return {'synthetic': True, 'purpose': 'software_demo_only_not_quality_benchmark',
            'candidates': [good,mature,pending]}
