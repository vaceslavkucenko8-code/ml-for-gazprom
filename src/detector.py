"""Evidence-based prototype. Feature assertions must be reviewed upstream.

No network access or LLM calls are made here. Rules are explicit, not trained
probabilities. The module validates evidence references, not their semantics.
"""
from datetime import date
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

VERSION = 'evidence-rules-0.1'
FEATURES = {'early_stage', 'limited_adoption', 'emerging_activity', 'relevance',
            'mass_adoption', 'established_market', 'industry_standard', 'hype_only', 'irrelevant'}
TRUSTED_TYPES = {'research', 'official', 'regulator', 'university', 'patent', 'professional_media'}
LABELS = {
    'early_stage': 'Подтверждена ранняя стадия',
    'limited_adoption': 'Подтверждён ограниченный масштаб внедрения',
    'emerging_activity': 'Есть новые события развития технологии',
    'relevance': 'Подтверждена связь с запросом',
    'trusted_source': 'Есть источник с обоснованным доверием',
    'independent_sources': 'Есть не менее двух независимых источников',
    'mass_adoption': 'Подтверждено массовое внедрение',
    'established_market': 'Подтверждён сформированный рынок',
    'industry_standard': 'Кандидат уже является отраслевым стандартом',
    'hype_only': 'Подтверждён информационный шум без технологического содержания',
    'irrelevant': 'Кандидат не соответствует запросу',
}
WEIGHTS = {'early_stage': 25, 'limited_adoption': 25, 'emerging_activity': 15,
           'relevance': 15, 'trusted_source': 10, 'independent_sources': 10}


def normalized_url(url):
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.netloc:
        raise ValueError('Source URL must be HTTP(S) with a host')
    query = [(k,v) for k,v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith('utm_') and k.lower() not in ('fbclid','gclid')]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or '/', urlencode(sorted(query)), ''))


def required_text(record, key):
    if not isinstance(record.get(key), str) or not record[key].strip():
        raise ValueError(f'{key} must be non-empty text')
    return record[key]


def evidence_features(candidate, as_of):
    cutoff = date.fromisoformat(as_of)
    for key in ('candidate_id', 'name_ru', 'query'):
        required_text(candidate, key)
    sources = {}
    for s in candidate.get('sources', []):
        sid = required_text(s, 'source_id')
        if sid in sources:
            raise ValueError(f'Duplicate source_id: {sid}')
        normalized_url(required_text(s, 'url'))
        required_text(s, 'title_original')
        required_text(s, 'text')
        sources[sid] = s
    accepted, rejected, seen = [], [], set()
    for ev in candidate.get('evidence', []):
        eid = required_text(ev, 'evidence_id')
        if eid in seen:
            raise ValueError(f'Duplicate evidence_id: {eid}')
        seen.add(eid)
        feature = ev.get('feature')
        if feature not in FEATURES:
            raise ValueError(f'Unknown evidence feature: {feature}')
        source = sources.get(ev.get('source_id'))
        reason = None
        if source is None:
            reason = 'unknown_source'
        elif ev.get('reviewed') is not True:
            reason = 'semantic_review_required'
        elif ev.get('scope_matches_candidate') is not True:
            reason = 'candidate_scope_not_confirmed'
        elif not isinstance(ev.get('quote'), str) or not ev['quote'].strip() or ev['quote'] not in source['text']:
            reason = 'quote_not_found_in_source'
        elif feature == 'relevance' and ev.get('query') != candidate['query']:
            reason = 'relevance_review_for_different_query'
        elif feature != 'relevance' and ev.get('event_status') != 'observed':
            reason = 'not_an_observed_fact'
        else:
            published = source.get('published_at')
            observed = ev.get('observed_at')
            if not published or not observed:
                reason = 'date_required_for_as_of_assessment'
            else:
                try:
                    if date.fromisoformat(published) > cutoff or date.fromisoformat(observed) > cutoff:
                        reason = 'future_evidence'
                except (ValueError, TypeError):
                    reason = 'invalid_date'
        if reason:
            rejected.append({'evidence_id': eid, 'reason': reason})
        else:
            accepted.append(ev)
    trusted_ids = {sid for sid,s in sources.items()
                   if s.get('source_type') in TRUSTED_TYPES and s.get('trust_level') in ('high','medium')
                   and isinstance(s.get('trust_reason'), str) and s['trust_reason'].strip()}
    # Conservative: only claims backed by an eligible source affect decisions.
    decision_evidence = [e for e in accepted if e['source_id'] in trusted_ids]
    flags = {feature: any(e['feature']==feature for e in decision_evidence) for feature in FEATURES}
    support_ids = {e['source_id'] for e in decision_evidence
                   if e['feature'] in ('early_stage','limited_adoption','emerging_activity')}
    flags['trusted_source'] = bool(support_ids)
    # Both distinct URLs and explicit reviewed publisher independence are needed.
    units = {sources[sid].get('independence_group') for sid in support_ids
             if sources[sid].get('independence_reviewed') is True and sources[sid].get('independence_group')}
    canonical = {normalized_url(sources[sid]['url']) for sid in support_ids}
    flags['independent_sources'] = len(units)>=2 and len(canonical)>=2
    return flags, accepted, rejected, decision_evidence, sources


def assess(candidate, as_of):
    flags, accepted, rejected, evidence, sources = evidence_features(candidate, as_of)
    predictors = []
    for feature, weight in WEIGHTS.items():
        if flags[feature]:
            predictors.append({'feature': feature, 'contribution': weight,
                               'explanation_ru': LABELS[feature],
                               'evidence_ids': [e['evidence_id'] for e in evidence if e['feature']==feature]})
    score = sum(p['contribution'] for p in predictors)
    exclusions = [f for f in ('irrelevant','mass_adoption','established_market','industry_standard','hype_only') if flags[f]]
    missing = [f for f in ('early_stage','limited_adoption','relevance','trusted_source') if not flags[f]]
    if exclusions:
        decision = 'reject'
        explanation = 'Исключён: ' + '; '.join(LABELS[f] for f in exclusions) + '.'
    elif missing or score<70:
        decision = 'needs_review'
        explanation = 'Недостаточно подтверждённых сведений для отбора. Нужна проверка: ' + ', '.join(missing or ['score_below_threshold']) + '.'
    else:
        decision = 'weak'
        explanation = 'Кандидат прошёл правила предварительного отбора: ранняя стадия, ограниченное внедрение и релевантность подтверждены источниками. Требуется экспертная оценка результата.'
    return {'candidate_id': candidate['candidate_id'], 'name_ru': candidate['name_ru'], 'query': candidate['query'],
            'as_of': as_of, 'decision': decision, 'score': score,
            'score_kind': 'evidence_heuristic_0_100', 'probability_weak': None,
            'model_version': VERSION, 'threshold': 70, 'predictors': predictors,
            'exclusion_reasons': [{'feature': f, 'explanation_ru': LABELS[f],
                'evidence_ids': [e['evidence_id'] for e in evidence if e['feature']==f]} for f in exclusions],
            'missing_features': missing, 'explanation_ru': explanation,
            'accepted_evidence_ids': [e['evidence_id'] for e in accepted], 'discarded_evidence': rejected,
            'source_ids': sorted({e['source_id'] for e in evidence}),
            'limitations': ['Признаки размечаются и проверяются вне этого модуля; совпадение цитаты не доказывает смысл утверждения.',
                            'Score — эвристический рейтинг доказательств, не обученная вероятность.',
                            'Качество детекции на скрытой разметке не измерено.']}


def rank_candidates(candidates, as_of, limit=15, assessor=None):
    if type(limit) is not int or limit<1:
        raise ValueError('limit must be a positive integer')
    ids = [c.get('candidate_id') for c in candidates]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate candidate_id')
    results = [(assessor or assess)(c, as_of) for c in candidates]
    group_by_id = {c['candidate_id']: c.get('technology_group_id') or c['candidate_id'] for c in candidates}
    selected, duplicate_groups, used = [], [], set()
    for result in sorted((r for r in results if r['decision']=='weak'), key=lambda r: (-r['score'], r['candidate_id'])):
        group = group_by_id[result['candidate_id']]
        if group in used:
            duplicate_groups.append(result['candidate_id'])
        else:
            used.add(group)
            selected.append(result)
    return {'as_of': as_of, 'requested_limit': limit, 'results': selected[:limit],
            'all_assessments': results, 'suppressed_same_group': duplicate_groups,
            'insufficient_results': len(selected)<limit,
            'counts': {d: sum(r['decision']==d for r in results) for d in ('weak','reject','needs_review')}}
