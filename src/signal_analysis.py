"""Separate maturity, corroboration and chronology without inventing truth labels."""
from datetime import date
from .evidence_overlap import repetition, independent_count
from .claim_scope import scoped_confirmation
from .source_provenance import relations
from .claim_conflicts import compare_stages

VERSION = 'signal-analysis-0.6'
STAGES = {'early_signal':'Ранний сигнал', 'emerging_trend':'Признаки формирующегося тренда',
          'mature':'Зрелая технология', 'uncertain':'Стадия не установлена',
          'conflicting':'Возможное расхождение стадий — требуется проверка'}
SUPPORT = {'corroborated':'Есть независимые подтверждения признака',
           'preliminary':'Предварительные основания', 'insufficient':'Недостаточно доказательств',
           'promotional_only':'Только рекламные или вторичные материалы',
           'conflicting':'Требуется разбор противоречий'}


def evaluate(candidate, assessment, as_of):
    cutoff=date.fromisoformat(as_of)
    sources={s['source_id']:s for s in candidate.get('sources',[])}
    claims=[]; rejected=[];seen=set()
    for c in assessment.get('claims',[]):
        s=sources.get(c.get('source_id'),{})
        q=c.get('quote','')
        try: published=date.fromisoformat(s.get('published_at',''))
        except (ValueError,TypeError): published=None
        if not q or q not in s.get('text','') or published is None or published>cutoff:
            rejected.append(c.get('evidence_id'));continue
        key=(c['source_id'],c['feature'],q)
        if key in seen:continue
        seen.add(key);claims.append(c)
    features={c['feature'] for c in claims}
    overlaps=repetition(sources, claims)
    provenance=relations(sources)
    counting_pairs=overlaps+[{'source_ids':p['source_ids'],'feature':None} for p in provenance if p['kind']=='same_work']
    groups={}
    for c in claims:
        s=sources[c['source_id']]
        if s.get('independence_reviewed') is True and s.get('independence_group'):
            groups.setdefault(c['feature'],set()).add(c['source_id'])
    counts={f:independent_count(ids,sources,counting_pairs,f) for f,ids in groups.items()}
    scope_groups=scoped_confirmation(claims,sources,counting_pairs)
    shared=sorted({f for group in scope_groups for f in group['features'] if f in ('early_stage','limited_adoption','mass_adoption','industry_standard')})
    trend_groups=[]
    for group in scope_groups:
        if {'early_stage','limited_adoption'}<=set(group['features']) and all(group['chronology_by_feature'][f]['span_days']>=90 for f in ('early_stage','limited_adoption')):
            trend_groups.append(group)
    dates=sorted({sources[c['source_id']]['published_at'] for c in claims if c['feature'] in ('early_stage','limited_adoption')})
    span=(date.fromisoformat(dates[-1])-date.fromisoformat(dates[0])).days if len(dates)>1 else 0
    mature=bool(features & {'mass_adoption','industry_standard'})
    early=bool(features & {'early_stage','limited_adoption'})
    conflict_pairs=compare_stages(claims,sources)
    if mature and early:
        stage='conflicting' if any(p['status']=='potential_conflict' for p in conflict_pairs) else 'uncertain'
    elif mature and 'relevance' in features: stage='mature'
    elif {'early_stage','limited_adoption','relevance'}<=features:
        stage='emerging_trend' if trend_groups else 'early_signal'
    else:stage='uncertain'
    low={'press_release','blog','aggregator','forum','social_media'}
    promotional=bool(sources) and all(s.get('source_type') in low for s in sources.values())
    if stage=='conflicting':support='conflicting'
    elif shared:support='corroborated'
    elif claims and features-{'relevance'}:support='preliminary'
    elif promotional:support='promotional_only'
    else:support='insufficient'
    risks=[]
    if conflict_pairs and stage=='uncertain':risks.append('В материалах указаны разные стадии, но общий предмет не установлен. До проверки нельзя выбрать единую стадию технологии.')
    if any(p['kind']=='same_work' for p in provenance):risks.append('Несколько материалов имеют идентификатор одной научной работы; они объединены при подсчёте подтверждений.')
    if any(p['kind']=='shared_reference' for p in provenance):risks.append('Найдены ссылки на общую научную работу. Требуется проверить, являются ли материалы пересказами или самостоятельными исследованиями.')
    if any(n>=2 for n in counts.values()) and not shared:risks.append('Есть отдельные источники, но общий предмет утверждений не установлен; одинаковый признак стадии не доказывает связь с одной технологией.')
    if overlaps:risks.append('Найдены текстовые повторы между источниками: они не считаются дополнительными независимыми подтверждениями соответствующего признака.')
    if not shared:risks.append('Независимость подтверждений не установлена; разные сайты могут перепечатывать один источник.')
    if promotional:risks.append('Рекламные и вторичные материалы требуют первичного подтверждения.')
    if stage=='conflicting':risks.append('Проверьте, относятся ли разные стадии к одному методу, времени и масштабу.')
    if rejected:risks.append('Часть свидетельств не совпала с текстом или не имеет допустимой даты.')
    return {'version':VERSION,'stage':stage,'stage_ru':STAGES[stage],
            'support':support,'support_ru':SUPPORT[support],
            'disinformation':'not_established','disinformation_ru':'Намеренный вброс не установлен',
            'claims':claims,'rejected_evidence_ids':rejected,
            'corroborated_features':shared,'source_count':len(sources),
            'repetition_pairs':overlaps,'independent_evidence_counts':counts,
            'scope_groups':scope_groups,'trend_scope_groups':trend_groups,
            'provenance_relations':provenance,
            'stage_comparisons':conflict_pairs,
            'timeline':{'publication_dates':dates,'span_days':span,
                        'meaning':'Даты публикаций, не даты внедрения; рост активности не измерен'},
            'risks':risks,'probability':None,
            'limitations':['Экспериментальная оценка по извлечённым утверждениям, не независимая проверка истинности.',
                           'Предмет сопоставляется по двум общим предметным словам цитат: это лексическая эвристика, не понимание смысла. Масштаб, отрицания и идентичность метода требуют отдельной проверки.',
                           'Текстовые совпадения — повод проверить перепечатку; отсутствие совпадений не доказывает независимость. Переводы и смысловые перефразирования могут быть пропущены.',
                           'Признаки тренда: для каждого из двух признаков нужны независимые группы с интервалом первых публикаций от 90 дней; перепечатки не продлевают интервал. Порог не откалиброван.']}
