"""Read-only, source-linked presentation of existing pipeline outputs.

No model calls, fabricated case studies, probabilities or implicit human approval.
"""
import json
import re
from urllib.parse import urlsplit

FEATURES = {'early_stage':'Ранняя стадия', 'limited_adoption':'Ограниченное внедрение',
            'relevance':'Связь с запросом', 'mass_adoption':'Массовое внедрение',
            'industry_standard':'Отраслевой стандарт', 'trusted_source':'Подходящий источник',
            'emerging_activity':'Новые события развития', 'established_market':'Сформированный рынок'}
REASONS = {
 'maturity_subject_not_established':'Зрелость относится к компоненту или её связь с исследуемым методом не установлена',
 'insufficient_explicit_evidence':'Недостаточно явных доказательств для отбора',
 'explicit_maturity_claim':'В источнике найдены признаки зрелой технологии',
 'explicit_early_and_limited_scope_claims':'Найдены признаки ранней стадии, ограниченного внедрения и связи с запросом',
 'conflicting_stage_claims':'Источники содержат противоречивые признаки стадии',
 'fetch_failed':'Не удалось загрузить источник', 'no_source_text':'Нет исходного текста',
 'insufficient_topic_overlap':'Недостаточное совпадение с темой запроса',
 'future_publication':'Публикация позже даты оценки', 'outside_time_window':'Публикация за пределами временного окна',
 'missing_or_future_publication_date':'Дата публикации неизвестна или позже даты оценки',
 'invalid_publication_date':'Некорректная дата публикации',
 'insufficient_source_trust':'Тип или доверие источника не подходят для автоматического отбора',
 'not_original_source_text':'Перевод, резюме или поисковый фрагмент не служат исходным доказательством',
 'future_or_modal_claim':'План или предположение, а не состоявшееся событие',
 'negated_claim':'В утверждении найдено отрицание', 'negated_limitation':'Отрицается ограниченность внедрения',
}
TERMS = [('quantum','Квантовые технологии'),('biosensor','Биосенсоры'),('glucose','Мониторинг глюкозы'),
 ('battery','Аккумуляторы'),('batteries','Аккумуляторы'),('hydrophone','Гидрофоны'),
 ('cryptograph','Криптография'),('sensor','Сенсорные технологии'),('photonic','Фотоника'),
 ('federated','Федеративное обучение'),('robot','Робототехника')]


def safe_url(value):
    try:
        p=urlsplit(value or '')
        return value if p.scheme in ('http','https') and p.netloc else None
    except ValueError:
        return None


def russian_terms(text):
    terms = [('federated learning','федеративное обучение'), ('fraud detection','обнаружение мошенничества'),
             ('privacy','конфиденциальность данных'), ('quantum','квантовые технологии'),
             ('navigation','навигация'), ('sensor','сенсоры'), ('glucose','глюкоза'),
             ('hydrophone','гидрофоны'), ('cryptograph','криптография'), ('prototype','прототип'),
             ('laboratory','лабораторные испытания'), ('efficiency','эффективность'),
             ('accuracy','точность'), ('cost','затраты'), ('energy','энергия'), ('simulation','моделирование')]
    return [ru for en,ru in terms if en in text.casefold()]

def reason(value):
    return REASONS.get(value, 'Дополнительное ограничение: '+str(value))


def make_card(candidate, assessment, index, manual=None):
    sources=[]
    valid_claims=[]
    raw_sources={s['source_id']:s for s in candidate.get('sources',[])}
    for claim in assessment.get('claims',[]):
        source=raw_sources.get(claim.get('source_id'),{})
        quote=claim.get('quote','')
        if quote and quote in source.get('text',''):
            valid_claims.append({**claim,'label_ru':FEATURES.get(claim['feature'],claim['feature']),
                                 'source_text_match':True})
    for source in raw_sources.values():
        metadata=source.get('retrieval_metadata') or {}
        labels=list(dict.fromkeys(c['label_ru'] for c in valid_claims if c['source_id']==source['source_id']))
        sources.append({**source,'url':safe_url(source.get('url')),
            'auto_translated':bool(metadata.get('is_translated')),
            'generated_summary':bool(metadata.get('is_generated_summary')),
            'interpretation_ru':('Правила обнаружили в тексте: '+', '.join(labels)+'. Смысл и применимость требуют проверки.'
                                 if labels else 'Автоматический анализ не нашёл достаточных признаков для отбора. Это не доказывает отсутствие ранней стадии.'),
            'interpretation_kind':'Автоматическая интерпретация по правилам, не полный перевод'})
    name=candidate.get('name_ru','')
    if re.search('[А-Яа-я]',name):
        title=name; title_kind='Название из данных'
    else:
        topic=next((ru for en,ru in TERMS if en in name.casefold()),'Исследование технологии')
        title=f'{topic} · кандидат {index}'
        title_kind='Автоматический тематический заголовок; оригинал ниже'
    p2=candidate.get('participant2_candidate') or {}
    evidence={e['evidence_id']:e for e in p2.get('evidence',[])}
    def excerpts(field):
        values=[]
        for fact in p2.get(field,[]):
            for eid in fact.get('evidence_ids',[]):
                e=evidence.get(eid,{})
                source=raw_sources.get(e.get('doc_id'),{})
                quote=e.get('quote','')
                if (not quote or quote not in source.get('text','') or not safe_url(source.get('url'))):
                    continue
                values.append({'quote':quote,'source_id':source['source_id'],
                    'status_ru':'План или предположение' if e.get('is_future_looking') else 'Фрагмент для проверки аналитиком',
                    'interpretation_ru':('Темы фрагмента: '+', '.join(russian_terms(quote))+'.' if russian_terms(quote) else 'Фрагмент помечен экстрактором как возможное преимущество или пример применения. Его смысл требует проверки.'),
                    'verified_case':False, 'is_translated':bool(e.get('is_translated')),
                    'is_generated_summary':bool(e.get('is_generated_summary'))})
        return values[:4]
    flags=assessment.get('flags',{})
    parts=[{'feature':f,'label_ru':FEATURES[f],'weight':w,'contribution':w if flags.get(f) else 0,
            'evidence_ids':[c['evidence_id'] for c in valid_claims if c['feature']==f]}
           for f,w in [('early_stage',30),('limited_adoption',40),('relevance',30)]]
    return {'candidate_id':candidate['candidate_id'],'title_ru':title,'title_kind':title_kind,'title_original':name,
        'technical_terms_ru':russian_terms(' '.join(s.get('text','') for s in sources)),
        'description_ru':f'Материалы по запросу «{candidate.get("query", "")}». '+reason(assessment.get('reason'))+'.',
        'description_kind':'Автоматическое резюме статуса по правилам',
        'description_evidence_ids':[c['evidence_id'] for c in valid_claims],
        'advantages':excerpts('advantages'),'cases':excerpts('case_examples'),
        'assessment':assessment,'reason_ru':reason(assessment.get('reason')),
        'missing_ru':[FEATURES.get(f,f) for f in assessment.get('missing_features',[])],
        'claims':valid_claims,'sources':sources,'score_parts':parts,
        'ignored':[{**i,'reason_ru':reason(i.get('reason'))} for i in assessment.get('ignored',[])],
        'manual_assessment':({**manual,'explanation_ru':__import__('functools').reduce(lambda text, pair: text.replace(pair[0],pair[1]), FEATURES.items(), manual.get('explanation_ru',''))} if manual else None),
        'unmatched_claims':len(assessment.get('claims',[]))-len(valid_claims)}


def load_view(folder, *, translate_cards=True, candidate_id=None):
    def read(name,default=None):
        p=folder/name
        if p.exists(): return json.loads(p.read_text(encoding='utf-8'))
        if default is not None: return default
        raise FileNotFoundError(name)
    auto=read('automatic.json'); data=read('detector_input.json'); summary=read('summary.json')
    result=read('result.json',{'results':auto.get('top15',[])})
    manual={a['candidate_id']:a for a in read('assessment.json',{}).get('all_assessments',[])}
    amap={a['candidate_id']:a for a in auto['assessments']}
    cards=[make_card(c,amap[c['candidate_id']],i+1,manual.get(c['candidate_id']))
           for i,c in enumerate(data['candidates']) if c['candidate_id'] in amap and (candidate_id is None or c['candidate_id']==candidate_id)]
    from .local_translation import localize
    from .signal_analysis import evaluate
    for card in cards:
        candidate=next(c for c in data['candidates'] if c['candidate_id']==card['candidate_id'])
        card['signal_analysis']=evaluate(candidate,card['assessment'],summary['as_of'])
    if translate_cards:
        cards=[localize(c) for c in cards]
    # Per-run editorial notes stay separate from detection and independent labels.
    reviews=read('application_review.json',{})
    notes={r['candidate_id']:r for r in reviews.get('items',[])}
    for card in cards:
        if card['candidate_id'] in notes:
            card['application_review']={**notes[card['candidate_id']], 'as_of':reviews.get('as_of')}
            if notes[card['candidate_id']].get('title_ru'):
                card['title_ru']=notes[card['candidate_id']]['title_ru']
                card['title_kind']='Название из обзора ассистента по первоисточникам'

    ids={c['candidate_id'] for c in cards}
    top=list(dict.fromkeys(r['candidate_id'] for r in result['results'] if r['candidate_id'] in ids))[:15]
    # Free TOP-15 slots are filled with explicitly marked preliminary candidates;
    # the strict automatic decision of every candidate stays unchanged.
    from .automatic import preliminary_candidates
    preliminary=[a['candidate_id'] for a in preliminary_candidates(auto['assessments'],top,15-len(top))
                 if a['candidate_id'] in ids] if candidate_id is None else []
    return {'query':summary['query'],'as_of':summary['as_of'],'cards':cards,'top_ids':top,
        'preliminary_ids':preliminary,
        'summary':summary,'excluded_documents':[{**d,'url':safe_url(d.get('url')),'reason_ru':reason(d['reason'])}
                                               for d in summary.get('excluded_documents',[])],
        'model_version':auto.get('model_version'),'ranking_model':auto.get('ranking_model'),
        'limitations':(['Поиск достиг лимита времени. Собранные материалы сохранены; охват источников неполный.'] if summary.get('budget_exhausted') else []) + ['Автоматические гипотезы требуют экспертной проверки.',
                       'Независимые бинарные метки отсутствуют. Precision, Recall и F1 не измерены.',
                       'Перевод фрагментов выполняется локально и может содержать ошибки. Оригинальные цитаты сохраняются.']}
