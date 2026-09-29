"""Reproducible query -> sources -> candidates -> evidence assessment CLI."""
import argparse
from datetime import date, datetime, timezone, timedelta
import hashlib
import html
import json
from pathlib import Path
import re

from .detector import rank_candidates
from .participant2_adapter import adapt_export
from .automatic import assess_automatically
from .topic_terms import search_equivalent
from .source_ranker import predict as predict_source_priority
from .retrieval.schemas import SourceDocument
from .retrieval.fetch import Fetcher
from .retrieval.search import SearchOrchestrator, _topic_keywords
from .retrieval.connectors import DEFAULT_CONNECTORS
from .retrieval.deduplicate import deduplicate_and_score
from .extraction.candidates import extract_candidates
from .extraction.relevance import assess_candidates


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def topic_match(text, query):
    keywords = list(dict.fromkeys(_topic_keywords(query)))
    if not keywords:
        return False
    from .automatic import words
    tokens = re.findall(r'\w+', text.casefold())
    equivalents = words(text)
    # Count original query terms, not expanded aliases: "compute" and
    # "computing" cannot supply two different topical requirements.
    found = sum(bool(words(k) & equivalents) or
                (bool(re.search('[а-я]', k)) and len(k)>5 and any(t.startswith(k[:-2]) for t in tokens))
                for k in keywords)
    return found >= min(2, len(keywords))


def stable_candidates(query, docs):
    candidates = extract_candidates(query, sorted(docs, key=lambda d: d.doc_id))
    for c in candidates:
        c.candidate_id = 'cand_' + digest(query + '\n' + '\n'.join(sorted(c.source_doc_ids)))
        mapping = {}
        for evidence in c.evidence:
            new_id = 'ev_' + digest(evidence.doc_id + '\n' + evidence.quote + '\n' + evidence.claim_type.value)
            mapping[evidence.evidence_id] = new_id
            evidence.evidence_id = new_id
        # Repeated sentences cannot create duplicate evidence identifiers.
        c.evidence = list({e.evidence_id: e for e in c.evidence}.values())
        for fact in c.stage_facts + c.advantages + c.case_examples:
            fact.evidence_ids = list(dict.fromkeys(mapping[i] for i in fact.evidence_ids))
    return candidates


def select_documents(documents, query, as_of, max_age_days):
    cutoff = date.fromisoformat(as_of)
    selected, excluded = [], []
    for d in documents:
        reason = None
        if d.fetch_status.value != 'ok':
            reason = 'fetch_failed'
        elif not d.text.strip() or d.raw_metadata.get('text_scope') == 'search_snippet':
            reason = 'no_source_text'
        elif not topic_match((d.title or '') + '\n' + d.text, query):
            reason = 'insufficient_topic_overlap'
        elif d.published_at and d.published_at.date() > cutoff:
            reason = 'future_publication'
        elif max_age_days is not None and d.published_at and (cutoff - d.published_at.date()).days > max_age_days:
            reason = 'outside_time_window'
        if reason:
            excluded.append({'doc_id': d.doc_id, 'url': d.url, 'reason': reason})
        else:
            selected.append(d)
    return selected, excluded


def make_report(payload, adapted, assessment, preliminary, summary, automatic=None):
    esc = lambda x: html.escape(str(x if x is not None else 'неизвестно'))
    by_id = {a['candidate_id']: a for a in assessment['all_assessments']}
    labels = {'weak': 'Прошёл правила отбора', 'reject': 'Исключён', 'needs_review': 'Нужна проверка'}
    cards = []
    auto_by_id = {r['candidate_id']:r for r in (automatic or {}).get('assessments',[])}
    for c in adapted['candidates']:
        a = by_id[c['candidate_id']]
        pre = preliminary.get(c['candidate_id'], {})
        sources = ''.join(f'<li><a href="{esc(s["url"])}" rel="noopener noreferrer">{esc(s["title_original"])}</a> '
                          f'— {esc(s["published_at"])}; {esc(s["source_type"])}; доверие: {esc(s["trust_level"])}</li>' for s in c['sources'])
        evidence = ''.join(f'<li><code>{esc(e["evidence_id"])}</code><blockquote>{esc(e["quote"])}</blockquote></li>' for e in c['extracted_evidence_pending'])
        full_text = ''.join(f'<details><summary>{esc(s["title_original"])}</summary><pre>{esc(s["text"])}</pre></details>' for s in c['sources'])
        auto=auto_by_id.get(c['candidate_id'],{})
        auto_label={'likely_weak':'Вероятный слабый сигнал','likely_mature':'Вероятно зрелая технология','needs_review':'Недостаточно однозначных фактов'}.get(auto.get('decision'),'Не выполнен')
        auto_claims=''.join(f'<li>{esc(e["feature"])}: {esc(e["quote"])}</li>' for e in auto.get('claims',[]))
        auto_html=f'<p><b>Автоматический анализ: {auto_label}</b> (экспериментальные правила, не экспертная метка).</p><details><summary>Основания автоматического анализа</summary><ul>{auto_claims}</ul></details>'
        if auto.get('learned_priority'):
            learned=auto['learned_priority']
            auto_html+=f'<p>Обученная модель ранжирования: {learned["score"]:.2f} / 7. Прогноз приоритета по источникам, не вероятность класса.</p>'
        cards.append(f'<article><h2>{esc(c["name_ru"])}</h2>{auto_html}<p>Оценка по проверенным фактам: <b>{labels[a["decision"]]}</b> · Балл доказательств {a["score"]}/100</p>'
                     f'<p>{esc(a["explanation_ru"])}</p><p>Предварительный приоритет изучения: {esc(pre.get("score"))} '
                     '(эвристика поиска, не вероятность и не итоговый класс).</p><p>ID: <code>' + esc(c['candidate_id']) + '</code></p>'
                     f'<h3>Источники</h3><ul>{sources}</ul><h3>Извлечённые цитаты для проверки</h3><ul>{evidence}</ul>{full_text}</article>')
    errors = ''.join(f'<li>{esc(e)}</li>' for e in summary.get('errors', []))
    return ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Поиск технологических сигналов</title><style>' \
        'body{font:17px/1.55 system-ui;background:#f3f5f9;color:#172640;max-width:1050px;margin:40px auto;padding:0 24px}' \
        'article,header{background:white;padding:24px;border-radius:14px;margin:20px 0}h1{font-size:32px}h2{font-size:23px}' \
        'a{color:#1858ab}pre{white-space:pre-wrap;overflow-wrap:anywhere}blockquote{border-left:3px solid #668ab6;margin-left:0;padding-left:15px}' \
        'code{overflow-wrap:anywhere}details{margin:12px 0}</style><header><h1>Технологические сигналы</h1>' \
        f'<p>Запрос: <b>{esc(payload.get("query"))}</b> · Дата оценки: {esc(assessment["as_of"])}</p>' \
        f'<p>Кандидатов: {len(cards)}. Автоматически предложено: {len((automatic or {}).get("top15",[]))}. По проверенным фактам отобрано: {len(assessment["results"])}.</p>' \
        '<p>Недостаток доказательств не означает отрицательный класс. Поиск и предварительный приоритет автоматизированы; '
        'смысл цитат, масштаб внедрения и применимость проверяются отдельно.</p>' \
        f'<details><summary>Ошибки источников ({len(summary.get("errors", []))})</summary><ul>{errors}</ul></details></header>' \
        + (''.join(cards) or '<article>Подходящие документы не получены. Посмотрите summary.json: причины исключений и ошибки источников.</article>') + '</html>')


def process_export(payload, as_of, out, reviews=None, max_age_days=540, filter_query=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    docs = [SourceDocument.model_validate(d) for d in payload['documents']]
    selected, excluded = select_documents(docs, filter_query or payload['query'], as_of, max_age_days)
    selected = deduplicate_and_score(selected)
    candidates = stable_candidates(payload['query'], selected)
    export = {**payload, 'documents': [d.model_dump(mode='json') for d in selected],
              'candidates': [c.model_dump(mode='json') for c in candidates]}
    adapted = adapt_export(export, reviews)
    automatic=assess_automatically(adapted['candidates'],as_of,filter_query or payload.get('search_query'))
    ranker_path=Path(__file__).resolve().parents[1]/'artifacts/source_ranker.json'
    if ranker_path.exists():
        ranker=json.loads(ranker_path.read_text(encoding='utf-8'))
        candidates_by_id={c['candidate_id']:c for c in adapted['candidates']}
        for result in automatic['assessments']:
            result['learned_priority']=predict_source_priority(candidates_by_id[result['candidate_id']]['sources'],ranker,as_of)
        if ranker['deployment_enabled']:
            automatic['top15']=sorted((r for r in automatic['assessments'] if r['decision']=='likely_weak'),
                key=lambda r:(-r['score'],-r['learned_priority']['score'],r['candidate_id']))[:15]
            automatic['ranking_model']=ranker['model_version']
    assessment = rank_candidates(adapted['candidates'], as_of)
    preliminary = {a.candidate_id: a.model_dump(mode='json') for a in assess_candidates(
        candidates, selected, now=datetime.fromisoformat(as_of).replace(tzinfo=timezone.utc))}
    summary = {**payload.get('summary', {}), 'query': payload['query'], 'as_of': as_of,
               'retrieved_documents': len(docs), 'eligible_documents': len(selected),
               'candidates': len(candidates), 'excluded_documents': excluded, 'max_age_days': max_age_days,
               'counts': assessment['counts'], 'automatic_counts':automatic['counts'], 'preliminary_scores_are_not_final_labels': True}
    write_json(out/'retrieved.json', payload)
    write_json(out/'export.json', export)
    write_json(out/'detector_input.json', adapted)
    write_json(out/'assessment.json', assessment)
    write_json(out/'preliminary.json', preliminary)
    write_json(out/'automatic.json',automatic)
    write_json(out/'result.json',{'query':payload['query'],'as_of':as_of,'results':automatic['top15'],
                               'candidates':adapted['candidates'],'counts':automatic['counts'],
                               'requires_manual_approval':False,
                               'decision_kind':'automatic_hypothesis_not_expert_ground_truth'})
    write_json(out/'reviews.json', reviews or [])
    (out/'report.html').write_text(make_report(export, adapted, assessment, preliminary, summary,automatic), encoding='utf-8')
    write_json(out/'summary.json', summary)
    from .storage import persist_run
    storage=persist_run(out)
    write_json(out/'completion.json',{'completed':True,**storage})
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir()) if f.is_file()}
    write_json(out/'manifest.json', hashes)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Поиск и оценка технологических сигналов')
    parser.add_argument('--query', help='Запрос пользователя')
    parser.add_argument('--search-query', help='Дополнительная формулировка для поиска, например на английском')
    parser.add_argument('--input', type=Path, help='Повторная обработка сохранённого retrieved.json без сети')
    parser.add_argument('--reviews', type=Path)
    parser.add_argument('--as-of', default=date.today().isoformat())
    parser.add_argument('--out', type=Path)
    parser.add_argument('--connectors', default='mit_news,arxiv,crossref,hackernews')
    parser.add_argument('--results-per-connector', type=int, default=6)
    parser.add_argument('--max-queries', type=int, default=1)
    parser.add_argument('--search-budget-seconds', type=int, default=240)
    parser.add_argument('--enrich-sources', action='store_true', help='Загрузить связанные первичные работы и полные тексты arXiv')
    parser.add_argument('--max-age-days', type=int, default=540)
    args = parser.parse_args()
    date.fromisoformat(args.as_of)
    if not 1 <= args.results_per_connector <= 30 or not 1 <= args.max_queries <= 6 or args.max_age_days < 0 or not 10 <= args.search_budget_seconds <= 900:
        parser.error('Некорректные ограничения поиска')
    out = args.out or Path('runs')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if out.exists():
        parser.error('Папка результата уже существует; выберите новую, чтобы сохранить предыдущий запуск')
    if args.input:
        payload = json.loads(args.input.read_text(encoding='utf-8'))
    else:
        if not args.query or not args.query.strip():
            parser.error('Нужен --query или --input')
        registry = {c.name: c for c in DEFAULT_CONNECTORS}
        names = args.connectors.split(',')
        if any(n not in registry for n in names):
            parser.error('Неизвестный коннектор')
        fetcher = Fetcher(max_retries=0, connect_timeout=5, read_timeout=12, min_delay_per_domain=3)
        effective_query = args.search_query or search_equivalent(args.query)
        connectors = [registry[n](fetcher=fetcher) for n in names]
        publication_start = (date.fromisoformat(args.as_of) - timedelta(days=args.max_age_days)).isoformat()
        for connector in connectors:
            connector.publication_window = (publication_start, args.as_of)
        run = SearchOrchestrator(connectors=connectors, fetcher=fetcher,
                    max_results_per_connector=args.results_per_connector).run(effective_query, args.max_queries, max_seconds=args.search_budget_seconds)
        payload = {'query': args.query, 'search_query': effective_query, 'synthetic': False,
                   'query_method': 'explicit' if args.search_query else ('bilingual_dictionary' if effective_query != args.query else 'original'),
                   'summary': run.summary(), 'documents': [d.model_dump(mode='json') for d in run.documents]}
        payload['summary']['search_limits'] = {'results_per_connector_per_query': args.results_per_connector,
                                               'query_variants': args.max_queries,
                                               'budget_seconds': args.search_budget_seconds}
        payload['summary']['publication_window'] = {'from': publication_start, 'until': args.as_of,
                                                   'server_filtered_connectors': [n for n in names if n in ('arxiv', 'crossref')]}
    if args.enrich_sources:
        from .retrieval.enrichment import enrich_payload
        payload = enrich_payload(payload, args.as_of,
                                 Fetcher(max_retries=0, connect_timeout=5, read_timeout=12, min_delay_per_domain=3),
                                 max_age_days=args.max_age_days, search_query=args.search_query)
    reviews = json.loads(args.reviews.read_text(encoding='utf-8')) if args.reviews else []
    summary = process_export(payload, args.as_of, out, reviews, args.max_age_days,
                             args.search_query or payload.get('search_query'))
    print(json.dumps(summary, ensure_ascii=False))
    print(f'Report: {out / "report.html"}')


if __name__ == '__main__':
    main()
