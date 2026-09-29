"""Bounded retrieval of original papers; citations never establish independence."""
import copy
import time
from urllib.parse import urlsplit
from .schemas import SourceDocument
from .connectors.base import SearchHit
from .parse import parse_html_document
from .publication_metadata import recover_arxiv_date
from ..source_provenance import work_id


def targets(document):
    """Only known scholarly identifiers, no arbitrary URLs taken from prose."""
    own = work_id(document.url)
    if own and own.startswith('arxiv:') and '/html/' not in document.url:
        # Keep the exact version if one was supplied by the source API.
        identifier = urlsplit(document.url).path.rsplit('/', 1)[-1].removesuffix('.pdf')
        yield 'https://arxiv.org/html/' + identifier, 'same_work_full_text'
    for link in document.raw_metadata.get('research_links', []):
        key = work_id(link)
        if key and key != own:
            if key.startswith('doi:'):
                yield 'https://doi.org/' + key[4:], 'cited_work'
            elif key.startswith('arxiv:'):
                yield 'https://arxiv.org/html/' + key[6:], 'cited_work'


def enrich_payload(payload, as_of, fetcher, max_documents=12, max_seconds=120,
                   max_age_days=540, search_query=None):
    from ..pipeline import select_documents, stable_candidates
    from ..participant2_adapter import adapt_export
    from ..automatic import assess_automatically
    from .deduplicate import deduplicate_and_score
    result = copy.deepcopy(payload)
    query = search_query or payload.get('search_query') or payload['query']
    docs = [SourceDocument.model_validate(d) for d in payload['documents']]
    selected, _ = select_documents(docs, query, as_of, max_age_days)
    selected = deduplicate_and_score(selected)
    candidates = stable_candidates(payload['query'], selected)
    adapted = adapt_export({**payload, 'documents': [d.model_dump(mode='json') for d in selected],
                            'candidates': [c.model_dump(mode='json') for c in candidates]})
    assessment = assess_automatically(adapted['candidates'], as_of, query)
    priorities = {a['candidate_id']: a for a in assessment['assessments']}
    preferred = set()
    for c in adapted['candidates']:
        a = priorities[c['candidate_id']]
        if a['decision'] == 'needs_review' and a['flags'].get('early_stage'):
            preferred.update(s['source_id'] for s in c['sources'])
    # Adapter IDs may be namespaced; use original document identity from metadata.
    preferred_docs = {s['retrieval_metadata']['doc_id'] for c in adapted['candidates']
                      for s in c['sources'] if s['source_id'] in preferred}
    selected.sort(key=lambda d: (d.doc_id not in preferred_docs, d.doc_id))
    seen = {d.url for d in docs}
    attempts = []
    deadline = time.monotonic() + max_seconds
    exhausted = False
    for parent in selected:
        for url, relation in targets(parent):
            if url in seen:
                continue
            if len(attempts) >= max_documents or time.monotonic() >= deadline:
                exhausted = True
                break
            seen.add(url)
            entry = {'url': url, 'parent_doc_id': parent.doc_id, 'relation': relation}
            try:
                response = fetcher.get(url)
                hit = SearchHit(url=url, connector='scholarly_followup', query=query,
                                raw={'discovered_from': parent.url, 'discovery_relation': relation})
                doc = recover_arxiv_date(parse_html_document(response, hit), fetcher)
                if doc.fetch_status.value == 'ok' and doc.text.strip():
                    # Full text can supply a date only via its own publisher metadata.
                    # Never copy the news article's publication date to the paper.
                    result['documents'].append(doc.model_dump(mode='json'))
                    entry.update(status='fetched', doc_id=doc.doc_id)
                else:
                    entry.update(status=doc.fetch_status.value, error=doc.fetch_error or 'No original text')
            except Exception as exc:
                entry.update(status='error', error=str(exc))
            attempts.append(entry)
        if exhausted:
            break
    result.setdefault('summary', {})['enrichment'] = {
        'attempts': attempts, 'fetched': sum(a['status'] == 'fetched' for a in attempts),
        'limit_reached': exhausted, 'max_documents': max_documents, 'max_seconds': max_seconds,
        'independence_verified': False,
    }
    errors = result['summary'].setdefault('errors', [])
    for attempt in attempts:
        if attempt['status'] != 'fetched':
            errors.append({'connector': 'scholarly_followup', 'query': query,
                           'message': attempt['url'] + ': ' + attempt.get('error', attempt['status'])})
    return result
