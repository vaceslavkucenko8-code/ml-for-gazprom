from datetime import datetime, timezone
import json
from types import SimpleNamespace
import pytest
from src.pipeline import process_export, select_documents, stable_candidates, topic_match
from src.retrieval.schemas import SourceDocument, SourceType
from src.retrieval.connectors.base import SearchHit
from src.retrieval.parse import parse_api_hit
from src.retrieval.connectors.hackernews import HackerNewsConnector
from src.retrieval.schemas import FetchStatus


def doc(**changes):
    values = dict(doc_id='test-document', url='https://example.org/paper', normalized_url='https://example.org/paper',
        title='Quantum navigation sensor prototype', text='Quantum navigation sensors were tested as a laboratory prototype.',
        published_at=datetime(2026,9,1,tzinfo=timezone.utc), connector='test', query='quantum navigation',
        source_type=SourceType.ACADEMIC, raw_metadata={'text_scope': 'abstract_or_description'})
    return SourceDocument(**(values | changes))


def test_filter_future_old_snippet_and_partial_topic():
    docs=[doc(), doc(doc_id='future', published_at=datetime(2027,1,1,tzinfo=timezone.utc)),
          doc(doc_id='old',published_at=datetime(2015,1,1,tzinfo=timezone.utc)),
          doc(doc_id='snippet',raw_metadata={'text_scope':'search_snippet'}),
          doc(doc_id='off-topic',title='Quantum computing',text='Quantum cloud platform for computations.')]
    selected,excluded=select_documents(docs,'quantum navigation','2026-09-20',540)
    assert [d.doc_id for d in selected]==['test-document']
    assert {e['reason'] for e in excluded}=={'future_publication','outside_time_window','no_source_text','insufficient_topic_overlap'}


def test_ids_stable_between_replays():
    a,b=stable_candidates('quantum navigation',[doc()]),stable_candidates('quantum navigation',[doc()])
    assert a[0].candidate_id==b[0].candidate_id
    assert [e.evidence_id for e in a[0].evidence]==[e.evidence_id for e in b[0].evidence]


def test_api_abstract_tags_removed_and_original_retained():
    hit=SearchHit(url='https://example.org',query='test',connector='test',title='Test',
                  full_text='<jats:p>Quantum <b>sensor</b> prototype.</jats:p>',raw={'date_estimated':True})
    parsed=parse_api_hit(hit,SourceType.ACADEMIC)
    assert '<' not in parsed.text
    assert parsed.raw_metadata['api_original_text']==hit.full_text
    assert parsed.published_at_is_estimated


def test_hn_submission_is_not_article_publication():
    data={'hits':[{'url':'https://example.org/paper','created_at':'2026-09-01T12:00:00Z','title':'Test'}]}
    fetcher=SimpleNamespace(get=lambda *a,**kw:SimpleNamespace(status=FetchStatus.OK,content=json.dumps(data)))
    hit=HackerNewsConnector(fetcher).search('test')[0]
    assert hit.published_at is None
    assert hit.raw['discovered_at']=='2026-09-01T12:00:00Z'


def test_report_pipeline_replay_and_no_overwrite(tmp_path):
    payload={'query':'quantum navigation','synthetic':True,'documents':[doc().model_dump(mode='json')]}
    out=tmp_path/'run'
    summary=process_export(payload,'2026-09-20',out)
    assert summary['counts']=={'weak':0,'reject':0,'needs_review':1}
    assert (out/'report.html').exists()
    assert (out/'manifest.json').exists()
    with pytest.raises(FileExistsError):
        process_export(payload,'2026-09-20',out)


def test_report_escapes_external_html(tmp_path):
    payload={'query':'quantum navigation','documents':[doc(title='<script>alert(1)</script> quantum navigation').model_dump(mode='json')]}
    process_export(payload,'2026-09-20',tmp_path/'run')
    report=(tmp_path/'run/report.html').read_text(encoding='utf-8')
    assert '<script>' not in report
    assert '&lt;script&gt;' in report


def test_distinct_quantum_studies_not_merged():
    from src.extraction.candidates import cluster_documents_into_candidates
    a=doc(title='Advancing Interplanetary Navigation with Quantum Inertial Sensors')
    b=doc(doc_id='b',title='GHZ-state Ramsey Interferometry on PKTron and IBM Quantum Hardware: Toward Quantum-enhanced GPS-denied Navigation Sensors')
    assert len(cluster_documents_into_candidates([a,b]))==2


def test_full_page_not_mislabeled_as_search_snippet():
    from src.retrieval.parse import parse_html_document
    from src.retrieval.fetch import FetchResult
    hit=SearchHit(url='https://example.org',query='test',connector='hackernews',title='Test',raw={'text_scope':'search_snippet'})
    fetched=FetchResult(url=hit.url,status=FetchStatus.OK,content='<html><title>Test</title><article><p>'+('A quantum sensor prototype was tested. '*20)+'</p></article></html>')
    assert parse_html_document(fetched,hit).raw_metadata['text_scope']=='full_page'


def test_substring_domain_does_not_grant_source_trust():
    from src.retrieval.parse import guess_source_type
    assert guess_source_type('https://notarxiv.org/paper')[0] != SourceType.ACADEMIC
    assert guess_source_type('https://arxiv.org.other.example/paper')[0] != SourceType.ACADEMIC
    assert guess_source_type('https://www.arxiv.org/paper')[0] == SourceType.ACADEMIC


def test_persistence_failure_not_reported_as_completed(tmp_path,monkeypatch):
    from src import storage
    from src.local_app import Application
    def fail(*args,**kwargs):raise RuntimeError('Database unavailable')
    monkeypatch.setattr(storage,'persist_run',fail)
    payload={'query':'quantum navigation','documents':[doc().model_dump(mode='json')]}
    with pytest.raises(RuntimeError):process_export(payload,'2026-09-20',tmp_path/'failed-run')
    app=Application(tmp_path)
    assert app.list_runs()[0]['status']=='failed'
    app.pool.shutdown()
