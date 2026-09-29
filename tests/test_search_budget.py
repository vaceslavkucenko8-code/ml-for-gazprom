from types import SimpleNamespace
from src.retrieval.search import SearchOrchestrator
from src.retrieval.connectors.base import SearchHit
from src.local_app import Application
from src.pipeline import topic_match
import pytest


def test_topic_inflections_do_not_count_as_multiple_query_terms():
    assert topic_match('A photonic chip can compute with light.', 'photonic computing')
    assert topic_match('Photonic computers use optical signals.', 'Фотонные вычисления')
    assert not topic_match('Computers compute computations.', 'photonic computing')


def test_budget_preserves_collected_documents(monkeypatch):
    clock = [0]
    monkeypatch.setattr('src.retrieval.search.time.monotonic', lambda: clock[0])
    connector = SimpleNamespace(name='test', is_available=lambda: True,
        search=lambda *a, **k: [SearchHit(url='https://example.org/a'), SearchHit(url='https://example.org/b')])
    search = SearchOrchestrator(connectors=[connector])
    doc = SimpleNamespace(fetch_status=SimpleNamespace(value='ok'), title='quantum sensor', text='quantum sensor')
    def read(*args):
        clock[0] += 11
        return doc
    monkeypatch.setattr(search, '_to_document', read)
    result = search.run('quantum sensor', max_seconds=10)
    assert result.documents == [doc]
    assert result.budget_exhausted
    assert len(result.queries_attempted) == 1


def test_all_sources_get_base_query_before_variants():
    calls = []
    def connector(name):
        def find(query, **kwargs):
            calls.append((name, query))
            return []
        return SimpleNamespace(name=name, is_available=lambda: True, search=find)
    run = SearchOrchestrator(connectors=[connector('a'), connector('b')]).run('quantum sensor', 2)
    assert calls[:2] == [('a', 'quantum sensor'), ('b', 'quantum sensor')]
    assert len(calls) == 4 and not run.budget_exhausted


def test_extended_search_dispatch_and_validation(tmp_path, monkeypatch):
    app = Application(tmp_path)
    calls = []
    monkeypatch.setattr(app.pool, 'submit', lambda *args: calls.append(args))
    with pytest.raises(ValueError):
        app.submit({'query': 'quantum sensor', 'search_mode': 'unbounded'})
    rid = app.submit({'query': 'quantum sensor'})
    assert calls[0][-1] == 'extended'
    command = []
    def execute(args, **kwargs):
        command.extend(args)
        assert kwargs['timeout'] == 600
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('src.local_app.subprocess.run', execute)
    app._search(rid, 'quantum sensor', '', '2026-09-28', 'extended')
    assert command[command.index('--results-per-connector') + 1] == '20'
    assert command[command.index('--max-queries') + 1] == '3'
    assert command[command.index('--search-budget-seconds') + 1] == '360'
    assert '--enrich-sources' in command
    app.pool.shutdown()
