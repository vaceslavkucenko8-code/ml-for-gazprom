import json
from types import SimpleNamespace
from test_pipeline import doc
from src.continuous_ingestion import run_cycle
from src.corpus_store import CorpusStore


def test_topic_and_updated_source_reach_detector(tmp_path):
    document=doc(text='This quantum navigation prototype is limited to laboratory experiments.')
    class Search:
        def run(self,*args,**kwargs):
            return SimpleNamespace(documents=[document],errors=[],connectors_used=['fixture'],connectors_skipped=[])
    with CorpusStore(tmp_path/'corpus.db') as store:
        def cycle():
            return run_cycle(store,Search(),seed_topics={'test':'quantum navigation'},max_seed_topics=1,max_adaptive_topics=0,as_of='2026-09-27')
        first=cycle()
        assert first['status']=='ok'
        assert store.candidates()[0]['automatic_decision']=='likely_weak'
        document.text='This quantum navigation system is widely deployed in industry.'
        second=cycle()
        assert second['status']=='ok' and second['counts']['new_candidates']==0
        assert store.candidates()[0]['automatic_decision']=='likely_mature'
        current=store.conn.execute('SELECT last_seen_cycle,payload_json FROM documents').fetchone()
        assert current['last_seen_cycle']==second['cycle_id']
        assert json.loads(current['payload_json'])['text']==document.text
        previous=store.conn.execute('SELECT payload_json FROM document_versions').fetchone()
        assert 'limited to laboratory' in previous[0]
