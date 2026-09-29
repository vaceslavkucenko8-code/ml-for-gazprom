"""PostgreSQL stores immutable run snapshots, raw documents and analyses."""
import json
import os
from pathlib import Path

DDL='''
CREATE TABLE IF NOT EXISTS signal_runs (
  run_id text PRIMARY KEY, query text NOT NULL, as_of date NOT NULL,
  summary jsonb NOT NULL, result jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS signal_documents (
  run_id text REFERENCES signal_runs(run_id) ON DELETE CASCADE,
  doc_id text NOT NULL, payload jsonb NOT NULL, PRIMARY KEY(run_id,doc_id)
);
CREATE TABLE IF NOT EXISTS signal_candidates (
  run_id text REFERENCES signal_runs(run_id) ON DELETE CASCADE,
  candidate_id text NOT NULL, payload jsonb NOT NULL, PRIMARY KEY(run_id,candidate_id)
);
CREATE TABLE IF NOT EXISTS signal_reviews (
  run_id text REFERENCES signal_runs(run_id) ON DELETE CASCADE,
  position integer NOT NULL, payload jsonb NOT NULL, PRIMARY KEY(run_id,position)
);
'''


def persist_run(folder,database_url=None):
    database_url=database_url or os.environ.get('DATABASE_URL')
    if not database_url:return {'storage':'files'}
    import psycopg
    from psycopg.types.json import Jsonb
    folder=Path(folder)
    read=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
    summary=read('summary.json');result=read('result.json')
    with psycopg.connect(database_url,connect_timeout=10) as conn:
        conn.execute(DDL)
        existing=conn.execute('SELECT result FROM signal_runs WHERE run_id=%s',(folder.name,)).fetchone()
        if existing:
            if existing[0]!=result:raise ValueError('Refusing to overwrite a different run in PostgreSQL')
            return {'storage':'postgresql','already_saved':True}
        conn.execute('INSERT INTO signal_runs(run_id,query,as_of,summary,result) VALUES (%s,%s,%s,%s,%s)',
                     (folder.name,summary['query'],summary['as_of'],Jsonb(summary),Jsonb(result)))
        docs={d['doc_id']:d for d in read('retrieved.json')['documents']}
        for doc in docs.values():
            conn.execute('INSERT INTO signal_documents VALUES (%s,%s,%s)',(folder.name,doc['doc_id'],Jsonb(doc)))
        for c in result['candidates']:
            conn.execute('INSERT INTO signal_candidates VALUES (%s,%s,%s)',(folder.name,c['candidate_id'],Jsonb(c)))
        for i,r in enumerate(read('reviews.json')):
            conn.execute('INSERT INTO signal_reviews VALUES (%s,%s,%s)',(folder.name,i,Jsonb(r)))
    return {'storage':'postgresql','already_saved':False}


def database_health(database_url=None):
    url=database_url or os.environ.get('DATABASE_URL')
    if not url:return {'storage':'files','ok':True}
    import psycopg
    try:
        with psycopg.connect(url,connect_timeout=3) as conn:conn.execute('SELECT 1')
        return {'storage':'postgresql','ok':True}
    except Exception:
        return {'storage':'postgresql','ok':False}
