import json
from pathlib import Path
import pytest
from src.local_app import Application
from src.pipeline import process_export
from test_pipeline import doc


def test_review_creates_new_version_and_preserves_original(tmp_path):
    app=Application(tmp_path)
    payload={'query':'quantum navigation','synthetic':True,'documents':[doc().model_dump(mode='json')]}
    process_export(payload,'2026-09-20',tmp_path/'first')
    before=(tmp_path/'first/assessment.json').read_bytes()
    c=json.loads((tmp_path/'first/detector_input.json').read_text(encoding='utf-8'))['candidates'][0]
    e=c['extracted_evidence_pending'][0]
    reviews=[{'candidate_id':c['candidate_id'],'evidence_id':e['evidence_id'],
              'feature':f,'reviewed':True,'scope_matches_candidate':True,'event_status':'observed',
              'observed_at':'2026-09-01','reviewer':'automated test','rationale':'Synthetic software fixture',
              'query':c['query']} for f in ('early_stage','limited_adoption','relevance')]
    rid=app.review({'run_id':'first','reviews':reviews})
    result=json.loads((tmp_path/rid/'assessment.json').read_text(encoding='utf-8'))
    assert result['counts']['weak']==1
    assert (tmp_path/'first/assessment.json').read_bytes()==before
    assert len(app.list_runs())==2
    app.pool.shutdown()


def test_invalid_path_and_unconfirmed_review_rejected(tmp_path):
    app=Application(tmp_path)
    with pytest.raises(ValueError):
        app.folder('../outside')
    with pytest.raises(ValueError):
        app.submit({'query':''})
    app.pool.shutdown()


def test_interrupted_job_survives_restart_as_failed(tmp_path):
    app=Application(tmp_path)
    app.jobs['interrupted']={'run_id':'interrupted','status':'running','query':'quantum navigation','deadline_at':0}
    app._save_job('interrupted')
    app.pool.shutdown()
    restarted=Application(tmp_path)
    assert restarted.list_runs()[0]['status']=='failed'
    assert 'остановлен' in restarted.list_runs()[0]['error']
    assert json.loads((tmp_path/'.jobs/interrupted.json').read_text(encoding='utf-8'))['status']=='failed'
    restarted.pool.shutdown()


def test_failed_job_survives_restart(tmp_path):
    app=Application(tmp_path)
    app.jobs['failed']={'run_id':'failed','status':'failed','error':'network unavailable'}
    app._save_job('failed');app.pool.shutdown()
    restarted=Application(tmp_path)
    assert restarted.list_runs()[0]['error']=='network unavailable'
    restarted.pool.shutdown()


def test_second_reader_preserves_active_worker_and_sees_completion(tmp_path):
    import time
    worker = Application(tmp_path)
    worker.jobs['live'] = {'run_id': 'live', 'status': 'running', 'deadline_at': time.time() + 660}
    worker._save_job('live')
    reader = Application(tmp_path)
    assert reader.list_runs()[0]['status'] == 'running'
    worker.jobs['live']['status'] = 'done'
    worker._save_job('live')
    assert reader.list_runs()[0]['status'] == 'done'
    worker.pool.shutdown()
    reader.pool.shutdown()
