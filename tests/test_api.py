from fastapi.testclient import TestClient
from src.api import create_app


def test_api_health_and_no_unapproved_writes(tmp_path,monkeypatch):
    monkeypatch.delenv('DATABASE_URL',raising=False)
    app=create_app(tmp_path)
    with TestClient(app) as client:
        assert client.get('/health').json()=={'storage':'files','ok':True}
        assert client.get('/').status_code==200
        assert client.post('/api/search',json={'query':'test'}).status_code==403
        assert client.get('/api/runs').json()==[]
        assert client.get('/runs/no/something.secret').status_code==404


def test_api_validates_query_and_dispatches(tmp_path,monkeypatch):
    app=create_app(tmp_path)
    called=[]
    monkeypatch.setattr(app.state.application.pool,'submit',lambda *args:called.append(args))
    with TestClient(app) as client:
        headers={'X-Local-Token':app.state.application.token}
        assert client.post('/api/search',headers=headers,json={'query':''}).status_code==400
        response=client.post('/api/search',headers=headers,json={'query':'quantum sensors'})
        assert response.status_code==202 and response.json()['run_id']
        assert len(called)==1
