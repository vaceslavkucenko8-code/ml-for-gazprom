import json
from src.presentation_view import make_card, load_view, safe_url
from src.api import create_app
from fastapi.testclient import TestClient


def fixture():
    s={'source_id':'s','url':'https://example.org/paper','title_original':'A sensor prototype',
       'text':'Our sensor prototype is limited to laboratory testing.','language':'en',
       'source_type':'research','trust_level':'high','retrieval_metadata':{}}
    c={'candidate_id':'c','name_ru':s['title_original'],'query':'Датчики','sources':[s],
       'participant2_candidate':{'evidence':[{'evidence_id':'e','doc_id':'s','quote':s['text'],'is_future_looking':True}],
        'case_examples':[{'text':'Unsupported business case','evidence_ids':['e']}],
        'advantages':[{'text':'Invented advantage','evidence_ids':['absent']}]}}
    a={'candidate_id':'c','decision':'needs_review','reason':'insufficient_explicit_evidence','score':30,
       'flags':{'early_stage':True},'claims':[{'evidence_id':'e','source_id':'s','feature':'early_stage','quote':s['text']}]}
    return c,a


def test_card_never_invents_cases_or_probability():
    c,a=fixture(); card=make_card(c,a,1)
    assert card['advantages']==[]
    assert card['cases'][0]['verified_case'] is False
    assert card['cases'][0]['status_ru']=='План или предположение'
    assert 'Unsupported business case' not in json.dumps(card)
    assert sum(p['contribution'] for p in card['score_parts'])==30
    assert card['claims'][0]['quote'] in c['sources'][0]['text']
    assert 'confidence' not in card


def test_unmatched_quotes_and_unsafe_links():
    c,a=fixture();a['claims'][0]['quote']='fabricated';c['sources'][0]['url']='javascript:alert(1)'
    card=make_card(c,a,1)
    assert card['claims']==[] and card['unmatched_claims']==1
    assert card['sources'][0]['url'] is None and card['cases']==[]
    assert safe_url('data:text/html,a') is None


def test_translation_provenance():
    c,a=fixture();c['sources'][0]['retrieval_metadata']={'is_translated':True,'is_generated_summary':True}
    source=make_card(c,a,1)['sources'][0]
    assert source['auto_translated'] and source['generated_summary']


def test_api_uses_final_order_and_supports_missing_and_empty(tmp_path):
    c,a=fixture();folder=tmp_path/'run';folder.mkdir()
    candidates=[{**c,'candidate_id':f'c{i}'} for i in range(20)]
    assessments=[{**a,'candidate_id':f'c{i}'} for i in range(20)]
    for name,data in {'automatic.json':{'assessments':assessments},'detector_input.json':{'candidates':candidates},
                      'summary.json':{'query':'Датчики','as_of':'2026-09-23'},
                      'result.json':{'results':list(reversed(assessments))}}.items():
        (folder/name).write_text(json.dumps(data))
    with TestClient(create_app(tmp_path)) as client:
        r=client.get('/api/runs/run/view');assert r.status_code==200
        assert r.json()['top_ids']==[f'c{i}' for i in range(19,4,-1)]
        assert client.get('/api/runs/missing/view').status_code==404
        assert client.get('/review').status_code==200
    (folder/'result.json').write_text('{"results":[]}')
    assert load_view(folder)['top_ids']==[]

def test_fast_view_skips_translation_and_preserves_assessment(tmp_path, monkeypatch):
    from src import local_translation
    c,a=fixture()
    for name,data in {'automatic.json':{'assessments':[a]},'detector_input.json':{'candidates':[c]},'summary.json':{'query':'Датчики','as_of':'2026-09-28'},'application_review.json':{'as_of':'2026-09-28','items':[{'candidate_id':'c','demonstrated':'Лабораторный опыт','independent_label':False}]}}.items():
        (tmp_path/name).write_text(json.dumps(data),encoding='utf-8')
    def forbidden(card):
        raise AssertionError('List must not invoke translation')
    monkeypatch.setattr(local_translation,'localize',forbidden)
    view=load_view(tmp_path,translate_cards=False)
    assert view['cards'][0]['assessment']==a
    assert view['cards'][0]['application_review']['independent_label'] is False
    assert load_view(tmp_path,translate_cards=False,candidate_id='absent')['cards']==[]
