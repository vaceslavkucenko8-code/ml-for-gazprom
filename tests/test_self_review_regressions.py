import pytest
from bs4 import BeautifulSoup
from src.automatic import analyse
from src.retrieval.parse import _extract_published_at, parse_html_document, guess_source_type
from src.retrieval.fetch import FetchResult, decode_response
from src.retrieval.connectors.base import SearchHit
from src.retrieval.schemas import FetchStatus, SourceType


def source(title, text, published='2026-06-01'):
    return dict(source_id=title, title_original=title, text=text, published_at=published,
                source_type='research', trust_level='high', trust_reason='Test fixture', url='https://example.org')


def assess(sources, query='quantum sensor'):
    return analyse(dict(candidate_id='c', query=query, sources=sources), '2026-09-24')


def test_unrelated_paper_cannot_supply_missing_positive_features():
    r=assess([source('Quantum sensor', 'The quantum sensor measures magnetic fields.'),
              source('Robotic actuator', 'Our prototype is limited to laboratory experiments.')])
    assert r['decision']=='needs_review'
    assert not r['flags']['early_stage']


def test_old_prototype_does_not_establish_current_novelty():
    r=assess([source('Quantum sensor','This quantum sensor prototype is limited to laboratory experiments.', '2022-01-01')])
    assert r['decision']=='needs_review'
    assert not r['flags']['early_stage']


def test_historical_maturity_is_still_relevant():
    assert assess([source('Quantum sensor','This quantum sensor is widely deployed.', '2022-01-01')])['decision']=='likely_mature'


def test_three_letter_technology_acronym_is_not_discarded():
    assert assess([source('OCS','This OCS prototype is limited to laboratory experiments.')], 'OCS')['decision']=='likely_weak'


@pytest.mark.parametrize('value',['2026','2026-06','June 2026'])
def test_partial_date_does_not_invent_day(value):
    assert _extract_published_at(BeautifulSoup('', 'html.parser'), {'date':value})==(None,False)


@pytest.mark.parametrize('html',[
    '<header><time datetime="2026-06-05">June 5</time></header><article>Research text.</article>',
    '<meta name="citation_publication_date" content="2026/06/05"><article>Research text.</article>',
])
def test_publication_dates_survive_article_extraction(html):
    hit=SearchHit(url='https://arxiv.org/abs/2606.07470',title='Research',connector='arxiv',query='Research')
    d=parse_html_document(FetchResult(url=hit.url,status=FetchStatus.OK,content=html),hit)
    assert d.published_at.date().isoformat()=='2026-06-05'


def test_frontiers_domain_boundary():
    assert guess_source_type('https://www.frontiersin.org/article')[0]==SourceType.ACADEMIC
    assert guess_source_type('https://frontiersin.org.fake.example/article')[0]!=SourceType.ACADEMIC


@pytest.mark.parametrize('encoding,header,meta',[
    ('utf-8','text/html','<meta charset="utf-8">'),
    ('cp1251','text/html; charset=windows-1251',''),
    ('cp1251','text/html','<meta charset="windows-1251">'),
])
def test_russian_html_keeps_original_characters(encoding,header,meta):
    text=meta+'<p>Прототип работает только в лаборатории.</p>'
    assert decode_response(text.encode(encoding),header)==text
