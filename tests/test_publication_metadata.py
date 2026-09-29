from unittest.mock import Mock
from types import SimpleNamespace
from src.retrieval.schemas import SourceDocument
from src.retrieval.publication_metadata import recover_arxiv_date


def test_date_from_matching_publisher_metadata_preserves_text():
    d=SourceDocument(doc_id='x',url='https://arxiv.org/html/2606.17200v1',normalized_url='https://arxiv.org/html/2606.17200v1',title='A sensor',text='Original text')
    f=Mock();f.get.return_value=SimpleNamespace(status=SimpleNamespace(value='ok'),content='<meta name="citation_title" content="A sensor"><meta name="citation_date" content="2026/06/15">')
    r=recover_arxiv_date(d,f)
    assert r.published_at.date().isoformat()=='2026-06-15'
    assert r.text=='Original text' and not r.published_at_is_estimated
    assert f.get.call_args.args[0]=='https://arxiv.org/abs/2606.17200v1'


def test_mismatching_title_does_not_create_date():
    d=SourceDocument(doc_id='x',url='https://arxiv.org/html/2606.17200v1',normalized_url='https://arxiv.org/html/2606.17200v1',title='A sensor')
    f=Mock();f.get.return_value=SimpleNamespace(status=SimpleNamespace(value='ok'),content='<meta name="citation_title" content="Other paper"><meta name="citation_date" content="2026/06/15">')
    assert recover_arxiv_date(d,f).published_at is None
