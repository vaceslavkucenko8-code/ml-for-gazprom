from bs4 import BeautifulSoup
from src.retrieval.parse import _extract_main_text


def test_nested_containers_do_not_duplicate_article():
    soup=BeautifulSoup('<main><article><p>Actual scientific evidence.</p><article>Nested evidence.</article></article></main>','html.parser')
    text=_extract_main_text(soup)
    assert text.count('Actual scientific evidence.')==1
    assert text.count('Nested evidence.')==1


def test_mit_body_excludes_related_topics():
    soup=BeautifulSoup('<main><article><div class="news-article--content--body--inner"><p>Actual evidence.</p></div><div>Related Topics: photonic computing</div></article></main>','html.parser')
    assert _extract_main_text(soup)=='Actual evidence.'
