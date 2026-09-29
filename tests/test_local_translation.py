from pathlib import Path
import pytest
from src.local_translation import translate, localize, MODEL
from src.presentation_view import make_card
from test_presentation_view import fixture

@pytest.mark.skipif(not (MODEL/'model/model.bin').exists(),reason='Optional translation model not installed')
def test_local_translation_preserves_original_and_labels_machine_output():
    c,a=fixture();before=c['sources'][0]['text'];card=localize(make_card(c,a,1))
    assert card['sources'][0]['text']==before
    assert card['title_original']==c['name_ru']
    assert 'Автоматический перевод' in card['title_kind']
    assert card['sources'][0]['display_translation_model']
    assert any('а'<=ch.lower()<='я' for ch in card['description_ru'])
    assert card['assessment']['score']==a['score']
    assert card['cases'][0]['verified_case'] is False


def test_unsupported_language_never_claims_translation():
    assert translate('Bonjour','fr') is None
