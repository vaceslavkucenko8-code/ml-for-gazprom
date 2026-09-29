"""Local English-to-Russian translation for display only. Never changes evidence."""
from functools import lru_cache
import logging
from pathlib import Path
import re
import threading
from .display_excerpt import select_excerpt

MODEL=Path(__file__).resolve().parents[1]/'models/translation-en-ru'
MODEL_ID='argos-en-ru-1.9-ctranslate2-int8'
_lock=threading.Lock()

@lru_cache(maxsize=1)
def engine():
    if not (MODEL/'model/model.bin').exists():return None
    try:
        import ctranslate2
        import sentencepiece
        return (sentencepiece.SentencePieceProcessor(model_file=str(MODEL/'sentencepiece.model')),
                ctranslate2.Translator(str(MODEL/'model'),device='cpu',compute_type='int8',inter_threads=1,intra_threads=2))
    except (ImportError,RuntimeError,OSError):
        logging.exception('Local translation unavailable')
        return None

@lru_cache(maxsize=4096)
def translate(text,language='en'):
    if not text or language!='en':return None
    e=engine()
    if e is None:return None
    tokenizer,translator=e
    # Translate complete sentences; cap the display excerpt, not the saved source.
    chunks=re.split(r'(?<=[.!?])\s+',text)
    selected=[]
    for chunk in chunks:
        if sum(map(len,selected))+len(chunk)>1800 and selected:break
        selected.append(chunk[:1800])
    try:
        with _lock:
            results=translator.translate_batch([tokenizer.encode(x,out_type=str) for x in selected],beam_size=2,max_decoding_length=512)
        return ' '.join(tokenizer.decode(r.hypotheses[0]) for r in results)
    except (RuntimeError,ValueError):
        logging.exception('Local translation failed')
        return None


def localize(card):
    original=card['sources'][0] if card['sources'] else None
    language=original.get('language') if original else None
    title=translate(card['title_original'],language)
    if title:
        card['title_ru']=title
        card['title_kind']='Автоматический перевод заголовка · '+MODEL_ID
    for source in card['sources']:
        text=source.get('text','')
        spans=select_excerpt(text,source.get('title_original',''))
        excerpt=' '.join(s['quote'] for s in spans)
        source['excerpt_spans']=spans
        translated=translate(excerpt,source.get('language'))
        source['excerpt_original']=excerpt
        source['excerpt_ru']=translated if translated else (excerpt if source.get('language')=='ru' else None)
        source['display_translation_model']=MODEL_ID if translated else None
    if original and original.get('excerpt_ru'):
        card['description_ru']=original['excerpt_ru']
        card['description_source_id']=original['source_id']
        card['description_kind']=('Автоматический перевод тематических фрагментов источника · '+MODEL_ID
                                  if original.get('display_translation_model') else 'Фрагмент оригинального русского текста')
    sources={s['source_id']:s for s in card['sources']}
    for item in card['advantages']+card['cases']+card['claims']:
        src=sources.get(item['source_id'],{})
        translated=translate(item['quote'],src.get('language'))
        item['quote_ru']=translated if translated else (item['quote'] if src.get('language')=='ru' else None)
        item['display_translation_model']=MODEL_ID if translated else None
    card['translation_available']=engine() is not None
    return card
