"""Restore the normal source-policy stage when evaluating raw source caches."""
import copy
import json
from pathlib import Path
from bs4 import BeautifulSoup
from .retrieval.schemas import SourceDocument
from .retrieval.parse import guess_source_type, _extract_published_at
from .retrieval.deduplicate import assign_trust
from .participant2_adapter import TYPE_MAP


def prepare(root=Path('.')):
    corpus=copy.deepcopy(json.loads((root/'data/self_review/corpus.json').read_text(encoding='utf-8')))
    for case in corpus['cases']:
        for source in case['candidate']['sources']:
            doc=SourceDocument.model_validate(source['retrieval_metadata'])
            kind,confidence,reason=guess_source_type(doc.url,doc.connector)
            if confidence>(doc.source_type_confidence or 0):
                doc.source_type,doc.source_type_confidence,doc.source_type_rationale=kind,confidence,reason
            if doc.published_at is None:
                doc.published_at,_=_extract_published_at(BeautifulSoup('','html.parser'),doc.raw_metadata.get('meta',{}))
            assign_trust(doc,corroborating_domains=0)
            source.update(source_type=TYPE_MAP.get(doc.source_type.value,'unknown'),
                published_at=doc.published_at.date().isoformat() if doc.published_at and not doc.published_at_is_estimated else None,
                trust_level=doc.trust_level.value,trust_reason=doc.trust_explanation,
                retrieval_metadata=doc.model_dump(mode='json'))
    corpus['preparation']='Source policy applied without invented corroboration; recover dates only from cached publisher metadata'
    (root/'data/self_review/corpus_prepared.json').write_text(json.dumps(corpus,ensure_ascii=False,indent=2),encoding='utf-8')
    return corpus


if __name__=='__main__':prepare()
