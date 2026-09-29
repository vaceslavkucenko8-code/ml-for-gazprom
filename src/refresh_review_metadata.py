"""Save a separate dated development corpus; original snapshots stay intact."""
import json
from pathlib import Path
from .retrieval.schemas import SourceDocument
from .retrieval.fetch import Fetcher
from .retrieval.publication_metadata import recover_arxiv_date
from .automatic import analyse


def main():
    source=Path('data/self_review/corpus_prepared.json')
    corpus=json.loads(source.read_text(encoding='utf-8'))
    output=Path('artifacts/metadata-recovery')
    output.mkdir(exist_ok=False)
    fetcher=Fetcher(max_retries=0,connect_timeout=5,read_timeout=12,min_delay_per_domain=3)
    recovered=[];cache={}
    for case in corpus['cases']:
        for s in case['candidate']['sources']:
            if s.get('published_at'):
                continue
            doc=SourceDocument.model_validate(s['retrieval_metadata'])
            if doc.url not in cache:
                cache[doc.url]=recover_arxiv_date(doc,fetcher)
            updated=cache[doc.url]
            if updated.raw_metadata.get('publication_date_recovery'):
                s['published_at']=updated.published_at.date().isoformat()
                s['retrieval_metadata']=updated.model_dump(mode='json')
                recovered.append({'case_id':case['case_id'],'url':doc.url,'date':s['published_at']})
    assessments=[analyse(c['candidate'],c['as_of']) for c in corpus['cases']]
    for name,value in [('corpus.json',corpus),('assessments.json',assessments),('recovered.json',recovered)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    from collections import Counter
    print(json.dumps({'recovered':len(recovered),'decisions':dict(Counter(r['decision'] for r in assessments))}))


if __name__=='__main__':main()
