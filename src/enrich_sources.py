"""Resumeable retrieval of workbook citations without inferring class labels."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from .retrieval.fetch import Fetcher, _DomainThrottle
from .retrieval.connectors.base import SearchHit
from .retrieval.parse import parse_html_document
from .retrieval.schemas import FetchStatus


def cache_key(url):
    return hashlib.sha256(url.encode('utf-8')).hexdigest()


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def collect_one(link,cache,throttle,retry_failed=False,fetcher_factory=Fetcher):
    url=link['url'];path=cache/f'{cache_key(url)}.json'
    if path.exists():
        stored=json.loads(path.read_text(encoding='utf-8'))
        if not retry_failed or stored['status']=='retrieved_unverified':
            return stored,True
    fetched_at=datetime.now(timezone.utc).isoformat()
    record={'url':url,'retrieved_at':fetched_at,'status':'error','document':None,'error':None}
    try:
        parts=urlsplit(url)
        if parts.scheme not in ('http','https') or not parts.hostname:
            raise ValueError('Citation must use HTTP(S)')
        fetcher=fetcher_factory(max_retries=0,connect_timeout=4,read_timeout=10)
        fetcher.throttle=throttle
        response=fetcher.get(url)
        record.update(http_status=response.status_code,final_url=response.final_url,content_type=response.content_type)
        if response.status!=FetchStatus.OK:
            record.update(status=response.status.value,error=response.error)
        elif any(t in (response.content_type or '').lower() for t in ('application/pdf','image/','application/octet-stream')):
            record.update(status='unsupported_content',error='Binary resource requires separate extraction; not interpreted as HTML')
        else:
            doc=parse_html_document(response,SearchHit(url=url,title=link.get('title_raw'),connector='workbook-citation',query=''))
            record['document']=doc.model_dump(mode='json')
            record['text_sha256']=hashlib.sha256(doc.text.encode('utf-8')).hexdigest()
            record['status']='retrieved_unverified' if len(doc.text.strip())>=300 else 'insufficient_text'
            record['quality_warnings']=[]
            if not doc.published_at:record['quality_warnings'].append('publication_date_missing')
            if doc.published_at_is_estimated:record['quality_warnings'].append('publication_date_estimated')
            if doc.source_type_confidence is None or doc.source_type_confidence<0.6:record['quality_warnings'].append('source_type_requires_review')
            if any(s in doc.text.casefold() for s in ('verify you are human','just a moment','access denied','enable javascript and cookies')):
                record['status']='possible_challenge_page'
                record['quality_warnings'].append('content_may_be_antibot_page')
    except Exception as exc:
        record.update(status='error',error=str(exc))
    # Preserve the previous failed attempt before an explicitly requested retry.
    if retry_failed and path.exists():
        history=cache/'history'/f'{path.stem}-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")}.json'
        history.parent.mkdir(exist_ok=True)
        history.write_bytes(path.read_bytes())
    atomic_json(path,record)
    return record,False


def enrich(rows,output,workers=4,retry_failed=False,fetcher_factory=Fetcher):
    output=Path(output);cache=output/'cache';cache.mkdir(parents=True,exist_ok=True)
    unique={link['url']:link for row in rows for link in row['sources']}
    records={};cached=0;throttle=_DomainThrottle(1.5)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(collect_one,link,cache,throttle,retry_failed,fetcher_factory):url for url,link in unique.items()}
        for future in as_completed(futures):
            record,was_cached=future.result();records[record['url']]=record;cached+=int(was_cached)
            if len(records)%20==0 or len(records)==len(unique):
                print(f'Retrieved/cached {len(records)}/{len(unique)}',flush=True)
    statuses=Counter(r['status'] for r in records.values())
    enriched=[]
    for row in rows:
        links=[{'url':link['url'],'cache_file':f'cache/{cache_key(link["url"])}.json',
                'status':records[link['url']]['status']} for link in row['sources']]
        enriched.append({'signal_id':row['signal_id'],'technology':row['technology'],'domain':row['domain'],
                         'sources':links,'binary_label':None,'semantic_review_status':'pending'})
    report={'records':len(rows),'unique_urls':len(unique),'cached_urls':cached,'statuses':dict(statuses),
            'signals_with_retrieved_text':sum(any(s['status']=='retrieved_unverified' for s in r['sources']) for r in enriched),
            'signals_without_retrieved_text':[r['signal_id'] for r in enriched if not any(s['status']=='retrieved_unverified' for s in r['sources'])],
            'limitations':['HTTP success and text length do not prove authenticity, relevance or validity of workbook claims.',
                           'No score/stage/rationale was converted into evidence or binary labels.']}
    atomic_json(output/'linked_sources.json',enriched)
    atomic_json(output/'summary.json',report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=Path('data/processed/signals_clean.json'))
    parser.add_argument('--output',type=Path,default=Path('data/enrichment'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args()
    if not 1<=args.workers<=8:parser.error('workers must be 1..8')
    report=enrich(json.loads(args.input.read_text(encoding='utf-8')),args.output,args.workers,args.retry_failed)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
