"""Document identity and citation hints; citations do not prove dependence."""
import re
from itertools import combinations
from urllib.parse import unquote, urlsplit


def work_id(value):
    value = unquote(str(value or '')).strip()
    if value.lower().startswith('10.'):
        return 'doi:' + value.lower().rstrip(' .') if re.fullmatch(r'10\.\d{4,9}/\S+', value) else None
    try:
        url = urlsplit(value)
    except ValueError:
        return None
    host = (url.hostname or '').lower()
    if host in ('doi.org', 'dx.doi.org'):
        return work_id(url.path.lstrip('/'))
    if host in ('arxiv.org', 'www.arxiv.org'):
        match = re.fullmatch(r'/(?:abs|html|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?/?', url.path)
        if match:
            return 'arxiv:' + match[1]
    return None


def identity(source):
    raw = source.get('retrieval_metadata', {}).get('raw_metadata', {})
    # Own citation metadata is distinct from outgoing bibliography links.
    values = [source.get('url'), raw.get('doi'), raw.get('meta', {}).get('citation_doi')]
    return {key for value in values if (key := work_id(value))}


def relations(sources):
    result = []
    for left, right in combinations(sorted(sources), 2):
        own_a, own_b = identity(sources[left]), identity(sources[right])
        same = own_a & own_b
        if same:
            result.append({'source_ids': [left, right], 'kind': 'same_work', 'work_ids': sorted(same)})
        cited = []
        for sid in (left, right):
            raw = sources[sid].get('retrieval_metadata', {}).get('raw_metadata', {})
            cited.append({key for link in raw.get('research_links', []) if (key := work_id(link))})
        common = (cited[0] & cited[1]) | (cited[0] & own_b) | (cited[1] & own_a)
        if common - same:
            result.append({'source_ids': [left, right], 'kind': 'shared_reference', 'work_ids': sorted(common - same)})
    return result
