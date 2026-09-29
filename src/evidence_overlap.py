"""Conservative lexical repetition checks, not semantic truth or provenance."""
import re
import unicodedata
from itertools import combinations


def tokens(text):
    return re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold())


def similarity(left, right):
    a, b = tokens(left), tokens(right)
    if min(len(a), len(b)) < 8:
        return 0.0
    if a == b:
        return 1.0
    # Containment catches a copied quote embedded in a longer paragraph.
    x = {tuple(a[i:i+4]) for i in range(len(a)-3)}
    y = {tuple(b[i:i+4]) for i in range(len(b)-3)}
    return len(x & y) / min(len(x), len(y))


def repetition(sources, claims):
    pairs = []
    for left, right in combinations(sorted(sources), 2):
        score = similarity(sources[left].get('text', ''), sources[right].get('text', ''))
        if score >= 0.8:
            pairs.append({'source_ids': [left, right], 'kind': 'text_overlap',
                          'similarity': round(score, 3), 'feature': None})
        for a in claims:
            if a['source_id'] != left:
                continue
            for b in claims:
                if b['source_id'] != right or a['feature'] != b['feature']:
                    continue
                score = similarity(a['quote'], b['quote'])
                if score >= 0.8:
                    item = {'source_ids': [left, right], 'kind': 'quote_overlap',
                            'similarity': round(score, 3), 'feature': a['feature']}
                    if item not in pairs:
                        pairs.append(item)
    return pairs


def independent_groups(ids, sources, pairs, feature):
    """Collapse both reviewed common origin and detected lexical repetition."""
    parent = {sid: sid for sid in ids}
    def root(sid):
        while parent[sid] != sid:
            sid = parent[sid]
        return sid
    for left, right in combinations(sorted(ids), 2):
        same_origin = sources[left]['independence_group'] == sources[right]['independence_group']
        overlap = any(p['source_ids'] == [left, right] and
                      p['feature'] in (None, feature) for p in pairs)
        if same_origin or overlap:
            parent[root(right)] = root(left)
    groups = {}
    for sid in sorted(ids):
        groups.setdefault(root(sid), []).append(sid)
    return sorted(groups.values())


def independent_count(ids, sources, pairs, feature):
    return len(independent_groups(ids, sources, pairs, feature))
