"""Select attributable topical sentences, without rewriting source evidence."""
import re
from .automatic import words

NOISE = re.compile(r'cookie|privacy policy|sign in|subscribe|all rights reserved|breadcrumb|official websites use|secure .*websites|was this page helpful', re.I)


def select_excerpt(text, title, limit=1500):
    topic = words(title)
    candidates = []
    for match in re.finditer(r'[^\n.!?]+[.!?]?', text):
        quote = match.group().strip()
        if len(quote) < 40 or len(quote) > limit or NOISE.search(quote):
            continue
        overlap = len(words(quote) & topic)
        if overlap < min(2, max(1, len(topic))):
            continue
        start = match.start() + len(match.group()) - len(match.group().lstrip())
        candidates.append((overlap, start, quote))
    chosen = []
    size = 0
    for _, start, quote in sorted(candidates, key=lambda x: (-x[0], x[1])):
        if size + len(quote) <= limit:
            chosen.append({'start': start, 'end': start + len(quote), 'quote': quote})
            size += len(quote)
        if len(chosen) == 2:
            break
    return sorted(chosen, key=lambda x: x['start'])
