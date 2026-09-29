"""Small explicit bilingual vocabulary; expands topic matching, never evidence."""
import re

# Only lexical equivalents. No technology labels or stage assumptions.
EQUIVALENTS = {
    'роев': ('swarm',), 'робототех': ('robotic',),
    'квантов': ('quantum',), 'сенсор': ('sensor',), 'датчик': ('sensor',),
    'фотон': ('photonic',), 'оптическ': ('optical',), 'гидрофон': ('hydrophone',),
    'криптограф': ('cryptography',), 'постквантов': ('postquantum',),
    'федеративн': ('federated',), 'обучени': ('learning',),
    'мошеннич': ('fraud',), 'сжат': ('compression',), 'модел': ('model',),
    'конфиденциальн': ('confidential',), 'вычислен': ('computing',),
    'спекулятивн': ('speculative',), 'декодирован': ('decoding',),
    'нейроморф': ('neuromorphic',), 'биосенсор': ('biosensor',),
    'аккумулятор': ('battery',), 'батаре': ('battery',), 'робот': ('robot',),
    'навигаци': ('navigation',), 'финансов': ('financial',),
    'обнаружени': ('detection',), 'магнит': ('magnetic',),
}


def expand(tokens):
    result = set(tokens)
    for token in tokens:
        if re.search('[а-я]', token):
            for stem, equivalents in EQUIVALENTS.items():
                if token.startswith(stem):
                    result.discard(token)
                    result.update(equivalents)
                    break
    return result


def search_equivalent(query):
    """Use English only when every substantive Cyrillic token is known."""
    stop = {'для', 'по', 'и', 'в', 'на', 'с', 'при', 'о'}
    tokens = re.findall(r'\w+', query.casefold())
    translated = []
    changed = False
    for token in tokens:
        if token in stop:
            continue
        mapped = expand({token})
        if re.search('[а-я]', token):
            if token in mapped:
                return query
            changed = True
        translated.extend(sorted(mapped))
    return ' '.join(translated) if changed and translated else query
