"""Словари ключевых маркеров для правило-ориентированного извлечения
кандидатов (см. candidates.py).

Подход осознанно rule-based, а не "чёрный ящик" LLM: это прямо соответствует
духу ТЗ про интерпретируемость ("система обязана показывать, по каким именно
признакам..."). Извлечение здесь готовит СЫРЬЁ (факты + доказательства) для
модели Участника 1, а не финальную классификацию, поэтому простые,
объяснимые правила — осознанный выбор, не ограничение.

Интерфейс extraction/candidates.py спроектирован так, что этот
rule-based экстрактор можно заменить/дополнить LLM-based (см.
`CandidateExtractor` ABC в candidates.py) без изменения остального конвейера.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Маркеры стадии развития (funding / pilot / patent / publication / regulatory)
# --------------------------------------------------------------------------- #

STAGE_MARKERS: dict[str, list[str]] = {
    "pre-seed / seed финансирование": [
        "pre-seed", "seed funding", "seed round", "посевн", "предпосевн",
    ],
    "финансирование Series A/B/C": [
        "series a", "series b", "series c", "раунд series", "раунд финансирования",
    ],
    "выход из stealth": [
        "emerges from stealth", "out of stealth", "вышла из stealth", "вышел из тени",
    ],
    "пилот / прототип / PoC": [
        "pilot project", "proof of concept", "prototype", "пилотный проект",
        "прототип", "пилотное внедрение", "испыта", "trial", "demonstrat",
        "продемонстрировал", "испытала",
    ],
    "научная публикация / препринт": [
        "preprint", "peer-reviewed", "published in", "препринт",
        "научн", "публикаци", "journal",
    ],
    "патентная заявка": [
        "patent application", "patent filed", "патентн", "заявк",
    ],
    "регуляторное одобрение": [
        "fda approval", "regulatory approval", "одобрен", "сертифика", "лицензи",
    ],
    "серийное производство / коммерческий запуск": [
        "commercial launch", "mass production", "general availability",
        "серийн", "коммерческ", "запуск продаж",
    ],
}

FUTURE_MARKERS_RU = [
    "планиру", "собира", "намерен", "к 2027", "к 2028", "ожида", "будет",
    "в перспективе", "готовится", "планируется",
]
FUTURE_MARKERS_EN = [
    "plans to", "is expected to", "will launch", "intends to", "aims to",
    "roadmap", "by 2027", "by 2028", "upcoming", "expected in",
]

# --------------------------------------------------------------------------- #
# Маркеры преимуществ / кейс-примеров (п. 2.5 ТЗ)
# --------------------------------------------------------------------------- #

ADVANTAGE_MARKERS_RU = [
    "преимуществ", "позволяет", "снижает", "ускоряет", "повышает точность",
    "без необходимости", "в отличие от", "дешевле", "эффективнее",
]
ADVANTAGE_MARKERS_EN = [
    "advantage", "enables", "reduces", "improves", "unlike", "without the need",
    "faster than", "cheaper than", "more efficient",
    "energy-efficient alternative", "allows researchers", "higher-resolution", "high-throughput",
]

CASE_MARKERS_RU = [
    "испытан", "внедрен", "применил", "использует", "в партнёрстве с",
    "совместно с", "на заводе", "в компании",
]
CASE_MARKERS_EN = [
    "deployed at", "piloted with", "in partnership with", "used by",
    "installed at", "adopted by",
    "could be used for", "could also be used", "plan to use", "could help identify",
]

# --------------------------------------------------------------------------- #
# Грубая доменная таксономия (соответствует колонке "Область" в датасете)
# --------------------------------------------------------------------------- #

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "Защита ИИ": ["security", "red team", "безопасност", "защит", "уязвим", "attack"],
    "Edge": ["edge", "on-device", "устройств", "iot", "мку", "mcu"],
    "Инфраструктура ИИ": ["data center", "дата-центр", "gpu", "чип", "compute", "инференс"],
    "Робототехника": ["robot", "робот", "манипулятор", "автоном"],
    "Финтех": ["fintech", "финтех", "payment", "платеж", "банк", "crypto", "блокчейн"],
    "Квантовые технологии": ["quantum", "квант"],
    "Материалы": ["material", "материал", "sensor", "сенсор"],
    "Энергетика": ["energy", "энерг", "battery", "аккумулятор", "реактор"],
}


import re as _re


def guess_domain(text: str) -> str | None:
    """Определяет область по частоте вхождений ключевых слов с границами
    слова (чтобы "edge" не срабатывал внутри "quantumEDGE"), суммируя ВСЕ
    вхождения, а не только факт присутствия — так более частотная тема
    текста перевешивает случайное однократное совпадение другой категории.
    """
    lowered = text.lower()
    best_domain, best_score = None, 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        for kw in keywords:
            pattern = r"\b" + _re.escape(kw) + r"\w*"
            score += len(_re.findall(pattern, lowered))
        if score > best_score:
            best_domain, best_score = domain, score
    return best_domain
