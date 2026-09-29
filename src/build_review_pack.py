"""Prepare real-data annotation inputs; never invent evidence or class labels."""
import csv
import json
from pathlib import Path
from .prepare_data import write_json

NOTES = {
  5: 'Разные стадии у лидера и остальных компаний; сохранить различие.',
  12: 'План мощности к октябрю 2026 не является состоявшимся фактом на 15.09.2026.',
  13: 'Разделить новую услугу сертификации и уже существующий стандарт.',
  32: 'CS-4 и CS-6 находятся на разных стадиях; проверить границы технологии.',
  34: 'Слово «пилот» сопровождается отсутствием физических установок; рабочий пилот не подтверждён.',
  59: 'Лётные тесты 2027 указаны как будущее событие.',
  70: 'Первые полисы — признак начала коммерциализации, а не просто лабораторный прототип.',
  80: 'В тренде заявлено массовое производство, в стадии — ранняя серия. Требуется сверка объекта и источника.',
  90: 'Доклинические и клинические шаги нельзя автоматически приравнивать к коммерческому внедрению.',
  100: 'Разделить зрелость криптографического стандарта и раннюю стадию конкретной программы миграции.'
}


def run():
    rows = json.loads(Path('data/processed/signals_clean.json').read_text(encoding='utf-8'))
    folds = {}
    with Path('artifacts/oof_predictions.csv').open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            folds[int(row['signal_id'])] = row['group_id']
    out = Path('data/review'); out.mkdir(exist_ok=True)
    candidates = []
    for row in rows:
        sources = [{'source_id': f'{row["signal_id"]}-src-{i}', 'url': link['url'],
                    'title_original': link['title_raw'], 'text': '[Текст источника ещё не загружен]',
                    'published_at': None, 'language': None, 'source_type': 'unknown',
                    'trust_level': 'unknown', 'trust_reason': '', 'independence_group': None,
                    'independence_reviewed': False} for i,link in enumerate(row['sources'],1)]
        candidates.append({'candidate_id': str(row['signal_id']), 'name_ru': row['technology'],
                           'query': row['domain'], 'technology_group_id': folds[row['signal_id']],
                           'sources': sources, 'evidence': [],
                           'review_status': 'pending_source_review',
                           'review_note_ru': NOTES.get(row['signal_id'], 'Проверить стадию, масштаб внедрения, связь с запросом и источники.'),
                           'workbook_context_not_verified': {'stage': row['stage_raw'], 'trend': row['trend_raw'],
                                                            'rationale': row['rationale_raw']}})
    if not (out/'candidates_pending.json').exists():
        write_json(out/'candidates_pending.json', {'synthetic': False, 'candidates': candidates})
    labels = out/'labels_template.csv'
    if not labels.exists():
        with labels.open('w', encoding='utf-8-sig', newline='') as stream:
            writer=csv.DictWriter(stream, fieldnames=['candidate_id','label','label_origin','reviewer','rationale'])
            writer.writeheader()
            writer.writerows({'candidate_id': r['candidate_id']} for r in candidates)
    write_json(out/'priority_review.json', [{'candidate_id': str(k), 'note_ru':v} for k,v in NOTES.items()])
    print(f'Review pack: {len(candidates)} real candidates; binary labels left empty.')


if __name__ == '__main__':
    run()
