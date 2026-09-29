"""Fixed, grouped out-of-fold comparison; no tuning on validation outcomes."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from .model import build_features, predict
from .prepare_data import stage_features, write_json


def group_ids(rows, config):
    parent = {r['signal_id']: r['signal_id'] for r in rows}
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    def join(a, b):
        parent[find(max(a, b))] = find(min(a, b))
    for family in config['families']:
        for member in family['ids'][1:]:
            join(family['ids'][0], member)
    seen = {}
    for r in rows:
        for source in r['sources']:
            if source['url'] in seen:
                join(seen[source['url']], r['signal_id'])
            seen[source['url']] = r['signal_id']
    return np.array([find(r['signal_id']) for r in rows])


def metrics(y, pred):
    rounded = np.floor(np.clip(pred, 3, 7) + 0.5)
    return {'mae': float(np.mean(abs(y-pred))),
            'rmse': float(np.sqrt(np.mean((y-pred)**2))),
            'exact_rounded_rating': float(np.mean(rounded == y))}


def bootstrap_mae(y, predictions, groups):
    rng = np.random.default_rng(42)
    unique = np.unique(groups)
    errors = abs(y-predictions)
    values = []
    for _ in range(2000):
        indices = np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(unique, len(unique))])
        values.append(float(errors[indices].mean()))
    return [float(v) for v in np.quantile(values, [.025, .975])]


def run():
    source = Path('data/processed/signals_clean.json')
    rows = json.loads(source.read_text(encoding='utf-8'))
    config = json.loads(Path('data/technology_groups.json').read_text(encoding='utf-8'))
    groups = group_ids(rows, config)
    y = np.array([r['reference_score'] for r in rows], dtype=float)
    splits = list(GroupKFold(n_splits=5).split(rows, y, groups))
    variants = ['median', 'text_domain', 'structured', 'combined']
    predictions = {name: np.zeros(len(rows)) for name in variants}
    fold_ids = np.zeros(len(rows), dtype=int)
    fold_metrics = []
    for fold, (train, test) in enumerate(splits):
        assert not set(groups[train]) & set(groups[test])
        fold_ids[test] = fold
        info = {'fold': fold, 'train_count': len(train), 'test_count': len(test),
                'test_score_counts': dict(Counter(str(int(v)) for v in y[test])), 'metrics': {}}
        for mode in variants:
            if mode == 'median':
                pred = np.repeat(np.median(y[train]), len(test))
            else:
                model = make_pipeline(DictVectorizer(), Ridge(alpha=1.0, solver='lsqr', tol=1e-8))
                model.fit([build_features(rows[i], mode) for i in train], y[train])
                pred = np.clip(model.predict([build_features(rows[i], mode) for i in test]), 3, 7)
            predictions[mode][test] = pred
            info['metrics'][mode] = metrics(y[test], pred)
        fold_metrics.append(info)
    summary = {mode: {**metrics(y, pred), 'mae_group_bootstrap_95': bootstrap_mae(y, pred, groups)}
               for mode, pred in predictions.items()}
    artifacts = Path('artifacts'); artifacts.mkdir(exist_ok=True)
    report = {'task': 'auxiliary_reference_rating_regression', 'ground_truth': 'reference_score, not weak-signal label',
              'rows': len(rows), 'groups': len(set(groups)), 'sklearn_version': sklearn.__version__,
              'split': 'GroupKFold(5), no tuning; topic families + identical URLs',
              'alpha': 1.0, 'metrics': summary, 'folds': fold_metrics,
              'by_domain': {domain: {mode: metrics(y[[r['domain']==domain for r in rows]], pred[[r['domain']==domain for r in rows]]) for mode,pred in predictions.items()} for domain in sorted({r['domain'] for r in rows})},
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'limitations': ['OOF evaluation is exploratory, not an untouched external test.', 'Group bootstrap intervals are approximate; overlapping training folds introduce dependence.', 'Stage and trend are analyst-created components of the target rating.', 'Topic grouping is conservative and incomplete; company/event leakage may remain.', 'No binary detection accuracy or calibrated confidence is estimated.']}
    write_json(artifacts/'evaluation.json', report)
    with (artifacts/'oof_predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['signal_id','group_id','fold','reference_score',*variants])
        writer.writeheader()
        for i, r in enumerate(rows):
            writer.writerow({'signal_id': r['signal_id'], 'group_id': int(groups[i]), 'fold': int(fold_ids[i]),
                             'reference_score': int(y[i]), **{mode: float(pred[i]) for mode,pred in predictions.items()}})
    # Structured variant chosen for interpretable deployment, not by OOF rank.
    final = make_pipeline(DictVectorizer(), Ridge(alpha=1.0, solver='lsqr', tol=1e-8))
    final.fit([build_features(r, 'structured') for r in rows], y)
    vectorizer, estimator = final.steps[0][1], final.steps[1][1]
    portable = {'model_version': 'rating-ridge-0.1', 'mode': 'structured', 'rating_bounds': [3,7],
                'intercept': float(estimator.intercept_),
                'weights': dict(zip(vectorizer.get_feature_names_out(), map(float, estimator.coef_))),
                'training_records': len(rows), 'input_sha256': report['source_sha256'],
                'purpose': 'auxiliary rating prediction; not weak-signal detection'}
    for r in rows:
        expected = np.clip(final.predict([build_features(r, 'structured')])[0],3,7)
        assert abs(predict(r, portable)['score']-expected)<1e-8
    write_json(artifacts/'rating_model.json', portable)
    write_json(artifacts/'example_input.json', {k: rows[0][k] for k in ('technology','domain','stage_raw','trend_raw')})
    write_json(artifacts/'example_prediction.json', predict(rows[0], portable))
    catalog = []
    for stage in sorted({r['stage_raw'] for r in rows}):
        flags = stage_features(stage)
        catalog.append({'stage_raw': stage, 'signal_ids': [r['signal_id'] for r in rows if r['stage_raw']==stage],
                        **flags, 'review_status': 'text_normalization_only_not_source_verified',
                        'note': 'Переходы и различия между компаниями сохранены; не выбирать единственную стадию без источников.' if flags['stage_multiple'] else 'Стадия извлечена из формулировки, требует фактической проверки при открытом поиске.'})
    write_json(Path('data/processed/stage_catalog.json'), catalog)
    lines = ['# Результаты первого обучения', '', 'Задача: вспомогательный прогноз балла 3–7 из XLSX. Это не детекция слабого сигнала.', '',
             f'Данные: {len(rows)} технологий; {len(set(groups))} групп; 5 групповых фолдов. Каждая запись проверена моделью, которая её группу не видела. Параметр регуляризации alpha=1 задан заранее; подбора параметров не было.', '',
             '| Вариант | MAE, баллы ↓ | RMSE ↓ | Точное совпадение округлённого балла |', '|---|---:|---:|---:|']
    for name,m in summary.items():
        lines.append(f'| {name} | {m["mae"]:.3f} | {m["rmse"]:.3f} | {m["exact_rounded_rating"]:.0%} |')
    lines += ['', '## Как читать сравнение', '',
              '- median: медиана балла только обучающей части.',
              '- text_domain: слова названия и область, без стадии и тренда.',
              '- structured: упоминания стадий и качественная категория тренда.',
              '- combined: оба набора вместе.',
              '', 'Стадия и тренд входят в смысл исходного балла. Хороший результат structured показывает воспроизведение этой зависимости, но не умение отличать слабые сигналы от зрелости/шума.',
              '', 'Для интеграции сохранён интерпретируемый structured-вариант, дообученный на всех 100 строках. Его оценки на этих строках не использовать как тестовые. Для проверки смотреть только oof_predictions.csv.',
              '', '47 формулировок стадий сохранены в stage_catalog.json. Неоднозначные переходы не схлопываются в один класс. Источники в этом этапе не перепроверялись.',
              '', '## Ограничения', '', *['- '+x for x in report['limitations']], '',
              'Технические источники: [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html), [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).']
    Path('docs/model_evaluation.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'groups': report['groups'], 'metrics': summary}, indent=2))


if __name__ == '__main__':
    run()
