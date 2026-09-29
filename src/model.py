"""Auxiliary rating regression. This module does not classify weak signals."""
import argparse
import json
import math
import re
from pathlib import Path
from .prepare_data import stage_features, trend_category


def build_features(record, mode='structured'):
    allowed = {'structured', 'text_domain', 'combined'}
    if mode not in allowed:
        raise ValueError(f'Unknown mode {mode}')
    features = {}
    if mode in ('text_domain', 'combined'):
        if not isinstance(record.get('technology'), str) or not record['technology'].strip():
            raise ValueError('technology must be non-empty text')
        tokens = set(re.findall(r'[\w]{2,}', record['technology'].casefold()))
        norm = math.sqrt(len(tokens)) or 1
        features.update({'word:' + token: 1 / norm for token in tokens})
        features['domain:' + str(record.get('domain', 'unknown'))] = 1.0
    if mode in ('structured', 'combined'):
        for key in ('stage_raw', 'trend_raw'):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f'{key} must be non-empty text')
        features.update({key: float(value) for key, value in stage_features(record['stage_raw']).items()})
        features['trend:' + trend_category(record['trend_raw'])] = 1.0
    return features


def predict(record, model):
    features = build_features(record, model['mode'])
    contributions = [{'feature': key, 'value': value,
                      'contribution': value * model['weights'].get(key, 0.0)}
                     for key, value in features.items() if value]
    raw = model['intercept'] + sum(x['contribution'] for x in contributions)
    lo, hi = model['rating_bounds']
    score = min(hi, max(lo, raw))
    return {'candidate_id': record.get('signal_id'), 'score': score,
            'raw_score': raw, 'intercept': model['intercept'],
            'score_kind': 'predicted_reference_rating', 'probability_weak': None,
            'model_version': model['model_version'], 'decision': 'not_evaluated',
            'contributions': sorted(contributions, key=lambda x: abs(x['contribution']), reverse=True),
            'unknown_features': sorted(k for k in features if k not in model['weights']),
            'explanation_ru': 'Прогноз балла исходной подборки. Это не вероятность слабого сигнала. Вклады признаков суммируются с intercept до ограничения диапазоном 3–7.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('artifacts/rating_model.json'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/prediction.json'))
    args = parser.parse_args()
    result = predict(json.loads(args.input.read_text(encoding='utf-8')),
                     json.loads(args.model.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Predicted reference rating: {result["score"]:.3f}')


if __name__ == '__main__':
    main()
