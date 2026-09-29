"""CLI/backend entry point: no web server or external credentials needed."""
import argparse
import json
from pathlib import Path
from .detector import rank_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--as-of', required=True, help='YYYY-MM-DD')
    parser.add_argument('--limit', type=int, default=15)
    parser.add_argument('--classifier', type=Path, help='Optional classifier trained on explicit real labels')
    parser.add_argument('--output', type=Path, default=Path('artifacts/detection_results.json'))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    candidates = data['candidates'] if isinstance(data, dict) else data
    if not isinstance(candidates, list):
        raise ValueError('Input must contain a candidate list')
    assessor = None
    if args.classifier:
        from .binary_inference import predict_with_classifier
        model = json.loads(args.classifier.read_text(encoding='utf-8'))
        assessor = lambda candidate, as_of: predict_with_classifier(candidate, model, as_of)
    result = rank_candidates(candidates, args.as_of, args.limit, assessor)
    result['input_is_synthetic'] = bool(isinstance(data, dict) and data.get('synthetic'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'selected': len(result['results']), 'counts': result['counts'],
                      'synthetic': result['input_is_synthetic']}))


if __name__ == '__main__':
    main()
