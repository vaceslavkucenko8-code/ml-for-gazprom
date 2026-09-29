import json
import unittest
from pathlib import Path

from src.model import build_features, predict


class ModelTests(unittest.TestCase):
    def test_target_and_id_do_not_change_features(self):
        row = {'technology': 'новая технология', 'domain': 'Edge',
               'stage_raw': 'Пилот', 'trend_raw': 'Растёт — тест'}
        changed = {**row, 'signal_id': 999, 'reference_score': 7,
                   'rationale_raw': 'готовый ответ'}
        for mode in ('structured', 'text_domain', 'combined'):
            self.assertEqual(build_features(row, mode), build_features(changed, mode))

    def test_missing_input_rejected(self):
        with self.assertRaises(ValueError):
            build_features({'stage_raw': 'Пилот'}, 'structured')

    @unittest.skipUnless(Path('artifacts/rating_model.json').exists(), 'Training model not published')
    def test_contributions_reconstruct_raw_prediction(self):
        model = json.loads(Path('artifacts/rating_model.json').read_text(encoding='utf-8'))
        row = {'stage_raw': 'Пилот', 'trend_raw': 'Растёт'}
        result = predict(row, model)
        self.assertAlmostEqual(result['raw_score'], result['intercept'] + sum(x['contribution'] for x in result['contributions']))
        self.assertIsNone(result['probability_weak'])
        self.assertEqual(result['decision'], 'not_evaluated')

    @unittest.skipUnless(Path('artifacts/oof_predictions.csv').exists(), 'Dataset predictions not published')
    def test_evaluation_groups_do_not_cross_folds(self):
        import csv
        with Path('artifacts/oof_predictions.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 100)
        self.assertEqual(len({r['signal_id'] for r in rows}), 100)
        groups = {}
        for row in rows:
            groups.setdefault(row['group_id'], set()).add(row['fold'])
        self.assertTrue(all(len(folds) == 1 for folds in groups.values()))


if __name__ == '__main__':
    unittest.main()
