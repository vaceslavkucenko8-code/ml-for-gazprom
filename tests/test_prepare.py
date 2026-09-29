import tempfile
import unittest
from pathlib import Path

import openpyxl

from src.prepare_data import HEADERS, FIELDS, prepare, stage_features, trend_category, source_links, validate


class PreparationTests(unittest.TestCase):
    def test_transition_preserves_both_stages(self):
        result = stage_features('Исследование → Прототип/PoC')
        self.assertTrue(result['stage_research'])
        self.assertTrue(result['stage_prototype'])
        self.assertTrue(result['stage_multiple'])
        self.assertTrue(result['stage_has_transition'])

    def test_unknown_not_implicitly_early(self):
        self.assertTrue(stage_features('Нет сведений')['stage_unknown'])
        self.assertEqual(trend_category('Нет сведений'), 'unknown')

    def test_mixed_case_and_sources(self):
        self.assertEqual(trend_category('Стабильный/Растёт — описание'), 'stable_or_growth')
        self.assertEqual(len(source_links('[A](https://a.org/x); [B](https://b.org/y)')), 2)

    def test_duplicate_id_rejected(self):
        record = dict(zip(FIELDS, [1, 'a', 'b', 'c', 'd', 'e', 'f', 4, 'g']))
        record['excel_row'] = 3
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate([record, record.copy()])

    def test_changed_header_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.xlsx'
            book = openpyxl.Workbook()
            book.active.title = 'Слабые сигналы'
            book.active.cell(2, 2, 'unexpected')
            book.save(path)
            book.close()
            with self.assertRaisesRegex(ValueError, 'Unexpected headers'):
                prepare(path, Path(temp) / 'out')


if __name__ == '__main__':
    unittest.main()
