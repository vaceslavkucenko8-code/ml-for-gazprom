import csv
import json
import tempfile
import unittest
from pathlib import Path

from fixtures import candidate
from src.train_detector import train, load_training
from src.binary_inference import predict_with_classifier


class TrainingTests(unittest.TestCase):
    def test_no_implicit_binary_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'labels.csv'
            path.write_text('candidate_id,label,label_origin,reviewer,rationale\nx,,,,\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'explicitly labeled'):
                load_training([candidate('x')],path)

    def test_synthetic_training_lifecycle_is_marked_and_not_deployed(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'labels.csv'
            rows=[]
            with path.open('w',encoding='utf-8',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['candidate_id','label','label_origin','reviewer','rationale'])
                for i in range(12):
                    c=candidate(str(i), 'mass_adoption' if i%2 else None)
                    c['sources'][0]['url']=f'https://example.org/fixture-{i}'
                    rows.append(c)
                    writer.writerow([str(i),0 if i%2 else 1,'synthetic_test','automated-test','Fictional software fixture'])
            report=train(rows,path,'2026-09-15',Path(temp)/'model')
            self.assertTrue(report['synthetic_test_only'])
            artifact=json.loads((Path(temp)/'model/classifier.json').read_text(encoding='utf-8'))
            with self.assertRaisesRegex(ValueError,'Synthetic-test'):
                predict_with_classifier(rows[0],artifact,'2026-09-15')

    def test_learned_score_cannot_override_maturity(self):
        model={'intercept':100,'weights':{},'model_version':'test', 'label_origins':['internal_human']}
        result=predict_with_classifier(candidate(extra_feature='mass_adoption'),model,'2026-09-15')
        self.assertEqual(result['decision'],'reject')

    def test_learned_score_cannot_fill_missing_evidence(self):
        item=candidate();item['evidence']=[]
        model={'intercept':100,'weights':{},'model_version':'test','label_origins':['internal_human']}
        self.assertEqual(predict_with_classifier(item,model,'2026-09-15')['decision'],'needs_review')


if __name__=='__main__':unittest.main()
