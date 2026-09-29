import unittest
from copy import deepcopy
from fixtures import candidate
from src.detector import assess, rank_candidates

AS_OF='2026-09-15'


class DetectorTests(unittest.TestCase):
    def test_evidenced_candidate(self):
        result=assess(candidate(),AS_OF)
        self.assertEqual(result['decision'],'weak')
        self.assertIsNone(result['probability_weak'])

    def test_maturity_overrides_rating(self):
        result=assess(candidate(extra_feature='mass_adoption'),AS_OF)
        self.assertEqual(result['decision'],'reject')
        self.assertEqual(result['exclusion_reasons'][0]['feature'],'mass_adoption')

    def test_unreviewed_negated_or_ambiguous_claim_not_used(self):
        item=candidate(extra_feature='mass_adoption')
        item['evidence'][-1]['reviewed']=False
        self.assertEqual(assess(item,AS_OF)['decision'],'weak')

    def test_future_source_is_not_current_evidence(self):
        item=candidate(); item['sources'][0]['published_at']='2027-01-01'
        self.assertEqual(assess(item,AS_OF)['decision'],'needs_review')

    def test_planned_pilot_not_observed(self):
        item=candidate(); item['evidence'][0]['event_status']='planned'
        self.assertEqual(assess(item,AS_OF)['decision'],'needs_review')

    def test_unknown_date(self):
        item=candidate(); item['sources'][0]['published_at']=None
        self.assertEqual(assess(item,AS_OF)['decision'],'needs_review')

    def test_fabricated_quote(self):
        item=candidate(); item['evidence'][0]['quote']='Нет такого фрагмента'
        self.assertEqual(assess(item,AS_OF)['decision'],'needs_review')

    def test_different_scope_maturity_not_used(self):
        item=candidate(extra_feature='industry_standard')
        item['evidence'][-1]['scope_matches_candidate']=False
        self.assertEqual(assess(item,AS_OF)['decision'],'weak')

    def test_relevance_to_other_query(self):
        item=candidate(); item['query']='медицинские роботы'
        self.assertEqual(assess(item,AS_OF)['decision'],'needs_review')

    def test_promotional_only_sources(self):
        item=candidate(); item['sources'][0]['source_type']='press_release'
        self.assertEqual(assess(item,AS_OF)['decision'],'needs_review')

    def test_duplicate_urls_do_not_give_independence_bonus(self):
        item=candidate()
        second=deepcopy(item['sources'][0]); second.update(source_id='s2',independence_group='different',url='https://example.org/fictional-lab?utm_source=copy')
        item['sources'].append(second)
        ev=deepcopy(item['evidence'][0]);ev.update(evidence_id='extra',source_id='s2');item['evidence'].append(ev)
        result=assess(item,AS_OF)
        self.assertNotIn('independent_sources',[x['feature'] for x in result['predictors']])

    def test_unknown_evidence_reference(self):
        item=candidate();item['evidence'][0]['source_id']='missing'
        self.assertEqual(assess(item,AS_OF)['decision'],'needs_review')

    def test_duplicate_id_rejected(self):
        with self.assertRaises(ValueError):
            rank_candidates([candidate(),candidate()],AS_OF)

    def test_same_technology_only_once_and_no_padding(self):
        a,b=candidate('a'),candidate('b');b['technology_group_id']='a'
        result=rank_candidates([a,b],AS_OF)
        self.assertEqual(len(result['results']),1)
        self.assertTrue(result['insufficient_results'])

    def test_invalid_limit(self):
        with self.assertRaises(ValueError):
            rank_candidates([],AS_OF,0)


if __name__=='__main__':
    unittest.main()
