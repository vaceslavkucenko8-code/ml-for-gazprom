from src.signal_analysis import evaluate


def sample(dates):
    texts=['Quantum navigation prototype is limited to laboratory use.',
           'Trials of quantum navigation remain experimental with restricted deployment.',
           'Quantum navigation prototype is limited to laboratory use.']
    sources=[{'source_id':str(i),'text':text,'published_at':dt,
              'independence_reviewed':True,'independence_group':str(i)}
             for i,(text,dt) in enumerate(zip(texts,dates))]
    claims=[{'source_id':s['source_id'],'quote':s['text'],'feature':f}
            for s in sources for f in ('early_stage','limited_adoption','relevance')]
    return {'sources':sources},{'claims':claims}


def test_late_reprint_cannot_extend_trend_interval():
    c,a=sample(['2026-01-01','2026-01-03','2026-08-01'])
    result=evaluate(c,a,'2026-09-27')
    assert result['stage']=='early_signal'
    g=next(g for g in result['scope_groups'] if g['subject_terms']==['navigation','quantum'])
    assert g['chronology_by_feature']['early_stage']['span_days']==2
    assert ['0','2'] in g['chronology_by_feature']['early_stage']['source_groups']


def test_actual_later_independent_group_retains_trend():
    c,a=sample(['2026-01-01','2026-06-01','2026-08-01'])
    assert evaluate(c,a,'2026-09-27')['stage']=='emerging_trend'


def test_separate_features_cannot_supply_each_others_time_span():
    c,a=sample(['2026-01-01','2026-01-03'])
    c['sources'] += [{**s,'source_id':s['source_id']+'b','published_at':'2026-08-01',
                      'independence_group':s['source_id']+'b','text':s['text']+' Additional observation.'}
                     for s in list(c['sources'])]
    a['claims']=[x for x in a['claims'] if x['feature']!='limited_adoption']
    a['claims'] += [{'source_id':s['source_id'],'quote':s['text'],'feature':'limited_adoption'} for s in c['sources'][2:]]
    assert evaluate(c,a,'2026-09-27')['stage']=='early_signal'
