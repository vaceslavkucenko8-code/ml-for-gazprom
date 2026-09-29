from src.signal_analysis import evaluate


def fixture():
    text='A sensor prototype is limited to laboratory use.'
    s={'source_id':'a','text':text,'source_type':'research','published_at':'2026-01-01'}
    c={'sources':[s]}
    a={'claims':[{'source_id':'a','feature':f,'quote':text,'evidence_id':f} for f in ('early_stage','limited_adoption','relevance')]}
    return c,a


def test_one_source_is_preliminary_not_trend_or_disinformation():
    c,a=fixture();r=evaluate(c,a,'2026-09-26')
    assert r['stage']=='early_signal' and r['support']=='preliminary'
    assert r['disinformation']=='not_established'


def test_multiple_hosts_without_verified_independence_not_trend():
    c,a=fixture();c['sources'].append({**c['sources'][0],'source_id':'b','published_at':'2026-06-01'})
    a['claims'] += [{**x,'source_id':'b'} for x in list(a['claims'])]
    assert evaluate(c,a,'2026-09-26')['stage']=='early_signal'
    for s in c['sources']:s.update(independence_reviewed=True,independence_group=s['source_id'])
    r=evaluate(c,a,'2026-09-26')
    assert r['stage']=='early_signal'
    assert r['repetition_pairs'] and r['independent_evidence_counts']['early_stage']==1
    different='An experimental sensor prototype remains restricted to small controlled research trials.'
    c['sources'][1]['text']=different
    for claim in a['claims']:
        if claim['source_id']=='b':claim['quote']=different
    # "sensor" alone does not identify a concrete technology.
    assert evaluate(c,a,'2026-09-26')['stage']=='early_signal'


def test_same_explicit_subject_can_be_confirmed_without_merging_other_technologies():
    c,a=fixture()
    texts=['A quantum navigation prototype is limited to laboratory use.',
           'Researchers tested quantum navigation; deployment remains restricted to controlled trials.']
    c['sources']=[{'source_id':sid,'text':text,'source_type':'research',
                   'published_at':dt,'independence_reviewed':True,'independence_group':sid}
                  for sid,text,dt in zip('ab',texts,['2026-01-01','2026-06-01'])]
    a['claims']=[{'source_id':s['source_id'],'feature':f,'quote':s['text']}
                 for s in c['sources'] for f in ('early_stage','limited_adoption','relevance')]
    r=evaluate(c,a,'2026-09-26')
    assert r['stage']=='emerging_trend'
    assert ['navigation','quantum'] in [g['subject_terms'] for g in r['trend_scope_groups']]
    c['sources'][1]['text']=texts[1].replace('quantum navigation','optical computing')
    for x in a['claims']:
        if x['source_id']=='b':x['quote']=c['sources'][1]['text']
    assert evaluate(c,a,'2026-09-26')['stage']=='early_signal'


def test_unverified_context_cannot_create_a_shared_subject():
    from src.claim_scope import subject_terms
    claim={'quote':'The prototype is limited to laboratory use.',
           'scope_context':{'quote':'quantum navigation'}}
    assert subject_terms(claim,{'text':claim['quote']})==[]


def test_repetition_is_transitive_and_cannot_prove_independence():
    from src.evidence_overlap import independent_count, similarity
    sources={k:{'independence_group':k} for k in 'abc'}
    pairs=[{'source_ids':['a','b'],'feature':None}, {'source_ids':['b','c'],'feature':'early_stage'}]
    assert independent_count(set(sources),sources,pairs,'early_stage')==1
    assert independent_count(set(sources),sources,pairs,'limited_adoption')==2
    assert similarity('A SENSOR prototype is limited to laboratory use!', 'A sensor prototype is limited to laboratory use.')==1
    assert similarity('short common title','short common title')==0


def test_copied_quote_in_otherwise_different_articles_is_collapsed():
    c,a=fixture()
    quote=c['sources'][0]['text']
    c['sources'][0]['text']='Unique original background research introduction. '+quote
    c['sources'].append({**c['sources'][0],'source_id':'b','text':quote+' Separate editorial analysis and discussion follows.', 'published_at':'2026-06-01'})
    a['claims'] += [{**x,'source_id':'b'} for x in list(a['claims'])]
    for s in c['sources']:s.update(independence_reviewed=True,independence_group=s['source_id'])
    r=evaluate(c,a,'2026-09-26')
    assert r['stage']=='early_signal'
    assert any(p['kind']=='quote_overlap' for p in r['repetition_pairs'])


def test_fabricated_evidence_and_future_sources_not_used():
    c,a=fixture();c['sources'][0]['published_at']='2027-01-01'
    assert evaluate(c,a,'2026-09-26')['stage']=='uncertain'
    c['sources'][0]['published_at']='2026-01-01'
    for x in a['claims']:x['quote']='invented'
    assert evaluate(c,a,'2026-09-26')['support']=='insufficient'


def test_promotional_only_does_not_mean_false():
    c,a=fixture();c['sources'][0]['source_type']='press_release'
    r=evaluate(c,{'claims':[]},'2026-09-26')
    assert r['support']=='promotional_only' and r['disinformation']=='not_established'
