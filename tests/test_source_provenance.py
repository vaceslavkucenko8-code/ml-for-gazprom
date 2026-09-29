from src.source_provenance import work_id, relations
from src.signal_analysis import evaluate


def test_identity_normalization_and_host_boundaries():
    assert work_id('https://doi.org/10.1234/ABC') == 'doi:10.1234/abc'
    assert work_id('https://arxiv.org/pdf/2601.12345v2.pdf') == 'arxiv:2601.12345'
    assert work_id('https://arxiv.org.evil.test/abs/2601.12345') is None
    assert work_id('https://doi.org.evil.test/10.1234/ABC') is None


def test_common_citation_does_not_prove_same_work():
    s={k:{'retrieval_metadata':{'raw_metadata':{'research_links':['https://doi.org/10.1234/common']}}} for k in 'ab'}
    assert relations(s)==[{'source_ids':['a','b'],'kind':'shared_reference','work_ids':['doi:10.1234/common']}]


def test_two_versions_of_one_work_cannot_confirm_trend():
    sources=[]; claims=[]
    for sid,version,text,dt in [('a',1,'Quantum navigation prototype is limited to laboratory use.','2026-01-01'),
                                ('b',2,'Trials of quantum navigation remain experimental with restricted deployment.','2026-06-01')]:
        sources.append({'source_id':sid,'url':f'https://arxiv.org/abs/2601.12345v{version}',
                        'text':text,'published_at':dt,'independence_reviewed':True,'independence_group':sid})
        claims.extend({'source_id':sid,'quote':text,'feature':f} for f in ('early_stage','limited_adoption','relevance'))
    result=evaluate({'sources':sources},{'claims':claims},'2026-09-27')
    assert result['stage']=='early_signal'
    assert result['independent_evidence_counts']['early_stage']==1
    assert result['provenance_relations'][0]['kind']=='same_work'
