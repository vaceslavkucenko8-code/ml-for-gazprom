"""Conservative automatic evidence rules; never impersonate human review.

Predictions are experimental and must not be used as training ground truth.
Publication date dates the assertion, not an inferred event date.
"""
from datetime import date
import hashlib
import re

VERSION = 'automatic-evidence-0.8'
MAX_POSITIVE_SOURCE_AGE_DAYS = 540
PATTERNS = {
    'early_stage': r'\bprototype\b|proof[- ]of[- ]concept|\bpilot (?:project|trial|study)\b|\bwe (?:propose|demonstrate|report)\b|прототип\w*|пилотн\w*|предлагаем\w*',
    'limited_adoption': r'(?:only (?:in )?|limited to |confined to |restricted to )(?:(?:a|the|our)\s+)?(?:laboratory|lab|simulation|research|pilot|testbed)\b|not (?:yet )?(?:commercially available|commercially deployed)|no commercially available \w+|только (?:в лаборатор\w*|в симуляци\w*|на испытательн\w*)|ограничен\w* (?:лаборатор\w*|пилот\w*)|не (?:доступ\w* коммерчески|внедрен\w* коммерчески)',
    'mass_adoption': r'\bmass production\b|\bwidely (?:deployed|adopted|used)\b|\bmillions of (?:users|devices|customers)\b|массов\w* (?:производств\w*|внедрен\w*)|миллион\w* (?:пользовател\w*|устройств\w*)',
    'industry_standard': r'\b(?:is|became|remains) (?:an? |the )?(?:industry|international) standard\b|\b(?:finalized|completed|final) (?:[\w-]+\s+){0,3}standards?\b|является (?:отраслевым|международным) стандартом',
}
FUTURE = re.compile(r'\b(?:will|would|could|might|may)\b|\b(?:plans?|aims?|intends?) to\b|expected to|планир\w*|намерен\w*|ожида\w*|будет|будут|может|могут',re.I)
NEGATION = re.compile(r'\b(?:not|never|no|without|neither)\b|\bне\b|\bнет\b|\bбез\b',re.I)
CONTRAST = re.compile(r'\b(?:unlike|whereas|compared|previous|conventional|existing)\b|в отличие|по сравнению|традиционн\w*',re.I)
STOP = {'the','and','for','with','from','this','that','into','based','using','toward','towards','для','при','как','это','или'}
PATTERNS['limited_adoption'] += (
    r'|\b(?:has|have) yet to (?:see|achieve|reach) (?:widespread|broad|large-scale) '
    r'(?:commercial|industrial)(?: or (?:commercial|industrial))? (?:application|adoption|deployment)\b'
    r'|\b(?:has|have) not (?:yet )?(?:achieved|reached) (?:widespread|broad) '
    r'(?:commercial|industrial) (?:adoption|deployment)\b'
)
PATTERNS['early_stage'] += r'|\bresearch\b[^.!?]{0,100}\brel(?:ied|ies) on simulations?\b'
PATTERNS['early_stage'] += r'|\b(?:researchers|engineers) (?:have )?(?:demonstrated|created|built)\b'


def words(text):
    from src.topic_terms import expand
    tokens={t for t in re.findall(r'\w+',text.casefold()) if len(t)>2 and t not in STOP}
    normalized={t[:-1] if t.isascii() and len(t)>4 and t.endswith('s') and not t.endswith('ss') else t for t in tokens}
    computing={'compute','computing','computation','computational','computer'}
    if normalized & computing:
        normalized |= computing
    return expand(normalized)


def sentences(text):
    # Keep exact source offsets/whitespace and avoid mixing distinct statements.
    for match in re.finditer(r'[^.!?]+(?:[.!?]|$)', text):
        quote=match.group().strip()
        if len(quote)>=20:
            start=match.start()+len(match.group())-len(match.group().lstrip())
            # A completed demonstration followed by an explicitly future clause
            # is not itself a future claim. Preserve exact slices and offsets.
            boundary=re.search(r',\s+but\s+(?:future|in the future)\b',quote,re.I)
            if boundary:
                left=quote[:boundary.start()]
                right_start=boundary.start()+1
                while right_start<len(quote) and quote[right_start].isspace():right_start+=1
                if len(left)>=20:yield start,left
                if len(quote[right_start:])>=20:yield start+right_start,quote[right_start:]
            else:
                yield start,quote


def analyse(candidate, as_of, search_query=None):
    cutoff=date.fromisoformat(as_of)
    claims,ignored=[],[]
    topic=words(search_query or candidate['query'])
    for source in candidate.get('sources',[]):
        sid=source['source_id']
        metadata=source.get('retrieval_metadata',{})
        raw=metadata.get('raw_metadata',{})
        published=source.get('published_at')
        reason=None
        try:
            if not published or date.fromisoformat(published)>cutoff:
                reason='missing_or_future_publication_date'
        except (ValueError,TypeError):
            reason='invalid_publication_date'
        if source.get('source_type') not in ('research','official','university','patent','professional_media') or source.get('trust_level') not in ('high','medium') or not source.get('trust_reason'):
            reason='insufficient_source_trust'
        if metadata.get('is_generated_summary') or metadata.get('is_translated') or raw.get('text_scope')=='search_snippet':
            reason='not_original_source_text'
        if reason:
            ignored.append({'source_id':sid,'reason':reason});continue
        title=words(source['title_original'])
        # The sentence can supply a query term omitted from the headline.
        title_relevant=bool(title&topic)
        # First-person claims in another paper do not describe this candidate.
        if not topic or not title & topic:
            ignored.append({'source_id':sid,'reason':'source_topic_not_established'});continue
        stale=(cutoff-date.fromisoformat(published)).days>MAX_POSITIVE_SOURCE_AGE_DAYS
        previous_topic_sentence = None
        for offset,quote in sentences(source['text']):
            if re.search(r'Share\s+this news article|Related (?:Topics|Links)|all rights reserved|privacy policy',quote,re.I):
                previous_topic_sentence=None
                continue
            qwords=words(quote)
            direct_topic = len(qwords&topic)>=min(2,len(topic))
            # Resolve only an explicit adjacent reference, without chaining it.
            refers_to_previous = bool(previous_topic_sentence and re.match(r'Research in this field\b',quote,re.I))
            context = previous_topic_sentence if refers_to_previous else None
            previous_topic_sentence = {'quote':quote,'start':offset,'end':offset+len(quote)} if direct_topic else None
            # Scientific first-person statements refer to this work; avoid background comparisons.
            scoped=bool(len(qwords&title)>=min(2,max(1,len(topic))) or len(qwords&topic)>=min(2,len(topic)) or re.search(r'\b(?:we|our|this work|this (?:prototype|system|sensor))\b|\b(?:мы|наш\w*|этот прототип)\b',quote,re.I))
            scoped = scoped or refers_to_previous
            if not scoped or CONTRAST.search(quote):
                continue
            features=[]
            if title_relevant and len(qwords&topic)>=min(2,len(topic)):
                features.append(('relevance','lexical_query_and_title_overlap'))
            for feature,pattern in PATTERNS.items():
                distinguishing = topic & {'llm', 'mems', 'rag', 'vla', 'tee', 'rl'}
                if feature in ('mass_adoption', 'industry_standard') and distinguishing and not distinguishing <= qwords:
                    if re.search(pattern,quote,re.I):
                        ignored.append({'source_id':sid,'quote':quote,'feature':feature,'reason':'maturity_subject_not_established'})
                    continue
                match=re.search(pattern,quote,re.I)
                if not match:
                    continue
                # Historical maturity remains evidence against novelty; historical
                # prototypes cannot establish the current early-stage status.
                if stale and feature in ('early_stage','limited_adoption'):
                    ignored.append({'source_id':sid,'feature':feature,'reason':'stale_positive_evidence'});continue
                if FUTURE.search(quote):
                    ignored.append({'source_id':sid,'quote':quote,'feature':feature,'reason':'future_or_modal_claim'});continue
                # Limited-adoption rules explicitly include negative commercialization assertions.
                before=quote[max(0,match.start()-90):match.start()]
                if feature!='limited_adoption' and NEGATION.search(quote):
                    ignored.append({'source_id':sid,'quote':quote,'feature':feature,'reason':'negated_claim'});continue
                # Negation of "limited to" means the opposite; do not infer limited deployment.
                if feature=='limited_adoption' and NEGATION.search(before):
                    ignored.append({'source_id':sid,'quote':quote,'feature':feature,'reason':'negated_limitation'});continue
                features.append((feature,'explicit_text_pattern'))
            for feature,rule in features:
                eid=hashlib.sha256(f'{sid}:{offset}:{feature}'.encode()).hexdigest()[:20]
                claims.append({'evidence_id':'auto_'+eid,'source_id':sid,'feature':feature,'quote':quote,
                               'scope_context':context,
                               'start':offset,'end':offset+len(quote),'asserted_at':published,
                               'event_date':None,'origin':'automatic_rule','human_reviewed':False,'rule':rule})
    flags={feature:any(e['feature']==feature for e in claims) for feature in (*PATTERNS,'relevance')}
    mature=flags['mass_adoption'] or flags['industry_standard']
    conflict=mature and flags['early_stage']
    missing=[f for f in ('early_stage','limited_adoption','relevance') if not flags[f]]
    if conflict:
        decision='needs_review';reason='conflicting_stage_claims'
    elif mature and flags['relevance']:
        decision='likely_mature';reason='explicit_maturity_claim'
    elif not missing:
        decision='likely_weak';reason='explicit_early_and_limited_scope_claims'
    else:
        decision='needs_review';reason='insufficient_explicit_evidence'
    score=sum(weight for feature,weight in [('early_stage',30),('limited_adoption',40),('relevance',30)] if flags[feature])
    result = {'candidate_id':candidate['candidate_id'],'decision':decision,'reason':reason,'score':score,
            'score_kind':'automatic_rule_support_not_probability','probability_weak':None,
            'model_version':VERSION,'flags':flags,'missing_features':missing,'claims':claims,'ignored':ignored,
            'requires_expert_validation':True}
    from src.signal_analysis import evaluate
    result['signal_analysis']=evaluate(candidate,result,as_of)
    return result


def preliminary_candidates(assessments, exclude_ids=(), limit=15):
    """Candidates that can fill free TOP-15 slots, clearly marked as preliminary.

    The strict decision rule is unchanged: these stay ``needs_review``. A candidate
    qualifies only with an explicit early-stage claim on a source matching the query,
    and no conflicting or maturity claims. Explicit limited adoption was not found.
    """
    if limit<=0:
        return []
    exclude=set(exclude_ids)
    pool=[a for a in assessments
          if a['candidate_id'] not in exclude and a['decision']=='needs_review'
          and a.get('reason')=='insufficient_explicit_evidence'
          and a.get('flags',{}).get('early_stage') and a.get('flags',{}).get('relevance')
          and not a['flags'].get('mass_adoption') and not a['flags'].get('industry_standard')]
    return sorted(pool,key=lambda a:(-a['score'],a['candidate_id']))[:limit]


def assess_automatically(candidates,as_of,search_query=None):
    results=[analyse(c,as_of,search_query) for c in candidates]
    return {'as_of':as_of,'model_version':VERSION,'assessments':results,
            'top15':[r for r in sorted(results,key=lambda r:(-r['score'],r['candidate_id'])) if r['decision']=='likely_weak'][:15],
            'counts':{key:sum(r['decision']==key for r in results) for key in ('likely_weak','likely_mature','needs_review')}}
