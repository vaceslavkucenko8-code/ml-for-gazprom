"""Learned source-based priority, separate from binary detection and probability."""
import math
import re
from datetime import date

PATTERNS={
 'early':r'prototype|proof.of.concept|pre.seed|pilot|прототип|пилот|исследован',
 'limited':r'controlled availability|limited (?:deployment|availability)|not yet commercially|no commercially available|ограниченн\w* внедрен',
 'mature':r'mass production|widely deployed|millions of users|industry standard|массов\w* внедрен|серийн\w* производств',
 'future':r'plans to|expected to|will launch|планиру|ожидается|будет',
 'funding':r'funding|seed round|series [abc]|финансирован|инвестици',
}


def features(documents,as_of):
    cutoff=date.fromisoformat(as_of)
    docs=[d for d in documents if d.get('text','').strip()]
    result={k:0.0 for k in [*PATTERNS,'log_sources','dated_fraction','recent_fraction','mean_log_length']}
    if not docs:return result
    n=len(docs)
    result['log_sources']=math.log1p(n)
    for d in docs:
        text=(d.get('title') or d.get('title_original') or '')+'\n'+d['text']
        for key,pattern in PATTERNS.items():
            result[key]+=float(bool(re.search(pattern,text,re.I)))/n
        result['mean_log_length']+=math.log1p(len(text))/n
        published=d.get('published_at')
        if published:
            try:
                age=(cutoff-date.fromisoformat(published[:10])).days
                if age>=0:
                    result['dated_fraction']+=1/n
                    result['recent_fraction']+=float(age<=365)/n
            except ValueError:pass
    return result


def predict(documents,model,as_of):
    values=features(documents,as_of)
    terms=[{'feature':k,'value':v,'contribution':v*model['weights'].get(k,0)} for k,v in values.items()]
    raw=model['intercept']+sum(t['contribution'] for t in terms)
    return {'score':min(7,max(3,raw)),'raw_score':raw,'score_kind':'learned_reference_priority_3_7',
            'probability_weak':None,'model_version':model['model_version'],
            'deployment_enabled':model['deployment_enabled'],
            'contributions':sorted(terms,key=lambda t:-abs(t['contribution']))}
