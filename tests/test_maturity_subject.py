from src.automatic import analyse
from test_automatic import candidate


def test_language_maturity_not_llm_generation_maturity():
    c=candidate('Structured Text is a widely used programming language for controllers.',title_original='LLM Structured Text generation')
    c['query']='LLM Structured Text generation'
    result=analyse(c,'2026-09-20')
    assert not result['flags']['mass_adoption']
    assert result['decision']=='needs_review'


def test_explicit_llm_method_maturity_still_detected():
    c=candidate('LLM Structured Text generation is widely used in production controllers.',title_original='LLM Structured Text generation')
    c['query']='LLM Structured Text generation'
    assert analyse(c,'2026-09-20')['decision']=='likely_mature'
