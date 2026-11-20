"""Retrieval of PDF-extracted UNVERIFIED law draft records.

This module does not establish whether a statute is in force, whether it applies
at a particular date, or whether an observed scene violates it.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / 'data' / 'law_articles.json'


def _tokens(value):
    compact = re.sub(r'\s+', '', value.lower())
    return [compact[i:i+2] for i in range(max(0,len(compact)-1))]


def load_law_articles(path=DEFAULT_PATH):
    with Path(path).open(encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data,list):
        raise ValueError('law_articles.json must be a JSON list')
    return data


def search_law_articles(query, top_k=5, path=DEFAULT_PATH, min_score=0.0):
    """Small dependency-free character-bigram TF-IDF lookup for manual review."""
    records = load_law_articles(path)
    if not query.strip() or not records:
        return []
    texts = [
        ' '.join([r.get('law_name',''), r.get('article_number',''),
                  r.get('article_title','')+' '+r.get('article_title',''),
                  r.get('article_text','')])
        for r in records
    ]
    docs = [Counter(_tokens(text)) for text in texts]
    df = Counter()
    for counts in docs:
        df.update(counts.keys())
    q = Counter(_tokens(query))
    n = len(records)
    weights = {term: math.log((n+1)/(df.get(term,0)+1))+1 for term in q}
    scored = []
    for r, doc in zip(records,docs):
        dot = sum(q[t] * doc.get(t,0) * w * w for t,w in weights.items())
        denom = math.sqrt(sum((v*(math.log((n+1)/(df[t]+1))+1))**2 for t,v in doc.items())) or 1
        qdenom = math.sqrt(sum((v*weights[t])**2 for t,v in q.items())) or 1
        score = dot/(denom*qdenom)
        if score > 0 and score >= min_score:
            scored.append({'score':round(score,4),**r})
    return sorted(scored,key=lambda x:x['score'],reverse=True)[:top_k]


if __name__ == '__main__':
    for query in ['안전난간 추락 방지', '개구부 방호 조치', '용접 화재 방지', '소화기 내용연수']:
        print(f'\nQ: {query}')
        for item in search_law_articles(query, top_k=3):
            print(f"  {item['score']:.3f} | {item['law_name']} {item['article_number']} | {item['article_title']} | p.{item['source_pages']} | {item['verification_status']}")
