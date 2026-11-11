"""Exploratory comparison of lexical reranking and abstention against existing TF-IDF.
Requires project law_rag.py and reviewed evaluation/rag/law_eval_queries.csv.
No ground-truth labels are used in ranking or abstention.
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == 'evaluation' else Path.cwd()
sys.path.insert(0, str(ROOT))
from law_rag import search_law_articles, load_law_articles

STOP = {"건설현장", "근로자", "작업", "작업장", "관련", "기준", "및", "등", "의", "시", "대한", "관리", "안전", "조치"}


def terms(s):
    return {t for t in re.findall(r'[가-힣a-zA-Z0-9]+', (s or '').lower()) if len(t) >= 2 and t not in STOP}


def rerank(query, documents, min_score=0.08):
    qt = terms(query)
    scored = []
    for doc in documents:
        base = doc['score']
        title = terms(doc.get('article_title', ''))
        body = terms(doc.get('article_text', ''))
        title_hits = len(qt & title)
        body_hits = len(qt & body)
        # Body may contain unrelated generic expressions. Require stronger matching to abstain less naively.
        title_ratio = title_hits / max(1, len(qt))
        body_ratio = body_hits / max(1, len(qt))
        score = 0.70 * base + 0.22 * title_ratio + 0.08 * body_ratio
        # Experimental, fixed parameters; not law-applicability verification.
        accept = base >= min_score and (title_hits >= 1 or body_hits >= 2 or base >= 0.25)
        if accept:
            scored.append((score, doc))
    scored.sort(key=lambda x: (-x[0], x[1]['id']))
    return [d for _, d in scored[:3]]


def metrics(rows, fn):
    ap, ar, rr = [], [], []
    reject_ok = 0
    reject_n = 0
    per = []
    for row in rows:
        gold = {x.strip() for x in row['gold_ids'].split(';') if x.strip()}
        pred = fn(row['query'])
        ids = [x['id'] for x in pred]
        if gold:
            hits = len(gold.intersection(ids))
            ap.append(hits/3)
            ar.append(hits/len(gold))
            rr.append(next((1/i for i, ident in enumerate(ids, 1) if ident in gold), 0.0))
        else:
            reject_n += 1
            reject_ok += int(not ids)
        per.append({'query_id': row['query_id'], 'query': row['query'],
                    'gold_ids': ';'.join(sorted(gold)), 'predicted_ids': ';'.join(ids),
                    'answerable': str(bool(gold)).lower()})
    return {'answerable': len(ap), 'no_answer': reject_n,
            'precision_at_3': sum(ap)/len(ap) if ap else None,
            'recall_at_3': sum(ar)/len(ar) if ar else None,
            'mrr_at_3': sum(rr)/len(rr) if rr else None,
            'no_answer_accuracy': reject_ok/reject_n if reject_n else None}, per


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queries', default='evaluation/rag/law_eval_queries.csv')
    p.add_argument('--output', default='evaluation/rag/rerank_comparison')
    args = p.parse_args()
    with open(args.queries, newline='', encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['review_status'].strip().lower() == 'approved']
    if not rows:
        raise SystemExit('No approved evaluation queries')
    all_k = len(load_law_articles())
    methods = {
        'baseline': lambda q: search_law_articles(q, top_k=3, min_score=0),
        'filtered_008': lambda q: search_law_articles(q, top_k=3, min_score=0.08),
        'rerank_candidate': lambda q: rerank(q, search_law_articles(q, top_k=all_k, min_score=0)),
    }
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    summary, details = [], []
    for name, fn in methods.items():
        summary_row, cases = metrics(rows, fn)
        summary.append({'method': name, 'approved': len(rows), **summary_row})
        details.extend({'method': name, **c} for c in cases)
    for filename, records in [('summary.csv', summary), ('details.csv', details)]:
        with (outdir/filename).open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=records[0].keys()); w.writeheader(); w.writerows(records)
    for x in summary:
        print(x)
    print('saved:', outdir)
    print('Exploratory: do not claim held-out improvement; L015 is excluded while pending.')

if __name__ == '__main__':
    main()
