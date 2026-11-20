"""Law retrieval evaluation with human-reviewed, corpus-scoped gold labels.

Run from Capstone1-AI project root:
  python evaluation/evaluate_law_rag.py init
  # fill evaluation/rag/law_eval_queries.csv manually, then:
  python evaluation/evaluate_law_rag.py evaluate

Only rows with review_status='approved' are scored. Gold labels must not
be inferred from retrieval outputs. No-answer means no relevant article in
*this 45-document corpus*, not no law exists in the legal system.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from law_rag import load_law_articles, search_law_articles

DEFAULT_GOLD = ROOT / 'evaluation' / 'rag' / 'law_eval_queries.csv'
DEFAULT_DETAIL = ROOT / 'evaluation' / 'rag' / 'law_eval_details.csv'
DEFAULT_SUMMARY = ROOT / 'evaluation' / 'rag' / 'law_eval_summary.csv'

# Questions are proposed for annotator review, not verified relevance judgments.
QUESTIONS = [
    ('L001', '안전난간 구조 및 설치 요건'),
    ('L002', '개구부 추락 방호 조치'),
    ('L003', '작업발판 설치 및 고정'),
    ('L004', '비계 구조 및 설치 기준'),
    ('L005', '작업장 통로 및 자재 적치'),
    ('L006', '용접 작업 화재 예방'),
    ('L007', '용접 작업 화재감시자 배치'),
    ('L008', '이동식 크레인 인양 작업 안전조치'),
    ('L009', '지게차 운전자 시야 가림 적재'),
    ('L010', '통로 바닥 이동전선 안전관리'),
    ('L011', '안전대 사용 및 추락 예방'),
    ('L012', '소화기 내용연수 관리'),
    ('L013', '밀폐공간 산소농도 측정'),
    ('L014', '물질안전보건자료 게시'),
    ('L015', '고압가스 용기 검사'),
]
FIELDS = ['query_id', 'query', 'gold_ids', 'corpus_answerable', 'review_status', 'notes']


def write_csv(path: Path, fields: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def init(path: Path):
    if path.exists():
        raise SystemExit(f'{path} already exists; will not overwrite annotation work.')
    rows = [dict(query_id=qid, query=q, gold_ids='', corpus_answerable='',
                 review_status='pending', notes='PDF 초안의 조문 범위 및 공식 법령 확인 후 정답 기재')
            for qid, q in QUESTIONS]
    write_csv(path, FIELDS, rows)
    print(f'Created {path} with {len(rows)} UNLABELLED queries.')
    print('For each row, annotate gold_ids (semicolon-separated), corpus_answerable true/false, review_status approved.')
    print('Do NOT infer gold_ids from top search results. All extracted laws remain unverified.')


def safe_div(a, b):
    return a / b if b else None


def evaluate(gold_path: Path, detail_path: Path, summary_path: Path, k: int, threshold: float):
    corpus = load_law_articles()
    valid_ids = {a['id'] for a in corpus}
    with gold_path.open('r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or any(f not in reader.fieldnames for f in FIELDS):
            raise SystemExit('Missing required evaluation CSV columns')
        source = list(reader)
    ids = set()
    approved = []
    for line, row in enumerate(source, 2):
        qid = row['query_id'].strip()
        if not qid or qid in ids:
            raise SystemExit(f'CSV line {line}: empty or duplicate query_id')
        ids.add(qid)
        if row['review_status'].strip().lower() != 'approved':
            continue
        ans = row['corpus_answerable'].strip().lower()
        if ans not in {'true', 'false'}:
            raise SystemExit(f'CSV line {line}: approved row needs corpus_answerable true/false')
        gold = set(x.strip() for x in row['gold_ids'].split(';') if x.strip())
        if (ans == 'true' and not gold) or (ans == 'false' and gold):
            raise SystemExit(f'CSV line {line}: answerable must match gold_ids emptiness')
        unknown = gold - valid_ids
        if unknown:
            raise SystemExit(f'CSV line {line}: unknown corpus IDs {sorted(unknown)}')
        if not row['query'].strip():
            raise SystemExit(f'CSV line {line}: empty query')
        approved.append((qid, row['query'].strip(), gold, ans == 'true'))

    if not approved:
        raise SystemExit('No approved labels yet. Review CSV and set review_status=approved. Nothing scored.')

    detail = []
    summ = []
    for method, min_score in [('baseline', 0.0), ('filtered', threshold)]:
        answerable_rows = []
        unanswered_rows = []
        for qid, query, gold, answerable in approved:
            hits = search_law_articles(query, top_k=k, min_score=min_score)
            pred = [h['id'] for h in hits]
            if answerable:
                tp = len(gold.intersection(pred))
                p = tp / k  # fixed K denominator; empty slots count as nonrelevant
                recall = tp / len(gold)
                rr = next((1 / (i + 1) for i, x in enumerate(pred) if x in gold), 0.0)
                answerable_rows.append((p, recall, rr))
                no_answer_correct = ''
            else:
                p = recall = rr = ''
                no_answer_correct = int(len(pred) == 0)
                unanswered_rows.append(no_answer_correct)
            detail.append({
                'method': method, 'query_id': qid, 'query': query,
                'answerable_in_corpus': str(answerable).lower(),
                'gold_ids': ';'.join(sorted(gold)), 'retrieved_ids': ';'.join(pred),
                'precision_at_k': p, 'recall_at_k': recall, 'reciprocal_rank': rr,
                'correct_no_answer': no_answer_correct,
            })
        summ.append({
            'method': method, 'top_k': k, 'min_score': min_score,
            'approved_queries': len(approved),
            'answerable_queries': len(answerable_rows),
            'no_answer_queries': len(unanswered_rows),
            'precision_at_k': safe_div(sum(x[0] for x in answerable_rows), len(answerable_rows)),
            'recall_at_k': safe_div(sum(x[1] for x in answerable_rows), len(answerable_rows)),
            'mrr_at_k': safe_div(sum(x[2] for x in answerable_rows), len(answerable_rows)),
            'no_answer_accuracy': safe_div(sum(unanswered_rows), len(unanswered_rows)),
        })
    write_csv(detail_path, list(detail[0]), detail)
    write_csv(summary_path, list(summ[0]), summ)
    print(f'Approved: {len(approved)}/{len(source)} queries; outputs: {detail_path}, {summary_path}')
    for row in summ:
        print(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['init', 'evaluate'])
    p.add_argument('--gold', type=Path, default=DEFAULT_GOLD)
    p.add_argument('--details', type=Path, default=DEFAULT_DETAIL)
    p.add_argument('--summary', type=Path, default=DEFAULT_SUMMARY)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--threshold', type=float, default=0.08)
    a = p.parse_args()
    if a.k < 1 or not 0 <= a.threshold <= 1:
        p.error('k >= 1 and threshold between 0 and 1 are required')
    if a.command == 'init':
        init(a.gold)
    else:
        evaluate(a.gold, a.details, a.summary, a.k, a.threshold)


if __name__ == '__main__':
    main()
