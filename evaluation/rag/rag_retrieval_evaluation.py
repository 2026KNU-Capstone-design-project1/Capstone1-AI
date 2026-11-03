"""건설현장 안전 기준 RAG 검색 성능 평가 (Qwen 생성 평가와 분리).

원본 Capstone_LLM.ipynb에서 검색 평가 로직만 정리한 재현용 스크립트.
평가 단위: rule(판단 기준), case(개별 사례); 지표: Recall@K, Precision@K, MRR@K.

실행:
    python rag_retrieval_evaluation.py --data risk_data_clean.csv \
        --queries rag_test_queries_reviewed.csv --output-dir evaluation/rag_results

선택적 확장 실험:
    python rag_retrieval_evaluation.py --data risk_data_clean.csv \
        --queries rag_test_queries_reviewed.csv --weight-sweep --with-reranker

필요 패키지: pandas numpy sentence-transformers rank-bm25 torch
Reranker 사용 시 추가 패키지: transformers
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RULE_COLUMNS = ["대분류", "위험요소", "판단항목", "정상", "위반", "예상 위험", "법적사항"]
CASE_COLUMNS = ["라벨", "위험요소", "판단항목", "개선내용", "안전대책", "법적사항"]
REQUIRED_COLUMNS = list(dict.fromkeys(RULE_COLUMNS + CASE_COLUMNS))
GOLD_COLUMNS = ["query_id", "query", "relevant_rule_ids", "relevant_case_ids"]


def load_documents(data_path: Path, deduplicate: bool = True):
    """원래 노트북과 동일한 rule/case 문서 구조·ID를 재구성한다.

    ID는 중복 제거 *전에* 부여한다. 중복 문서는 먼저 등장한 ID를 대표로 삼는다.
    """
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    missing = [name for name in REQUIRED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"위험 데이터셋에 없는 열: {missing}")
    df = df.copy()
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    rules = df[RULE_COLUMNS].drop_duplicates().reset_index(drop=True)
    cases = df[CASE_COLUMNS].copy()
    documents = []

    for index, row in rules.iterrows():
        content = "\n".join([
            "[위험성평가 판단 기준]",
            f"대분류: {row['대분류']}",
            f"위험요소: {row['위험요소']}",
            f"판단항목: {row['판단항목']}",
            f"정상 기준: {row['정상']}",
            f"위반 기준: {row['위반']}",
            f"예상 위험: {row['예상 위험']}",
            f"법적사항: {row['법적사항']}",
        ])
        documents.append({"id": f"rule_{index + 1}", "type": "rule", "content": content})

    for _, row in cases.iterrows():
        content = "\n".join([
            "[위험성평가 사례]",
            f"위험요소: {row['위험요소']}",
            f"판단항목: {row['판단항목']}",
            f"개선내용: {row['개선내용']}",
            f"안전대책: {row['안전대책']}",
            f"법적사항: {row['법적사항']}",
        ])
        documents.append({"id": f"case_{row['라벨']}", "type": "case", "content": content})

    seen, canonical_ids, unique = {}, {}, []
    for doc in documents:
        key = (doc["type"], doc["content"])
        if key not in seen:
            seen[key] = doc["id"]
            unique.append(doc)
        canonical_ids[doc["id"]] = seen[key]

    if not deduplicate:
        return documents, {doc["id"]: doc["id"] for doc in documents}
    return unique, canonical_ids


def parse_ids(value):
    """정답 문서 ID를 ';' 구분 문자열/리스트에서 읽어 온다."""
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    if pd.isna(value) or not str(value).strip():
        return []
    return [token.strip() for token in str(value).split(";") if token.strip()]


def load_test_queries(path: Path, canonical_ids: dict, available_ids: set):
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [col for col in GOLD_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"평가 질문 CSV에 없는 열: {missing}")
    if df["query_id"].duplicated().any():
        raise ValueError("query_id가 중복되었습니다.")
    tests = []
    for _, row in df.iterrows():
        relevant = {}
        for doc_type in ("rule", "case"):
            raw_ids = parse_ids(row[f"relevant_{doc_type}_ids"])
            relevant[doc_type] = list(dict.fromkeys(canonical_ids.get(x, x) for x in raw_ids))
            unknown = set(relevant[doc_type]) - available_ids
            if unknown:
                raise ValueError(
                    f"{row['query_id']}: {doc_type} 정답 ID가 문서에 없음: {sorted(unknown)}. "
                    "원본 CSV/문서 생성 로직/정답 파일의 버전을 확인하세요."
                )
        if not str(row["query"]).strip():
            raise ValueError(f"{row['query_id']}: 빈 질문")
        tests.append({
            "query_id": row["query_id"], "query": str(row["query"]),
            "category": row.get("category", ""), "relevant": relevant,
        })
    if not tests:
        raise ValueError("평가할 질문이 없습니다.")
    return tests


def tokenize_korean(text: str):
    """노트북과 동일한 정규식 단어 토큰 + 문자 2-gram."""
    words = re.findall(r"[가-힣a-zA-Z0-9]+", str(text).lower())
    tokens = words.copy()
    for word in words:
        tokens.extend(word[i:i + 2] for i in range(len(word) - 1))
    return tokens


def rrf(ranked_lists, weights=None, k=60, top_k=5):
    """순위 기반 융합. 점수 스케일이 다른 검색 결과도 결합 가능."""
    weights = weights or [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("ranked_lists와 weights의 길이가 다릅니다.")
    fused = defaultdict(float)
    for ranked, weight in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] += weight / (k + rank)
    return sorted(fused, key=lambda doc_id: (-fused[doc_id], doc_id))[:top_k]


def calculate_metrics(retrieved_ids, relevant_ids, k=5):
    """노트북의 기존 정의: Precision@K 분모는 K, MRR은 첫 정답의 역순위."""
    retrieved = list(retrieved_ids)[:k]
    relevant = set(relevant_ids)
    if not relevant:
        # 정답이 0개면 평가 평균에서 제외해야 하므로 NaN으로 표시
        return {"recall": np.nan, "precision": np.nan, "mrr": np.nan}
    hits = len(set(retrieved) & relevant)
    mrr = next((1.0 / rank for rank, item in enumerate(retrieved, 1) if item in relevant), 0.0)
    return {"recall": hits / len(relevant), "precision": hits / k, "mrr": mrr}


class Retriever:
    def __init__(self, documents, embedding_model_name=EMBEDDING_MODEL, device=None):
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.documents = documents
        self.indices = {
            doc_type: [i for i, doc in enumerate(documents) if doc["type"] == doc_type]
            for doc_type in ("rule", "case")
        }
        for doc_type, indices in self.indices.items():
            if not indices:
                raise ValueError(f"{doc_type} 문서가 없습니다.")
        self.model = SentenceTransformer(embedding_model_name, device=device)
        self.embeddings = self.model.encode(
            [doc["content"] for doc in documents],
            batch_size=8, show_progress_bar=True, normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self.bm25 = {
            kind: BM25Okapi([tokenize_korean(documents[i]["content"]) for i in indices])
            for kind, indices in self.indices.items()
        }
        self.query_cache = {}

    def base_rankings(self, query, doc_type):
        if query not in self.query_cache:
            self.query_cache[query] = self.model.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True
            )[0]
        indices = self.indices[doc_type]
        dense_scores = self.embeddings[indices] @ self.query_cache[query]
        sparse_scores = self.bm25[doc_type].get_scores(tokenize_korean(query))
        dense_order = np.argsort(-dense_scores, kind="stable")
        sparse_order = np.argsort(-sparse_scores, kind="stable")
        dense = [self.documents[indices[i]]["id"] for i in dense_order]
        sparse = [self.documents[indices[i]]["id"] for i in sparse_order]
        return dense, sparse


class Reranker:
    def __init__(self, model_name=RERANKER_MODEL, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def rank(self, query, ids, by_id, batch_size=8):
        scored = []
        for start in range(0, len(ids), batch_size):
            chunk = ids[start:start + batch_size]
            pairs = [(query, by_id[doc_id]["content"]) for doc_id in chunk]
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**inputs).logits.view(-1).float().cpu().tolist()
            scored.extend(zip(chunk, logits))
        return [doc_id for doc_id, _ in sorted(scored, key=lambda item: -item[1])]


def evaluate(retriever, tests, top_k=5, candidate_k=10,
             weight_sweep=False, reranker=None):
    detail = []
    by_id = {doc["id"]: doc for doc in retriever.documents}
    if candidate_k < top_k:
        raise ValueError("candidate_k는 top_k 이상이어야 합니다.")

    for test in tests:
        for doc_type in ("rule", "case"):
            relevant = test["relevant"][doc_type]
            if not relevant:
                continue  # 기준 정답이 없는 항목은 해당 유형 평가에서 제외
            dense, sparse = retriever.base_rankings(test["query"], doc_type)
            hybrid = rrf([dense[:top_k], sparse[:top_k]], top_k=top_k)
            methods = {
                "BGE-M3": dense[:top_k],
                "BM25": sparse[:top_k],
                "Hybrid-RRF": hybrid,
            }
            if weight_sweep:
                for dense_weight in (1.0, 1.5, 2.0, 3.0):
                    methods[f"Weighted-RRF-{dense_weight:g}:1"] = rrf(
                        [dense[:candidate_k], sparse[:candidate_k]],
                        weights=[dense_weight, 1.0], top_k=top_k
                    )
            if reranker is not None:
                # 원본 노트북과 동일하게 BGE/BM25 후보 각각 상위 candidate_k 합집합
                candidate_ids = list(dict.fromkeys(dense[:candidate_k] + sparse[:candidate_k]))
                reranked = reranker.rank(test["query"], candidate_ids, by_id)
                methods["Reranker"] = reranked[:top_k]
                hybrid_candidates = rrf(
                    [dense[:candidate_k], sparse[:candidate_k]], top_k=candidate_k
                )
                methods["Fusion"] = rrf(
                    [hybrid_candidates, reranked[:candidate_k]], top_k=top_k
                )

            for method, retrieved in methods.items():
                scores = calculate_metrics(retrieved, relevant, k=top_k)
                detail.append({
                    "query_id": test["query_id"], "category": test["category"],
                    "type": doc_type, "method": method, "query": test["query"],
                    "relevant_ids": ";".join(relevant),
                    "retrieved_ids": ";".join(retrieved),
                    "hit_ids": ";".join(sorted(set(retrieved) & set(relevant))),
                    **scores,
                })

    detail_df = pd.DataFrame(detail)
    summary_df = (detail_df.groupby(["type", "method"], as_index=False)
                  [["recall", "precision", "mrr"]].mean())
    return detail_df, summary_df


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("risk_data_clean.csv"))
    parser.add_argument("--queries", type=Path, default=Path("rag_test_queries_reviewed.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/rag_results"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--device", default=None, help="cuda 또는 cpu (기본: 자동 감지)")
    parser.add_argument("--weight-sweep", action="store_true")
    parser.add_argument("--with-reranker", action="store_true")
    parser.add_argument("--no-dedup", action="store_true", help="중복 제거 전 검색 성능 평가")
    args = parser.parse_args(argv)
    if args.top_k < 1 or args.candidate_k < args.top_k:
        parser.error("0 < top-k <= candidate-k 이어야 합니다.")

    documents, canonical_ids = load_documents(args.data, deduplicate=not args.no_dedup)
    tests = load_test_queries(args.queries, canonical_ids, {d["id"] for d in documents})
    print(f"문서 {len(documents)}개 / 평가 질문 {len(tests)}개")
    retriever = Retriever(documents, device=args.device)
    reranker = Reranker(device=args.device) if args.with_reranker else None
    detail, summary = evaluate(
        retriever, tests, top_k=args.top_k, candidate_k=args.candidate_k,
        weight_sweep=args.weight_sweep, reranker=reranker,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "rag_evaluation_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "rag_evaluation_summary.csv", index=False, encoding="utf-8-sig")
    print("\n검색 성능 요약")
    print(summary.to_string(index=False))
    print(f"\n결과 저장 위치: {args.output_dir.resolve()}")
    return detail, summary


if __name__ == "__main__":
    main()
