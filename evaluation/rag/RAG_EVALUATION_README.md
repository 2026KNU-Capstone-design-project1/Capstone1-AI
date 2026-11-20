# RAG 검색 성능 평가 — 정리본

`Capstone_LLM.ipynb`의 검색 평가 실험을 재현하기 좋게 정리했습니다. 원본 124개 셀 중 모델 시연, 데이터 탐색, 반복 출력, 보고서 생성 디버깅 코드는 제외했습니다.

## 파일

- `rag_retrieval_evaluation.py`: 실행 가능한 검색 성능 평가 스크립트
- `rag_test_queries_reviewed.csv`: 노트북의 25개 질문 + 수동 검토 결과와 중복 사례 문서 ID 보정
- `Capstone_LLM_RAG_clean.ipynb`: 결과를 확인하기 위한 간단한 Jupyter 실행 노트북 (위 두 파일 필요)
- **별도 필요**: 기존 프로젝트의 `risk_data_clean.csv` (제공된 노트북에는 파일 내용이 없음)

## 설치

```bash
pip install numpy pandas sentence-transformers rank-bm25 torch
# Reranker까지 실행할 경우
pip install transformers
```

## 실행

```bash
python rag_retrieval_evaluation.py \
  --data risk_data_clean.csv \
  --queries rag_test_queries_reviewed.csv \
  --output-dir evaluation/rag_results
```

추가 비교 실험 (모델 다운로드 / 하드웨어 리소스 필요):

```bash
python rag_retrieval_evaluation.py --data risk_data_clean.csv \
  --queries rag_test_queries_reviewed.csv \
  --weight-sweep --with-reranker
```

실행 결과: `rag_evaluation_detail.csv`, `rag_evaluation_summary.csv`.

## 평가 조건

- **검색 문서**: 판단 기준(rule)과 개별 사례(case)를 구분
- **검색 모델**: BAAI/bge-m3, BM25, RRF(기본 검색의 경우 각 상위 5개), 옵션 가중치 RRF, BAAI/bge-reranker-v2-m3, Fusion
- **평가**: Recall@5, Precision@5, MRR@5 (질문 및 문서 유형별 평균)
- **중복 처리**: 문서 내용을 기준으로 완전 일치 중복 제거, 원본 ID 유지, 제거된 ID는 최초 등장 문서 ID로 재매핑
- **정답 데이터**: 후반부 수동 수정(Q10 `rule_92` 제외 포함)을 반영. 이 정답은 노트북 저자의 수동 라벨링이며 별도 전문가 검증을 뜻하지 않음

## CI/CD 주의

기본 실행도 임베딩 모델 다운로드 및 모델 추론이 필요합니다. **일반 PR마다 필수 통과하는 CI 작업으로 돌리기보다는, 수동 실행 또는 모델 캐시가 있는 전용 러너에서 실행**하는 것을 권장합니다.

원본 `risk_data_clean.csv`가 이번 업로드에 포함되지 않아 전체 검색 성능 수치는 재실행·검증하지 못했습니다. 정답 ID 검증과 모듈 문법 확인은 수행했습니다. 프로젝트의 데이터셋 및 라이브러리 버전으로 재현 결과를 확인해야 합니다.
