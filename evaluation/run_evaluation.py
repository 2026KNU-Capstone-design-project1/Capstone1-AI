
import sys
import json
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag_test import search_criteria, generate_assessment


EVALUATION_DIR = Path(__file__).resolve().parent
TEST_CSV = EVALUATION_DIR / "test_cases.csv"
SEARCH_METHOD = "word"

RESULT_CSV = EVALUATION_DIR / f"evaluation_results_{SEARCH_METHOD}.csv"

REQUIRED_FIELDS = [
    "위험요소",
    "예상 재해유형",
    "위험도",
    "위험도 판단 근거",
    "위험내용",
    "관련 안전 기준",
    "법적사항",
    "개선대책",
    "안전대책",
    "위험도 평가 시 추가 확인사항",
]

ALLOWED_RISK_LEVELS = ["상", "중", "하", "추가 확인 필요"]


def validate_assessment(data):
    """생성된 평가 결과의 필수 항목과 위험도 값을 검증한다."""

    if not isinstance(data, dict):
        return False, "JSON 객체가 아님"

    missing = [
        field for field in REQUIRED_FIELDS
        if field not in data
    ]

    if missing:
        return False, f"필수 항목 누락: {missing}"

    if data["위험도"] not in ALLOWED_RISK_LEVELS:
        return False, f"잘못된 위험도: {data['위험도']}"

    return True, ""


def parse_relevant_rows(value):
    """CSV의 정답 기준 행 번호를 정수 집합으로 변환한다."""

    if pd.isna(value) or str(value).strip() == "":
        return set()

    return {
        int(row.strip())
        for row in str(value).split(";")
        if row.strip()
    }


def evaluate_retrieval(retrieval_results, relevant_rows, top_k=5):
    """검색 결과와 정답 기준을 비교하여 Precision@K와 Recall@K를 계산한다."""

    retrieved_rows = {
        int(item["기준_행"])
        for item in retrieval_results[:top_k]
    }

    if not relevant_rows:
        return {
            "검색 평가 가능": False,
            "Precision@5": None,
            "Recall@5": None,
            "정답 검색 수": 0,
        }

    correct_rows = retrieved_rows & relevant_rows

    precision = (
        len(correct_rows) / len(retrieved_rows)
        if retrieved_rows else 0.0
    )

    recall = len(correct_rows) / len(relevant_rows)

    return {
        "검색 평가 가능": True,
        "Precision@5": round(precision, 4),
        "Recall@5": round(recall, 4),
        "정답 검색 수": len(correct_rows),
    }


def run_evaluation():
    test_df = pd.read_csv(TEST_CSV, encoding="utf-8-sig")

    if "relevant_rows" not in test_df.columns:
        raise ValueError(
            "test_cases.csv에 relevant_rows 컬럼이 없습니다."
        )

    results = []

    print(f"전체 테스트 수: {len(test_df)}")

    for index, row in test_df.iterrows():
        test_id = row["test_id"]
        scene = row["scene"]
        relevant_rows = parse_relevant_rows(row["relevant_rows"])

        print(f"\n===== {test_id} ({index + 1}/{len(test_df)}) =====")
        print(f"현장 상황: {scene}")

        retrieval_results = []
        assessment_data = {}
        raw_response = ""
        error_message = ""
        validation_message = ""
        validation = False

        search_time = 0.0
        generation_time = 0.0

        total_start = time.perf_counter()

        try:
            search_start = time.perf_counter()

            retrieval_results = search_criteria(
                scene,
                top_k=5,
                method=SEARCH_METHOD,
            )

            search_time = time.perf_counter() - search_start

            print(f"검색 결과 수: {len(retrieval_results)}")
            print(f"검색 시간: {search_time:.4f}초")

            retrieval_metrics = evaluate_retrieval(
                retrieval_results,
                relevant_rows,
                top_k=5,
            )

            if retrieval_results:
                generation_start = time.perf_counter()

                assessment_data = generate_assessment(
                    scene,
                    retrieval_results,
                    return_json=True,
                )

                generation_time = time.perf_counter() - generation_start

                raw_response = json.dumps(
                    assessment_data,
                    ensure_ascii=False,
                )

                validation, validation_message = validate_assessment(
                    assessment_data
                )

            else:
                validation_message = "관련 안전 기준 검색 실패"

            print(f"평가 생성 시간: {generation_time:.4f}초")
            print(f"JSON 검증 결과: {validation}")

            if retrieval_metrics["검색 평가 가능"]:
                print(
                    f"Precision@5: "
                    f"{retrieval_metrics['Precision@5']:.2%}"
                )
                print(
                    f"Recall@5: "
                    f"{retrieval_metrics['Recall@5']:.2%}"
                )
            else:
                print("검색 성능 평가 불가: 정답 기준이 지정되지 않음")

            if validation_message:
                print(f"검증 내용: {validation_message}")

        except Exception as e:
            validation = False
            error_message = f"{type(e).__name__}: {e}"
            validation_message = error_message

            retrieval_metrics = evaluate_retrieval(
                retrieval_results,
                relevant_rows,
                top_k=5,
            )

            print(f"오류 발생: {error_message}")

        total_time = time.perf_counter() - total_start

        retrieved_rows = [
            {
                "기준_행": item["기준_행"],
                "유사도": item["유사도"],
            }
            for item in retrieval_results
        ]

        result = {
            "test_id": test_id,
            "scene": scene,
            "검색 결과 수": len(retrieval_results),
            "검색 기준 상세": json.dumps(
                retrieved_rows,
                ensure_ascii=False,
            ),
            "정답 기준 행": json.dumps(
                sorted(relevant_rows),
                ensure_ascii=False,
            ),
            "정답 검색 수": retrieval_metrics["정답 검색 수"],
            "검색 평가 가능": retrieval_metrics["검색 평가 가능"],
            "Precision@5": retrieval_metrics["Precision@5"],
            "Recall@5": retrieval_metrics["Recall@5"],
            "RAG 검색 시간(초)": round(search_time, 4),
            "평가 생성 시간(초)": round(generation_time, 4),
            "전체 처리 시간(초)": round(total_time, 4),
            "JSON 원본 응답": raw_response,
            "JSON 및 필드 검증": validation,
            "검증 실패 사유": validation_message,
            "오류": error_message,
        }

        for field in REQUIRED_FIELDS:
            result[field] = (
                assessment_data.get(field, "")
                if isinstance(assessment_data, dict)
                else ""
            )

        results.append(result)

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        RESULT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    total_count = len(results)
    success_count = sum(
        result["JSON 및 필드 검증"] for result in results
    )

    evaluated_results = [
        result for result in results
        if result["검색 평가 가능"]
    ]

    print("\n===== 성능 측정 요약 =====")
    print(f"전체 테스트: {total_count}건")
    print(f"JSON 검증 성공: {success_count}건")

    if evaluated_results:
        avg_precision = sum(
            result["Precision@5"]
            for result in evaluated_results
        ) / len(evaluated_results)

        avg_recall = sum(
            result["Recall@5"]
            for result in evaluated_results
        ) / len(evaluated_results)

        print(f"평가 가능한 테스트: {len(evaluated_results)}건")
        print(f"평균 Precision@5: {avg_precision:.2%}")
        print(f"평균 Recall@5: {avg_recall:.2%}")

    if total_count:
        print(
            "평균 전체 처리 시간: "
            f"{sum(r['전체 처리 시간(초)'] for r in results) / total_count:.2f}초"
        )
        print(
            "평균 RAG 검색 시간: "
            f"{sum(r['RAG 검색 시간(초)'] for r in results) / total_count:.4f}초"
        )
        print(
            "평균 평가 생성 시간: "
            f"{sum(r['평가 생성 시간(초)'] for r in results) / total_count:.2f}초"
        )

    print(f"\n결과 저장 위치: {RESULT_CSV}")


if __name__ == "__main__":
    run_evaluation()
