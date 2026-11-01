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
RESULT_CSV = EVALUATION_DIR / "evaluation_results.csv"

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


def run_evaluation():
    test_df = pd.read_csv(TEST_CSV, encoding="utf-8-sig")

    results = []

    print(f"전체 테스트 수: {len(test_df)}")

    for index, row in test_df.iterrows():
        test_id = row["test_id"]
        scene = row["scene"]

        print(f"\n===== {test_id} ({index + 1}/{len(test_df)}) =====")
        print(f"현장 상황: {scene}")

        retrieval_results = []
        assessment_data = {}
        raw_response = ""
        error_message = ""

        total_start = time.perf_counter()
        search_time = 0.0
        generation_time = 0.0

        try:
            search_start = time.perf_counter()
            retrieval_results = search_criteria(scene, top_k=5)
            search_time = time.perf_counter() - search_start

            print(f"검색 결과 수: {len(retrieval_results)}")
            print(f"검색 시간: {search_time:.4f}초")

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
                validation = False
                validation_message = "관련 안전 기준 검색 실패"

            print(f"평가 생성 시간: {generation_time:.4f}초")
            print(f"검증 결과: {validation}")

            if validation_message:
                print(f"검증 내용: {validation_message}")

        except Exception as e:
            validation = False
            error_message = f"{type(e).__name__}: {e}"
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
            "RAG 검색 시간(초)": round(search_time, 4),
            "평가 생성 시간(초)": round(generation_time, 4),
            "전체 처리 시간(초)": round(total_time, 4),
            "JSON 원본 응답": raw_response,
            "JSON 및 필드 검증": validation,
            "검증 실패 사유": validation_message if retrieval_results else "검색 실패",
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

    print("\n===== 성능 측정 요약 =====")
    print(f"전체 테스트: {total_count}건")
    print(f"검증 성공: {success_count}건")

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
