import json
import ollama
import time

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "risk_data_clean.csv"

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

search_colums = [
    "대분류", "위험요소", "판단항목", "정상", "위반", "예상 위험", "개선내용", "안전대책", "법적사항"
]

missing = [col for col in search_colums if col not in df.columns]
if missing:
    raise ValueError(f"csv에 없는 column: {missing}")

df[search_colums] = df[search_colums].fillna("").astype(str)
documents = df[search_colums].agg(" ".join, axis=1).tolist()

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
)

document_vectors = vectorizer.fit_transform(documents)

def search_criteria(query, top_k=5):
    """현장 상황과 관련 있는 기준을 상위 top_k개 검색한다."""
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector, document_vectors
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0.1:
            continue

        row = df.iloc[idx]

        results.append({
            "기준_행": int(idx),
            "유사도": round(float(scores[idx]), 4),
            **{col: row[col] for col in search_colums},
        })

    return results



def generate_assessment(scene, results):
    """검색된 안전 기준을 근거로 위험성평가 결과를 생성한다."""

    if not results:
        return "관련 안전 기준을 찾지 못했습니다. 추가 검토가 필요합니다."

    criteria_text = json.dumps(
        results,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    # LLM이 반환해야 하는 JSON 구조
    assessment_schema = {
        "type": "object",
        "properties": {
            "위험요소": {"type": "string"},
            "예상 재해유형": {"type": "string"},
            "위험도": {
                "type": "string",
                "enum": ["상", "중", "하", "추가 확인 필요"],
            },
            "위험도 판단 근거": {"type": "string"},
            "위험내용": {"type": "string"},
            "관련 안전 기준": {"type": "string"},
            "법적사항": {"type": "string"},
            "개선대책": {"type": "string"},
            "안전대책": {"type": "string"},
            "위험도 평가 시 추가 확인사항": {"type": "string"},
        },
        "required": [
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
        ],
        "additionalProperties": False,
    }

    system_prompt = """
        너는 한국어로 산업현장 위험성평가를 작성하는 AI다.

        반드시 지정된 JSON 스키마에 맞는 JSON 객체만 반환한다.
        JSON 바깥에 설명이나 마크다운을 추가하지 않는다.

        규칙:
        1. 모든 값은 한국어로 작성한다.
        2. 검색된 안전 기준과 현장 상황을 근거로 작성한다.
        3. 개선대책에는 현재 문제를 시정하는 구체적인 조치를 작성한다.
        4. 안전대책에는 위험 재발을 방지하는 예방 조치를 작성한다.
        5. 법적사항은 검색된 자료를 참고하되, 관련성이 불확실하면
        '관련 법령 추가 확인 필요'라고 작성한다.
        6. 위험도 평가 기준이 제공되지 않았거나 정보가 부족하면
        위험도를 임의로 확정하지 않고 '추가 확인 필요'로 작성한다.
        7. 위험도 판단 근거와 추가 확인사항에는 판단에 필요한 정보를 작성한다.
        8. 검색 자료에 없는 법령, 수치 및 사실을 만들어내지 않는다.
    """

    user_prompt = f"""
        [현장 상황]
        {scene}

        [검색된 안전 기준]
        {criteria_text}

        위 자료를 바탕으로 위험성평가 결과를 작성하라.
        모든 필수 필드를 빠짐없이 작성하라.
    """

    print("Qwen 호출 시작", flush=True)
    start = time.perf_counter()

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format=assessment_schema,
        think=False,
        stream=False,
        options={
            "temperature": 0,
            "num_predict": 600,
        },
    )

    elapsed = time.perf_counter() - start
    print(f"Qwen 호출 완료: {elapsed:.2f}초", flush=True)

    # 응답을 먼저 확인한 다음 JSON으로 변환한다.
    content = (response.message.content or "").strip()

    if not content:
        thinking = getattr(response.message, "thinking", None) or ""
        raise RuntimeError(
            "Qwen이 빈 응답을 반환했습니다.\n"
            f"Thinking 내용: {thinking!r}"
        )

    try:
        answer_data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(
            "Qwen 응답이 올바른 JSON 형식이 아닙니다.\n"
            f"원본 응답: {content!r}"
        ) from e

    # 필수 필드 누락 여부 확인
    missing = [
        key for key in assessment_schema["required"]
        if key not in answer_data
    ]

    if missing:
        raise ValueError(f"Qwen 응답에 필수 항목이 없습니다: {missing}")

    # JSON 객체를 사람이 읽기 쉬운 형태로 출력
    answer = "\n".join(
        f"{key}: {answer_data[key]}"
        for key in assessment_schema["required"]
    )

    return answer


if __name__ == "__main__":
    scene = "일자가 지난 위험성평가서가 다수 배치되어 있다."

    results = search_criteria(scene)

    print(f"전체 기준 수: {len(df)}")
    print(f"검색 결과 수: {len(results)}")

    for i, item in enumerate(results, start=1):
        print(f"\n===== 검색 결과 {i} =====")
        print(f"유사도: {item['유사도']}")

        for col in search_colums:
            print(f"{col}: {item[col]}")

    print("\n===== Qwen 위험성평가 결과 =====")

    assessment = generate_assessment(scene, results)
    print(assessment)