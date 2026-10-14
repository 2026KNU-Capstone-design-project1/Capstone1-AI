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
    """검색된 안전 기준을 근거로 정해진 형식의 평가 결과를 생성한다."""

    if not results:
        return "관련 안전 기준을 찾지 못했습니다. 추가 검토가 필요합니다."

    criteria_text = json.dumps(
        results,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    system_prompt = """
        너는 한국어로 산업현장 위험성평가 결과를 작성하는 AI다.

        반드시 다음 규칙을 지켜라.
        1. 최종 답변은 한국어로만 작성한다.
        2. 문제를 분석하는 과정이나 서론을 출력하지 않는다.
        3. 지정된 출력 항목과 순서를 반드시 지킨다.
        4. 검색된 안전 기준에 근거하여 작성한다.
        5. 근거에 없는 법령, 수치, 사실을 만들어내지 않는다.
        6. 근거가 부족한 항목은 '추가 확인 필요'라고 작성한다.
        7. 위험도 평가 기준이 없으면 위험 수준을 임의로 결정하지 않는다.
        8. 각 항목은 간결하고 구체적으로 작성한다.

        반드시 아래 형식으로만 답변한다.

        위험요소:
        예상 재해유형:
        위험내용:
        관련 안전 기준:
        법적사항:
        개선대책:
        위험도 평가 시 추가 확인사항:
        """

    user_prompt = f"""
        [현장 상황]
        {scene}

        [검색된 안전 기준]
        {criteria_text}

        위 자료를 바탕으로 위험성평가 결과를 작성하라.
        위에서 지정한 7개 항목 외에는 출력하지 마라.
        """

    start = time.perf_counter()
    print("Qwen 호출 시작", flush=True)

            
    assessment_schema = {
        "type": "object",
        "properties": {
            "위험요소": {"type": "string"},
            "예상 재해유형": {"type": "string"},
            "위험내용": {"type": "string"},
            "관련 안전 기준": {"type": "string"},
            "법적사항": {"type": "string"},
            "개선대책": {"type": "string"},
            "위험도 평가 시 추가 확인사항": {"type": "string"},
        },
        "required": [
            "위험요소",
            "예상 재해유형",
            "위험내용",
            "관련 안전 기준",
            "법적사항",
            "개선대책",
            "위험도 평가 시 추가 확인사항",
        ],
        "additionalProperties": False,
    }

    start = time.perf_counter()
    print("Qwen 호출 시작", flush=True)

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
            "num_predict": 400,
        },
    )

    print(
        f"Qwen 호출 완료: {time.perf_counter() - start:.2f}초",
        flush=True,
    )

    answer_data = json.loads(response.message.content)

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