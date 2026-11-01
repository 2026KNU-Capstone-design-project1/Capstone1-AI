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

documents = (
    df[search_colums]
    .agg(" ".join, axis=1)
    .tolist()
)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
)

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    token_pattern=r"(?u)\b\w+\b",
)

char_document_vectors = char_vectorizer.fit_transform(documents)
word_document_vectors = word_vectorizer.fit_transform(documents)

def search_criteria(query, top_k=5, method="char"):
    """검색 방식에 따라 현장 상황과 관련 있는 기준을 검색한다."""

    if method == "char":
        selected_vectorizer = char_vectorizer
        selected_document_vectors = char_document_vectors

    elif method == "word":
        selected_vectorizer = word_vectorizer
        selected_document_vectors = word_document_vectors

    else:
        raise ValueError(
            "method는 'char' 또는 'word'만 사용할 수 있습니다."
        )

    query_vector = selected_vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        selected_document_vectors
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

def generate_assessment(scene, results, return_json=False):
    """규칙 판정 딕셔너리를 근거로 위험성평가 결과를 생성한다."""
    if not isinstance(results, dict):
        raise TypeError("results는 규칙 판정 결과 딕셔너리여야 합니다.")
    if not results:
        return "규칙 판정 결과가 없습니다. 추가 검토가 필요합니다."

    scene_text = json.dumps(scene, ensure_ascii=False, indent=2, default=str)
    criteria_text = json.dumps(
        results,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

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
        2. 입력된 현장 영상 객체 탐지 정보와 규칙 판정 결과만 근거로 사용한다.
        3. rule_id, hazard, violation_status, reason 등은 규칙 판정의 결과이며
           텍스트 검색 결과나 유사도 점수가 아니다.
        4. 규칙 판정의 위반 여부를 임의로 변경하지 않는다.
        5. 개선대책에는 현재 문제를 시정하는 구체적인 조치를 작성하고,
           안전대책에는 재발 방지를 위한 예방 조치를 작성한다.
        6. 법적사항에 확인된 조항이 없다면 '관련 법령 추가 확인 필요'로 작성한다.
           확인되지 않은 법령이나 조항을 만들어내지 않는다.
        7. likelihood, severity, risk_score 중 하나라도 null이거나,
           risk_level이 '미확정'이면 위험도는 '추가 확인 필요'로 작성한다.
           점수와 위험도 등급을 임의로 계산하거나 보완하지 않는다.
        8. 위험도 판단 근거와 추가 확인사항에는 확인된 사실과 부족한 정보를 구분한다.
        9. 입력에 없는 사실, 수치, 법령 및 객체 상태를 만들어내지 않는다.
        10. 규칙 판정 결과의 위반 여부와 위험도 평가의 확정 여부는 구분한다.
    """

    user_prompt = f"""
        [현장 영상 객체 탐지 정보]
        {scene_text}

        [규칙 기반 위험 판정 결과]
        {criteria_text}

        위 입력을 바탕으로 위험성평가 결과를 작성하라.
        규칙 판정 결과를 텍스트 검색 결과로 취급하지 마라.
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
            "num_predict": 1200,
        },
    )

    elapsed = time.perf_counter() - start
    print(f"Qwen 호출 완료: {elapsed:.2f}초", flush=True)

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

    missing = [
        key for key in assessment_schema["required"]
        if key not in answer_data
    ]

    if missing:
        raise ValueError(f"Qwen 응답에 필수 항목이 없습니다: {missing}")

    if return_json:
        return answer_data

    answer = "\n".join(
        f"{key}: {answer_data[key]}"
        for key in assessment_schema["required"]
    )

    return answer

if __name__ == "__main__":
    scene = {
        "detected_objects": [
            {
                "class": "guardrail",
                "confidence": 0.91,
                "bbox": [100, 120, 300, 250]
            },
            {
                "class": "worker",
                "confidence": 0.95,
                "bbox": [220, 180, 280, 350]
            },
            {
                "class": "edge",
                "confidence": 0.88,
                "bbox": [80, 100, 400, 120]
            }
        ],
        "visibility": "clear"
    }

    results = {
        "rule_id": "F-01",
        "hazard": "안전난간",
        "violation_status": "위반",
        "reason": "추락 위험 단부의 방호조치 미실시",
        "likelihood": None,
        "severity": None,
        "risk_score": None,
        "risk_level": "미확정",
        "review_required": True
    }

    print(f"전체 기준 수: {len(df)}")
    print("\n===== 규칙 기반 위험 판정 결과 =====")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    print("\n===== Qwen 위험성평가 결과 =====")
    assessment = generate_assessment(scene, results)
    print(assessment)