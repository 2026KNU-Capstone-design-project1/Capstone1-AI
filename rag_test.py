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

search_columns = [
    "대분류",
    "위험요소",
    "판단항목",
    "정상",
    "위반",
    "예상 위험",
    "개선내용",
    "안전대책",
    "법적사항",
]

missing = [col for col in search_columns if col not in df.columns]

if missing:
    raise ValueError(f"CSV에 없는 column: {missing}")


df[search_columns] = df[search_columns].fillna("").astype(str)

search_weights = {
    "대분류": 2,
    "위험요소": 5,
    "판단항목": 5,
    "정상": 2,
    "위반": 2,
    "예상 위험": 2,
    "개선내용": 1,
    "안전대책": 1,
    "법적사항": 1,
}

def build_weighted_document(row):
    parts = []

    for column, weight in search_weights.items():
        value = row[column].strip()

        if value:
            parts.extend([value] * weight)

    return " ".join(parts)


documents = df.apply(
    build_weighted_document,
    axis=1
).tolist()

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
            **{col: row[col] for col in search_columns},
        })

    return results

def search_hazard_knowledge(
    hazard_code: str,
) -> list[dict]:
    """위험 코드에 정확히 일치하는 참고 지식을 검색한다.

    검증되지 않은 초안은 원본 검증 상태를 유지한다.
    """

    knowledge_path = BASE_DIR / "data" / "hazard_knowledge.json"

    with knowledge_path.open(
        encoding="utf-8"
    ) as f:
        knowledge = json.load(f)

    results = []

    for item in knowledge.get("items", []):
        if item.get("hazard_code") == hazard_code:
            results.append({
                **item,
                "retrieval_method": "hazard_code_exact",
            })

    return results

def generate_assessment(scene, results, return_json=False):
    """
    검색 결과 목록(list) 또는 규칙 판정 딕셔너리(dict)를
    근거로 위험성평가 결과를 생성한다.
    """

    if not isinstance(results, (list, dict)):
        raise TypeError(
            "results는 검색 결과 목록(list) 또는 "
            "규칙 판정 딕셔너리(dict)여야 합니다."
        )

    if not results:
        return "관련 결과가 없습니다. 추가 검토가 필요합니다."

    if isinstance(results, dict):
        input_label = "규칙 기반 위험 판정 결과"

        input_instruction = """
        입력은 규칙 엔진의 판정 결과다.
        위반 여부와 판정 근거를 임의로 변경하지 마라.
        """

        risk_fields = [
            "likelihood",
            "severity",
            "risk_score",
            "risk_level",
        ]

        valid_levels = {"상", "중", "하"}

        risk_information_sufficient = (
            all(
                field in results and results[field] is not None
                for field in risk_fields
            )
            and type(results["likelihood"]) is int
            and results["likelihood"] in (1, 2, 3)
            and type(results["severity"]) is int
            and results["severity"] in (1, 2, 3)
            and type(results["risk_score"]) is int
            and results["risk_score"] == (
                results["likelihood"] * results["severity"]
            )
            and results["risk_level"] in valid_levels
        )

    else:
        input_label = "검색된 안전 기준 및 유사 사례"

        input_instruction = """
        입력은 RAG 검색으로 찾은 참고 자료다.
        검색 결과 자체를 현재 현장의 확정된 판정으로 취급하지 마라.
        검색 자료는 안전 기준과 개선대책을 참고하는 용도로만 사용하라.
        위험도 산정 결과가 별도로 제공되지 않았다면
        위험도는 반드시 '추가 확인 필요'로 작성하라.
        """

        risk_information_sufficient = False

    scene_text = json.dumps(
        scene,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

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
    2. 입력된 현장 영상 객체 탐지 정보와 제공된 자료만 근거로 사용한다.
    3. 규칙 판정 결과의 위반 여부를 임의로 변경하지 않는다.
    4. RAG 검색 결과는 참고 자료이며 현재 현장의 확정 판정이 아니다.
    5. 개선대책에는 현재 문제를 시정하는 구체적인 조치를 작성한다.
    6. 안전대책에는 재발 방지를 위한 예방 조치를 작성한다.
    7. 법적사항에 확인된 조항이 없다면
       '관련 법령 추가 확인 필요'로 작성한다.
    8. 확인되지 않은 법령이나 조항을 만들어내지 않는다.
    9. 위험도 산정 정보가 부족하거나 미확정이면
       위험도는 '추가 확인 필요'로 작성한다.
    10. 점수와 위험도 등급을 임의로 계산하거나 보완하지 않는다.
    11. 위험도 판단 근거와 추가 확인사항에는
        확인된 사실과 부족한 정보를 구분한다.
    12. 입력에 없는 사실, 수치, 법령 및 객체 상태를 만들어내지 않는다.
    13. 규칙 판정 결과의 위반 여부와 위험도 평가의 확정 여부는 구분한다.
        14. evidence_context가 제공되면 이를 관측 근거로 활용한다.
    15. evidence_status가 'matched'이면 YOLO와 공간관계 규칙이
        동일 위험 후보를 제시한 것으로 설명한다.
        실제 위반이 확정되었다고 표현하지 않는다.
    16. evidence_status가 'detection_only'이면 YOLO 단독 위험 후보이며,
        공간관계 규칙의 확인 근거가 없다고 설명한다.
        공간관계 규칙이 위험을 부정했다고 단정하지 않는다.
    17. evidence_status가 'spatial_only'이면 공간관계 규칙 단독
        위험 후보로 설명한다. YOLO가 안전하다고 판단했다고 쓰지 않는다.
    18. evidence_status가 'undetermined'이면 공간관계 규칙에서
        판단할 수 없었던 위험 후보로 설명하고 추가 확인을 요청한다.
    19. spatial_findings의 status가 '위험'이어도 이는 위험 후보이며,
        실제 법령 위반이나 위험도 등급의 확정 근거가 아니다.
    20. spatial_findings가 비어 있으면 정상 상태로 해석하지 않는다.
        규칙 미지원, 조건 불충족, 객체 미탐지 중 어떤 원인인지
        제공되지 않았다면 임의로 결정하지 않는다.
    21. related_objects와 related_relations에 있는 객체명,
        관계 유형, 수치는 실제 제공된 내용만 사용한다.
    22. gap은 화면 대각선 기준의 정규화된 거리다.
        실제 거리(m)로 변환하거나 법적 이격거리 충족 여부를 판단하지 않는다.
    23. IoU와 객체 포함 비율은 바운딩박스의 기하학적 관계이며,
        실제 물리적 접촉이나 인양 작업 상태의 확정 근거가 아니다.
    24. 객체 탐지 confidence를 위반 확률이나 위험 발생 확률로
        해석하지 않는다.
        25. evidence_context의 evaluation_status와 reason_code를
        공간관계 규칙의 평가 상태 및 판단 원인으로 해석한다.
        이 값은 실제 법령 위반 여부나 L×S 위험등급을 의미하지 않는다.

    26. reason_code가 'rule_matched'이면
        공간관계 규칙에서 위험 후보 조건이 충족되었다고 설명한다.
        agrees_with_model이 true이면 YOLO 위험 후보와
        공간관계 규칙의 판단이 일치했다고 설명한다.
        실제 위반이 확정되었다고 표현하지 않는다.

    27. reason_code가 'rule_not_supported'이면
        해당 위험요소의 공간관계 판단 규칙이 지원되지 않아
        규칙 기반 평가를 수행할 수 없었다고 설명한다.
        위험이 없거나 정상이라고 판단하지 않는다.

    28. evaluation_status가 'not_evaluable'이면
        공간관계 규칙 평가가 불가능하거나 보류된 상태로 설명한다.
        구체적인 원인은 제공된 reason_code를 따른다.

    29. checks에 포함된 각 규칙의 status와 reason_code를 확인한다.
        프레임 전체의 evaluation_status만 보고 모든 위험요소가
        같은 상태라고 판단하지 않는다.

    30. findings가 비어 있더라도 정상 상태라고 단정하지 않는다.
        reason_code가 제공되었다면 해당 원인을 우선 설명한다.

    31. 위험내용에는 현재 detection.hazard와 관측된 객체,
        공간관계를 우선 반영한다.
        RAG 검색 결과의 유사 사례를 현재 영상에서 관측된
        사실처럼 작성하지 않는다.

    32. 개선대책과 추가 확인사항은 현재 위험 후보에 맞게 작성한다.
        규칙 미지원이나 근거 부족 상태에서는
        실제 위험 여부를 확인하기 위한 점검사항을 구체적으로 제시한다.
    33. 검색 결과의 verification_status가 'unverified'이면
        해당 자료는 검증되지 않은 안전조치 참고 초안이다.
        검증된 법적 의무, 공식 안전기준 또는 확정된 규정으로 표현하지 않는다.

    34. unverified 자료의 legal_basis가 null이면
        법적사항은 반드시 '관련 법령 추가 확인 필요'로 작성한다.
        다른 위험유형의 법령을 대신 가져오지 않는다.

    35. 위험내용은 검색된 유사 사례보다
        현재 detection.hazard와 evidence_context를 우선하여 작성한다.
        RAG 자료의 개선내용을 영상에서 실제 관측한 사실로 작성하지 않는다.

    36. evidence_context에 related_objects가 있으면
        해당 객체 이름을 위험내용에 반영한다.
        related_relations가 있으면 실제 관측된 공간관계 유형을 설명한다.
        단, 바운딩박스의 포함·중첩만으로 실제 인양이나 접촉을 확정하지 않는다.

    37. 개선대책은 현재 위험 후보에 해당하는
        RAG 자료의 improvement를 참고하여 작성한다.
        위험 상태가 미확인이면 필요한 확인과 조건부 시정조치를 구분한다.

    38. 위험도 평가 시 추가 확인사항은
        해당 위험 코드의 additional_checks를 우선 참고한다.
        일반적인 방호시설 점검사항으로 대체하지 않는다.
    """

    user_prompt = f"""
    [현장 영상 객체 탐지 정보]
    {scene_text}

    [{input_label}]
    {criteria_text}

    {input_instruction}

    위 입력을 바탕으로 위험성평가 결과를 작성하라.
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

    print(f"\nQwen 호출 완료: {elapsed:.2f}초", flush=True)

    # Ollama 성능 측정
    print("===== Ollama 성능 분석 =====")

    print(f"모델 로딩 시간: {response.load_duration / 1e9:.2f}초")

    print(
        f"입력 토큰 처리 시간: "
        f"{response.prompt_eval_duration / 1e9:.2f}초"
    )

    print(f"입력 토큰 수: {response.prompt_eval_count}")

    print(
        f"출력 토큰 생성 시간: "
        f"{response.eval_duration / 1e9:.2f}초"
    )

    print(f"출력 토큰 수: {response.eval_count}")

    if response.eval_duration and response.eval_count:
        tokens_per_second = (
            response.eval_count / (response.eval_duration / 1e9)
        )
        print(f"토큰 생성 속도: {tokens_per_second:.2f} tokens/s")


    # 응답 내용 확인
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
        raise ValueError(
            f"Qwen 응답에 필수 항목이 없습니다: {missing}"
        )

    extra = [
    key for key in answer_data
    if key not in assessment_schema["properties"]
]

    if extra:
        raise ValueError(f"Qwen 응답에 불필요한 항목이 있습니다: {extra}")

    if not risk_information_sufficient:
        answer_data["위험도"] = "추가 확인 필요"

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