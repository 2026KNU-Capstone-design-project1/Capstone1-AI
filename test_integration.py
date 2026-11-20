
import json

from assessment_pipeline import run_assessment


# 1. 안전난간 관련 관측 정보
# 실제 탐지 결과가 아닌 테스트용 가상 데이터
scene = {
    "edge_detected": True,
    "visibility": "clear",
    "guardrail_present": False,
    "guardrail_continuous": None,
    "guardrail_protects_edge": None,
    "alternative_protection": False,
    "violation_suspected": False,
}

# 2. 위험도 평가 근거
# 고위험 상황을 가정한 테스트 값
risk_evidence = {
    "worker_exposure": "direct",
    "expected_consequence": "fatal",
    "evidence_sufficient": True,
}

# 3. RAG 검색 질문
query = "굴착부 추락 위험 단부 안전난간 미설치"

# 4. 전체 파이프라인 실행
result = run_assessment(
    hazard_type="guardrail",
    scene=scene,
    risk_evidence=risk_evidence,
    query=query,
)

# 5. 실행 결과 출력
print("\n===== 통합 위험성평가 결과 =====")
print(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )
)
