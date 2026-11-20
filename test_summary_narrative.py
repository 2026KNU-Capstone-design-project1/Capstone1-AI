
import json

from full_report_builder import build_full_report
from summary_narrative import generate_summary_narrative


# 실제 현장 데이터가 아닌 가상 테스트 데이터
assessments = [
    {
        "risk_id": "RISK-001",
        "hazard": "안전난간 미설치",
        "violation_status": "위반",
        "risk_level": "상",
        "area": "A구역",
        "action_status": "미처리",
        "rule_result": {"hazard": "안전난간"},
        "report": {
            "예상 재해유형": "추락",
            "위험내용": "굴착부 단부의 안전난간 미설치",
            "개선대책": "굴착부 단부에 안전난간 설치",
        },
    },
    {
        "risk_id": "RISK-002",
        "hazard": "개구부 방호 상태 미확인",
        "violation_status": "미확인",
        "risk_level": "미확정",
        "area": "B구역",
        "action_status": None,
        "rule_result": {"hazard": "개구부"},
        "report": {
            "예상 재해유형": "추락",
            "위험내용": "방호시설 설치 상태 확인 불가",
            "개선대책": "현장에서 방호시설 설치 여부 확인",
        },
    },
]

full_report = build_full_report(
    report_id="REPORT-TEST-001",
    site_info={"site_name": "테스트 건설 현장"},
    assessments=assessments,
    report_period="2026-11",
)

print("Qwen 종합 보고서 생성 시작", flush=True)

narrative = generate_summary_narrative(full_report)

print(
    json.dumps(narrative, ensure_ascii=False, indent=2)
)

assert isinstance(narrative["executive_summary"], str)
assert isinstance(narrative["key_findings"], list)
assert isinstance(narrative["area_type_analysis"], str)
assert isinstance(narrative["management_opinion"], str)

print("종합 보고서 Qwen 생성 테스트 성공")

from summary_narrative import build_verified_insights

verified_insights = build_verified_insights(full_report)

print("\n검증된 핵심 인사이트:")
for insight in verified_insights:
    print("-", insight)