from rule_engine import evaluate_scene
from risk_assessment import assess_risk_from_evidence


def evaluate_hazard_risk(hazard_type, scene, risk_evidence):
    """
    위험요소별 판정 결과와 위험도 산정을 연결한다.

    hazard_type:
        guardrail / opening / work_platform

    scene:
        영상 분석 결과 및 현장 관측 정보

    risk_evidence:
        작업자 노출 상태, 예상 피해 수준,
        근거 충분 여부
    """

    rule_result = evaluate_scene(hazard_type, scene)

    evidence = risk_evidence.copy()
    evidence["hazard_type"] = hazard_type

    risk_result = assess_risk_from_evidence(
        rule_result,
        evidence
    )

    return {
        "rule_result": rule_result,
        "risk_assessment": risk_result
    }
