
import pytest

from risk_pipeline import evaluate_hazard_risk


# 안전난간 기본 입력
def make_guardrail_scene(**overrides):
    scene = {
        "edge_detected": True,
        "visibility": "clear",
        "guardrail_present": True,
        "guardrail_continuous": True,
        "guardrail_protects_edge": True,
        "alternative_protection": False,
        "violation_suspected": False,
    }
    scene.update(overrides)
    return scene


# 개구부 기본 입력
def make_opening_scene(**overrides):
    scene = {
        "opening_detected": True,
        "visibility": "clear",
        "cover_present": True,
        "guardrail_present": False,
        "protection_complete": True,
        "violation_suspected": False,
    }
    scene.update(overrides)
    return scene


# 위험도 평가 근거 기본 입력
def make_risk_evidence(**overrides):
    evidence = {
        "worker_exposure": "direct",
        "expected_consequence": "fatal",
        "evidence_sufficient": True,
    }
    evidence.update(overrides)
    return evidence


# 1. 안전난간 정상 상태 통합 테스트
def test_evaluate_hazard_risk_guardrail_normal():
    result = evaluate_hazard_risk(
        "guardrail",
        make_guardrail_scene(),
        make_risk_evidence(),
    )

    assert result["rule_result"]["violation_status"] == "관측상 정상"
    assert result["risk_assessment"]["risk_level"] == "평가 제외"
    assert result["risk_assessment"]["risk_score"] is None
    assert result["risk_assessment"]["review_required"] is False


# 2. 개구부 위반 상태 통합 테스트
def test_evaluate_hazard_risk_opening_violation():
    scene = make_opening_scene(
        cover_present=False,
        guardrail_present=False,
    )

    result = evaluate_hazard_risk(
        "opening",
        scene,
        make_risk_evidence(),
    )

    assert result["rule_result"]["violation_status"] == "위반"
    assert result["rule_result"]["rule_id"] == "F-04"

    assert result["risk_assessment"]["risk_score"] == 9
    assert result["risk_assessment"]["risk_level"] == "상"
    assert result["risk_assessment"]["review_required"] is False


# 3. 작업자 노출 및 피해 수준에 따른 위험도 계산
def test_evaluate_hazard_risk_calculates_risk_score():
    scene = make_opening_scene(
        cover_present=False,
        guardrail_present=False,
    )

    evidence = make_risk_evidence(
        worker_exposure="possible",
        expected_consequence="serious",
    )

    result = evaluate_hazard_risk("opening", scene, evidence)

    assert result["risk_assessment"]["likelihood"] == 2
    assert result["risk_assessment"]["severity"] == 2
    assert result["risk_assessment"]["risk_score"] == 4
    assert result["risk_assessment"]["risk_level"] == "중"


# 4. 위험도 평가 근거가 부족한 경우
def test_evaluate_hazard_risk_insufficient_evidence():
    scene = make_opening_scene(
        cover_present=False,
        guardrail_present=False,
    )

    evidence = make_risk_evidence(
        evidence_sufficient=False,
    )

    result = evaluate_hazard_risk("opening", scene, evidence)

    assert result["rule_result"]["violation_status"] == "위반"
    assert result["risk_assessment"]["risk_score"] is None
    assert result["risk_assessment"]["risk_level"] == "미확정"
    assert result["risk_assessment"]["review_required"] is True


# 5. 영상이 가려져 위반 여부를 확인할 수 없는 경우
def test_evaluate_hazard_risk_uncertain_scene():
    scene = make_opening_scene(visibility="blocked")

    result = evaluate_hazard_risk(
        "opening",
        scene,
        make_risk_evidence(),
    )

    assert result["rule_result"]["violation_status"] == "미확인"
    assert result["risk_assessment"]["risk_level"] == "미확정"
    assert result["risk_assessment"]["risk_score"] is None
    assert result["risk_assessment"]["review_required"] is True


# 6. 지원하지 않는 위험요소 입력
def test_evaluate_hazard_risk_unsupported_hazard():
    with pytest.raises(ValueError):
        evaluate_hazard_risk(
            "unknown",
            {},
            make_risk_evidence(),
        )


# 7. 입력된 risk_evidence 원본이 변경되지 않는지 확인
def test_evaluate_hazard_risk_does_not_modify_original_evidence():
    scene = make_opening_scene(
        cover_present=False,
        guardrail_present=False,
    )

    evidence = make_risk_evidence()
    original_evidence = evidence.copy()

    evaluate_hazard_risk("opening", scene, evidence)

    assert evidence == original_evidence
    assert "hazard_type" not in evidence
