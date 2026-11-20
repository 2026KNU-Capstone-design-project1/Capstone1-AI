
import pytest

from risk_assessment import (
    calculate_risk_score,
    classify_risk_level,
    assess_risk,
    evaluate_likelihood,
    evaluate_severity,
    assess_risk_from_evidence,
)


# 1. 위험도 점수 계산 테스트
@pytest.mark.parametrize(
    "likelihood, severity, expected",
    [
        (1, 1, 1),
        (1, 3, 3),
        (2, 2, 4),
        (2, 3, 6),
        (3, 3, 9),
    ],
)
def test_calculate_risk_score(likelihood, severity, expected):
    assert calculate_risk_score(likelihood, severity) == expected


@pytest.mark.parametrize(
    "likelihood, severity",
    [
        (0, 1),
        (4, 1),
        (1, 0),
        (1, 4),
        (1.5, 2),
        (True, 2),
        (2, False),
    ],
)
def test_calculate_risk_score_invalid_input(likelihood, severity):
    with pytest.raises(ValueError):
        calculate_risk_score(likelihood, severity)


# 2. 위험도 등급 분류 테스트
@pytest.mark.parametrize(
    "score, expected",
    [
        (1, "하"),
        (2, "하"),
        (3, "중"),
        (4, "중"),
        (6, "상"),
        (9, "상"),
    ],
)
def test_classify_risk_level(score, expected):
    assert classify_risk_level(score) == expected


@pytest.mark.parametrize("score", [0, 5, 7, 8, 10])
def test_classify_risk_level_invalid_score(score):
    with pytest.raises(ValueError):
        classify_risk_level(score)


# 3. 규칙 엔진 결과에 따른 위험도 평가 테스트
def test_assess_risk_normal():
    rule_result = {
        "hazard": "안전난간",
        "violation_status": "관측상 정상",
        "reason": "안전난간이 정상적으로 설치됨",
    }

    result = assess_risk(rule_result)

    assert result["risk_level"] == "평가 제외"
    assert result["risk_score"] is None
    assert result["review_required"] is False


def test_assess_risk_violation():
    rule_result = {
        "hazard": "개구부",
        "violation_status": "위반",
        "reason": "개구부에 안전난간이 설치되지 않음",
    }

    result = assess_risk(
        rule_result,
        likelihood=3,
        severity=3,
        evidence_sufficient=True,
    )

    assert result["risk_score"] == 9
    assert result["risk_level"] == "상"
    assert result["review_required"] is False


def test_assess_risk_insufficient_evidence():
    rule_result = {
        "hazard": "개구부",
        "violation_status": "위반",
        "reason": "안전난간 미설치",
    }

    result = assess_risk(
        rule_result,
        evidence_sufficient=False,
    )

    assert result["risk_score"] is None
    assert result["risk_level"] == "미확정"
    assert result["review_required"] is True


def test_assess_risk_suspected_violation():
    rule_result = {
        "hazard": "개구부",
        "violation_status": "위반 의심",
        "reason": "영상에서 안전난간 확인이 어려움",
    }

    result = assess_risk(rule_result)

    assert result["risk_level"] == "미확정"
    assert result["review_required"] is True


def test_assess_risk_invalid_rule_result():
    with pytest.raises(ValueError):
        assess_risk("잘못된 입력")


def test_assess_risk_invalid_status():
    rule_result = {
        "hazard": "개구부",
        "violation_status": "알 수 없는 상태",
        "reason": "테스트",
    }

    with pytest.raises(ValueError):
        assess_risk(rule_result)


# 4. 가능성(L) 분류 테스트
@pytest.mark.parametrize(
    "exposure, expected",
    [
        ("controlled", 1),
        ("possible", 2),
        ("direct", 3),
        ("unknown", None),
    ],
)
def test_evaluate_likelihood(exposure, expected):
    assert evaluate_likelihood(exposure) == expected


# 5. 피해 심각도(S) 분류 테스트
@pytest.mark.parametrize(
    "consequence, expected",
    [
        ("minor", 1),
        ("serious", 2),
        ("fatal", 3),
        ("unknown", None),
    ],
)
def test_evaluate_severity(consequence, expected):
    assert evaluate_severity(consequence) == expected


# 6. 구조화된 증거 기반 통합 평가 테스트
def test_assess_risk_from_evidence():
    rule_result = {
        "hazard": "개구부",
        "violation_status": "위반",
        "reason": "안전난간 미설치",
    }

    risk_evidence = {
        "hazard_type": "opening",
        "worker_exposure": "direct",
        "expected_consequence": "fatal",
        "evidence_sufficient": True,
    }

    result = assess_risk_from_evidence(rule_result, risk_evidence)

    assert result["likelihood"] == 3
    assert result["severity"] == 3
    assert result["risk_score"] == 9
    assert result["risk_level"] == "상"
    assert result["hazard_type"] == "opening"


def test_assess_risk_from_evidence_insufficient():
    rule_result = {
        "hazard": "개구부",
        "violation_status": "위반",
        "reason": "안전난간 미설치",
    }

    risk_evidence = {
        "hazard_type": "opening",
        "worker_exposure": "direct",
        "expected_consequence": "fatal",
        "evidence_sufficient": False,
    }

    result = assess_risk_from_evidence(rule_result, risk_evidence)

    assert result["risk_score"] is None
    assert result["risk_level"] == "미확정"
    assert result["review_required"] is True


def test_assess_risk_from_evidence_invalid_hazard():
    rule_result = {
        "hazard": "기타",
        "violation_status": "위반",
        "reason": "테스트",
    }

    risk_evidence = {
        "hazard_type": "unknown",
        "worker_exposure": "direct",
        "expected_consequence": "fatal",
        "evidence_sufficient": True,
    }

    with pytest.raises(ValueError):
        assess_risk_from_evidence(rule_result, risk_evidence)


def test_assess_risk_from_evidence_invalid_type():
    rule_result = {
        "hazard": "개구부",
        "violation_status": "위반",
        "reason": "테스트",
    }

    with pytest.raises(ValueError):
        assess_risk_from_evidence(rule_result, None)
