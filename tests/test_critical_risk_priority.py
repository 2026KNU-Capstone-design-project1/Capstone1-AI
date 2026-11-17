import pytest

from risk_assessment import assess_risk, assess_risk_from_evidence


def rule(status="위반"):
    return {
        "hazard": "개구부 방호",
        "violation_status": status,
        "reason": "테스트"
    }


@pytest.mark.parametrize(
    "likelihood,severity,expected_score,expected_level",
    [
        (1, 2, 2, "하"),
        (2, 2, 4, "중"),
        (3, 3, 9, "상"),
    ],
)
def test_regular_risk_unchanged(
    likelihood, severity, expected_score, expected_level
):
    result = assess_risk(
        rule(),
        likelihood=likelihood,
        severity=severity,
        evidence_sufficient=True,
    )
    assert result["risk_score"] == expected_score
    assert result["risk_level"] == expected_level
    assert "critical_priority_applied" not in result


@pytest.mark.parametrize(
    "likelihood,severity,expected_score",
    [(1, 2, 2), (2, 2, 4), (3, 3, 9)],
)
def test_confirmed_critical_priority(
    likelihood, severity, expected_score
):
    result = assess_risk(
        rule(),
        likelihood=likelihood,
        severity=severity,
        evidence_sufficient=True,
        critical_hazard_confirmed=True,
    )
    assert result["risk_score"] == expected_score
    assert result["risk_level"] == "상"
    assert result["critical_priority_applied"] is True


@pytest.mark.parametrize(
    "status,expected_level",
    [
        ("미확인", "미확정"),
        ("위반 의심", "미확정"),
        ("관측상 정상", "평가 제외"),
    ],
)
def test_critical_flag_does_not_override_status(
    status, expected_level
):
    result = assess_risk(
        rule(status),
        likelihood=1,
        severity=2,
        evidence_sufficient=True,
        critical_hazard_confirmed=True,
    )
    assert result["risk_level"] == expected_level
    assert result["risk_score"] is None


def test_insufficient_evidence_remains_undetermined():
    result = assess_risk(
        rule(),
        likelihood=1,
        severity=2,
        evidence_sufficient=False,
        critical_hazard_confirmed=True,
    )
    assert result["risk_level"] == "미확정"


def test_missing_scores_remain_undetermined():
    result = assess_risk(
        rule(),
        evidence_sufficient=True,
        critical_hazard_confirmed=True,
    )
    assert result["risk_level"] == "미확정"


def test_structured_evidence_applies_priority():
    result = assess_risk_from_evidence(
        rule(),
        {
            "hazard_type": "opening",
            "worker_exposure": "controlled",
            "expected_consequence": "serious",
            "evidence_sufficient": True,
            "critical_hazard_confirmed": True,
        },
    )
    assert result["risk_score"] == 2
    assert result["risk_level"] == "상"
    assert result["critical_priority_applied"] is True


def test_structured_evidence_insufficient_stays_undetermined():
    result = assess_risk_from_evidence(
        rule(),
        {
            "hazard_type": "opening",
            "worker_exposure": "controlled",
            "expected_consequence": "serious",
            "evidence_sufficient": False,
            "critical_hazard_confirmed": True,
        },
    )
    assert result["risk_level"] == "미확정"


@pytest.mark.parametrize("invalid", ["True", 1, None])
def test_invalid_priority_flag_rejected(invalid):
    with pytest.raises(ValueError):
        assess_risk(
            rule(),
            likelihood=1,
            severity=2,
            evidence_sufficient=True,
            critical_hazard_confirmed=invalid,
        )
