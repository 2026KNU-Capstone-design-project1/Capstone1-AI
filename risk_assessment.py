def calculate_risk_score(likelihood, severity):
    """가능성(L)과 피해 심각도(S)를 곱해 위험도 점수를 계산한다."""

    if not isinstance(likelihood, int) or isinstance(likelihood, bool):
        raise ValueError("가능성은 1~3 사이의 정수여야 합니다.")

    if not isinstance(severity, int) or isinstance(severity, bool):
        raise ValueError("중대성은 1~3 사이의 정수여야 합니다.")

    if not 1 <= likelihood <= 3:
        raise ValueError("가능성은 1~3 사이여야 합니다.")

    if not 1 <= severity <= 3:
        raise ValueError("중대성은 1~3 사이여야 합니다.")

    return likelihood * severity


def classify_risk_level(risk_score):
    """위험도 점수를 하·중·상 등급으로 분류한다."""

    if risk_score in (1, 2):
        return "하"

    if risk_score in (3, 4):
        return "중"

    if risk_score in (6, 9):
        return "상"

    raise ValueError(f"유효하지 않은 위험도 점수: {risk_score}")


def assess_risk(
    rule_result,
    likelihood=None,
    severity=None,
    evidence_sufficient=False,
    review_reason=None,
    critical_hazard_confirmed=False
):
    """
    안전 규칙 엔진 결과와 L/S 평가 근거를 받아 위험도를 산정한다.

    evidence_sufficient:
        L과 S를 평가할 근거가 충분한지 나타내는 값.
        실제 시스템에서는 영상 분석 및 추가 현장정보를 통해 판단해야 한다.
    """

    if not isinstance(rule_result, dict):
        raise ValueError("rule_result는 딕셔너리 형태여야 합니다.")

    if not isinstance(critical_hazard_confirmed, bool):
        raise ValueError("critical_hazard_confirmed는 bool이어야 합니다.")

    violation_status = rule_result.get("violation_status")
    hazard = rule_result.get("hazard")
    reason = rule_result.get("reason")

    if violation_status == "관측상 정상":
        return {
            "hazard": hazard,
            "violation_status": violation_status,
            "risk_score": None,
            "risk_level": "평가 제외",
            "reason": reason,
            "review_required": False,
            "review_reason": None
        }

    if violation_status in ("위반 의심", "미확인"):
        return {
            "hazard": hazard,
            "violation_status": violation_status,
            "risk_score": None,
            "risk_level": "미확정",
            "reason": reason,
            "review_required": True,
            "review_reason": review_reason or "위반 여부에 대한 추가 확인 필요"
        }

    if violation_status != "위반":
        raise ValueError(f"지원하지 않는 판정 상태: {violation_status}")

    if evidence_sufficient is not True:
        return {
            "hazard": hazard,
            "violation_status": violation_status,
            "risk_score": None,
            "risk_level": "미확정",
            "reason": reason,
            "review_required": True,
            "review_reason": review_reason or "가능성(L)과 중대성(S)의 판단 근거 부족"
        }

    # 근거가 충분하더라도 L/S가 모두 입력되어야 한다.
    if likelihood is None or severity is None:
        return {
            "hazard": hazard,
            "violation_status": violation_status,
            "risk_score": None,
            "risk_level": "미확정",
            "reason": reason,
            "review_required": True,
            "review_reason": "가능성(L)과 중대성(S) 점수 입력 필요"
        }

    risk_score = calculate_risk_score(likelihood, severity)
    risk_level = classify_risk_level(risk_score)
    if critical_hazard_confirmed:
        risk_level = "상"

    result = {
        "hazard": hazard,
        "violation_status": violation_status,
        "likelihood": likelihood,
        "severity": severity,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "reason": reason,
        "review_required": False,
        "review_reason": None
    }
    if critical_hazard_confirmed:
        result["critical_priority_applied"] = True
        result["critical_priority_reason"] = (
            "중대 위험 확인에 따른 위험등급 상향"
        )
    return result


# 위험요소별 평가 대상
SUPPORTED_HAZARDS = {
    "guardrail",
    "opening",
    "work_platform"
}

# 작업자 노출 상태 → 발생 가능성 L
LIKELIHOOD_MAP = {
    "controlled": 1,
    "possible": 2,
    "direct": 3
}

# 예상 피해 수준 → 피해 심각도 S
SEVERITY_MAP = {
    "minor": 1,
    "serious": 2,
    "fatal": 3
}


def evaluate_likelihood(worker_exposure):
    """작업자 노출 상태를 바탕으로 발생 가능성 L을 분류한다."""

    if worker_exposure not in LIKELIHOOD_MAP:
        return None

    return LIKELIHOOD_MAP[worker_exposure]


def evaluate_severity(expected_consequence):
    """예상 피해 수준을 바탕으로 피해 심각도 S를 분류한다."""

    if expected_consequence not in SEVERITY_MAP:
        return None

    return SEVERITY_MAP[expected_consequence]


def assess_risk_from_evidence(rule_result, risk_evidence):
    """
    구조화된 평가 정보를 이용해 L/S를 분류하고 위험도를 산정한다.
    """

    if not isinstance(risk_evidence, dict):
        raise ValueError("risk_evidence는 딕셔너리여야 합니다.")

    hazard_type = risk_evidence.get("hazard_type")
    worker_exposure = risk_evidence.get("worker_exposure")
    expected_consequence = risk_evidence.get("expected_consequence")
    evidence_sufficient = risk_evidence.get("evidence_sufficient")
    critical_hazard_confirmed = risk_evidence.get(
        "critical_hazard_confirmed", False
    )

    if hazard_type not in SUPPORTED_HAZARDS:
        raise ValueError(f"지원하지 않는 위험요소: {hazard_type}")

    if not isinstance(evidence_sufficient, bool):
        raise ValueError("evidence_sufficient는 True 또는 False여야 합니다.")

    if not isinstance(critical_hazard_confirmed, bool):
        raise ValueError("critical_hazard_confirmed는 bool이어야 합니다.")

    if evidence_sufficient is False:
        return assess_risk(
            rule_result,
            evidence_sufficient=False,
            review_reason="위험도 평가에 필요한 근거 부족"
        )

    likelihood = evaluate_likelihood(worker_exposure)
    severity = evaluate_severity(expected_consequence)

    if likelihood is None or severity is None:
        return assess_risk(
            rule_result,
            evidence_sufficient=False,
            review_reason="작업자 노출 상태 또는 예상 피해 수준 확인 필요"
        )

    result = assess_risk(
        rule_result,
        likelihood=likelihood,
        severity=severity,
        evidence_sufficient=True,
        critical_hazard_confirmed=critical_hazard_confirmed
    )

    result["hazard_type"] = hazard_type
    result["worker_exposure"] = worker_exposure
    result["expected_consequence"] = expected_consequence

    return result
