import csv
import itertools
from pathlib import Path

from risk_assessment import (
    assess_risk,
    assess_risk_from_evidence,
    calculate_risk_score,
)

HAZARDS = ["guardrail", "opening", "work_platform"]
EXPOSURES = {"controlled": 1, "possible": 2, "direct": 3}
CONSEQUENCES = {"minor": 1, "serious": 2, "fatal": 3}

OUT = Path("evaluation/results/risk_assessment")
OUT.mkdir(parents=True, exist_ok=True)

records = []
passed = 0

def expected_level(score):
    if score <= 2:
        return "하"
    if score <= 4:
        return "중"
    return "상"

def evaluate_case(
    case_id, hazard, status, exposure, consequence, sufficient,
    expected_score, expected_risk_level, expected_review
):
    global passed

    rule = {
        "hazard": hazard,
        "violation_status": status,
        "reason": "평가용 입력"
    }

    evidence = {
        "hazard_type": hazard,
        "worker_exposure": exposure,
        "expected_consequence": consequence,
        "evidence_sufficient": sufficient
    }

    result = assess_risk_from_evidence(rule, evidence)

    checks = {
        "score": result.get("risk_score") == expected_score,
        "level": result.get("risk_level") == expected_risk_level,
        "review": result.get("review_required") is expected_review
    }

    ok = all(checks.values())
    passed += int(ok)

    records.append({
        "case_id": case_id,
        "hazard_type": hazard,
        "violation_status": status,
        "worker_exposure": exposure,
        "expected_consequence": consequence,
        "evidence_sufficient": sufficient,
        "expected_score": expected_score,
        "actual_score": result.get("risk_score"),
        "expected_level": expected_risk_level,
        "actual_level": result.get("risk_level"),
        "expected_review": expected_review,
        "actual_review": result.get("review_required"),
        "passed": ok,
    })

    if not ok:
        raise AssertionError(f"{case_id}: {checks}, result={result}")

# A. 3개 위험유형 × L 3단계 × S 3단계 = 27개
for hazard, (exposure, likelihood), (consequence, severity) in itertools.product(
    HAZARDS,
    EXPOSURES.items(),
    CONSEQUENCES.items()
):
    score = likelihood * severity

    evaluate_case(
        f"M{len(records)+1:03d}",
        hazard, "위반",
        exposure, consequence, True,
        score, expected_level(score), False
    )

# B. 정상, 위반 의심, 미확인 = 9개
for hazard in HAZARDS:
    for status in ["관측상 정상", "위반 의심", "미확인"]:
        level = "평가 제외" if status == "관측상 정상" else "미확정"

        evaluate_case(
            f"M{len(records)+1:03d}",
            hazard, status,
            "direct", "fatal", True,
            None, level, status != "관측상 정상"
        )

# C. 위반이어도 증거가 불충분하면 미확정 = 3개
for hazard in HAZARDS:
    evaluate_case(
        f"M{len(records)+1:03d}",
        hazard, "위반",
        "direct", "fatal", False,
        None, "미확정", True
    )

# D. 증거 충분 표시지만 L 또는 S가 없으면 미확정 = 3개
for hazard in HAZARDS:
    evaluate_case(
        f"M{len(records)+1:03d}",
        hazard, "위반",
        None, "fatal", True,
        None, "미확정", True
    )

# E. 잘못된 점수 입력 거부 = 4개
for likelihood, severity in [
    (0, 1), (1, 4), (True, 2), (1, 1.0)
]:
    try:
        calculate_risk_score(likelihood, severity)
    except ValueError:
        passed += 1
    else:
        raise AssertionError(
            f"잘못된 점수 입력을 허용함: {likelihood}, {severity}"
        )

# F. 지원하지 않는 위험유형 거부 = 1개
try:
    assess_risk_from_evidence(
        {"hazard": "unknown", "violation_status": "위반"},
        {
            "hazard_type": "unknown",
            "worker_exposure": "direct",
            "expected_consequence": "fatal",
            "evidence_sufficient": True
        }
    )
except ValueError:
    passed += 1
else:
    raise AssertionError("지원하지 않는 위험유형을 허용함")

path = OUT / "risk_matrix_test_results.csv"

with path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print("=" * 55)
print("L×S 위험도 평가 규칙 검증")
print("=" * 55)
print("L×S 조합:", 27)
print("상태별 분기:", 9)
print("증거 부족:", 3)
print("L/S 누락:", 3)
print("잘못된 입력:", 4)
print("미지원 위험유형:", 1)
print(f"전체: {passed}/47 통과")
print("결과:", path)
