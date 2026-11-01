import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk_assessment import assess_risk

from risk_assessment import (
    assess_risk_from_evidence,
    evaluate_likelihood,
    evaluate_severity
)

rule_result = {
    "hazard": "안전난간",
    "violation_status": "위반",
    "reason": "추락 위험 단부의 방호조치 미실시"
}


def run_tests():
    passed = 0
    failed = 0

    print("===== 위험도 산정 테스트 =====\n")

    cases = [
        {
            "name": "위반 + L3/S3",
            "args": (rule_result, 3, 3),
            "kwargs": {"evidence_sufficient": True},
            "score": 9,
            "level": "상"
        },
        {
            "name": "위반 + L2/S2",
            "args": (rule_result, 2, 2),
            "kwargs": {"evidence_sufficient": True},
            "score": 4,
            "level": "중"
        },
        {
            "name": "위반 + L1/S1",
            "args": (rule_result, 1, 1),
            "kwargs": {"evidence_sufficient": True},
            "score": 1,
            "level": "하"
        },
        {
            "name": "위반 + 근거 부족",
            "args": (rule_result, 3, 3),
            "kwargs": {"evidence_sufficient": False},
            "score": None,
            "level": "미확정"
        },
        {
            "name": "위반 + L/S 미입력",
            "args": (rule_result, None, None),
            "kwargs": {"evidence_sufficient": True},
            "score": None,
            "level": "미확정"
        },
        {
            "name": "관측상 정상",
            "args": ({
                "hazard": "안전난간",
                "violation_status": "관측상 정상",
                "reason": "안전난간 설치 확인"
            },),
            "kwargs": {},
            "score": None,
            "level": "평가 제외"
        },
        {
            "name": "위반 의심",
            "args": ({
                "hazard": "안전난간",
                "violation_status": "위반 의심",
                "reason": "안전난간 상태 일부 가림"
            },),
            "kwargs": {},
            "score": None,
            "level": "미확정"
        },
        {
            "name": "미확인",
            "args": ({
                "hazard": "안전난간",
                "violation_status": "미확인",
                "reason": "영상에서 안전난간 확인 불가"
            },),
            "kwargs": {},
            "score": None,
            "level": "미확정"
        }
    ]

    for case in cases:
        try:
            result = assess_risk(*case["args"], **case["kwargs"])

            if (
                result["risk_score"] == case["score"]
                and result["risk_level"] == case["level"]
            ):
                print(f"[통과] {case['name']}")
                passed += 1
            else:
                print(f"[실패] {case['name']}")
                print("실제 결과:", result)
                failed += 1

        except Exception as e:
            print(f"[오류] {case['name']}: {e}")
            failed += 1

    print("\n===== 입력값 유효성 테스트 =====")

    invalid_cases = [
        ("가능성 범위 초과", 4, 3),
        ("중대성 범위 초과", 3, 0),
        ("가능성에 문자열 입력", "높음", 3)
    ]

    for name, likelihood, severity in invalid_cases:
        try:
            assess_risk(
                rule_result,
                likelihood,
                severity,
                evidence_sufficient=True
            )
            print(f"[실패] {name}: 잘못된 입력이 허용됨")
            failed += 1

        except ValueError:
            print(f"[통과] {name}: 잘못된 입력 차단")
            passed += 1

        except Exception as e:
            print(f"[실패] {name}: 예상하지 못한 오류 - {e}")
            failed += 1

    print("\n===== 최종 테스트 결과 =====")
    print(f"통과: {passed}")
    print(f"실패: {failed}")
    print(f"전체: {passed + failed}")


def run_evidence_tests():
    passed = 0
    failed = 0

    print("\n===== L/S 자동 분류 테스트 =====")

    likelihood_cases = [
        ("controlled", 1),
        ("possible", 2),
        ("direct", 3),
        ("unknown", None)
    ]

    severity_cases = [
        ("minor", 1),
        ("serious", 2),
        ("fatal", 3),
        ("unknown", None)
    ]

    for value, expected in likelihood_cases:
        result = evaluate_likelihood(value)

        if result == expected:
            print(f"[통과] L 분류: {value} → {result}")
            passed += 1
        else:
            print(f"[실패] L 분류: {value} → {result}")
            failed += 1

    for value, expected in severity_cases:
        result = evaluate_severity(value)

        if result == expected:
            print(f"[통과] S 분류: {value} → {result}")
            passed += 1
        else:
            print(f"[실패] S 분류: {value} → {result}")
            failed += 1

    print("\n===== 통합 위험도 평가 테스트 =====")

    evidence = {
        "hazard_type": "guardrail",
        "worker_exposure": "direct",
        "expected_consequence": "fatal",
        "evidence_sufficient": True
    }

    result = assess_risk_from_evidence(rule_result, evidence)

    if (
        result["likelihood"] == 3
        and result["severity"] == 3
        and result["risk_score"] == 9
        and result["risk_level"] == "상"
    ):
        print("[통과] 안전난간 통합 위험도 평가")
        passed += 1
    else:
        print("[실패] 안전난간 통합 위험도 평가")
        print(result)
        failed += 1

    evidence["evidence_sufficient"] = False

    result = assess_risk_from_evidence(rule_result, evidence)

    if result["risk_score"] is None and result["risk_level"] == "미확정":
        print("[통과] 근거 부족 시 평가 보류")
        passed += 1
    else:
        print("[실패] 근거 부족 시 평가 보류")
        failed += 1

    try:
        invalid_evidence = {
            "hazard_type": "electrical",
            "worker_exposure": "direct",
            "expected_consequence": "fatal",
            "evidence_sufficient": True
        }

        assess_risk_from_evidence(rule_result, invalid_evidence)

        print("[실패] 미지원 위험요소 차단")
        failed += 1

    except ValueError:
        print("[통과] 미지원 위험요소 차단")
        passed += 1

    print("\n===== L/S 자동 분류 테스트 결과 =====")
    print(f"통과: {passed}")
    print(f"실패: {failed}")
    print(f"전체: {passed + failed}")


if __name__ == "__main__":
    run_tests()
    run_evidence_tests()
