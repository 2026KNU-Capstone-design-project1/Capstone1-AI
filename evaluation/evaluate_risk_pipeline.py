from risk_pipeline import evaluate_hazard_risk


def run_pipeline_tests():
    tests = [
        {
            "name": "안전난간 위반",
            "hazard_type": "guardrail",
            "scene": {
                "edge_detected": True,
                "visibility": "clear",
                "guardrail_present": False,
                "guardrail_continuous": False,
                "guardrail_protects_edge": False,
                "alternative_protection": False,
                "violation_suspected": False
            }
        },
        {
            "name": "개구부 위반",
            "hazard_type": "opening",
            "scene": {
                "opening_detected": True,
                "visibility": "clear",
                "cover_present": False,
                "guardrail_present": False,
                "protection_complete": False,
                "violation_suspected": False
            }
        },
        {
            "name": "작업발판 위반",
            "hazard_type": "work_platform",
            "scene": {
                "visibility": "clear",
                "platform_present": False,
                "gap_detected": False,
                "fastening_confirmed": False,
                "violation_suspected": False
            }
        }
    ]

    risk_evidence = {
        "worker_exposure": "direct",
        "expected_consequence": "fatal",
        "evidence_sufficient": True
    }

    passed = 0
    failed = 0

    print("===== 위험요소별 통합 테스트 =====")

    for test in tests:
        try:
            result = evaluate_hazard_risk(
                test["hazard_type"],
                test["scene"],
                risk_evidence
            )

            rule = result["rule_result"]
            risk = result["risk_assessment"]

            if (
                rule["violation_status"] == "위반"
                and risk["risk_score"] == 9
                and risk["risk_level"] == "상"
            ):
                print(f'[통과] {test["name"]}')
                passed += 1
            else:
                print(f'[실패] {test["name"]}')
                print("판정 결과:", rule)
                print("위험도 결과:", risk)
                failed += 1

        except Exception as e:
            print(f'[오류] {test["name"]}: {e}')
            failed += 1

    print("\n===== 최종 결과 =====")
    print(f"통과: {passed}")
    print(f"실패: {failed}")
    print(f"전체: {passed + failed}")



def run_status_tests():
    cases = [
        {
            "name": "정상 상태",
            "scene": {
                "edge_detected": True,
                "visibility": "clear",
                "guardrail_present": True,
                "guardrail_continuous": True,
                "guardrail_protects_edge": True,
                "alternative_protection": False,
                "violation_suspected": False
            },
            "expected_status": "관측상 정상",
            "expected_level": "평가 제외"
        },
        {
            "name": "위반 의심 상태",
            "scene": {
                "edge_detected": True,
                "visibility": "clear",
                "guardrail_present": False,
                "guardrail_continuous": False,
                "guardrail_protects_edge": False,
                "alternative_protection": False,
                "violation_suspected": True
            },
            "expected_status": "위반 의심",
            "expected_level": "미확정"
        },
        {
            "name": "시야 불량 상태",
            "scene": {
                "edge_detected": True,
                "visibility": "occluded",
                "guardrail_present": None,
                "guardrail_continuous": None,
                "guardrail_protects_edge": None,
                "alternative_protection": None,
                "violation_suspected": False
            },
            "expected_status": "미확인",
            "expected_level": "미확정"
        }
    ]

    print("\n===== 판정 상태별 통합 테스트 =====")

    passed = 0
    failed = 0

    for case in cases:
        evidence = {
            "worker_exposure": "direct",
            "expected_consequence": "fatal",
            "evidence_sufficient": True
        }

        result = evaluate_hazard_risk(
            "guardrail",
            case["scene"],
            evidence
        )

        rule = result["rule_result"]
        risk = result["risk_assessment"]

        if (
            rule["violation_status"] == case["expected_status"]
            and risk["risk_level"] == case["expected_level"]
            and risk["risk_score"] is None
        ):
            print(f'[통과] {case["name"]}')
            passed += 1
        else:
            print(f'[실패] {case["name"]}')
            print("실제 판정:", rule["violation_status"])
            print("실제 위험도:", risk["risk_level"])
            failed += 1

    print("\n===== 판정 상태 테스트 결과 =====")
    print(f"통과: {passed}")
    print(f"실패: {failed}")
    print(f"전체: {passed + failed}")



def run_all_hazard_status_tests():
    cases = [
        {
            "name": "안전난간 정상",
            "hazard_type": "guardrail",
            "scene": {
                "edge_detected": True,
                "visibility": "clear",
                "guardrail_present": True,
                "guardrail_continuous": True,
                "guardrail_protects_edge": True,
                "alternative_protection": False,
                "violation_suspected": False
            },
            "expected": "관측상 정상"
        },
        {
            "name": "안전난간 위반 의심",
            "hazard_type": "guardrail",
            "scene": {
                "edge_detected": True,
                "visibility": "clear",
                "guardrail_present": False,
                "guardrail_continuous": False,
                "guardrail_protects_edge": False,
                "alternative_protection": False,
                "violation_suspected": True
            },
            "expected": "위반 의심"
        },
        {
            "name": "안전난간 미확인",
            "hazard_type": "guardrail",
            "scene": {
                "edge_detected": True,
                "visibility": "occluded",
                "guardrail_present": None,
                "guardrail_continuous": None,
                "guardrail_protects_edge": None,
                "alternative_protection": None,
                "violation_suspected": False
            },
            "expected": "미확인"
        },

        {
            "name": "개구부 정상",
            "hazard_type": "opening",
            "scene": {
                "opening_detected": True,
                "visibility": "clear",
                "cover_present": True,
                "guardrail_present": False,
                "protection_complete": True,
                "violation_suspected": False
            },
            "expected": "관측상 정상"
        },
        {
            "name": "개구부 위반 의심",
            "hazard_type": "opening",
            "scene": {
                "opening_detected": True,
                "visibility": "clear",
                "cover_present": False,
                "guardrail_present": False,
                "protection_complete": False,
                "violation_suspected": True
            },
            "expected": "위반 의심"
        },
        {
            "name": "개구부 미확인",
            "hazard_type": "opening",
            "scene": {
                "opening_detected": True,
                "visibility": "occluded",
                "cover_present": None,
                "guardrail_present": None,
                "protection_complete": None,
                "violation_suspected": False
            },
            "expected": "미확인"
        },

        {
            "name": "작업발판 정상",
            "hazard_type": "work_platform",
            "scene": {
                "visibility": "clear",
                "platform_present": True,
                "gap_detected": False,
                "fastening_confirmed": True,
                "violation_suspected": False
            },
            "expected": "관측상 정상"
        },
        {
            "name": "작업발판 위반 의심",
            "hazard_type": "work_platform",
            "scene": {
                "visibility": "clear",
                "platform_present": False,
                "gap_detected": False,
                "fastening_confirmed": False,
                "violation_suspected": True
            },
            "expected": "위반 의심"
        },
        {
            "name": "작업발판 미확인",
            "hazard_type": "work_platform",
            "scene": {
                "visibility": "occluded",
                "platform_present": None,
                "gap_detected": None,
                "fastening_confirmed": None,
                "violation_suspected": False
            },
            "expected": "미확인"
        }
    ]

    evidence = {
        "worker_exposure": "direct",
        "expected_consequence": "fatal",
        "evidence_sufficient": True
    }

    passed = 0
    failed = 0

    print("\n===== 전체 위험요소 판정 상태 테스트 =====")

    for case in cases:
        result = evaluate_hazard_risk(
            case["hazard_type"],
            case["scene"],
            evidence
        )

        rule = result["rule_result"]
        risk = result["risk_assessment"]

        expected_level = (
            "평가 제외"
            if case["expected"] == "관측상 정상"
            else "미확정"
        )

        if (
            rule["violation_status"] == case["expected"]
            and risk["risk_level"] == expected_level
            and risk["risk_score"] is None
        ):
            print(f'[통과] {case["name"]}')
            passed += 1
        else:
            print(f'[실패] {case["name"]}')
            print("판정 상태:", rule["violation_status"])
            print("위험도:", risk["risk_level"])
            failed += 1

    print("\n===== 전체 위험요소 판정 테스트 결과 =====")
    print(f"통과: {passed}")
    print(f"실패: {failed}")
    print(f"전체: {passed + failed}")


if __name__ == "__main__":
    run_pipeline_tests()
    run_status_tests()
    run_all_hazard_status_tests()