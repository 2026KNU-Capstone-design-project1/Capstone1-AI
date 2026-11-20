from rule_engine import evaluate_scene
from risk_assessment import assess_risk_from_evidence


def contains_value(result, expected):
    """결과 딕셔너리 안에 기대하는 값이 있는지 확인"""
    return expected in result.values()


def run_test(name, test_function):
    try:
        test_function()
        print(f"[통과] {name}")
        return True
    except Exception as e:
        print(f"[실패] {name}: {e}")
        return False


def make_guardrail_violation():
    scene = {
        "edge_detected": True,
        "visibility": "clear",
        "guardrail_present": False,
        "guardrail_continuous": True,
        "guardrail_protects_edge": True,
        "alternative_protection": False,
        "violation_suspected": False
    }
    return evaluate_scene("guardrail", scene)


def test_missing_information():
    scene = {
        "edge_detected": None,
        "visibility": "clear",
        "guardrail_present": None,
        "guardrail_continuous": None,
        "guardrail_protects_edge": None,
        "alternative_protection": None,
        "violation_suspected": False
    }
    result = evaluate_scene("guardrail", scene)
    assert contains_value(result, "미확인"), result


def test_occluded_video():
    scene = {
        "edge_detected": True,
        "visibility": "occluded",
        "guardrail_present": None,
        "guardrail_continuous": None,
        "guardrail_protects_edge": None,
        "alternative_protection": None,
        "violation_suspected": False
    }
    result = evaluate_scene("guardrail", scene)
    assert contains_value(result, "미확인"), result


def test_insufficient_evidence():
    rule_result = make_guardrail_violation()

    evidence = {
        "hazard_type": "guardrail",
        "worker_exposure": "direct",
        "expected_consequence": "fatal",
        "evidence_sufficient": False
    }

    result = assess_risk_from_evidence(rule_result, evidence)
    assert contains_value(result, "미확정"), result


def test_unknown_worker_exposure():
    rule_result = make_guardrail_violation()

    evidence = {
        "hazard_type": "guardrail",
        "worker_exposure": "unknown",
        "expected_consequence": "fatal",
        "evidence_sufficient": True
    }

    result = assess_risk_from_evidence(rule_result, evidence)
    assert contains_value(result, "미확정"), result


def test_unknown_consequence():
    rule_result = make_guardrail_violation()

    evidence = {
        "hazard_type": "guardrail",
        "worker_exposure": "direct",
        "expected_consequence": "unknown",
        "evidence_sufficient": True
    }

    result = assess_risk_from_evidence(rule_result, evidence)
    assert contains_value(result, "미확정"), result


def test_unsupported_hazard():
    result = evaluate_scene("unknown_hazard", {})
    assert contains_value(result, "미확인"), result


def run_all_tests():
    tests = [
        ("필수 정보 누락", test_missing_information),
        ("영상 가림", test_occluded_video),
        ("증거 불충분", test_insufficient_evidence),
        ("작업자 노출 정보 불명확", test_unknown_worker_exposure),
        ("예상 재해 결과 불명확", test_unknown_consequence),
        ("지원하지 않는 위험요소", test_unsupported_hazard),
    ]

    passed = 0
    failed = 0

    print("===== 예외 상황 테스트 =====")

    for name, test_function in tests:
        if run_test(name, test_function):
            passed += 1
        else:
            failed += 1

    print("\n===== 최종 결과 =====")
    print(f"통과: {passed}")
    print(f"실패: {failed}")
    print(f"전체: {len(tests)}")


if __name__ == "__main__":
    run_all_tests()