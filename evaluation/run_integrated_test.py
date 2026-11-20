import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from rule_engine import evaluate_scene


test_cases = [
    # 안전난간 테스트
    {
        "test_id": "I001",
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
        "test_id": "I002",
        "hazard_type": "guardrail",
        "scene": {
            "edge_detected": True,
            "visibility": "clear",
            "guardrail_present": False,
            "guardrail_continuous": False,
            "guardrail_protects_edge": False,
            "alternative_protection": False,
            "violation_suspected": False
        },
        "expected": "위반"
    },

    # 개구부 테스트
    {
        "test_id": "I003",
        "hazard_type": "opening",
        "scene": {
            "opening_detected": True,
            "visibility": "clear",
            "cover_present": False,
            "guardrail_present": False,
            "protection_complete": False,
            "violation_suspected": False
        },
        "expected": "위반"
    },
    {
        "test_id": "I004",
        "hazard_type": "opening",
        "scene": {
            "opening_detected": True,
            "visibility": "clear",
            "cover_present": True,
            "guardrail_present": True,
            "protection_complete": False,
            "violation_suspected": False
        },
        "expected": "위반"
    },
    {
        "test_id": "I005",
        "hazard_type": "opening",
        "scene": {
            "opening_detected": True,
            "visibility": "clear",
            "cover_present": True,
            "guardrail_present": True,
            "protection_complete": None,
            "violation_suspected": False
        },
        "expected": "미확인"
    },

    # 작업발판 테스트
    {
        "test_id": "I006",
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
        "test_id": "I007",
        "hazard_type": "work_platform",
        "scene": {
            "visibility": "clear",
            "platform_present": False,
            "gap_detected": False,
            "fastening_confirmed": False,
            "violation_suspected": False
        },
        "expected": "위반"
    },
    {
        "test_id": "I008",
        "hazard_type": "work_platform",
        "scene": {
            "visibility": "clear",
            "platform_present": True,
            "gap_detected": None,
            "fastening_confirmed": True,
            "violation_suspected": False
        },
        "expected": "미확인"
    }
]


correct = 0

print("\n===== 위험성평가 통합 테스트 =====")

for case in test_cases:

    result = evaluate_scene(
        case["hazard_type"],
        case["scene"]
    )

    expected = case["expected"]
    actual = result["violation_status"]

    is_correct = expected == actual

    if is_correct:
        correct += 1

    status = "정답" if is_correct else "오답"

    print(f"\n[{case['test_id']}] {status}")
    print(f"위험요소: {result['hazard']}")
    print(f"규칙 ID: {result['rule_id']}")
    print(f"예상 결과: {expected}")
    print(f"실제 결과: {actual}")
    print(f"판정 근거: {result['reason']}")


total = len(test_cases)
accuracy = correct / total * 100 if total > 0 else 0

print("\n===== 최종 결과 =====")
print(f"전체 테스트: {total}개")
print(f"정답: {correct}개")
print(f"오답: {total - correct}개")
print(f"판정 일치율: {accuracy:.2f}%")