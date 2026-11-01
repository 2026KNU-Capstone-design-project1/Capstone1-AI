# evaluation/run_opening_test.py

import pandas as pd
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from rule_engine import evaluate_opening

CSV_PATH = Path(__file__).resolve().parent / "opening_rule_test_cases.csv"

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")


def parse_bool(value):
    if pd.isna(value) or str(value).strip() == "":
        return None

    value = str(value).strip().lower()

    if value == "true":
        return True
    elif value == "false":
        return False
    else:
        raise ValueError(f"잘못된 Boolean 값: {value}")


correct = 0
results = []

for _, row in df.iterrows():

    scene = {
        "opening_detected": parse_bool(row["opening_detected"]),
        "visibility": row["visibility"],
        "cover_present": parse_bool(row["cover_present"]),
        "guardrail_present": parse_bool(row["guardrail_present"]),
        "protection_complete": parse_bool(row["protection_complete"]),
        "violation_suspected": parse_bool(row["violation_suspected"])
    }

    result = evaluate_opening(scene)

    expected = row["expected_status"]
    actual = result["violation_status"]

    is_correct = expected == actual

    if is_correct:
        correct += 1

    results.append({
        "test_id": row["test_id"],
        "expected": expected,
        "actual": actual,
        "correct": is_correct,
        "reason": result["reason"]
    })


print("\n===== 개구부 판정 테스트 결과 =====")

for result in results:
    status = "정답" if result["correct"] else "오답"

    print(f"\n[{result['test_id']}] {status}")
    print(f"예상 결과: {result['expected']}")
    print(f"실제 결과: {result['actual']}")
    print(f"판정 근거: {result['reason']}")


total = len(results)
accuracy = correct / total * 100 if total > 0 else 0

print("\n===== 최종 결과 =====")
print(f"전체 테스트: {total}개")
print(f"정답: {correct}개")
print(f"오답: {total - correct}개")
print(f"판정 일치율: {accuracy:.2f}%")