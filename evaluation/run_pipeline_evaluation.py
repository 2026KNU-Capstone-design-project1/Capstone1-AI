import csv
from pathlib import Path
from datetime import datetime

from risk_pipeline import evaluate_hazard_risk


RISK_EVIDENCE = {
    "worker_exposure": "direct",
    "expected_consequence": "fatal",
    "evidence_sufficient": True
}


TEST_CASES = [
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
        },
        "expected_status": "위반",
        "expected_level": "상"
    },
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
        "expected_status": "관측상 정상",
        "expected_level": "평가 제외"
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
        "expected_status": "위반 의심",
        "expected_level": "미확정"
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
        "expected_status": "미확인",
        "expected_level": "미확정"
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
        },
        "expected_status": "위반",
        "expected_level": "상"
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
        "expected_status": "관측상 정상",
        "expected_level": "평가 제외"
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
        "expected_status": "위반 의심",
        "expected_level": "미확정"
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
        "expected_status": "미확인",
        "expected_level": "미확정"
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
        },
        "expected_status": "위반",
        "expected_level": "상"
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
        "expected_status": "관측상 정상",
        "expected_level": "평가 제외"
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
        "expected_status": "위반 의심",
        "expected_level": "미확정"
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
        "expected_status": "미확인",
        "expected_level": "미확정"
    }
]

def summarize_results(results, output_dir):
    from collections import defaultdict

    hazard_summary = defaultdict(lambda: {
        "total": 0,
        "passed": 0,
        "failed": 0
    })

    status_summary = defaultdict(lambda: {
        "total": 0,
        "passed": 0,
        "failed": 0
    })

    failures = []

    for row in results:
        hazard = row["hazard_type"]
        status = row["expected_status"]

        for summary, key in (
            (hazard_summary, hazard),
            (status_summary, status)
        ):
            summary[key]["total"] += 1

            if row["result"] == "통과":
                summary[key]["passed"] += 1
            else:
                summary[key]["failed"] += 1

        if row["result"] == "실패":
            failures.append(row)

    print("\n===== 위험요소별 평가 요약 =====")

    for hazard, data in hazard_summary.items():
        rate = data["passed"] / data["total"] * 100

        print(
            f'{hazard}: '
            f'{data["passed"]}/{data["total"]} '
            f'({rate:.1f}%)'
        )

    print("\n===== 판정 상태별 평가 요약 =====")

    for status, data in status_summary.items():
        rate = data["passed"] / data["total"] * 100

        print(
            f'{status}: '
            f'{data["passed"]}/{data["total"]} '
            f'({rate:.1f}%)'
        )

    print("\n===== 실패 항목 =====")

    if failures:
        for row in failures:
            print(
                f'{row["test_name"]}: '
                f'기대={row["expected_status"]}, '
                f'실제={row["actual_status"]}'
            )
    else:
        print("실패 항목 없음")

    summary_file = output_dir / "evaluation_summary.csv"

    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "구분",
            "항목",
            "전체",
            "통과",
            "실패",
            "통과율(%)"
        ])

        for hazard, data in hazard_summary.items():
            rate = data["passed"] / data["total"] * 100

            writer.writerow([
                "위험요소",
                hazard,
                data["total"],
                data["passed"],
                data["failed"],
                round(rate, 2)
            ])

        for status, data in status_summary.items():
            rate = data["passed"] / data["total"] * 100

            writer.writerow([
                "판정 상태",
                status,
                data["total"],
                data["passed"],
                data["failed"],
                round(rate, 2)
            ])

    print(f"\n요약 CSV 저장 위치: {summary_file}")

def run_evaluation():
    results = []

    for case in TEST_CASES:
        result = evaluate_hazard_risk(
            case["hazard_type"],
            case["scene"],
            RISK_EVIDENCE
        )

        rule = result["rule_result"]
        risk = result["risk_assessment"]

        status_correct = (
            rule["violation_status"] == case["expected_status"]
        )
        level_correct = (
            risk["risk_level"] == case["expected_level"]
        )

        test_result = (
            "통과" if status_correct and level_correct else "실패"
        )

        row = {
            "test_name": case["name"],
            "hazard_type": case["hazard_type"],
            "expected_status": case["expected_status"],
            "actual_status": rule["violation_status"],
            "expected_level": case["expected_level"],
            "actual_level": risk["risk_level"],
            "risk_score": risk.get("risk_score"),
            "reason": rule["reason"],
            "review_required": risk["review_required"],
            "result": test_result
        }

        results.append(row)

        print(f'[{test_result}] {case["name"]}')

    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"pipeline_result_{timestamp}.csv"

    fieldnames = list(results[0].keys())

    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    passed = sum(1 for row in results if row["result"] == "통과")
    failed = len(results) - passed

    print("\n===== 평가 결과 =====")
    print(f"통과: {passed}")
    print(f"실패: {failed}")
    print(f"전체: {len(results)}")
    print(f"\nCSV 저장 위치: {output_file}")
    summarize_results(results, output_dir)

if __name__ == "__main__":
    run_evaluation()