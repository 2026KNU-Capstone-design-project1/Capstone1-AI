import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(".")
OUT = Path("evaluation/results/report_reliability")
OUT.mkdir(parents=True, exist_ok=True)

REPORT_KEYS = [
    "위험요소", "예상 재해유형", "위험도",
    "위험도 판단 근거", "위험내용",
    "관련 안전 기준", "법적사항",
    "개선대책", "안전대책",
    "위험도 평가 시 추가 확인사항",
]

VALID_VIOLATIONS = {
    "위반", "위반 의심", "관측상 정상", "미확인"
}
VALID_RISKS = {
    "상", "중", "하", "미확정", "평가 제외"
}

checks = []

def check(case, name, passed, details=""):
    checks.append({
        "case": case,
        "check": name,
        "passed": bool(passed),
        "details": str(details)
    })

def read_json(name):
    path = ROOT / name
    with path.open(encoding="utf-8") as f:
        return json.load(f)

def check_assessment(case, item, idx):
    prefix = f"assessment_{idx}"

    check(case, prefix + "_object", isinstance(item, dict))
    if not isinstance(item, dict):
        return

    check(
        case, prefix + "_hazard",
        isinstance(item.get("hazard"), str)
        and bool(item["hazard"].strip())
    )

    violation = item.get("violation_status")
    level = item.get("risk_level")

    check(
        case, prefix + "_violation_status",
        violation in VALID_VIOLATIONS, violation
    )
    check(
        case, prefix + "_risk_level",
        level in VALID_RISKS, level
    )

    if violation in {"미확인", "위반 의심"}:
        check(
            case, prefix + "_unconfirmed_risk",
            level == "미확정",
            f"violation={violation}, risk={level}"
        )

    report = item.get("report")
    check(
        case, prefix + "_report_object",
        isinstance(report, dict)
    )
    if not isinstance(report, dict):
        return

    for key in REPORT_KEYS:
        check(
            case,
            f"{prefix}_report_{key}",
            isinstance(report.get(key), str),
            f"value={str(report.get(key))[:100]}"
        )

    # 위험도 표시값은 보고서의 정해진 등급과 동일해야 한다.
    report_level = report.get("위험도", "")
    # 위험도 등급과 보고서 표현 간 의미적 일치 확인
    allowed_report_levels = {
        "상": {"상"},
        "중": {"중"},
        "하": {"하"},
        "미확정": {"미확정", "추가 확인 필요"},
        "평가 제외": {"평가 제외"},
    }

    check(
        case,
        prefix + "_report_risk_consistency",
        report_level in allowed_report_levels.get(level, set()),
        f"assessment={level}, report={report_level}"
    )

    if violation in {"미확인", "위반 의심"}:
        check(
            case, prefix + "_no_explicit_violation_claim",
            not any(
                phrase in str(report.get("위험내용", ""))
                for phrase in (
                    "법령 위반이 확인됨",
                    "법령 위반이 확정됨",
                    "위반 사실이 확인됨",
                )
            ),
            "명시적인 위반 확정 문구 검사"
        )

    legal = str(report.get("법적사항", ""))
    check(
        case, prefix + "_legal_basis_present",
        bool(legal.strip()), legal[:100]
    )

    # 법령 조문이 기재되어 있어도 실제 출처 검증은 별도 필요.
    if legal != "관련 법령 추가 확인 필요":
        check(
            case, prefix + "_legal_manual_review_needed",
            False,
            f"법령 출처/시행일/적용성 수동 검토 필요: {legal[:150]}"
        )

def check_summary(case, assessments, summary):
    expected = Counter(
        a.get("violation_status")
        for a in assessments
    )
    risks = Counter(
        a.get("risk_level")
        for a in assessments
    )

    check(
        case, "summary_total",
        summary.get("total_assessments") == len(assessments),
        f"summary={summary.get('total_assessments')}, "
        f"actual={len(assessments)}"
    )

    mappings = {
        "confirmed_violations": expected["위반"],
        "suspected_violations": expected["위반 의심"],
        "normal": expected["관측상 정상"],
        "unconfirmed": expected["미확인"],
        "high_risk": risks["상"],
        "medium_risk": risks["중"],
        "low_risk": risks["하"],
        "undetermined_risk": risks["미확정"],
    }

    for key, value in mappings.items():
        check(
            case, "summary_" + key,
            summary.get(key) == value,
            f"summary={summary.get(key)}, actual={value}"
        )

for filename in ("demo_result.json", "demo_result_multi.json"):
    data = read_json(filename)
    items = data.get("assessments")

    check(filename, "assessments_list", isinstance(items, list))

    if isinstance(items, list):
        for idx, item in enumerate(items, 1):
            check_assessment(filename, item, idx)

multi = read_json("demo_result_multi.json")
db = read_json("report_db_payload_multi.json")

multi_items = multi.get("assessments", [])
db_items = db.get("risk_items", [])

check(
    "db_payload", "item_count",
    len(db_items) == len(multi_items),
    f"db={len(db_items)}, api={len(multi_items)}"
)

for idx, (source, saved) in enumerate(
    zip(multi_items, db_items), 1
):
    for key in (
        "hazard", "violation_status",
        "risk_level", "confidence", "bbox"
    ):
        check(
            "db_payload",
            f"item_{idx}_{key}_preserved",
            source.get(key) == saved.get(key),
            f"source={source.get(key)}, db={saved.get(key)}"
        )

raw_summary = db.get("raw_summary_api_response", {}).get(
    "summary", {}
)
check_summary("db_payload", multi_items, raw_summary)

stored_summary = db.get("summary_metrics", {}).get(
    "summary", {}
)
check_summary("db_payload_stored", multi_items, stored_summary)

check(
    "db_payload", "assessment_count",
    db.get("assessment_count") == len(multi_items),
    f"count={db.get('assessment_count')}"
)

check(
    "db_payload", "full_report_count",
    db.get("raw_full_api_response", {})
      .get("page3_details", {})
      .get("total_items") == len(multi_items)
)

output = OUT / "report_reliability_results.csv"
with output.open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=["case", "check", "passed", "details"]
    )
    writer.writeheader()
    writer.writerows(checks)

passed = sum(c["passed"] for c in checks)
failed = [c for c in checks if not c["passed"]]

print("=" * 60)
print("통합 보고서 정합성 검사")
print("=" * 60)
print(f"전체 검사: {len(checks)}")
print(f"통과: {passed}")
print(f"실패/수동검토 필요: {len(failed)}")
print(f"통과율: {passed / len(checks):.2%}")

if failed:
    print("\n검토할 항목:")
    for c in failed:
        print(
            f'[{c["case"]}] {c["check"]}: '
            f'{c["details"][:180]}'
        )

print("\n저장:", output)
