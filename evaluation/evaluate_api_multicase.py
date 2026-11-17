import csv
import json
import time
from datetime import datetime
from pathlib import Path

import requests
from detection_adapter import build_assessment_requests

API_URL = "http://127.0.0.1:8000/assess"
TIMEOUT = 180

REPORT_KEYS = [
    "위험요소",
    "예상 재해유형",
    "위험도",
    "위험도 판단 근거",
    "위험내용",
    "관련 안전 기준",
    "법적사항",
    "개선대책",
    "안전대책",
    "위험도 평가 시 추가 확인사항",
]

VALID_VIOLATIONS = {
    "위반", "위반 의심", "관측상 정상", "미확인"
}
VALID_RISKS = {
    "상", "중", "하", "미확정", "평가 제외"
}
RISK_ALIASES = {
    "상": {"상"},
    "중": {"중"},
    "하": {"하"},
    "미확정": {"미확정", "추가 확인 필요"},
    "평가 제외": {"평가 제외"},
}

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out = Path("evaluation/results/api_multicase") / run_id
out.mkdir(parents=True, exist_ok=True)

with open("data/video_ai_sample.json", encoding="utf-8") as f:
    video = json.load(f)

payloads = build_assessment_requests(video)
rows = []

for i, payload in enumerate(payloads, 1):
    case_id = payload.get("assessment_id", f"case-{i}")

    print(f"\n[{i}/{len(payloads)}] {case_id}", flush=True)

    (out / f"request_{i:02d}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    detections = payload.get("detections", [])
    expected_hazards = [
        d.get("hazard")
        for d in detections
        if isinstance(d, dict)
    ]

    row = {
        "case_id": case_id,
        "expected_hazards": json.dumps(
            expected_hazards, ensure_ascii=False
        ),
        "actual_hazards": "",
        "http_status": "",
        "response_seconds": "",
        "request_success": False,
        "assessment_count_match": False,
        "schema_valid": False,
        "report_complete": False,
        "hazard_match": False,
        "risk_consistent": False,
        "conservative_status": False,
        "violation_status": "",
        "risk_level": "",
        "report_risk_level": "",
        "legal_basis": "",
        "legal_review_phrase": False,
        "error": "",
    }

    start = time.perf_counter()

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=TIMEOUT
        )

        row["response_seconds"] = round(
            time.perf_counter() - start, 3
        )
        row["http_status"] = response.status_code

        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            raise ValueError("응답 최상위 구조가 JSON 객체가 아님")

        items = data.get("assessments")

        if not isinstance(items, list):
            raise ValueError("assessments가 리스트가 아님")

        row["request_success"] = True

        (out / f"response_{i:02d}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        row["assessment_count_match"] = (
            len(items) == len(expected_hazards)
        )

        actual_hazards = [
            item.get("hazard") if isinstance(item, dict)
            else None
            for item in items
        ]
        row["actual_hazards"] = json.dumps(
            actual_hazards, ensure_ascii=False
        )

        row["hazard_match"] = (
            row["assessment_count_match"]
            and sorted(actual_hazards, key=str)
            == sorted(expected_hazards, key=str)
        )

        schema_ok = []
        report_ok = []
        risk_ok = []
        conservative_ok = []
        legal_phrases = []
        statuses = []
        levels = []
        report_levels = []
        legal_bases = []

        for item in items:
            if not isinstance(item, dict):
                schema_ok.append(False)
                report_ok.append(False)
                risk_ok.append(False)
                conservative_ok.append(False)
                legal_phrases.append(False)
                continue

            status = item.get("violation_status")
            level = item.get("risk_level")
            report = item.get("report")

            statuses.append(status)
            levels.append(level)

            schema_ok.append(
                isinstance(item.get("hazard"), str)
                and bool(item["hazard"].strip())
                and status in VALID_VIOLATIONS
                and level in VALID_RISKS
                and isinstance(report, dict)
            )

            complete = (
                isinstance(report, dict)
                and all(
                    isinstance(report.get(k), str)
                    and bool(report[k].strip())
                    for k in REPORT_KEYS
                )
            )
            report_ok.append(complete)

            if isinstance(report, dict):
                report_level = report.get("위험도")
                legal_basis = report.get("법적사항", "")
            else:
                report_level = None
                legal_basis = ""

            report_levels.append(report_level)
            legal_bases.append(legal_basis)

            risk_ok.append(
                report_level in RISK_ALIASES.get(level, set())
            )

            if status in {"미확인", "위반 의심"}:
                conservative_ok.append(level == "미확정")
            elif status == "관측상 정상":
                conservative_ok.append(level == "평가 제외")
            else:
                conservative_ok.append(True)

            # 법령 정확성 검사와 구분되는 보조 지표
            legal_phrases.append(
                isinstance(legal_basis, str)
                and "관련 법령 추가 확인 필요" in legal_basis
            )

        count_ok = row["assessment_count_match"]

        row["schema_valid"] = count_ok and all(schema_ok)
        row["report_complete"] = count_ok and all(report_ok)
        row["risk_consistent"] = count_ok and all(risk_ok)
        row["conservative_status"] = (
            count_ok and all(conservative_ok)
        )
        row["legal_review_phrase"] = (
            count_ok and all(legal_phrases)
        )

        row["violation_status"] = json.dumps(
            statuses, ensure_ascii=False
        )
        row["risk_level"] = json.dumps(
            levels, ensure_ascii=False
        )
        row["report_risk_level"] = json.dumps(
            report_levels, ensure_ascii=False
        )
        row["legal_basis"] = json.dumps(
            legal_bases, ensure_ascii=False
        )

    except Exception as e:
        if row["response_seconds"] == "":
            row["response_seconds"] = round(
                time.perf_counter() - start, 3
            )

        row["error"] = f"{type(e).__name__}: {e}"

    rows.append(row)

    print("  HTTP:", row["http_status"])
    print("  요청 성공:", row["request_success"])
    print("  위험요소 일치:", row["hazard_match"])
    print("  보고서 완전성:", row["report_complete"])
    print("  위험등급 정합성:", row["risk_consistent"])
    print("  보수적 판정:", row["conservative_status"])
    print("  응답시간:", row["response_seconds"], "초")

    if row["error"]:
        print("  오류:", row["error"])

csv_path = out / "multicase_results.csv"

with csv_path.open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(rows)


def rate(field):
    return round(
        sum(bool(r[field]) for r in rows) / len(rows),
        4
    ) if rows else None


times = [
    float(r["response_seconds"])
    for r in rows
    if r["response_seconds"] != ""
]

summary = {
    "total_cases": len(rows),
    "successful_cases": sum(
        bool(r["request_success"]) for r in rows
    ),
    "request_success_rate": rate("request_success"),
    "schema_valid_rate": rate("schema_valid"),
    "report_complete_rate": rate("report_complete"),
    "hazard_match_rate": rate("hazard_match"),
    "risk_consistency_rate": rate("risk_consistent"),
    "conservative_status_rate": rate("conservative_status"),
    "legal_review_phrase_rate": rate("legal_review_phrase"),
    "mean_response_seconds": (
        round(sum(times) / len(times), 3)
        if times else None
    ),
}

(out / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("서로 다른 위험요소 API 통합 평가")
print("=" * 60)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n결과 폴더:", out)
