import csv
import json
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests
from detection_adapter import build_assessment_requests

API_URL = "http://127.0.0.1:8000/assess"
N_RUNS = 10
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

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out = Path("evaluation/results/api_repeat") / run_id
out.mkdir(parents=True, exist_ok=True)

with open("data/video_ai_sample.json", encoding="utf-8") as f:
    video = json.load(f)

frame = video["results"][0]
payload = build_assessment_requests({
    "video_id": video.get("video_id"),
    "results": [frame],
})[0]

# 재현성을 위해 입력 요청 저장
(out / "request.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

expected_count = len(payload.get("detections", []))
rows = []

def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * percent / 100
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)

for i in range(1, N_RUNS + 1):
    record = {
        "run": i,
        "http_status": "",
        "response_seconds": "",
        "request_success": False,
        "assessment_count": "",
        "count_match": False,
        "schema_valid": False,
        "complete_report": False,
        "judgment_signature": "",
        "conservative_status": False,
        "legal_review_phrase": False,
        "error": "",
    }

    print(f"\n[{i}/{N_RUNS}] API 호출 중...", flush=True)
    start = time.perf_counter()

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=TIMEOUT,
        )
        record["response_seconds"] = round(
            time.perf_counter() - start, 3
        )
        record["http_status"] = response.status_code

        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            raise ValueError("JSON 최상위 구조가 객체가 아님")

        record["request_success"] = True

        (out / f"response_{i:02d}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        items = data.get("assessments")
        if not isinstance(items, list):
            raise ValueError("assessments가 리스트가 아님")

        record["assessment_count"] = len(items)
        record["count_match"] = len(items) == expected_count

        schema_flags = []
        report_flags = []
        conservative_flags = []
        legal_flags = []
        signatures = []

        for item in items:
            if not isinstance(item, dict):
                schema_flags.append(False)
                report_flags.append(False)
                conservative_flags.append(False)
                legal_flags.append(False)
                signatures.append(["INVALID"])
                continue

            violation = item.get("violation_status")
            risk = item.get("risk_level")
            report = item.get("report")

            schema_flags.append(
                isinstance(item.get("hazard"), str)
                and bool(item["hazard"].strip())
                and violation in VALID_VIOLATIONS
                and risk in VALID_RISKS
                and isinstance(report, dict)
            )

            report_flags.append(
                isinstance(report, dict)
                and all(
                    isinstance(report.get(k), str)
                    and bool(report[k].strip())
                    for k in REPORT_KEYS
                )
            )

            conservative_flags.append(
                risk == "미확정"
                if violation in {"미확인", "위반 의심"}
                else True
            )

            legal = (
                str(report.get("법적사항", ""))
                if isinstance(report, dict) else ""
            )
            legal_flags.append(
                "관련 법령 추가 확인 필요" in legal
            )

            signatures.append([
                item.get("hazard"),
                violation,
                risk,
            ])

        record["schema_valid"] = (
            record["count_match"]
            and all(schema_flags)
        )
        record["complete_report"] = (
            record["count_match"]
            and all(report_flags)
        )
        record["conservative_status"] = (
            record["count_match"]
            and all(conservative_flags)
        )
        record["legal_review_phrase"] = (
            record["count_match"]
            and all(legal_flags)
        )

        record["judgment_signature"] = json.dumps(
            signatures, ensure_ascii=False, sort_keys=True
        )

    except Exception as e:
        if record["response_seconds"] == "":
            record["response_seconds"] = round(
                time.perf_counter() - start, 3
            )
        record["error"] = f"{type(e).__name__}: {e}"

    rows.append(record)

    print(
        "  성공:", record["request_success"],
        "| 응답시간:", record["response_seconds"],
        "| 스키마:", record["schema_valid"],
        "| 판정:", record["judgment_signature"][:100],
        "| 오류:", record["error"][:100],
        flush=True
    )

csv_path = out / "api_repeat_results.csv"
with csv_path.open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

successful = [r for r in rows if r["request_success"]]
times = [
    float(r["response_seconds"]) for r in rows
    if r["response_seconds"] != ""
]
signatures = [
    r["judgment_signature"] for r in successful
    if r["schema_valid"]
]
signature_counts = Counter(signatures)

def ratio(count, total):
    return round(count / total, 4) if total else None

summary = {
    "attempted_requests": N_RUNS,
    "successful_requests": len(successful),
    "request_success_rate": ratio(len(successful), N_RUNS),
    "schema_valid_rate": ratio(
        sum(bool(r["schema_valid"]) for r in rows), N_RUNS
    ),
    "complete_report_rate": ratio(
        sum(bool(r["complete_report"]) for r in rows), N_RUNS
    ),
    "conservative_status_rate": ratio(
        sum(bool(r["conservative_status"]) for r in successful),
        len(successful)
    ),
    "legal_review_phrase_rate": ratio(
        sum(bool(r["legal_review_phrase"]) for r in successful),
        len(successful)
    ),
    "judgment_signatures": dict(signature_counts),
    "most_common_signature_rate": ratio(
        max(signature_counts.values(), default=0),
        len(signatures)
    ),
    "mean_response_seconds": (
        round(statistics.mean(times), 3) if times else None
    ),
    "p95_response_seconds": (
        round(percentile(times, 95), 3) if times else None
    ),
    "min_response_seconds": min(times) if times else None,
    "max_response_seconds": max(times) if times else None,
}

(out / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("API + Qwen 10회 반복 실행 결과")
print("=" * 60)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\n결과 폴더:", out)
