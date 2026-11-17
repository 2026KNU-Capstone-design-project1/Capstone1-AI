import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = os.getenv("REPORT_API_URL", "http://127.0.0.1:8000").rstrip("/")
INPUT = Path(os.getenv("ASSESSMENTS_FILE", "demo_result.json"))
OUTPUT = Path(os.getenv("REPORT_DB_OUTPUT", "report_db_payload.json"))
REPORT_ID = os.getenv("REPORT_ID", "DEMO-REPORT-001")
PERIOD = os.getenv("REPORT_PERIOD") or None


def post_json(path, body):
    response = requests.post(f"{BASE_URL}{path}", json=body, timeout=60)
    try:
        response.raise_for_status()
    except requests.HTTPError:
        raise RuntimeError(f"{path} failed: HTTP {response.status_code}: {response.text[:1200]}") from None
    return response.json()


def main():
    if not INPUT.exists():
        raise SystemExit(f"Missing {INPUT}; first run demo_ai.py to generate it.")

    with INPUT.open(encoding="utf-8") as f:
        assessed = json.load(f)

    items = assessed.get("assessments")
    if not isinstance(items, list):
        raise SystemExit("Input must be an /assess response with assessments array.")

    site_info = {
        "site_id": os.getenv("SITE_ID", "SITE-001"),
        "site_name": os.getenv("SITE_NAME", "테스트 건설현장"),
    }
    common = {
        "report_id": REPORT_ID,
        "site_info": site_info,
        "assessments": items,
    }

    summary = post_json("/reports/summary", common)
    full = post_json("/reports/full", {**common, "report_period": PERIOD})

    template_sections = {
        "cover_metadata": {
            "report_title": "종합 위험성평가 보고서",
            "site_info": full.get("site_info", site_info),
            "report_period": full.get("report_period", PERIOD),
            "branding": "VIRNECT / Industrial Safety",  # presentation metadata, not a risk finding
        },
        "executive_summary_and_overall_risk": full.get("page1_overview"),
        "area_category_repeated_risks_and_recommendations": full.get("page2_analysis"),
        "individual_risk_index_and_details": full.get("page3_details"),
        "data_quality": full.get("data_quality"),
        "narrative": full.get("narrative"),
    }

    risk_items = []
    for i, item in enumerate(items, 1):
        source = item.get("source") or assessed.get("source") or {}
        report = item.get("report") or {}
        risk_items.append({
            "assessment_id": assessed.get("assessment_id"),
            "risk_id": item.get("risk_id"),  # BE may need to assign if null
            "sequence": i,
            "hazard": item.get("hazard"),
            "hazard_type": item.get("hazard_type"),
            "location": item.get("location"),
            "area": item.get("area"),
            "work_name": item.get("work_name"),
            "image_url": item.get("image_url") or source.get("image_url"),
            "bbox": item.get("bbox"),
            "confidence": item.get("confidence"),
            "violation_status": item.get("violation_status"),
            "risk_level": item.get("risk_level"),
            "action_status": item.get("action_status"),
            "first_detected_at": item.get("first_detected_at"),
            "accident_type": report.get("예상 재해유형"),
            "risk_description": report.get("위험내용"),
            "risk_reason": report.get("위험도 판단 근거"),
            "safety_criteria": report.get("관련 안전 기준"),
            "legal_basis": report.get("법적사항"),
            "improvement": report.get("개선대책"),
            "safety_measures": report.get("안전대책"),
            "additional_checks": report.get("위험도 평가 시 추가 확인사항"),
            "report": report,  # preserve full original fields, even if extended later
            "original_assessment": item,
        })

    result = {
        "schema_version": "demo-report-export-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_id": REPORT_ID,
        "site_info": site_info,
        "report_period": PERIOD,
        "source_note": "Generated from /assess and /reports endpoints; NOT a DB save or PDF.",
        "assessment_count": len(items),
        "summary_metrics": {
            "summary": summary.get("summary"),
            "risk_distribution": summary.get("risk_distribution"),
            "violation_distribution": summary.get("violation_distribution"),
            "top_hazards": summary.get("top_hazards"),
            "priority_actions": summary.get("priority_actions"),
        },
        "report_template_sections": template_sections,
        "risk_items": risk_items,
        "raw_summary_api_response": summary,
        "raw_full_api_response": full,
    }
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved {OUTPUT} — {len(items)} assessed item(s)")
    print("Risk levels:", json.dumps(summary.get("risk_distribution"), ensure_ascii=False))
    print("Full sections:", ", ".join(k for k in ("page1_overview", "page2_analysis", "page3_details", "data_quality", "narrative") if k in full))
    print("Note: for an actual multi-item report, supply multiple /assess responses, not just one demo_result.json")


if __name__ == "__main__":
    main()