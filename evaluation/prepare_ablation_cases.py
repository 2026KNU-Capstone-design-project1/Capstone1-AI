import csv
import json
from pathlib import Path

from detection_adapter import adapt_video_results

SOURCE = Path("data/video_ai_sample.json")
OUTPUT = Path("evaluation/ablation_cases_draft.csv")

with SOURCE.open(encoding="utf-8") as f:
    data = json.load(f)

frames = adapt_video_results(data)
rows = []

for frame_index, frame in enumerate(frames):
    for item_index, item in enumerate(frame.get("fused_evidence", [])):
        detection = item.get("detection") or {}
        bbox = detection.get("bbox") or {}
        rule_checks = item.get("rule_checks") or []

        # 특정 위험에 대한 규칙 결과가 하나로 확정되는 경우만 사용
        statuses = {
            c.get("status")
            for c in rule_checks
            if c.get("status") in (
                "hazard", "no_hazard", "not_evaluable"
            )
        }

        rule_status = (
            next(iter(statuses))
            if len(statuses) == 1
            else ""
        )

        reason_codes = {
            c.get("reason_code")
            for c in rule_checks
            if c.get("reason_code")
        }

        reason_code = (
            next(iter(reason_codes))
            if len(reason_codes) == 1
            else ""
        )

        agreement = item.get("agrees_with_model")
        if agreement is None and len(rule_checks) == 1:
            agreement = rule_checks[0].get("agrees_with_model")

        source = frame.get("source") or {}
        frame_id = source.get("frame_index", frame_index)

        rows.append({
            "sample_id": f"sample_{frame_id}_{item_index}",
            "gt": "",
            "detection_present": str(bool(item.get("detection"))).lower(),
            "confidence": detection.get("confidence", ""),
            "bbox_width": bbox.get("width", ""),
            "bbox_height": bbox.get("height", ""),
            "rule_status": rule_status,
            "reason_code": reason_code,
            "agreement": (
                str(agreement).lower()
                if isinstance(agreement, bool)
                else ""
            ),
            "observation_complete": "",
            "hazard": item.get("hazard", ""),
        })

fields = [
    "sample_id", "gt", "detection_present",
    "confidence", "bbox_width", "bbox_height",
    "rule_status", "reason_code", "agreement",
    "observation_complete", "hazard",
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"생성 완료: {OUTPUT}")
print(f"위험 후보: {len(rows)}건")
print("gt와 observation_complete는 실제 데이터 검증 후 입력해야 합니다.")

for row in rows:
    print(
        row["sample_id"],
        "|", row["hazard"],
        "| rule:", row["rule_status"],
        "| agreement:", row["agreement"]
    )
