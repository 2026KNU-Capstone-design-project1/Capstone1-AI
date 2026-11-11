"""Generate A/B/C/D predictions for independently labelled evaluation units.

Input: evaluation/ablation_cases.csv (see README). Do not assign ground truth
from model outputs. An evaluation unit is an image/event + a specified hazard.
Output: evaluation/ablation_predictions.csv for calculate_metrics.py.

A: hazard candidate detected -> risk, otherwise normal (naive baseline)
B: relation rule hazard -> risk; no_hazard -> normal; non-evaluable -> review
C: A with detection-quality gate; no-detection requires explicit complete observation
D: B with same detection-quality gate, plus matched relationship evidence for positives

A and B cannot treat rule_not_supported as evidence of safety. B's abstentions
are structural, not the proposed uncertainty gate; report them separately.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_quality import assess_evidence_quality

FIELDS = ["sample_id", "gt", "detection_present", "confidence", "bbox_width", "bbox_height", "rule_status", "reason_code", "agreement", "observation_complete"]
PRED_FIELDS = ["sample_id", "gt", "A", "B", "C", "D"]
ALLOWED_RULES = {"hazard", "no_hazard", "not_evaluable"}


def parse_bool(value: str, field: str) -> bool:
    value = value.strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field}: expected true/false, got {value!r}")


def number_or_none(value: str) -> float | None:
    if not value.strip():
        return None
    return float(value)


def run_case(row: dict) -> dict:
    sample_id = row["sample_id"].strip()
    if not sample_id:
        raise ValueError("empty sample_id")
    gt = int(row["gt"])
    if gt not in (0, 1):
        raise ValueError("gt must be 0 or 1")
    detected = parse_bool(row["detection_present"], "detection_present")
    complete = parse_bool(row["observation_complete"], "observation_complete")
    status = row["rule_status"].strip()
    if status not in ALLOWED_RULES:
        raise ValueError(f"rule_status must be one of {sorted(ALLOWED_RULES)}")

    raw_agreement = row["agreement"].strip().lower()
    if raw_agreement not in {"true", "false", ""}:
        raise ValueError("agreement must be true, false, or blank")
    agreement = None if raw_agreement == "" else parse_bool(raw_agreement, "agreement")

    a = 1 if detected else 0
    b = {"hazard": 1, "no_hazard": 0, "not_evaluable": -1}[status]

    # No detection in a fully reviewed, labelled observation: permit baseline
    # negative; otherwise the quality gate abstains from inferring absence.
    if not detected:
        c = 0 if complete else -1
        d = 0 if complete and b == 0 else -1
    else:
        detection = {
            "confidence": number_or_none(row["confidence"]),
            "bbox": {
                "x": 0.0, "y": 0.0,
                "width": number_or_none(row["bbox_width"]),
                "height": number_or_none(row["bbox_height"]),
            },
        }
        # Width and height are expected normalised; x/y are dummy zero values
        # and do not encode an actual location. Only area is used here.
        fusion = {
            "detection": detection,
            "evidence_status": "matched" if agreement is True and status == "hazard" else "undetermined",
            "rule_checks": [
                {"status": status, "reason_code": row["reason_code"].strip()}
            ],
        }
        q = assess_evidence_quality(fusion, min_frames=None, require_relation=False)
        quality_ok = q["evidence_sufficient"]
        c = a if quality_ok else -1
        if b == -1 or not quality_ok:
            d = -1
        elif b == 1:
            relation_ok = assess_evidence_quality(
                fusion, min_frames=None, require_relation=True
            )["evidence_sufficient"]
            d = 1 if relation_ok else -1
        else:
            # A supported no_hazard rule is a negative conclusion, not a model/rule
            # hazard match. Quality can validate the observation, not the negative.
            d = 0

    return {"sample_id": sample_id, "gt": gt, "A": a, "B": b, "C": c, "D": d}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("evaluation/ablation_cases.csv"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/ablation_predictions.csv"))
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Missing {args.input}. Create labelled rows using ablation_cases_template.csv; do not invent labels.")
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(f not in reader.fieldnames for f in FIELDS):
            raise SystemExit("Missing CSV columns: " + ", ".join(f for f in FIELDS if f not in (reader.fieldnames or [])))
        rows = list(reader)
    if not rows:
        raise SystemExit("No labelled evaluation rows; output not created.")
    ids, results = set(), []
    for line, row in enumerate(rows, 2):
        try:
            result = run_case(row)
            if result["sample_id"] in ids:
                raise ValueError("duplicate sample_id")
            ids.add(result["sample_id"])
            results.append(result)
        except (ValueError, TypeError, KeyError) as exc:
            raise SystemExit(f"CSV line {line}: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRED_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} labelled predictions to {args.output}")
    for m in ["A", "B", "C", "D"]:
        print(f"  {m}: review={sum(r[m] == -1 for r in results)}, risk={sum(r[m] == 1 for r in results)}, normal={sum(r[m] == 0 for r in results)}")


if __name__ == "__main__":
    main()
