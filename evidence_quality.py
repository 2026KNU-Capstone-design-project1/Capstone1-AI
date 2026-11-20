"""Observation-quality gate. It does NOT verify a safety violation.

Use min_frames=None for frame-independent evaluation and
require_relation=False for the object-presence baseline (condition C).
"""
from __future__ import annotations

import math
from typing import Any


def _unit_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def assess_evidence_quality(
    fused_item: dict,
    confidence_threshold: float = 0.5,
    min_frames: int | None = 3,
    min_area_ratio: float = 0.01,
    require_relation: bool = True,
) -> dict:
    """Return whether all enabled observation checks passed.

    Note: 'evidence_sufficient' means quality gate passed, *not* that a
    violation or a risk level is established.
    """
    if not isinstance(fused_item, dict):
        raise TypeError("fused_item must be a dict")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if not 0 <= min_area_ratio <= 1:
        raise ValueError("min_area_ratio must be in [0, 1]")
    if min_frames is not None and (isinstance(min_frames, bool) or not isinstance(min_frames, int) or min_frames < 1):
        raise ValueError("min_frames must be a positive int or None")

    detection = fused_item.get("detection")
    detection = detection if isinstance(detection, dict) else {}
    checks = {}
    reasons = []

    conf = _unit_number(detection.get("confidence"))
    passed = conf is not None and conf >= confidence_threshold
    checks["confidence"] = {"value": conf, "threshold": confidence_threshold, "passed": passed}
    if not passed:
        reasons.append("탐지 신뢰도 부족 또는 누락")

    bbox = detection.get("bbox")
    area = None
    if isinstance(bbox, dict):
        w, h = _unit_number(bbox.get("width")), _unit_number(bbox.get("height"))
        x, y = _unit_number(bbox.get("x")), _unit_number(bbox.get("y"))
        if (w is not None and h is not None and w > 0 and h > 0
                and (x is None or x + w <= 1.000001)
                and (y is None or y + h <= 1.000001)):
            area = w * h
    passed = area is not None and area >= min_area_ratio
    checks["bbox_area"] = {"value": area, "threshold": min_area_ratio, "passed": passed}
    if not passed:
        reasons.append("탐지 박스 면적 부족 또는 누락")

    if min_frames is not None:
        frames = detection.get("frames")
        count = None
        if isinstance(frames, list):
            ids = {f.get("frame_index") for f in frames if isinstance(f, dict) and f.get("frame_index") is not None}
            count = len(ids)
        passed = count is not None and count >= min_frames
        checks["observation_frames"] = {"value": count, "threshold": min_frames, "passed": passed}
        if not passed:
            reasons.append("관측 프레임 수 부족 또는 누락")

    if require_relation:
        status = fused_item.get("evidence_status")
        passed = status == "matched"
        checks["model_rule_agreement"] = {"value": status, "passed": passed}
        if not passed:
            reasons.append("탐지 모델과 관계 규칙의 일치 근거 부족")

        rule_checks = fused_item.get("rule_checks") or []
        supported = any(
            isinstance(c, dict) and c.get("status") == "hazard"
            and c.get("reason_code") == "rule_matched"
            for c in rule_checks
        )
        checks["rule_support"] = {"passed": supported}
        if not supported:
            reasons.append("적용 가능한 위험 관계 규칙의 긍정 근거 없음")

    enough = all(c["passed"] for c in checks.values())
    return {
        "hazard": fused_item.get("hazard"),
        "evidence_sufficient": enough,
        "decision": "평가 가능" if enough else "검토 필요",
        "review_required": not enough,
        "review_reasons": reasons,
        "checks": checks,
    }
