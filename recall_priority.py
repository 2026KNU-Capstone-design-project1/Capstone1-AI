"""YOLO 우선·공간관계 보완(E) 위험 경보 정책.

경보 후보를 선택하는 함수이며, 실제 위반 여부나
L×S 위험등급을 확정하지 않는다.
"""

from typing import Any

POLICY_NAME = "yolo_first_spatial_fallback"


def select_recall_priority_candidates(
    fused_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """E 정책에 따라 위험 경보 후보를 선택한다.

    YOLO 위험 후보가 하나라도 있으면 YOLO 후보만 사용한다.
    YOLO 후보가 없으면 공간관계 위험 후보를 사용한다.
    판단불가 또는 위험 판정 근거가 없는 항목은 경보하지 않는다.
    """
    if not isinstance(fused_evidence, list):
        raise TypeError("fused_evidence는 리스트여야 합니다.")

    yolo_candidates = [
        item
        for item in fused_evidence
        if item.get("detection") is not None
    ]

    if yolo_candidates:
        return [
            {
                **item,
                "risk_alert_policy": POLICY_NAME,
                "risk_alert_source": "yolo",
            }
            for item in yolo_candidates
        ]

    spatial_candidates = [
        item
        for item in fused_evidence
        if any(
            isinstance(finding, dict)
            and finding.get("status") == "위험"
            for finding in (item.get("spatial_findings") or [])
        )
    ]

    return [
        {
            **item,
            "risk_alert_policy": POLICY_NAME,
            "risk_alert_source": "spatial_fallback",
        }
        for item in spatial_candidates
    ]
