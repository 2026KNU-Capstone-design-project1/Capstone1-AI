from typing import Any
from evidence_fusion import fuse_evidence
from evidence_context import build_evidence_context

def adapt_video_results(payload: dict[str, Any]) -> list[dict]:
    """영상 AI 결과를 프레임 단위로 정리한다.

    탐지·공간관계 결과를 보존하며,
    실제 위반 여부나 위험등급을 임의로 확정하지 않는다.
    """
    if not isinstance(payload.get("results"), list):
        raise ValueError("results는 배열이어야 합니다.")

    adapted = []

    for frame in payload["results"]:
        source = {
            "source_type": frame["source_type"],
            "photo_id": frame.get("photo_id"),
            "video_id": frame.get("video_id"),
            "frame_index": frame.get("frame_index"),
            "frame_time_ms": frame.get("frame_time_ms"),
            "image_url": frame.get("image_url"),
        }

        relations = frame.get("relations") or {}

        adapted.append({
            "source": source,
            "detections": frame.get("detections") or [],
            "spatial_objects": relations.get("objects") or [],
            "spatial_relations": relations.get("relations") or [],
            "continuity": relations.get("continuity") or [],
            "spatial_findings": relations.get("findings") or [],
            "evaluation_status": relations.get("evaluation_status"),
            "reason_code": relations.get("reason_code"),
            "rule_checks": relations.get("checks") or [],
        })

    for frame in adapted:
        frame["fused_evidence"] = fuse_evidence(frame)
        
    return adapted

def build_assessment_requests(payload: dict) -> list[dict]:
    """영상 AI의 프레임별 결과를 /assess 요청 형식으로 변환한다."""

    frames = adapt_video_results(payload)
    requests = []

    for frame in frames:
        source = frame["source"]
        detections = []

        for item in frame["fused_evidence"]:
            detection = item.get("detection") or {}

            context = build_evidence_context(frame, item)

            detections.append({
                "hazard": item["hazard"],
                "location": detection.get("location"),
                "evidence": detection.get("evidence"),
                "bbox": detection.get("bbox"),
                "confidence": detection.get("confidence", 0.0),
                "hazard_type": None,
                "rule_features": None,
                "risk_evidence": None,
                "evidence_context": context,
            })

        requests.append({
            "assessment_id": (
                f"video-{source.get('video_id')}-"
                f"frame-{source.get('frame_index')}"
            ),
            "site_info": None,
            "work_info": None,
            "source": source,
            "detections": detections,
        })

    return requests