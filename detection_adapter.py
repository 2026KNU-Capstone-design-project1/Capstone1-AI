from typing import Any
from evidence_fusion import fuse_evidence
from recall_priority import select_recall_priority_candidates
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
        if isinstance(frame.get("source"), dict):
            source = frame["source"].copy()
        else:
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
            "assessment_id": frame.get("assessment_id"),
            "site_info": frame.get("site_info"),
            "work_info": frame.get("work_info"),

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
        frame["risk_alert_candidates"] = (
            select_recall_priority_candidates(
                frame["fused_evidence"]
            )
        )
        
    return adapted

def build_assessment_requests(payload: dict) -> list[dict]:
    """영상 AI의 프레임별 결과를 /assess 요청 형식으로 변환한다."""

    frames = adapt_video_results(payload)
    requests = []

    for frame in frames:
        source = frame["source"]
        detections = []

        for item in frame["risk_alert_candidates"]:
            detection = item.get("detection") or {}

            # 규칙(공간관계) 후보는 YOLO 박스가 없으므로 spatial_findings 의 bbox 를 쓴다.
            # 이 bbox 는 규칙에 걸린 객체를 감싸는 영역이다
            # (결여: 기준 객체, 위험 인접: 두 객체, 상태 객체: 그 객체 · x, y, width, height 0~1 정규화).
            spatial_bbox = next(
                (
                    f.get("bbox")
                    for f in item.get("spatial_findings") or []
                    if f.get("status") == "위험" and f.get("bbox")
                ),
                None,
            )

            context = build_evidence_context(frame, item)

            detections.append({
                "hazard": item["hazard"],
                "location": detection.get("location"),
                "evidence": detection.get("evidence"),
                "bbox": detection.get("bbox") or spatial_bbox,
                # 규칙 후보의 0.0 은 '신뢰도 0' 이 아니라 'YOLO confidence 미제공' 의미
                # (출처는 evidence_context.risk_alert_source 로 구분)
                "confidence": detection.get("confidence", 0.0),
                "hazard_type": None,
                "rule_features": None,
                "risk_evidence": None,
                "evidence_context": context,
            })

        requests.append({
            "assessment_id": frame.get("assessment_id") or (
                f"video-{source.get('video_id')}-frame-{source.get('frame_index')}"
                if source.get("source_type") == "video"
                else f"photo-{source.get('photo_id')}"
            ),
            "site_info": frame.get("site_info"),
            "work_info": frame.get("work_info"),
            "source": source,
            "detections": detections,
        })

    return requests