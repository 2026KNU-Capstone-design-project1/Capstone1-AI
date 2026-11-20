"""영상 AI 결과 → /assess 요청 변환 (C 정책: YOLO 우선 · 공간관계 보완) 테스트.

ai_server 의 /predict/video 응답에 들어가는 assessment_requests 가
아래 상황에서 기대대로 만들어지는지 확인한다.

| 상황                           | 기대 결과                         |
| YOLO 위험 탐지 1건             | /assess 요청 1건                  |
| YOLO 위험 0건 + 규칙 위험 1건  | 요청 1건 (규칙 후보)              |
| YOLO와 규칙이 같은 위험 탐지   | 중복 없이 1건 (matched)           |
| YOLO 위험과 규칙 위험이 다름   | C 정책: YOLO 후보만 (합집합 아님) |
| 규칙 후보에 bbox 있음          | 요청 bbox 에 findings bbox 반영   |
"""

import pytest

from detection_adapter import build_assessment_requests

WELD = "용접기 옆 소화기 미배치"
SCAFFOLD = "이동식비계 안전난간 설치 불량"
WELD_BOX = {"x": 0.3152, "y": 0.5052, "width": 0.0509, "height": 0.0742}
YOLO_BOX = {"x": 0.3864, "y": 0.2878, "width": 0.2299, "height": 0.2629}


def yolo(hazard, bbox=YOLO_BOX, confidence=0.98):
    return {"hazard": hazard, "location": "사진 가운데", "evidence": f"{hazard} 근거",
            "bbox": bbox, "confidence": confidence}


def finding(hazard, code="N-33", bbox=WELD_BOX, status="위험"):
    return {"code": code, "type": "결여", "status": status, "objects": [0],
            "missing": ["SO-40"], "hazard": hazard, "bbox": bbox}


def check(hazard, code="N-33", detected=False):
    return {"code": code, "hazard": hazard, "detected_by_model": detected, "status": "hazard",
            "reason_code": "rule_matched", "agrees_with_model": True if detected else None}


def frame(detections, findings=(), checks=(), frame_index=6360):
    """ai_server 가 내보내는 AI 결과 한 장면 (영상)."""
    return {
        "source_type": "video", "photo_id": None, "video_id": 906,
        "frame_index": frame_index, "frame_time_ms": frame_index * 1000 // 30,
        "image_url": f"https://example.test/frames/video_906/frame_{frame_index}.jpg",
        "detections": list(detections),
        "relations": {
            "objects": [{"id": 0, "code": "SO-24", "name": "용접기", "category": "공구",
                         "bbox": [0.3152, 0.5052, 0.0509, 0.0742], "confidence": 0.83}],
            "relations": [], "continuity": [],
            "findings": list(findings),
            "evaluation_status": "hazard" if findings else "no_hazard",
            "reason_code": "rule_matched" if findings else "condition_not_met",
            "checks": list(checks),
        },
    }


def requests_for(*frames):
    return build_assessment_requests({"results": list(frames)})


def test_yolo_only_makes_one_request():
    reqs = requests_for(frame([yolo(SCAFFOLD)]))
    assert len(reqs) == 1
    [det] = reqs[0]["detections"]
    assert det["hazard"] == SCAFFOLD
    assert det["bbox"] == YOLO_BOX
    assert det["evidence_context"]["risk_alert_source"] == "yolo"


def test_rule_only_makes_one_request():
    """YOLO 0건 + 규칙 1건 → 요청 1건 (C 정책 공간관계 보완)."""
    reqs = requests_for(frame([], [finding(WELD)], [check(WELD)]))
    assert len(reqs) == 1
    [det] = reqs[0]["detections"]
    assert det["hazard"] == WELD
    ctx = det["evidence_context"]
    assert ctx["risk_alert_source"] == "spatial_fallback"
    assert ctx["evidence_status"] == "spatial_only"
    assert ctx["risk_alert_policy"] == "yolo_first_spatial_fallback"


def test_same_hazard_from_yolo_and_rule_is_not_duplicated():
    reqs = requests_for(frame([yolo(WELD)], [finding(WELD)], [check(WELD, detected=True)]))
    [det] = reqs[0]["detections"]
    assert det["hazard"] == WELD
    assert det["bbox"] == YOLO_BOX  # YOLO 박스 우선
    assert det["evidence_context"]["evidence_status"] == "matched"
    assert det["evidence_context"]["risk_alert_source"] == "yolo"


def test_different_hazards_follow_yolo_first_not_union():
    """YOLO 위험(A)과 규칙 위험(B)이 다르면 C 정책은 YOLO 후보만 쓴다 (합집합 D 정책이 아님)."""
    reqs = requests_for(frame([yolo(SCAFFOLD)], [finding(WELD)], [check(WELD)]))
    hazards = [d["hazard"] for d in reqs[0]["detections"]]
    assert hazards == [SCAFFOLD]


def test_rule_candidate_uses_findings_bbox_and_zero_confidence():
    reqs = requests_for(frame([], [finding(WELD)], [check(WELD)]))
    [det] = reqs[0]["detections"]
    assert det["bbox"] == WELD_BOX
    # 0.0 은 'YOLO confidence 미제공' 의미 (스키마 호환을 위해 숫자로 유지)
    assert det["confidence"] == 0.0


def test_mixed_video_keeps_one_request_per_scene():
    reqs = requests_for(
        frame([], [finding(WELD)], [check(WELD)], frame_index=6360),
        frame([yolo(WELD)], [finding(WELD)], [check(WELD, detected=True)], frame_index=6795),
    )
    assert [r["source"]["frame_index"] for r in reqs] == [6360, 6795]
    assert [len(r["detections"]) for r in reqs] == [1, 1]
    assert [r["detections"][0]["evidence_context"]["risk_alert_source"] for r in reqs] == ["spatial_fallback", "yolo"]


def test_requests_match_assess_schema():
    """/assess 입력 형식(main.AssessmentRequest)을 그대로 통과하는지 (ollama 가 없으면 건너뜀)."""
    pytest.importorskip("ollama")
    from main import AssessmentRequest

    reqs = requests_for(frame([], [finding(WELD)], [check(WELD)]), frame([yolo(SCAFFOLD)], frame_index=10))
    for req in reqs:
        AssessmentRequest(**req)
