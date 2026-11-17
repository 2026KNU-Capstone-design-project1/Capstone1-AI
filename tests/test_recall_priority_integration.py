from recall_priority import select_recall_priority_candidates
from detection_adapter import (
    adapt_video_results,
    build_assessment_requests,
)


def yolo(hazard):
    return {
        "hazard": hazard,
        "detection": {
            "hazard": hazard,
            "confidence": 0.8,
        },
        "spatial_findings": [],
    }


def spatial(hazard, status="위험"):
    return {
        "hazard": hazard,
        "detection": None,
        "spatial_findings": [
            {
                "hazard": hazard,
                "status": status,
                "code": "N-03",
            }
        ],
    }


def test_yolo_has_priority():
    result = select_recall_priority_candidates([
        yolo("위험 A"),
        spatial("위험 B"),
    ])

    assert len(result) == 1
    assert result[0]["hazard"] == "위험 A"
    assert result[0]["risk_alert_source"] == "yolo"


def test_spatial_fallback():
    result = select_recall_priority_candidates([
        spatial("위험 B"),
    ])

    assert len(result) == 1
    assert result[0]["risk_alert_source"] == "spatial_fallback"


def test_undetermined_is_not_alert():
    result = select_recall_priority_candidates([
        spatial("위험 B", status="판단 불가"),
        {
            "hazard": "위험 C",
            "detection": None,
            "spatial_findings": [],
            "evidence_status": "undetermined",
        },
    ])

    assert result == []


def test_adapter_to_assess_request():
    payload = {
        "results": [
            {
                "assessment_id": "recall-test-1",
                "source": {
                    "source_type": "photo",
                    "photo_id": "test-photo",
                },
                "detections": [],
                "relations": {
                    "objects": [],
                    "relations": [],
                    "continuity": [],
                    "findings": [
                        {
                            "hazard": "개구부 덮개 설치 불량",
                            "code": "N-03",
                            "status": "위험",
                            "type": "상태 객체",
                            "objects": [],
                        }
                    ],
                    "checks": [],
                    "evaluation_status": "hazard",
                    "reason_code": "rule_matched",
                },
            }
        ]
    }

    frames = adapt_video_results(payload)
    assert len(frames[0]["risk_alert_candidates"]) == 1

    requests = build_assessment_requests(payload)
    assert len(requests) == 1
    assert len(requests[0]["detections"]) == 1

    detection = requests[0]["detections"][0]

    assert detection["hazard"] == "개구부 덮개 설치 불량"

    context = detection["evidence_context"]
    assert context["risk_alert_source"] == "spatial_fallback"
    assert context["violation_status"] == "미확인"
    assert context["risk_level"] == "미확정"
