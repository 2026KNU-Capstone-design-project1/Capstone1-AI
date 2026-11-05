from fastapi.testclient import TestClient
import main


client = TestClient(main.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_assess_multiple_hazards(monkeypatch):
    def fake_generate_assessment(scene, results, return_json=False):
        return {
            "위험요소": "테스트 위험요소",
            "위험도": "추가 확인 필요",
            "위험도 판단 근거": "테스트 데이터",
            "개선대책": "테스트 개선대책",
            "안전대책": "테스트 안전대책",
            "위험도 평가 시 추가 확인사항": "현장 확인 필요",
        }

    # 실제 CSV 검색 대신 가짜 검색 결과 사용
    def fake_search_criteria(query, top_k=5, method="char"):
        return [{"위험요소": "테스트", "유사도": 0.9}]

    monkeypatch.setattr(
        main, "generate_assessment", fake_generate_assessment
    )
    monkeypatch.setattr(
        main, "search_criteria", fake_search_criteria
    )

    payload = {
        "assessment_id": "RA-AUTO-001",
        "source": {
            "source_type": "video",
            "video_id": 3,
            "frame_index": 367,
        },
        "detections": [
            {
                "hazard": "굴착부 안전난간 미설치",
                "confidence": 0.92,
                "hazard_type": "guardrail",
                "rule_features": {
                    "edge_detected": True,
                    "visibility": "clear",
                    "guardrail_present": False,
                    "guardrail_continuous": None,
                    "guardrail_protects_edge": None,
                    "alternative_protection": False,
                    "violation_suspected": False,
                },
                "risk_evidence": {
                    "worker_exposure": "direct",
                    "expected_consequence": "fatal",
                    "evidence_sufficient": True,
                },
            },
            {
                "hazard": "바닥 개구부 방호조치 미확인",
                "confidence": 0.81,
                "hazard_type": "opening",
                "rule_features": {
                    "opening_detected": True,
                    "visibility": "occluded",
                    "cover_present": None,
                    "guardrail_present": None,
                    "protection_complete": None,
                    "violation_suspected": True,
                },
                "risk_evidence": {
                    "worker_exposure": "possible",
                    "expected_consequence": "fatal",
                    "evidence_sufficient": False,
                },
            },
        ],
    }

    response = client.post("/assess", json=payload)

    assert response.status_code == 200

    data = response.json()
    assessments = data["assessments"]

    assert len(assessments) == 2

    assert assessments[0]["violation_status"] == "위반"
    assert assessments[0]["risk_level"] == "상"
    assert assessments[0]["risk_assessment"]["risk_score"] == 9
    assert assessments[0]["report"]["위험도"] == "상"

    assert assessments[1]["violation_status"] == "미확인"
    assert assessments[1]["risk_level"] == "미확정"
    assert assessments[1]["report"]["위험도"] == "추가 확인 필요"

    assert "미설치 또는 불량이 확인되면" in (
        assessments[1]["report"]["개선대책"]
    )
