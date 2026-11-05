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


def test_summary_report_api():
    payload = {
        "report_id": "SUMMARY-AUTO-001",
        "site_info": {
            "site_id": "SITE-001",
            "site_name": "테스트 건설 현장",
        },
        "assessments": [
            {
                "hazard": "안전난간 미설치",
                "violation_status": "위반",
                "risk_level": "상",
                "rule_result": {"hazard": "안전난간"},
                "report": {
                    "개선대책": "안전난간 즉시 설치"
                },
            },
            {
                "hazard": "개구부 방호조치 미확인",
                "violation_status": "미확인",
                "risk_level": "미확정",
                "rule_result": {"hazard": "개구부"},
                "report": {
                    "개선대책": "방호시설 설치 여부 확인"
                },
            },
        ],
    }

    response = client.post("/reports/summary", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["report_id"] == "SUMMARY-AUTO-001"
    assert data["summary"]["total_assessments"] == 2
    assert data["summary"]["confirmed_violations"] == 1
    assert data["summary"]["unconfirmed"] == 1

    assert data["risk_distribution"]["상"] == 1
    assert data["risk_distribution"]["미확정"] == 1

    assert len(data["priority_actions"]) == 2
    assert data["priority_actions"][0]["hazard"] == "안전난간 미설치"
    assert data["priority_actions"][0]["risk_level"] == "상"
    assert data["priority_actions"][1]["risk_level"] == "미확정"

    
def test_full_report_api():
    payload = {
        "report_id": "REPORT-AUTO-001",
        "site_info": {
            "site_id": "SITE-001",
            "site_name": "테스트 건설 현장",
        },
        "report_period": "2026-11",
        "assessments": [
            {
                "risk_id": "RISK-001",
                "hazard": "안전난간 미설치",
                "violation_status": "위반",
                "risk_level": "상",
                "area": "A구역",
                "action_status": "미처리",
                "rule_result": {"hazard": "안전난간"},
                "report": {
                    "개선대책": "안전난간 설치"
                },
            },
            {
                "risk_id": "RISK-002",
                "hazard": "개구부 방호 상태 미확인",
                "violation_status": "미확인",
                "risk_level": "미확정",
                "area": "B구역",
                "action_status": None,
                "rule_result": {"hazard": "개구부"},
                "report": {
                    "개선대책": "방호시설 확인"
                },
            },
        ],
    }

    response = client.post("/reports/full", json=payload)

    assert response.status_code == 200

    data = response.json()

    stats = data["page1_overview"]["statistics"]

    assert stats["total_assessments"] == 2
    assert stats["high_risk_count"] == 1
    assert stats["unresolved_count"] == 1

    assert len(
        data["page2_analysis"]["area_distribution"]
    ) == 2

    assert len(
        data["page2_analysis"]["type_distribution"]
    ) == 2

    details = data["page3_details"]

    assert details["total_items"] == 2
    assert details["items"][0]["risk_id"] == "RISK-001"

    narrative = data["narrative"]

    assert "2건" in narrative["executive_summary"]
    assert "고위험 1건" in narrative["key_findings"][0]
    assert "A구역" in narrative["area_type_analysis"]
    assert "B구역" in narrative["area_type_analysis"]
    assert "미확정" in narrative["management_opinion"]


def test_assessment_response_to_full_report(monkeypatch):
    """개별 평가 응답을 종합 보고서에 그대로 전달할 수 있는지 확인."""

    def fake_generate_assessment(scene, results, return_json=False):
        return {
            "위험요소": "안전난간",
            "위험도": "추가 확인 필요",
            "위험도 판단 근거": "테스트 근거",
            "개선대책": "안전난간 설치",
            "안전대책": "현장 점검",
        }

    def fake_search_criteria(query, top_k=5, method="char"):
        return [{"위험요소": "안전난간", "유사도": 0.9}]

    monkeypatch.setattr(
        main, "generate_assessment", fake_generate_assessment
    )
    monkeypatch.setattr(
        main, "search_criteria", fake_search_criteria
    )

    assess_response = client.post(
        "/assess",
        json={
            "assessment_id": "RA-AUTO-002",
            "site_info": {
                "site_id": "SITE-001",
                "site_name": "테스트 건설 현장",
            },
            "work_info": {
                "work_name": "굴착 작업",
                "location": "A구역",
            },
            "source": {
                "source_type": "video",
                "video_id": 3,
                "image_url": "s3://example/frame.jpg",
            },
            "detections": [
                {
                    "hazard": "안전난간 미설치",
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
                }
            ],
        },
    )

    assert assess_response.status_code == 200

    assessments = assess_response.json()["assessments"]

    summary_response = client.post(
        "/reports/summary",
        json={
            "report_id": "SUMMARY-AUTO-002",
            "site_info": {"site_id": "SITE-001"},
            "assessments": assessments,
        },
    )

    assert summary_response.status_code == 200

    summary_data = summary_response.json()

    assert summary_data["summary"]["total_assessments"] == 1
    assert summary_data["summary"]["confirmed_violations"] == 1
    assert summary_data["risk_distribution"]["상"] == 1
    assert summary_data["priority_actions"][0]["hazard"] == "안전난간 미설치"

    report_response = client.post(
        "/reports/full",
        json={
            "report_id": "REPORT-AUTO-002",
            "site_info": {"site_id": "SITE-001"},
            "report_period": "2026-11",
            "assessments": assessments,
        },
    )

    assert report_response.status_code == 200

    data = report_response.json()

    assert data["page1_overview"]["statistics"]["total_assessments"] == 1
    assert data["page1_overview"]["statistics"]["high_risk_count"] == 1

    detail = data["page3_details"]["items"][0]

    assert detail["hazard"] == "안전난간 미설치"
    assert detail["work_name"] == "굴착 작업"
    assert detail["image_url"] == "s3://example/frame.jpg"
    assert detail["risk_level"] == "상"
    assert detail["risk_score"] == 9