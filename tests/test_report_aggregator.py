import pytest

from report_aggregator import aggregate_assessments


def test_aggregate_multiple_assessments():
    assessments = [
        {
            "hazard": "굴착부 안전난간 미설치",
            "violation_status": "위반",
            "risk_level": "상",
            "rule_result": {
                "hazard": "안전난간"
            },
        },
        {
            "hazard": "바닥 개구부 방호조치 미확인",
            "violation_status": "미확인",
            "risk_level": "미확정",
            "rule_result": {
                "hazard": "개구부"
            },
        },
    ]

    result = aggregate_assessments(assessments)

    assert result["summary"]["total_assessments"] == 2
    assert result["summary"]["confirmed_violations"] == 1
    assert result["summary"]["unconfirmed"] == 1

    assert result["risk_distribution"]["상"] == 1
    assert result["risk_distribution"]["미확정"] == 1

    assert result["top_hazards"] == [
        {"hazard": "개구부", "count": 1},
        {"hazard": "안전난간", "count": 1},
    ]


def test_unknown_risk_is_not_classified_as_high():
    assessments = [
        {
            "hazard": "개구부",
            "violation_status": "미확인",
            "risk_level": "상",
        }
    ]

    result = aggregate_assessments(assessments)

    assert result["risk_distribution"]["상"] == 0
    assert result["risk_distribution"]["미확정"] == 1


def test_empty_assessments():
    result = aggregate_assessments([])

    assert result["summary"]["total_assessments"] == 0
    assert result["top_hazards"] == []


def test_invalid_input():
    with pytest.raises(TypeError):
        aggregate_assessments(None)


def test_priority_actions_order():
    assessments = [
        {
            "hazard": "개구부 방호 상태 미확인",
            "violation_status": "미확인",
            "risk_level": "미확정",
            "report": {
                "개선대책": "개구부 방호시설 설치 상태 확인"
            },
        },
        {
            "hazard": "작업발판 안전조치 미흡",
            "violation_status": "위반",
            "risk_level": "중",
            "report": {
                "개선대책": "작업발판 안전조치 보완"
            },
        },
        {
            "hazard": "안전난간 미설치",
            "violation_status": "위반",
            "risk_level": "상",
            "report": {
                "개선대책": "안전난간 즉시 설치"
            },
        },
    ]

    result = aggregate_assessments(assessments)
    actions = result["priority_actions"]

    assert len(actions) == 3

    assert [item["risk_level"] for item in actions] == [
        "상",
        "중",
        "미확정",
    ]

    # 위험요소와 개선대책이 올바르게 연결되는지 확인
    assert actions[0]["hazard"] == "안전난간 미설치"
    assert actions[0]["action"] == "안전난간 즉시 설치"

    assert actions[1]["hazard"] == "작업발판 안전조치 미흡"
    assert actions[1]["action"] == "작업발판 안전조치 보완"

    assert actions[2]["hazard"] == "개구부 방호 상태 미확인"
    assert actions[2]["action"] == "개구부 방호시설 설치 상태 확인"
