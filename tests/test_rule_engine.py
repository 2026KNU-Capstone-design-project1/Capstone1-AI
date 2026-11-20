
import pytest

from rule_engine import (
    evaluate_guardrail,
    evaluate_opening,
    evaluate_work_platform,
    evaluate_scene,
)


# 테스트용 기본 입력값
def make_guardrail_scene(**overrides):
    scene = {
        "edge_detected": True,
        "visibility": "clear",
        "guardrail_present": True,
        "guardrail_continuous": True,
        "guardrail_protects_edge": True,
        "alternative_protection": False,
        "violation_suspected": False,
    }
    scene.update(overrides)
    return scene


def make_opening_scene(**overrides):
    scene = {
        "opening_detected": True,
        "visibility": "clear",
        "cover_present": True,
        "guardrail_present": False,
        "protection_complete": True,
        "violation_suspected": False,
    }
    scene.update(overrides)
    return scene


def make_platform_scene(**overrides):
    scene = {
        "visibility": "clear",
        "platform_present": True,
        "gap_detected": False,
        "fastening_confirmed": True,
        "violation_suspected": False,
    }
    scene.update(overrides)
    return scene


# 1. 안전난간 평가 테스트
@pytest.mark.parametrize(
    "overrides, expected_status, expected_rule",
    [
        ({}, "관측상 정상", None),
        (
            {
                "edge_detected": False,
            },
            "관측상 정상",
            None,
        ),
        (
            {
                "guardrail_present": False,
                "alternative_protection": False,
            },
            "위반",
            "F-01",
        ),
        (
            {
                "guardrail_continuous": False,
            },
            "위반",
            "F-02",
        ),
        (
            {
                "guardrail_continuous": False,
                "violation_suspected": True,
            },
            "위반 의심",
            "F-02",
        ),
        (
            {
                "guardrail_protects_edge": False,
            },
            "위반",
            "F-03",
        ),
        (
            {
                "alternative_protection": True,
            },
            "관측상 정상",
            None,
        ),
        (
            {
                "edge_detected": None,
            },
            "미확인",
            None,
        ),
        (
            {
                "visibility": "blocked",
            },
            "미확인",
            None,
        ),
        (
            {
                "visibility": "blocked",
                "violation_suspected": True,
            },
            "위반 의심",
            "F-02",
        ),
        (
            {
                "guardrail_present": None,
            },
            "미확인",
            None,
        ),
        (
            {
                "guardrail_continuous": None,
            },
            "미확인",
            None,
        ),
    ],
)
def test_evaluate_guardrail(overrides, expected_status, expected_rule):
    scene = make_guardrail_scene(**overrides)

    result = evaluate_guardrail(scene)

    assert result["violation_status"] == expected_status
    assert result["rule_id"] == expected_rule
    assert result["hazard"] == "안전난간"
    assert isinstance(result["reason"], str)


# 2. 개구부 평가 테스트
@pytest.mark.parametrize(
    "overrides, expected_status, expected_rule",
    [
        ({}, "관측상 정상", None),
        (
            {
                "opening_detected": False,
            },
            "관측상 정상",
            None,
        ),
        (
            {
                "cover_present": False,
                "guardrail_present": False,
            },
            "위반",
            "F-04",
        ),
        (
            {
                "protection_complete": False,
            },
            "위반",
            "F-05",
        ),
        (
            {
                "protection_complete": False,
                "violation_suspected": True,
            },
            "위반 의심",
            "F-05",
        ),
        (
            {
                "visibility": "blocked",
            },
            "미확인",
            "F-06",
        ),
        (
            {
                "opening_detected": None,
            },
            "미확인",
            "F-06",
        ),
        (
            {
                "cover_present": None,
            },
            "미확인",
            "F-06",
        ),
        (
            {
                "guardrail_present": None,
            },
            "미확인",
            "F-06",
        ),
        (
            {
                "protection_complete": None,
            },
            "미확인",
            "F-06",
        ),
    ],
)
def test_evaluate_opening(overrides, expected_status, expected_rule):
    scene = make_opening_scene(**overrides)

    result = evaluate_opening(scene)

    assert result["violation_status"] == expected_status
    assert result["rule_id"] == expected_rule
    assert result["hazard"] == "개구부"
    assert isinstance(result["reason"], str)


# 3. 작업발판 평가 테스트
@pytest.mark.parametrize(
    "overrides, expected_status, expected_rule",
    [
        ({}, "관측상 정상", "F-10"),
        (
            {
                "platform_present": False,
            },
            "위반",
            "F-07",
        ),
        (
            {
                "gap_detected": True,
            },
            "위반",
            "F-08",
        ),
        (
            {
                "gap_detected": True,
                "violation_suspected": True,
            },
            "위반 의심",
            "F-08",
        ),
        (
            {
                "fastening_confirmed": False,
            },
            "위반",
            "F-09",
        ),
        (
            {
                "visibility": "blocked",
            },
            "미확인",
            "F-09",
        ),
        (
            {
                "platform_present": None,
            },
            "미확인",
            "F-09",
        ),
        (
            {
                "gap_detected": None,
            },
            "미확인",
            "F-09",
        ),
        (
            {
                "fastening_confirmed": None,
            },
            "미확인",
            "F-09",
        ),
    ],
)
def test_evaluate_work_platform(overrides, expected_status, expected_rule):
    scene = make_platform_scene(**overrides)

    result = evaluate_work_platform(scene)

    assert result["violation_status"] == expected_status
    assert result["rule_id"] == expected_rule
    assert result["hazard"] == "작업발판"
    assert isinstance(result["reason"], str)


# 4. 통합 평가 함수 테스트
@pytest.mark.parametrize(
    "hazard_type, scene_factory, expected_hazard",
    [
        ("guardrail", make_guardrail_scene, "안전난간"),
        ("opening", make_opening_scene, "개구부"),
        ("work_platform", make_platform_scene, "작업발판"),
    ],
)
def test_evaluate_scene(hazard_type, scene_factory, expected_hazard):
    result = evaluate_scene(hazard_type, scene_factory())

    assert result["hazard"] == expected_hazard
    assert result["violation_status"] == "관측상 정상"


def test_evaluate_scene_unsupported_hazard():
    result = evaluate_scene("unknown", {})

    assert result["rule_id"] is None
    assert result["hazard"] == "unknown"
    assert result["violation_status"] == "미확인"
    assert "지원하지 않는 위험요소 유형" in result["reason"]


def test_evaluate_scene_guardrail_violation():
    scene = make_guardrail_scene(
        guardrail_present=False,
        alternative_protection=False,
    )

    result = evaluate_scene("guardrail", scene)

    assert result["violation_status"] == "위반"
    assert result["rule_id"] == "F-01"


def test_evaluate_scene_opening_violation():
    scene = make_opening_scene(
        cover_present=False,
        guardrail_present=False,
    )

    result = evaluate_scene("opening", scene)

    assert result["violation_status"] == "위반"
    assert result["rule_id"] == "F-04"


def test_evaluate_scene_platform_violation():
    scene = make_platform_scene(platform_present=False)

    result = evaluate_scene("work_platform", scene)

    assert result["violation_status"] == "위반"
    assert result["rule_id"] == "F-07"
