def evaluate_guardrail(scene):
    """
    안전난간 위험 판정 함수

    판정 결과:
    - 관측상 정상
    - 위반
    - 위반 의심
    - 미확인
    """

    edge_detected = scene["edge_detected"]
    visibility = scene["visibility"]
    guardrail_present = scene["guardrail_present"]
    guardrail_continuous = scene["guardrail_continuous"]
    guardrail_protects_edge = scene["guardrail_protects_edge"]
    alternative_protection = scene["alternative_protection"]
    violation_suspected = scene["violation_suspected"]

    if edge_detected is None:
        return {
            "rule_id": None,
            "hazard": "안전난간",
            "violation_status": "미확인",
            "reason": "추락 위험 단부 확인 불가"
        }

    if visibility != "clear":
        if violation_suspected:
            return {
                "rule_id": "F-02",
                "hazard": "안전난간",
                "violation_status": "위반 의심",
                "reason": "안전난간 단절 의심 정황이 있으나 확인 불충분"
            }

        return {
            "rule_id": None,
            "hazard": "안전난간",
            "violation_status": "미확인",
            "reason": "영상의 가림 또는 관측 범위 부족"
        }

    if edge_detected is False:
        return {
            "rule_id": None,
            "hazard": "안전난간",
            "violation_status": "관측상 정상",
            "reason": "평가 대상 추락 위험 단부가 확인되지 않음"
        }

    if guardrail_present is None or alternative_protection is None:
        return {
            "rule_id": None,
            "hazard": "안전난간",
            "violation_status": "미확인",
            "reason": "방호시설의 존재 여부 확인 불가"
        }

    if not guardrail_present and not alternative_protection:
        return {
            "rule_id": "F-01",
            "hazard": "안전난간",
            "violation_status": "위반",
            "reason": "추락 위험 단부의 방호조치 미실시"
        }

    if alternative_protection:
        return {
            "rule_id": None,
            "hazard": "안전난간",
            "violation_status": "관측상 정상",
            "reason": "안전난간 외 적절한 대체 방호조치가 확인됨"
        }

    if guardrail_continuous is None or guardrail_protects_edge is None:
        return {
            "rule_id": None,
            "hazard": "안전난간",
            "violation_status": "미확인",
            "reason": "난간의 연속성 또는 보호 위치 확인 불가"
        }

    if not guardrail_continuous:
        if violation_suspected:
            return {
                "rule_id": "F-02",
                "hazard": "안전난간",
                "violation_status": "위반 의심",
                "reason": "안전난간 단절 정황 확인"
            }

        return {
            "rule_id": "F-02",
            "hazard": "안전난간",
            "violation_status": "위반",
            "reason": "안전난간 보호구간의 연속성 부족"
        }

    if not guardrail_protects_edge:
        return {
            "rule_id": "F-03",
            "hazard": "안전난간",
            "violation_status": "위반",
            "reason": "안전난간이 추락 위험 단부를 보호하지 못함"
        }

    return {
        "rule_id": None,
        "hazard": "안전난간",
        "violation_status": "관측상 정상",
        "reason": "관측 구간에서 안전난간의 보호 상태 확인"
    }

def evaluate_opening(scene):
    """
    개구부 위험 판정 함수

    판정 결과:
    - 관측상 정상
    - 위반
    - 위반 의심
    - 미확인
    """

    opening_detected = scene["opening_detected"]
    visibility = scene["visibility"]
    cover_present = scene["cover_present"]
    guardrail_present = scene["guardrail_present"]
    protection_complete = scene["protection_complete"]
    violation_suspected = scene["violation_suspected"]

    if visibility != "clear":
        return {
            "rule_id": "F-06",
            "hazard": "개구부",
            "violation_status": "미확인",
            "reason": "개구부 또는 방호시설이 가려져 확인 불가"
        }

    if opening_detected is None:
        return {
            "rule_id": "F-06",
            "hazard": "개구부",
            "violation_status": "미확인",
            "reason": "개구부 존재 여부 확인 불가"
        }

    if opening_detected is False:
        return {
            "rule_id": None,
            "hazard": "개구부",
            "violation_status": "관측상 정상",
            "reason": "평가 대상 구간에서 개구부가 확인되지 않음"
        }

    if cover_present is None or guardrail_present is None:
        return {
            "rule_id": "F-06",
            "hazard": "개구부",
            "violation_status": "미확인",
            "reason": "개구부 방호시설의 설치 여부 확인 불가"
        }

    if not cover_present and not guardrail_present:
        return {
            "rule_id": "F-04",
            "hazard": "개구부",
            "violation_status": "위반",
            "reason": "개구부에 덮개 또는 안전난간 등 방호조치가 확인되지 않음"
        }

    if protection_complete is None:
        return {
            "rule_id": "F-06",
            "hazard": "개구부",
            "violation_status": "미확인",
            "reason": "방호시설의 보호 범위 확인 불가"
        }

    if not protection_complete:
        if violation_suspected:
            return {
                "rule_id": "F-05",
                "hazard": "개구부",
                "violation_status": "위반 의심",
                "reason": "개구부 방호 범위가 불완전한 것으로 의심됨"
            }

        return {
            "rule_id": "F-05",
            "hazard": "개구부",
            "violation_status": "위반",
            "reason": "개구부 방호시설의 보호 범위 불완전"
        }

    return {
        "rule_id": None,
        "hazard": "개구부",
        "violation_status": "관측상 정상",
        "reason": "개구부 방호시설의 보호 범위가 확보된 것으로 관측됨"
    }

def evaluate_work_platform(scene):
    """
    작업발판 위험 판정 함수

    판정 결과:
    - 관측상 정상
    - 위반
    - 위반 의심
    - 미확인
    """

    visibility = scene["visibility"]
    platform_present = scene["platform_present"]
    gap_detected = scene["gap_detected"]
    fastening_confirmed = scene["fastening_confirmed"]
    violation_suspected = scene["violation_suspected"]

    if visibility != "clear":
        return {
            "rule_id": "F-09",
            "hazard": "작업발판",
            "violation_status": "미확인",
            "reason": "영상의 가림 또는 관측 범위 부족"
        }

    if platform_present is None:
        return {
            "rule_id": "F-09",
            "hazard": "작업발판",
            "violation_status": "미확인",
            "reason": "작업발판 설치 여부 확인 불가"
        }

    if platform_present is False:
        return {
            "rule_id": "F-07",
            "hazard": "작업발판",
            "violation_status": "위반",
            "reason": "작업발판 미설치"
        }

    if gap_detected is None:
        return {
            "rule_id": "F-09",
            "hazard": "작업발판",
            "violation_status": "미확인",
            "reason": "작업발판의 틈새 또는 누락 여부 확인 불가"
        }

    if gap_detected:
        if violation_suspected:
            return {
                "rule_id": "F-08",
                "hazard": "작업발판",
                "violation_status": "위반 의심",
                "reason": "작업발판의 틈새 또는 누락이 의심됨"
            }

        return {
            "rule_id": "F-08",
            "hazard": "작업발판",
            "violation_status": "위반",
            "reason": "작업발판의 틈새 또는 누락 확인"
        }

    if fastening_confirmed is None:
        return {
            "rule_id": "F-09",
            "hazard": "작업발판",
            "violation_status": "미확인",
            "reason": "작업발판의 고정 상태 확인 불가"
        }

    if fastening_confirmed is False:
        return {
            "rule_id": "F-09",
            "hazard": "작업발판",
            "violation_status": "위반",
            "reason": "작업발판의 고정 불량 확인"
        }

    return {
        "rule_id": "F-10",
        "hazard": "작업발판",
        "violation_status": "관측상 정상",
        "reason": "작업발판의 설치 상태와 고정 상태가 확인됨"
    }

def evaluate_scene(hazard_type, scene):
    """
    위험요소 종류에 따라 적절한 판정 함수를 실행하는 통합 함수

    hazard_type:
        - guardrail: 안전난간
        - opening: 개구부
        - work_platform: 작업발판
    """

    evaluators = {
        "guardrail": evaluate_guardrail,
        "opening": evaluate_opening,
        "work_platform": evaluate_work_platform
    }

    if hazard_type not in evaluators:
        return {
            "rule_id": None,
            "hazard": hazard_type,
            "violation_status": "미확인",
            "reason": f"지원하지 않는 위험요소 유형: {hazard_type}"
        }

    return evaluators[hazard_type](scene)