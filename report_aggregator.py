from collections import Counter


RISK_LEVELS = ("상", "중", "하", "미확정")


def aggregate_assessments(assessments: list[dict]) -> dict:
    """
    개별 위험성평가 결과를 집계한다.

    assessments:
        /assess API의 assessments 배열
    """

    if not isinstance(assessments, list):
        raise TypeError("assessments는 리스트여야 합니다.")

    risk_distribution = Counter()
    violation_distribution = Counter()
    hazard_distribution = Counter()

    for assessment in assessments:
        if not isinstance(assessment, dict):
            raise TypeError("평가 결과는 딕셔너리여야 합니다.")

        violation_status = assessment.get(
            "violation_status", "미확인"
        )

        if violation_status not in (
            "위반",
            "위반 의심",
            "관측상 정상",
            "미확인",
        ):
            violation_status = "미확인"

        violation_distribution[violation_status] += 1

        risk_level = assessment.get(
            "risk_level", "미확정"
        )

        if (
            violation_status in ("미확인", "위반 의심")
            or risk_level not in RISK_LEVELS
        ):
            risk_level = "미확정"

        risk_distribution[risk_level] += 1

        rule_result = assessment.get("rule_result")

        hazard = None
        if isinstance(rule_result, dict):
            hazard = rule_result.get("hazard")

        if not hazard:
            hazard = assessment.get("hazard") or "분류 미확인"

        hazard_distribution[str(hazard)] += 1

    total = len(assessments)

    summary = {
        "total_assessments": total,
        "confirmed_violations": violation_distribution["위반"],
        "suspected_violations": violation_distribution["위반 의심"],
        "normal": violation_distribution["관측상 정상"],
        "unconfirmed": violation_distribution["미확인"],
        "high_risk": risk_distribution["상"],
        "medium_risk": risk_distribution["중"],
        "low_risk": risk_distribution["하"],
        "undetermined_risk": risk_distribution["미확정"],
    }

    top_hazards = [
        {"hazard": hazard, "count": count}
        for hazard, count in sorted(
            hazard_distribution.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    risk_priority = {
        "상": 0,
        "중": 1,
        "하": 2,
        "미확정": 3,
    }

    priority_actions = []

    for assessment in assessments:
        hazard = assessment.get("hazard") or "분류 미확인"
        violation_status = assessment.get(
            "violation_status", "미확인"
        )
        risk_level = assessment.get(
            "risk_level", "미확정"
        )

        if (
            violation_status in ("미확인", "위반 의심")
            or risk_level not in RISK_LEVELS
        ):
            risk_level = "미확정"

        report = assessment.get("report") or {}

        priority_actions.append({
            "hazard": hazard,
            "risk_level": risk_level,
            "violation_status": violation_status,
            "action": (
                report.get("개선대책")
                or "현장 확인 후 필요한 안전조치 실시"
            ),
        })

    priority_actions.sort(
        key=lambda item: risk_priority[item["risk_level"]]
    )

    return {
        "summary": summary,
        "risk_distribution": {
            level: risk_distribution[level]
            for level in RISK_LEVELS
        },
        "violation_distribution": {
            status: violation_distribution[status]
            for status in (
                "위반",
                "위반 의심",
                "관측상 정상",
                "미확인",
            )
        },
        "top_hazards": top_hazards,
        "priority_actions": priority_actions,
    }