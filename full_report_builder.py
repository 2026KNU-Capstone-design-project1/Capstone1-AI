
from collections import Counter
from report_aggregator import aggregate_assessments


RISK_ORDER = {
    "상": 0,
    "중": 1,
    "하": 2,
    "미확정": 3,
}

RISK_LABELS = {
    "상": "HIGH",
    "중": "MEDIUM",
    "하": "LOW",
    "미확정": "UNDETERMINED",
}


def get_hazard_name(item):
    rule = item.get("rule_result") or {}
    return rule.get("hazard") or item.get("hazard") or "분류 미확인"


def get_risk_level(item):
    status = item.get("violation_status", "미확인")
    level = item.get("risk_level", "미확정")

    if status in ("미확인", "위반 의심"):
        return "미확정"

    if level not in RISK_ORDER:
        return "미확정"

    return level


def get_location(item):
    return (
        item.get("area")
        or item.get("location")
        or (item.get("detection") or {}).get("location")
        or "위치 미확인"
    )


def build_full_report(
    report_id: str,
    site_info: dict | None,
    assessments: list[dict],
    report_period: str | None = None,
) -> dict:
    """
    종합 위험성평가 보고서의 페이지별 데이터를 생성한다.

    assessments:
      개별 /assess 결과를 BE에서 모아 전달한 목록.
      선택적으로 risk_id, area, location, work_name,
      action_status, first_detected_at, source 등의
      부가 필드를 포함할 수 있다.
    """

    aggregated = aggregate_assessments(assessments)
    summary = aggregated["summary"]

    # -------------------------------
    # 1. 전체 위험 현황
    # -------------------------------
    total = summary["total_assessments"]
    high = summary["high_risk"]

    unresolved_statuses = {"미처리", "처리중"}
    known_statuses = {"미처리", "처리중", "처리완료"}

    unresolved_count = 0
    unknown_action_count = 0

    area_counter = Counter()
    type_counter = Counter()

    for item in assessments:
        action_status = item.get("action_status")

        if action_status in unresolved_statuses:
            unresolved_count += 1

        if action_status not in known_statuses:
            unknown_action_count += 1

        area_counter[get_location(item)] += 1
        type_counter[get_hazard_name(item)] += 1

    hazard_counts = aggregated["top_hazards"]

    if hazard_counts:
        max_count = max(item["count"] for item in hazard_counts)

        major_hazards = [
            item["hazard"]
            for item in hazard_counts
            if item["count"] == max_count
        ]
    else:
        major_hazards = []

    major_area = (
        area_counter.most_common(1)[0][0]
        if area_counter
        else None
    )

    # -------------------------------
    # 2. 우선 개선 위험 목록
    # -------------------------------
    sorted_items = sorted(
        assessments,
        key=lambda item: RISK_ORDER[get_risk_level(item)],
    )

    priority_risks = []

    for item in sorted_items:
        level = get_risk_level(item)
        report = item.get("report") or {}

        priority_risks.append({
            "risk_id": item.get("risk_id"),
            "hazard": item.get("hazard") or "분류 미확인",
            "location": get_location(item),
            "risk_level": level,
            "risk_label": RISK_LABELS[level],
            "violation_status": item.get(
                "violation_status", "미확인"
            ),
            "action_status": item.get("action_status"),
            "action": report.get("개선대책"),
        })

    # -------------------------------
    # 3. 종합 관리 권고
    # -------------------------------
    recommendations = []

    high_items = [
        item for item in priority_risks
        if item["risk_level"] == "상"
    ]

    if high_items:
        recommendations.append(
            f"확정된 고위험 평가 항목 {len(high_items)}건에 대해 "
            "방호조치와 현장 점검을 우선 실시해야 합니다."
        )

    if summary["undetermined_risk"] > 0:
        recommendations.append(
            f"위험도가 미확정인 평가 항목 "
            f"{summary['undetermined_risk']}건은 "
            "추가 관측 또는 현장 확인이 필요합니다."
        )

    if unresolved_count > 0:
        recommendations.append(
            f"미처리 또는 처리중으로 기록된 "
            f"{unresolved_count}건의 조치 상태를 "
            "추적 관리해야 합니다."
        )

    if unknown_action_count > 0:
        recommendations.append(
            f"조치 상태가 확인되지 않은 "
            f"{unknown_action_count}건은 "
            "처리 여부를 확인해야 합니다."
        )

    if not recommendations:
        recommendations.append(
            "현재 제공된 평가 기록을 지속적으로 점검하고 "
            "위험 및 조치 상태 변화를 추적 관리해야 합니다."
        )

    # -------------------------------
    # 4. 구역별 위험 분석
    # -------------------------------
    area_distribution = [
        {
            "area": area,
            "count": count,
            "percentage": round(count / total * 100, 1)
            if total else 0,
        }
        for area, count in sorted(
            area_counter.items(),
            key=lambda x: (-x[1], x[0]),
        )
    ]

    # -------------------------------
    # 5. 위험 유형별 분석
    # -------------------------------
    type_distribution = [
        {
            "hazard_type": hazard,
            "count": count,
            "percentage": round(count / total * 100, 1)
            if total else 0,
        }
        for hazard, count in sorted(
            type_counter.items(),
            key=lambda x: (-x[1], x[0]),
        )
    ]

    # -------------------------------
    # 6. 개별 위험 상세 요약
    # -------------------------------
    details = []

    for item in sorted_items:
        report = item.get("report") or {}
        source = item.get("source") or {}
        risk_result = item.get("risk_assessment") or {}

        level = get_risk_level(item)

        details.append({
            "risk_id": item.get("risk_id"),
            "hazard": item.get("hazard") or "분류 미확인",
            "hazard_type": get_hazard_name(item),
            "risk_level": level,
            "risk_label": RISK_LABELS[level],
            "violation_status": item.get(
                "violation_status", "미확인"
            ),
            "action_status": item.get("action_status"),
            "work_name": item.get("work_name"),
            "location": get_location(item),
            "first_detected_at": item.get("first_detected_at"),
            "image_url": (
                item.get("image_url")
                or source.get("image_url")
            ),
            "bbox": item.get("bbox"),
            "accident_type": report.get("예상 재해유형"),
            "risk_description": report.get("위험내용"),
            "risk_reason": report.get("위험도 판단 근거"),
            "likelihood": risk_result.get("likelihood"),
            "severity": risk_result.get("severity"),
            "risk_score": risk_result.get("risk_score"),
            "improvement": report.get("개선대책"),
            "safety_measures": report.get("안전대책"),
            "legal_basis": report.get("법적사항"),
            "additional_checks": report.get(
                "위험도 평가 시 추가 확인사항"
            ),
        })

    # -------------------------------
    # 7. 보고서 페이지별 구조
    # -------------------------------
    return {
        "report_id": report_id,
        "site_info": site_info,
        "report_period": report_period,

        "page1_overview": {
            "title": "종합 위험 현황 및 핵심 인사이트",
            "statistics": {
                "total_assessments": total,
                "high_risk_count": high,
                "unresolved_count": unresolved_count,
                "unknown_action_count": unknown_action_count,
                "major_hazard_type": major_hazards,
                "risk_distribution": aggregated[
                    "risk_distribution"
                ],
            },
            "key_insights": {
                "most_frequent_area": major_area,
                "most_frequent_hazard": major_hazards,
                "unconfirmed_count": summary["unconfirmed"],
                "high_risk_count": high,
            },
            "priority_risks": priority_risks,
            "management_recommendations": recommendations,
        },

        "page2_analysis": {
            "title": "구역별 및 유형별 위험 분석",
            "area_distribution": area_distribution,
            "type_distribution": type_distribution,
            "risk_distribution": aggregated[
                "risk_distribution"
            ],
            "management_recommendations": recommendations,
        },

        "page3_details": {
            "title": "개별 위험 상세 요약",
            "total_items": len(details),
            "items": details,
        },

        "data_quality": {
            "count_unit": "평가 항목",
            "unique_facility_count_available": False,
            "action_status_missing": unknown_action_count,
        },
    }
