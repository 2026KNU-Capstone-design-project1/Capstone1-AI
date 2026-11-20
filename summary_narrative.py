def build_report_narrative_context(full_report: dict) -> dict:
    """종합 보고서의 AI 설명문 생성에 필요한 근거 데이터 구성."""

    page1 = full_report["page1_overview"]
    page2 = full_report["page2_analysis"]

    return {
        "report_id": full_report["report_id"],
        "report_period": full_report.get("report_period"),
        "site_info": full_report.get("site_info"),

        "overall_statistics": page1["statistics"],
        "key_insights": page1["key_insights"],
        "priority_risks": page1["priority_risks"][:5],

        "area_distribution": page2["area_distribution"],
        "type_distribution": page2["type_distribution"],

        "management_recommendations": page1[
            "management_recommendations"
        ],

        "generation_rules": [
            "제공된 통계와 평가 결과만 사용한다.",
            "평가 건수와 위험도 등급을 임의로 변경하지 않는다.",
            "동일 건수의 위험유형은 공동 최다로 표현한다.",
            "미확정 위험요소를 확정 위반으로 단정하지 않는다.",
            "입력에 없는 작업 구역이나 현장 상황을 생성하지 않는다.",
            "관리 권고는 확인된 위험과 추가 확인 사항을 구분한다.",
        ],
    }

import json
from ollama import chat


SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string"},
        "key_findings": {
            "type": "array",
            "items": {"type": "string"},
        },
        "area_type_analysis": {"type": "string"},
        "management_opinion": {"type": "string"},
    },
    "required": [
        "executive_summary",
        "key_findings",
        "area_type_analysis",
        "management_opinion",
    ],
    "additionalProperties": False,
}


def generate_summary_narrative(
    full_report: dict,
    model: str = "qwen3:4b",
) -> dict:
    """
    종합 위험성평가 보고서의 자연어 설명문 생성.
    위험 건수 및 위험등급은 변경하지 않는다.
    """

    context = build_report_narrative_context(full_report)

    context.pop("key_insights", None)

    system_prompt = """
        당신은 건설현장 산업안전 위험성평가 보고서 작성 보조 AI입니다.

        제공된 JSON은 Python 집계 모듈이 계산한 결과입니다.
        다음 규칙을 반드시 준수하세요.

        1. 입력 JSON에 존재하는 사실만 사용합니다.
        2. 위험 건수, 위험등급, 위치, 조치 상태를 임의로 변경하지 않습니다.
        3. 동일한 최다 발생 건수를 가진 구역이나 위험유형은
        공동 최다라고 설명합니다.
        4. 미확정 항목은 위반으로 단정하지 않습니다.
        5. 입력에 없는 작업 환경, 사고, 날짜, 법령을 생성하지 않습니다.
        6. 평가 항목 수를 실제 시설물 수나 사고 발생 건수로
        표현하지 않습니다.
        7. 개선대책은 입력된 조치 내용을 우선 사용합니다.
        8. 확인되지 않은 사항은 추가 확인이 필요하다고 작성합니다.
        9. 한국어로 간결하고 전문적인 보고서 문체를 사용합니다.
        10. 구역별·유형별 percentage는 전체 평가 항목 중
            해당 항목의 비중을 의미합니다.
            이를 위험 발생률, 사고 발생률 또는 발생 확률로 표현하지 않습니다.

        11. 위험도 '미확정'은 고위험·중위험·저위험과
            구별되는 별도의 평가 상태입니다.
            위험도 등급을 합쳐서 공동 최다라고 표현하지 않습니다.

        12. 동일 건수의 구역이나 위험유형이 있을 경우에만
            공동 최다라고 표현합니다.
            입력된 평가 항목이 2건뿐이라면
            특정 구역이나 유형에 위험이 집중되었다고 단정하지 않습니다.

        13. '위험이 발생했다'보다는
            '위험요소가 관측되었다' 또는
            '평가 항목으로 집계되었다'라고 표현합니다.

        14. 입력에 없는 위험 증감률, 사고 이력,
            전월 대비 변화, 작업자 피해를 만들어내지 않습니다.

        반드시 다음 항목을 JSON으로 반환하세요.

        executive_summary:
        전체 위험 현황을 설명하는 2~3문장.

        key_findings:
        위험 현황에서 확인할 수 있는 핵심 발견 2~4개.
        문자열 배열로 작성합니다.

        area_type_analysis:
        구역별·위험유형별 발생 분포를 설명하는 2~4문장.

        management_opinion:
        확정 고위험 항목의 우선 개선과 미확정 항목의
        추가 확인을 구분해 작성하는 2~3문장.
    """

    response = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": json.dumps(
                    context,
                    ensure_ascii=False,
                ),
            },
        ],
        format=SUMMARY_SCHEMA,
        think=False,
        stream=False,
        options={
            "temperature": 0,
            "num_predict": 650,
        },
    )

    result = json.loads(response.message.content)

    required_strings = [
        "executive_summary",
        "area_type_analysis",
        "management_opinion",
    ]

    for field in required_strings:
        if not isinstance(result.get(field), str):
            raise ValueError(
                f"종합 보고서 생성 결과 오류: {field}"
            )

    if (
        not isinstance(result.get("key_findings"), list)
        or not all(
            isinstance(item, str)
            for item in result["key_findings"]
        )
    ):
        raise ValueError(
            "종합 보고서 생성 결과 오류: key_findings"
        )

    return result

def build_verified_insights(full_report: dict) -> list[str]:
    """실제 평가 결과에 근거한 핵심 인사이트를 생성한다."""

    page1 = full_report["page1_overview"]
    page2 = full_report["page2_analysis"]
    details = full_report["page3_details"]["items"]

    stats = page1["statistics"]
    insights = []

    insights.append(
        f"전체 평가 항목 {stats['total_assessments']}건 중 "
        f"고위험 {stats['high_risk_count']}건, "
        f"위험도 미확정 {stats['risk_distribution']['미확정']}건이 "
        "집계되었습니다."
    )

    for item in details:
        if item["risk_level"] == "상":
            insights.append(
                f"{item['location']}의 '{item['hazard']}' 항목이 "
                "위험도 '상'으로 평가되었습니다."
            )

    for item in details:
        if item["risk_level"] == "미확정":
            insights.append(
                f"{item['location']}의 '{item['hazard']}' 항목은 "
                "위험도가 미확정으로, 추가 현장 확인이 필요합니다."
            )

    areas = page2["area_distribution"]

    if areas:
        max_count = max(area["count"] for area in areas)
        top_areas = [
            area["area"]
            for area in areas
            if area["count"] == max_count
        ]

        if len(top_areas) > 1:
            insights.append(
                f"{', '.join(top_areas)}은(는) 각각 "
                f"{max_count}건으로 공동 최다 평가 구역입니다."
            )
        else:
            insights.append(
                f"{top_areas[0]}의 평가 항목이 "
                f"{max_count}건으로 가장 많았습니다."
            )

    return insights


def build_verified_narrative(full_report: dict) -> dict:
    """집계 결과에 근거해 종합 보고서 설명문을 생성한다."""

    page1 = full_report["page1_overview"]
    page2 = full_report["page2_analysis"]

    stats = page1["statistics"]
    risk_dist = stats["risk_distribution"]

    total = stats["total_assessments"]
    high = risk_dist["상"]
    medium = risk_dist["중"]
    low = risk_dist["하"]
    unknown = risk_dist["미확정"]

    executive_summary = (
        f"전체 평가 항목 {total}건 중 "
        f"위험도 상 {high}건, 중 {medium}건, "
        f"하 {low}건, 미확정 {unknown}건이 집계되었습니다."
    )

    key_findings = build_verified_insights(full_report)

    # 구역별 평가 항목 분포
    area_parts = [
        f"{item['area']} {item['count']}건 "
        f"({item['percentage']}%)"
        for item in page2["area_distribution"]
    ]

    # 위험유형별 평가 항목 분포
    type_parts = [
        f"{item['hazard_type']} {item['count']}건 "
        f"({item['percentage']}%)"
        for item in page2["type_distribution"]
    ]

    area_type_analysis = (
        "구역별 평가 항목 분포는 "
        + (", ".join(area_parts) if area_parts else "자료 없음")
        + "입니다. 위험유형별 평가 항목 분포는 "
        + (", ".join(type_parts) if type_parts else "자료 없음")
        + "입니다."
    )

    management_opinion = " ".join(
        page1["management_recommendations"]
    )

    return {
        "executive_summary": executive_summary,
        "key_findings": key_findings,
        "area_type_analysis": area_type_analysis,
        "management_opinion": management_opinion,
    }