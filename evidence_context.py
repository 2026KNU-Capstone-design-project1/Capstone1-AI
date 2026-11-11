from typing import Any


def build_evidence_context(
    frame: dict[str, Any],
    fused_item: dict[str, Any],
) -> dict[str, Any]:
    """위험 후보에 연결된 관측 근거를 보고서 입력용으로 구성한다."""

    objects = frame.get("spatial_objects") or []
    relations = frame.get("spatial_relations") or []
    findings = fused_item.get("spatial_findings") or []

    related_ids = set()

    for finding in findings:
        related_ids.update(finding.get("objects") or [])

    related_objects = [
        obj for obj in objects
        if obj.get("id") in related_ids
    ]

    related_relations = [
        rel for rel in relations
        if rel.get("a") in related_ids
        and rel.get("b") in related_ids
    ]

    return {
        "source": frame.get("source"),
        "hazard": fused_item.get("hazard"),
        "evidence_status": fused_item.get("evidence_status"),

        # 새로 추가
        "evaluation_status": fused_item.get("evaluation_status"),
        "reason_code": fused_item.get("reason_code"),
        "rule_checks": fused_item.get("rule_checks") or [],
        "agrees_with_model": fused_item.get("agrees_with_model"),

        "violation_status": "미확인",
        "risk_level": "미확정",
        "yolo_detection": fused_item.get("detection"),
        "spatial_findings": findings,
        "related_objects": related_objects,
        "related_relations": related_relations,
        "scene_objects": objects,
        "scene_relations": relations,
        "interpretation_note": (
            "YOLO와 공간관계 규칙의 판단은 위험 후보에 대한 "
            "관측 근거이며 실제 안전규정 위반을 확정하지 않는다. "
            "findings가 없다는 사실만으로 정상이라고 판단하지 않는다. "
            "evaluation_status와 reason_code를 참고하여 "
            "판단이 불가능한 이유를 구분한다. "
            "rule_not_supported는 규칙 미지원으로 "
            "위험이 없다는 의미가 아니다. "
            "gap은 실제 미터 단위 거리가 아니라 화면 기준 정규화 거리다."
        ),
    }

def build_observation_summary(context: dict) -> str:
    """확인된 영상 분석 결과를 근거 중심의 문장으로 구성한다.

    영상 AI의 위험 판단은 후보이며 실제 위반 확정이 아니다.
    """

    hazard = context.get("hazard") or "미지정 위험요소"
    status = context.get("evidence_status")
    reason = context.get("reason_code")

    sentences = []

    detection = context.get("yolo_detection")

    if detection:
        sentences.append(
            f"YOLO에서 '{hazard}' 위험 후보가 탐지되었다."
        )

    objects = context.get("related_objects") or []

    if objects:
        names = [
            obj.get("name")
            for obj in objects
            if obj.get("name")
        ]

        if names:
            sentences.append(
                f"객체 탐지 결과 {', '.join(names)}이(가) 관측되었다."
            )

    object_names = {
        obj.get("id"): obj.get("name", "미확인 객체")
        for obj in context.get("scene_objects") or []
    }

    for relation in context.get("related_relations") or []:
        a_name = object_names.get(
            relation.get("a"), "미확인 객체"
        )
        b_name = object_names.get(
            relation.get("b"), "미확인 객체"
        )
        relation_type = relation.get("type", "미확인")

        sentences.append(
            f"{a_name}와(과) {b_name}의 바운딩박스에서 "
            f"'{relation_type}' 관계가 관측되었다."
        )

    codes = list(dict.fromkeys(
        finding.get("code")
        for finding in context.get("spatial_findings") or []
        if finding.get("code")
    ))

    if codes:
        sentences.append(
            f"공간관계 규칙 {', '.join(codes)}에서 "
            "위험 후보가 제시되었다."
        )

        # 기하학적 관계와 규칙 기반 위험 판단 근거를 구분한다.
        for finding in context.get("spatial_findings") or []:
            if not isinstance(finding, dict):
                continue

            code = finding.get("code")
            relation_type = finding.get("type")
            finding_status = finding.get("status")

            if (
                code
                and relation_type
                and finding_status == "위험"
            ):
                sentences.append(
                    f"위험 판단 규칙 {code}은(는) "
                    f"'{relation_type}' 조건에 따라 "
                    "위험 후보를 제시하였다."
                )

    if status == "matched":
        sentences.append(
            "YOLO와 공간관계 규칙에서 동일한 위험 후보가 제시되었다."
        )
    elif status == "detection_only":
        sentences.append(
            "YOLO 위험 후보는 존재하지만 공간관계 규칙의 "
            "위험 후보 결과는 확인되지 않았다."
        )
    elif status == "spatial_only":
        sentences.append(
            "공간관계 규칙에서 위험 후보가 제시되었으나 "
            "YOLO의 동일 위험 후보는 확인되지 않았다."
        )
    elif status == "undetermined":
        if reason == "rule_not_supported":
            sentences.append(
                "해당 위험에 대한 공간관계 판단 규칙이 "
                "지원되지 않아 규칙 기반 평가를 수행하지 못했다."
            )
        else:
            sentences.append(
                "공간관계 규칙의 판단이 완료되지 않아 "
                "추가 확인이 필요하다."
            )

    sentences.append(
        "영상 분석 결과만으로 실제 안전규정 위반 여부를 "
        "확정할 수 없으므로 현장 확인이 필요하다."
    )

    return " ".join(sentences)