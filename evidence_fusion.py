def fuse_evidence(frame: dict) -> list[dict]:
    """YOLO와 공간관계 규칙의 위험 후보 및 판단 원인을 결합한다.

    공간관계 finding은 위험 후보이지 위반 확정이 아니다.
    """

    detections = frame.get("detections") or []
    findings = frame.get("spatial_findings") or []
    checks = frame.get("rule_checks") or []

    results = []
    matched_findings = set()
    matched_checks = set()

    def is_undetermined(status):
        return status in (
            "판단불가",
            "판단 불가",
            "not_evaluable",
        )

    def build_item(
        hazard,
        detection,
        related_findings,
        related_checks,
    ):
        risk_findings = [
            f for f in related_findings
            if f.get("status") == "위험"
        ]

        agreement_values = [
            c.get("agrees_with_model")
            for c in related_checks
        ]

        has_disagreement = False in agreement_values
        has_undetermined = any(
            is_undetermined(c.get("status"))
            for c in related_checks
        ) or any(
            is_undetermined(f.get("status"))
            for f in related_findings
        )

        if detection is not None and risk_findings:
            if has_disagreement:
                evidence_status = "disagreement"
            else:
                evidence_status = "matched"

        elif detection is not None:
            if has_undetermined:
                evidence_status = "undetermined"
            else:
                evidence_status = "detection_only"

        elif related_findings:
            if risk_findings:
                evidence_status = "spatial_only"
            else:
                evidence_status = "undetermined"

        else:
            evidence_status = "undetermined"

        primary_check = (
            related_checks[0] if related_checks else {}
        )

        return {
            "hazard": hazard,
            "evidence_status": evidence_status,
            "detection": detection,
            "spatial_findings": related_findings,
            "rule_checks": related_checks,
            "evaluation_status": (
                primary_check.get("status")
                or frame.get("evaluation_status")
            ),
            "reason_code": (
                primary_check.get("reason_code")
                or frame.get("reason_code")
            ),
            "agrees_with_model": (
                primary_check.get("agrees_with_model")
                if related_checks else None
            ),
            "violation_status": "미확인",
            "risk_level": "미확정",
        }

    for detection in detections:
        hazard = detection["hazard"]

        related_findings = []
        related_checks = []

        for i, finding in enumerate(findings):
            if finding.get("hazard") == hazard:
                related_findings.append(finding)
                matched_findings.add(i)

        for i, check in enumerate(checks):
            if check.get("hazard") == hazard:
                related_checks.append(check)
                matched_checks.add(i)

        results.append(
            build_item(
                hazard,
                detection,
                related_findings,
                related_checks,
            )
        )

    for i, finding in enumerate(findings):
        if i in matched_findings:
            continue

        hazard = finding.get("hazard", "위험요소 미확인")

        related_checks = []
        for j, check in enumerate(checks):
            if check.get("hazard") == hazard:
                related_checks.append(check)
                matched_checks.add(j)

        results.append(
            build_item(
                hazard,
                None,
                [finding],
                related_checks,
            )
        )

    for i, check in enumerate(checks):
        if i in matched_checks:
            continue

        results.append(
            build_item(
                check.get("hazard", "위험요소 미확인"),
                None,
                [],
                [check],
            )
        )

    return results