from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_test import (
    search_criteria,
    search_hazard_knowledge,
    generate_assessment,
)
from risk_pipeline import evaluate_hazard_risk
from report_aggregator import aggregate_assessments
from full_report_builder import build_full_report
from summary_narrative import build_verified_narrative
from schemas import RiskAssessmentItem
from evidence_context import build_observation_summary

app = FastAPI(
    title="Construction Risk Assessment AI",
    description="건설현장 위험성평가 AI API",
    version="1.0.0",
)
class Detection(BaseModel):
    hazard: str
    location: str | None = None
    evidence: str | None = None
    bbox: dict[str, float] | None = None
    confidence: float = Field(ge=0, le=1)

    hazard_type: Literal[
        "guardrail",
        "opening",
        "work_platform"
    ] | None = None

    rule_features: dict | None = None
    risk_evidence: dict | None = None
    evidence_context: dict | None = None

class SourceInfo(BaseModel):
    source_type: Literal["photo", "video"]
    photo_id: str | int | None = None
    video_id: str | int | None = None
    frame_index: int | None = None
    frame_time_ms: int | None = None
    image_url: str | None = None

class AssessmentRequest(BaseModel):
    assessment_id: str
    site_info: dict | None = None
    work_info: dict | None = None
    source: SourceInfo
    detections: list[Detection]

class SummaryReportRequest(BaseModel):
    report_id: str
    site_info: dict | None = None
    assessments: list[RiskAssessmentItem]


class FullReportRequest(BaseModel):
    report_id: str
    site_info: dict | None = None
    report_period: str | None = None
    assessments: list[RiskAssessmentItem]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "AI 서버 정상 실행 중",
    }

@app.post("/assess")
def assess(payload: AssessmentRequest):
    assessments = []

    for detection in payload.detections:
        
        rule_result = None
        risk_result = None

        if (
            detection.hazard_type is not None
            and detection.rule_features is not None
            and detection.risk_evidence is not None
        ):
            required_fields = {
                "guardrail": {
                    "edge_detected",
                    "visibility",
                    "guardrail_present",
                    "guardrail_continuous",
                    "guardrail_protects_edge",
                    "alternative_protection",
                    "violation_suspected",
                },
                "opening": {
                    "opening_detected",
                    "visibility",
                    "cover_present",
                    "guardrail_present",
                    "protection_complete",
                    "violation_suspected",
                },
                "work_platform": {
                    "visibility",
                    "platform_present",
                    "gap_detected",
                    "fastening_confirmed",
                    "violation_suspected",
                },
            }

            fields = required_fields.get(detection.hazard_type)

            if (
                fields is not None
                and fields.issubset(detection.rule_features)
                and isinstance(
                    detection.risk_evidence.get("evidence_sufficient"),
                    bool,
                )
            ):
                pipeline_result = evaluate_hazard_risk(
                    hazard_type=detection.hazard_type,
                    scene=detection.rule_features,
                    risk_evidence=detection.risk_evidence,
                )

                rule_result = pipeline_result["rule_result"]
                risk_result = pipeline_result["risk_assessment"]


        query = " ".join([
            detection.hazard,
            detection.location or "",
            detection.evidence or "",
        ])

        context = detection.evidence_context or {}
        checks = context.get("rule_checks") or []

        hazard_codes = list(dict.fromkeys(
            check["code"]
            for check in checks
            if isinstance(check, dict)
            and isinstance(check.get("code"), str)
            and check["code"].strip()
            and check.get("hazard") == detection.hazard
        ))

        if not hazard_codes and not checks:
            findings = context.get("spatial_findings") or []
            hazard_codes = list(dict.fromkeys(
                finding["code"]
                for finding in findings
                if isinstance(finding, dict)
                and isinstance(finding.get("code"), str)
                and finding["code"].strip()
                and finding.get("hazard") == detection.hazard
            ))

        search_results = []

        for code in hazard_codes:
            search_results.extend(
                search_hazard_knowledge(code)
            )

        if not search_results:
            search_results = search_criteria(
                query,
                top_k=5,
                method="char"
            )

        if not search_results:
            item = RiskAssessmentItem(
                risk_id=None,
                hazard=detection.hazard,
                hazard_type=detection.hazard_type,
                location=detection.location,
                area=(
                    payload.work_info.get("location")
                    if payload.work_info
                    else None
                ),
                work_name=(
                    payload.work_info.get("work_name")
                    if payload.work_info
                    else None
                ),
                image_url=payload.source.image_url,
                bbox=detection.bbox,
                confidence=detection.confidence,
                violation_status=(
                    rule_result.get("violation_status")
                    if rule_result is not None
                    else "미확인"
                ),
                risk_level=(
                    risk_result.get("risk_level", "미확정")
                    if risk_result is not None
                    else "미확정"
                ),
                rule_result=rule_result,
                risk_assessment=risk_result,
                report=None,
                action_status=None,
                first_detected_at=None,
            )

            assessments.append(item.model_dump())
            continue


        # Unverified law excerpts support retrieval/audit only, never legal conclusions.
        from law_rag import search_law_articles
        law_candidates = [
            {
                "id": law.get("id"),
                "law_name": law.get("law_name"),
                "article_number": law.get("article_number"),
                "article_title": law.get("article_title"),
                "article_excerpt": (law.get("article_text") or "")[:600],
                "source_pages": law.get("source_pages", []),
                "score": law.get("score"),
                "verification_status": law.get("verification_status", "unverified"),
                "official_source_url": law.get("official_source_url"),
                "effective_date": law.get("effective_date"),
            }
            for law in search_law_articles(query, top_k=3, min_score=0.08)
        ]

        scene = {
            "site_info": payload.site_info,
            "work_info": payload.work_info,
            "source": payload.source.model_dump(),
            "detection": detection.model_dump(),
            "evidence_context": detection.evidence_context,
            "rule_result": rule_result,
            "risk_assessment": risk_result,
            "retrieved_criteria": search_results,
            "retrieved_law_candidates": law_candidates,
        }

        report = generate_assessment(
            scene=scene,
            results=(
                risk_result
                if risk_result is not None
                else search_results
            ),
            return_json=True
        )

        context = detection.evidence_context

        if context:
            observation_summary = build_observation_summary(context)

            if observation_summary:
                report["위험내용"] = observation_summary

        unverified_knowledge = any(
            isinstance(result, dict)
            and result.get("verification_status") == "unverified"
            for result in search_results
        )

        # Deterministic legal provenance gate: LLM output is never enough.
        # Verified law must have an official reference and applicability date.
        verified_laws = [
            law for law in law_candidates
            if law.get("verification_status") == "verified"
            and law.get("official_source_url")
            and law.get("effective_date")
        ]
        verified_code_basis = [
            r for r in search_results
            if isinstance(r, dict)
            and r.get("retrieval_method") == "hazard_code_exact"
            and r.get("verification_status") == "verified"
            and r.get("legal_basis")
            and r.get("official_source_url")
            and r.get("effective_date")
        ]
        if verified_laws:
            report["법적사항"] = "관련 법령(현장 적용·위반 여부 별도 확인): " + "; ".join(
                f'{law["law_name"]} {law["article_number"]} ({law["article_title"]})'
                for law in verified_laws
            )
        elif verified_code_basis:
            report["법적사항"] = "관련 법령(현장 적용·위반 여부 별도 확인): " + "; ".join(
                str(r["legal_basis"]) for r in verified_code_basis
            )
        else:
            report["법적사항"] = "관련 법령 추가 확인 필요"

        matched_knowledge = [
            result
            for result in search_results
            if isinstance(result, dict)
            and result.get("retrieval_method") == "hazard_code_exact"
        ]

        if matched_knowledge:
            knowledge = matched_knowledge[0]

            safety_criteria = knowledge.get("safety_criteria") or []
            improvements = knowledge.get("improvement") or []
            additional_checks = knowledge.get("additional_checks") or []

            if safety_criteria:
                report["관련 안전 기준"] = (
                    "[검토용 안전조치 초안] "
                    + " ".join(safety_criteria)
                )

            if improvements:
                report["개선대책"] = " ".join(improvements)

            if safety_criteria:
                report["안전대책"] = " ".join(safety_criteria)

            if additional_checks:
                report["위험도 평가 시 추가 확인사항"] = (
                    " / ".join(additional_checks)
                )

        if risk_result is not None:
            risk_level = risk_result.get("risk_level")

            if risk_level in ("상", "중", "하"):
                report["위험도"] = risk_level
            else:
                report["위험도"] = "추가 확인 필요"

        violation_status = (
            rule_result.get("violation_status")
            if rule_result is not None
            else "미확인"
        )

        if violation_status in ("미확인", "위반 의심"):
            report["위험도"] = "추가 확인 필요"

            if rule_result is not None:
                report["위험도 판단 근거"] = (
                    rule_result.get("reason")
                    or "규칙 판단 근거에 대한 추가 확인 필요"
                )
            else:
                context = detection.evidence_context or {}
                evidence_status = context.get("evidence_status")

                if evidence_status == "matched":
                    report["위험도 판단 근거"] = (
                        "YOLO 위험 탐지와 공간관계 규칙에서 동일한 "
                        "위험 후보가 확인되었으나, 실제 위반 여부 및 "
                        "위험도 산정에 필요한 근거는 충분하지 않음"
                    )
                elif evidence_status == "detection_only":
                    report["위험도 판단 근거"] = (
                        "YOLO에서 위험 후보가 탐지되었으나 "
                        "공간관계 규칙의 위험 판단 근거가 확인되지 않음. "
                        "규칙 미지원, 조건 불충족 또는 객체 미탐지 "
                        "가능성이 있어 추가 확인 필요"
                    )
                elif evidence_status == "spatial_only":
                    report["위험도 판단 근거"] = (
                        "공간관계 규칙에서 위험 후보가 발견되었으나 "
                        "YOLO 위험 탐지 결과와의 일치가 확인되지 않음"
                    )
                elif evidence_status == "undetermined":
                    report["위험도 판단 근거"] = (
                        "공간관계 규칙에서 판단불가 결과가 발생하여 "
                        "실제 위험 여부에 대한 추가 확인 필요"
                    )
                else:
                    report["위험도 판단 근거"] = (
                        "위험 판단에 필요한 근거가 충분하지 않음"
                    )

            if not matched_knowledge:
                report["안전대책"] = (
                    "현장 확인 전까지 위험요소에 대한 예방조치를 검토하고, "
                    "실제 위험 상태가 확인되면 필요한 안전조치를 실시"
                )

                report["위험도 평가 시 추가 확인사항"] = (
                    "영상 탐지 결과의 정확성, 실제 작업 상태, "
                    "위험요소별 안전조치 이행 여부"
                )

        item = RiskAssessmentItem(
            risk_id=None, 
            hazard=detection.hazard,
            hazard_type=detection.hazard_type,
            location=detection.location,
            area=(
                payload.work_info.get("location")
                if payload.work_info
                else None
            ),
            work_name=(
                payload.work_info.get("work_name")
                if payload.work_info
                else None
            ),
            image_url=payload.source.image_url,
            bbox=detection.bbox,
            confidence=detection.confidence,
            violation_status=(
                rule_result["violation_status"]
                if rule_result is not None
                else "미확인"
            ),
            risk_level=(
                risk_result["risk_level"]
                if risk_result is not None
                else "미확정"
            ),
            rule_result=rule_result,
            risk_assessment=risk_result,
            report=report,
            action_status=None,  # BE에서 관리
            first_detected_at=None,  # BE에서 관리
        )

        assessments.append(item.model_dump())

    return {
        "assessment_id": payload.assessment_id,
        "source": payload.source.model_dump(),
        "assessments": assessments
    }

@app.post("/reports/summary")
def create_summary_report(payload: SummaryReportRequest):
    aggregated = aggregate_assessments(
        [item.model_dump() for item in payload.assessments]
    )
    
    return {
        "report_id": payload.report_id,
        "site_info": payload.site_info,
        "summary": aggregated["summary"],
        "risk_distribution": aggregated["risk_distribution"],
        "violation_distribution": aggregated["violation_distribution"],
        "top_hazards": aggregated["top_hazards"],
        "priority_actions": aggregated["priority_actions"],
    }


@app.post("/reports/full")
def create_full_report(payload: FullReportRequest):
    full_report = build_full_report(
        report_id=payload.report_id,
        site_info=payload.site_info,
        assessments=[
            item.model_dump() for item in payload.assessments
        ],
        report_period=payload.report_period,
    )

    narrative = build_verified_narrative(full_report)

    full_report["narrative"] = narrative

    return full_report