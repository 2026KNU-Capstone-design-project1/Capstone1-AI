from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_test import search_criteria, generate_assessment
from risk_pipeline import evaluate_hazard_risk
from report_aggregator import aggregate_assessments
from full_report_builder import build_full_report
from summary_narrative import build_verified_narrative
from schemas import RiskAssessmentItem

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


        scene = {
            "site_info": payload.site_info,
            "work_info": payload.work_info,
            "source": payload.source.model_dump(),
            "detection": detection.model_dump(),
            "rule_result": rule_result,
            "risk_assessment": risk_result,
            "retrieved_criteria": search_results,
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

            report["위험도 판단 근거"] = (
                rule_result.get("reason")
                if rule_result is not None
                else "위험 판단에 필요한 공간관계 및 방호시설 확인 정보가 부족함"
            )

            report["개선대책"] = (
                "현장 점검을 통해 방호시설 설치 및 정상 작동 여부를 "
                "우선 확인하고, 미설치 또는 불량이 확인되면 필요한 방호조치를 실시"
            )

            report["안전대책"] = (
                "확인 전까지 해당 위험구역에 대한 접근을 통제하고, "
                "방호시설 상태 확인 후 필요한 안전조치를 실시"
            )

            report["위험도 평가 시 추가 확인사항"] = (
                "방호시설 설치 여부, 관측 가림 여부, "
                "작업자 접근 상태 및 실제 작업 조건"
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