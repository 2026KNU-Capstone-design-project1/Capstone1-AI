from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_test import search_criteria, generate_assessment
from risk_pipeline import evaluate_hazard_risk

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
            assessments.append({
                "hazard": detection.hazard,
                "violation_status": "미확인",
                "risk_level": "미확정",
                "report": None,
                "message": "관련 안전 기준 검색 결과 없음"
            })
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

        
        # 규칙 엔진의 위험도를 최종 보고서에 우선 반영
        if risk_result is not None:
            risk_level = risk_result.get("risk_level")

            if risk_level in ("상", "중", "하"):
                report["위험도"] = risk_level
            else:
                report["위험도"] = "추가 확인 필요"

        # 위반 여부가 확정되지 않은 경우 단정적인 보고서 방지
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

        
        assessments.append({
            "hazard": detection.hazard,
            "violation_status": (
                rule_result["violation_status"]
                if rule_result is not None
                else "미확인"
            ),
            "risk_level": (
                risk_result["risk_level"]
                if risk_result is not None
                else "미확정"
            ),
            "rule_result": rule_result,
            "risk_assessment": risk_result,
            "retrieved_criteria_count": len(search_results),
            "report": report,
        })

    return {
        "assessment_id": payload.assessment_id,
        "source": payload.source.model_dump(),
        "assessments": assessments
    }
