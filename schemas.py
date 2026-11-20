from typing import Any, Literal

from pydantic import BaseModel, Field


class RiskAssessmentItem(BaseModel):
    """개별 위험성평가 결과 공통 스키마."""

    risk_id: str | None = None

    hazard: str
    hazard_type: str | None = None

    location: str | None = None
    area: str | None = None
    work_name: str | None = None

    image_url: str | None = None
    bbox: dict[str, float] | None = None
    confidence: float | None = Field(
        default=None, ge=0, le=1
    )

    violation_status: Literal[
        "위반",
        "위반 의심",
        "관측상 정상",
        "미확인",
    ] = "미확인"

    risk_level: Literal[
        "상",
        "중",
        "하",
        "미확정",
    ] = "미확정"

    rule_result: dict[str, Any] | None = None
    risk_assessment: dict[str, Any] | None = None
    report: dict[str, Any] | None = None

    action_status: Literal[
        "미처리",
        "처리중",
        "처리완료",
    ] | None = None

    first_detected_at: str | None = None