from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RecommendationContextSchema(BaseModel):
    device: str = "UNKNOWN"


class RecommendationRequestSchema(BaseModel):
    user_id: str
    session_id: str
    carousel_type: str = "FOR_YOU"
    limit: int = Field(default=20, ge=1, le=100)
    non_personalized_work_ids: list[str] = Field(default_factory=list)
    context: RecommendationContextSchema = Field(default_factory=RecommendationContextSchema)


class RecommendationItemSchema(BaseModel):
    work_id: str
    format_id: str
    title: str
    position: int
    reason_code: str
    is_exploration: bool
    propensity: float | None = None


class RecommendationMetadataSchema(BaseModel):
    model_version: str
    feature_version: str
    policy_version: str
    experiment_id: str
    arm: str


class RecommendationResponseSchema(BaseModel):
    request_id: str
    carousel_id: str
    items: list[RecommendationItemSchema]
    metadata: RecommendationMetadataSchema
    diagnostics: dict[str, Any]
