from __future__ import annotations

import uuid

from fastapi import APIRouter, Header, HTTPException, Request

from ..domain import RequestContext
from ..core.orchestrator import RecommendationTimeout
from .schemas import (
    RecommendationItemSchema,
    RecommendationMetadataSchema,
    RecommendationRequestSchema,
    RecommendationResponseSchema,
)

router = APIRouter()


@router.get("/health/live")
def liveness() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
def readiness(request: Request) -> dict[str, str]:
    if not hasattr(request.app.state, "container"):
        raise HTTPException(status_code=503, detail="service not initialized")
    return {"status": "ready"}


@router.post("/v1/recommendations/carousels", response_model=RecommendationResponseSchema)
def recommendations(
    payload: RecommendationRequestSchema,
    request: Request,
    request_id: str | None = Header(default=None, alias="X-Request-Id"),
) -> RecommendationResponseSchema:
    try:
        result = request.app.state.container.orchestrator.recommend(
            RequestContext(
                request_id=request_id or f"r_{uuid.uuid4().hex}",
                user_id=payload.user_id,
                session_id=payload.session_id,
                carousel_type=payload.carousel_type,
                limit=payload.limit,
                non_personalized_work_ids=frozenset(payload.non_personalized_work_ids),
                device=payload.context.device,
            )
        )
    except RecommendationTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    return RecommendationResponseSchema(
        request_id=result.request_id,
        carousel_id=result.carousel_id,
        items=[
            RecommendationItemSchema(
                work_id=rec.candidate.item.work_id,
                format_id=rec.candidate.item.format_id.value,
                title=rec.candidate.item.title,
                position=rec.position,
                reason_code=rec.reason_code,
                is_exploration=rec.candidate.is_exploration,
                propensity=rec.candidate.propensity,
            )
            for rec in result.recommendations
        ],
        metadata=RecommendationMetadataSchema(
            model_version=result.policy.model_version,
            feature_version=result.policy.feature_version,
            policy_version=result.policy.policy_version,
            experiment_id=result.policy.experiment_id,
            arm=result.policy.arm.value,
        ),
        diagnostics=result.diagnostics,
    )
