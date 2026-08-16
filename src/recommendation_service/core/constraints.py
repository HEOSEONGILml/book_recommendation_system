from __future__ import annotations

from ..domain import Candidate, RequestContext


class CarouselConstraint:
    """분석으로 도출한 캐러셀별 최소 조건을 적용한다."""

    def __init__(self, similarity_threshold: float = 0.55) -> None:
        self.similarity_threshold = similarity_threshold

    def apply(
        self, candidates: list[Candidate], context: RequestContext
    ) -> tuple[list[Candidate], int]:
        if context.carousel_type != "LIBRARY_SIMILAR":
            return candidates, 0
        kept = [candidate for candidate in candidates if candidate.source_score >= self.similarity_threshold]
        return kept, len(candidates) - len(kept)
