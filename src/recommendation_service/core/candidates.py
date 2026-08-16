from __future__ import annotations

from math import sqrt

from ..domain import Candidate, CandidateSource, Item, RequestContext, UserProfile
from ..ports import CatalogRepository
from .eligibility import EligibilityFilters


class CandidateGenerator:
    """캐러셀 목적에 맞는 후보를 만들고 공통 Candidate 형식으로 합친다."""

    def __init__(self, catalog: CatalogRepository, filters: EligibilityFilters) -> None:
        self.catalog = catalog
        self.filters = filters

    def generate(
        self, profile: UserProfile, context: RequestContext, limit: int = 100
    ) -> tuple[list[Candidate], list[Item]]:
        if context.carousel_type == "LIBRARY_SIMILAR":
            retrieved = self.catalog.similar_candidates(profile.library_work_ids, limit)
            eligible = self.filters.catalog(retrieved, profile, context)
            scored = self._content_similarity(eligible, profile, limit)
            source = CandidateSource.CONTENT
        else:
            retrieved = self.catalog.personalized_candidates(profile, limit)
            eligible = self.filters.catalog(retrieved, profile, context)
            scored = self._personalized(eligible, profile, limit)
            source = CandidateSource.CF

        if not scored:
            eligible = self.filters.catalog(
                self.catalog.personalized_candidates(profile, limit), profile, context
            )
            scored = [(item, item.popularity) for item in sorted(
                eligible, key=lambda current: current.popularity, reverse=True
            )[:limit]]
            source = CandidateSource.POPULAR
        exploration = self.filters.catalog(
            self.catalog.exploration_candidates(limit), profile, context
        )
        return [
            Candidate(item=item, sources={source}, source_score=score) for item, score in scored
        ], exploration

    @staticmethod
    def _personalized(
        items: list[Item], profile: UserProfile, limit: int
    ) -> list[tuple[Item, float]]:
        scored = [
            (
                item,
                0.65 * profile.genre_affinity.get(item.genre, 0.0)
                + 0.25 * item.popularity
                + 0.10 * max(item.embedding, default=0.0),
            )
            for item in items
        ]
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:limit]

    def _content_similarity(
        self,
        items: list[Item], profile: UserProfile, limit: int
    ) -> list[tuple[Item, float]]:
        anchors = [self.catalog.get_item(work_id) for work_id in profile.library_work_ids]
        anchors = [item for item in anchors if item is not None]
        if not anchors:
            return []
        matched = [
            (item, max(self._cosine(item.embedding, anchor.embedding) for anchor in anchors))
            for item in items
        ]
        return sorted(matched, key=lambda pair: pair[1], reverse=True)[:limit]

    @staticmethod
    def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        denominator = sqrt(sum(a * a for a in left)) * sqrt(sum(b * b for b in right))
        return numerator / denominator if denominator else 0.0
