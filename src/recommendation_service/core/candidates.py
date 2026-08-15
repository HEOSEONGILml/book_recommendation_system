from __future__ import annotations

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
        if context.carousel_type == "INTEREST_COLD_START":
            retrieved = self.catalog.cold_start_candidates(profile.onboarding_genres, limit)
            eligible = self.filters.catalog(retrieved, profile, context)
            scored = self._cold_start(eligible, profile, limit)
            source = CandidateSource.COLD_START
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

    @staticmethod
    def _cold_start(
        items: list[Item], profile: UserProfile, limit: int
    ) -> list[tuple[Item, float]]:
        if not profile.onboarding_genres:
            return []
        matched = [
            (item, 0.75 + 0.25 * item.popularity)
            for item in items
            if item.genre in profile.onboarding_genres
        ]
        return sorted(matched, key=lambda pair: pair[1], reverse=True)[:limit]
