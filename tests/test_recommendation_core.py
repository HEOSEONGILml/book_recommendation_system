from datetime import UTC, datetime, timedelta

from recommendation_service.adapters.memory import (
    InMemoryCatalogRepository, InMemoryUserRepository, StaticPolicyProvider, sample_catalog,
)
from recommendation_service.config import Settings
from recommendation_service.core.candidates import CandidateGenerator
from recommendation_service.core.constraints import CarouselConstraint
from recommendation_service.core.eligibility import EligibilityFilters
from recommendation_service.core.orchestrator import RecommendationOrchestrator
from recommendation_service.core.ranking import HeuristicRanker
from recommendation_service.core.slate import SlateComposer
from recommendation_service.core.reranking import TemporalXQuAD
from recommendation_service.domain import RequestContext, UserProfile


def build(profile: UserProfile, epsilon: float = 1.0) -> RecommendationOrchestrator:
    filters = EligibilityFilters()
    return RecommendationOrchestrator(
        users=InMemoryUserRepository([profile]),
        policies=StaticPolicyProvider(Settings(epsilon=epsilon)),
        candidates=CandidateGenerator(InMemoryCatalogRepository(sample_catalog()), filters),
        ranker=HeuristicRanker(), constraints=CarouselConstraint(0.55),
        filters=filters, reranker=TemporalXQuAD(), slate=SlateComposer(),
    )


def test_duplicate_filter_temporal_xquad_and_exploration() -> None:
    now = datetime.now(UTC)
    profile = UserProfile(
        "u1", age=17, genre_affinity={"소설": 1.0},
        consumed_work_ids={"w_001"}, exposure_history={"w_006": [now - timedelta(hours=1)]},
    )
    result = build(profile).recommend(RequestContext(
        "r1", "u1", "s1", "FOR_YOU", 8,
        non_personalized_work_ids=frozenset({"w_002"}), now=now,
    ))
    ids = [recommendation.candidate.item.work_id for recommendation in result.recommendations]
    assert len(ids) == len(set(ids)) == 8
    assert not {"w_001", "w_002"}.intersection(ids)
    assert sum(rec.candidate.is_exploration for rec in result.recommendations) == 1
    assert result.diagnostics["duplicates_removed"] >= 0
    assert {"feature_ms", "candidate_ms", "ranker_ms", "rerank_ms"} <= result.diagnostics.keys()


def test_library_similar_applies_similarity_constraint() -> None:
    profile = UserProfile("u2", library_work_ids={"w_005"})
    result = build(profile, epsilon=0).recommend(RequestContext(
        "r2", "u2", "s2", "LIBRARY_SIMILAR", 5,
    ))
    assert result.recommendations
    assert all(rec.candidate.source_score >= 0.55 for rec in result.recommendations)
    assert result.diagnostics["constraint_filtered"] >= 0
