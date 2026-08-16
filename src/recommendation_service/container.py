from __future__ import annotations

from dataclasses import dataclass

from .adapters.memory import (
    InMemoryCatalogRepository,
    InMemoryUserRepository,
    StaticPolicyProvider,
    sample_catalog,
)
from .config import Settings
from .core.candidates import CandidateGenerator
from .core.constraints import CarouselConstraint
from .core.eligibility import EligibilityFilters
from .core.orchestrator import RecommendationOrchestrator
from .core.ranking import HeuristicRanker, LinearModelRanker
from .core.slate import SlateComposer
from .core.reranking import TemporalXQuAD


@dataclass(slots=True)
class Container:
    settings: Settings
    catalog: InMemoryCatalogRepository
    users: InMemoryUserRepository
    orchestrator: RecommendationOrchestrator


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or Settings.from_env()
    catalog = InMemoryCatalogRepository(sample_catalog())
    users = InMemoryUserRepository()
    filters = EligibilityFilters()
    orchestrator = RecommendationOrchestrator(
        users=users,
        policies=StaticPolicyProvider(settings),
        candidates=CandidateGenerator(catalog, filters),
        ranker=_build_ranker(settings),
        constraints=CarouselConstraint(settings.similarity_threshold),
        filters=filters,
        reranker=TemporalXQuAD(),
        slate=SlateComposer(),
        timeout_ms=settings.request_timeout_ms,
    )
    return Container(settings, catalog, users, orchestrator)


def _build_ranker(settings: Settings):
    arm = settings.ranker_arm.upper()
    if arm == "A0":
        return HeuristicRanker()
    if not settings.model_manifest:
        raise ValueError(f"MILLIE_MODEL_MANIFEST is required for Arm {arm}")
    if arm == "A":
        return LinearModelRanker(settings.model_manifest)
    if arm == "B":
        from .core.ml_rankers import LightGBMRanker
        return LightGBMRanker(settings.model_manifest)
    if arm == "C":
        from .core.ml_rankers import DeepRanker
        return DeepRanker(settings.model_manifest)
    raise ValueError(f"unsupported ranker arm: {arm}")
