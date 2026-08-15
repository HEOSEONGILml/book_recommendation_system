from __future__ import annotations

import time
import uuid

from ..domain import Recommendation, RecommendationResult, RequestContext
from ..ports import PolicyProvider, Ranker, UserRepository
from .candidates import CandidateGenerator
from .eligibility import EligibilityFilters
from .slate import SlateComposer
from .reranking import TemporalXQuAD


class RecommendationOrchestrator:
    def __init__(
        self,
        users: UserRepository,
        policies: PolicyProvider,
        candidates: CandidateGenerator,
        ranker: Ranker,
        filters: EligibilityFilters,
        reranker: TemporalXQuAD,
        slate: SlateComposer,
        timeout_ms: int = 250,
    ) -> None:
        self.users = users
        self.policies = policies
        self.candidates = candidates
        self.ranker = ranker
        self.filters = filters
        self.reranker = reranker
        self.slate = slate
        self.timeout_ms = timeout_ms

    def recommend(self, context: RequestContext) -> RecommendationResult:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        profile = self.users.get_profile(context.user_id)
        policy = self.policies.resolve(context.user_id)
        timings["feature_ms"] = self._elapsed_ms(started)
        self._check_deadline(started)

        stage_started = time.perf_counter()
        generated, exploration_universe = self.candidates.generate(profile, context)
        timings["candidate_ms"] = self._elapsed_ms(stage_started)
        self._check_deadline(started)

        stage_started = time.perf_counter()
        self.ranker.score(generated, profile, context)
        timings["ranker_ms"] = self._elapsed_ms(stage_started)
        self._check_deadline(started)

        stage_started = time.perf_counter()
        deduplicated, duplicate_count = self.filters.remove_non_personalized_duplicates(
            generated, context
        )
        self.reranker.apply(deduplicated, profile, policy, context.now)
        selected = self.slate.compose(deduplicated, exploration_universe, context, policy)
        timings["rerank_ms"] = self._elapsed_ms(stage_started)
        self._check_deadline(started)

        recommendations = tuple(
            Recommendation(
                candidate=candidate,
                position=position,
                reason_code=(
                    "EPSILON_GREEDY"
                    if candidate.is_exploration
                    else min(source.value for source in candidate.sources).upper()
                ),
            )
            for position, candidate in enumerate(selected, start=1)
        )
        return RecommendationResult(
            request_id=context.request_id,
            carousel_id=f"c_{uuid.uuid4().hex}",
            recommendations=recommendations,
            policy=policy,
            diagnostics={
                "generated_candidates": len(generated),
                "duplicates_removed": duplicate_count,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                **timings,
            },
        )

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 3)

    def _check_deadline(self, started: float) -> None:
        if self._elapsed_ms(started) > self.timeout_ms:
            raise RecommendationTimeout(f"recommendation exceeded {self.timeout_ms}ms")


class RecommendationTimeout(RuntimeError):
    pass
