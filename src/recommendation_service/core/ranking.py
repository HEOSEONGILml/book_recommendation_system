from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

from ..domain import Candidate, RequestContext, UserProfile


FEATURE_NAMES = ("bias", "source_score", "genre_affinity", "onboarding_match", "popularity")


def build_features(candidate: Candidate, profile: UserProfile) -> dict[str, float]:
    return {
        "bias": 1.0,
        "source_score": candidate.source_score,
        "genre_affinity": profile.genre_affinity.get(candidate.item.genre, 0.0),
        "onboarding_match": float(candidate.item.genre in profile.onboarding_genres),
        "popularity": candidate.item.popularity,
    }


class HeuristicRanker:
    def score(
        self, candidates: Iterable[Candidate], profile: UserProfile, context: RequestContext
    ) -> None:
        for candidate in candidates:
            candidate.features = build_features(candidate, profile)
            candidate.rank_score = min(
                1.0,
                0.55 * candidate.source_score
                + 0.25 * candidate.features["genre_affinity"]
                + 0.20 * candidate.features["onboarding_match"],
            )


class LinearModelRanker:
    def __init__(self, artifact_path: str | Path) -> None:
        self.artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))

    def score(
        self, candidates: Iterable[Candidate], profile: UserProfile, context: RequestContext
    ) -> None:
        weights = self.artifact["weights"]
        for candidate in candidates:
            candidate.features = build_features(candidate, profile)
            raw = sum(weights.get(name, 0.0) * candidate.features[name] for name in FEATURE_NAMES)
            candidate.rank_score = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw))))
