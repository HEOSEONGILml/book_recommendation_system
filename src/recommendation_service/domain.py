from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class Format(StrEnum):
    EBOOK = "ebook"
    AUDIOBOOK = "audiobook"
    CHATBOOK = "chatbook"


class CandidateSource(StrEnum):
    CF = "cf"
    CONTENT = "content"
    POPULAR = "popular"
    EXPLORATION = "exploration"


class RankerArm(StrEnum):
    A0 = "A0"
    A = "A"
    B = "B"
    C = "C"


@dataclass(frozen=True, slots=True)
class Item:
    work_id: str
    format_id: Format
    title: str
    author_id: str
    genre: str
    popularity: float = 0.0
    embedding: tuple[float, ...] = ()
    rights_start: datetime = field(default_factory=lambda: datetime(2000, 1, 1, tzinfo=UTC))
    rights_end: datetime = field(default_factory=lambda: datetime(2100, 1, 1, tzinfo=UTC))
    age_rating: int = 0


@dataclass(slots=True)
class UserProfile:
    user_id: str
    age: int | None = None
    library_work_ids: set[str] = field(default_factory=set)
    consumed_work_ids: set[str] = field(default_factory=set)
    disliked_work_ids: set[str] = field(default_factory=set)
    genre_affinity: dict[str, float] = field(default_factory=dict)
    exposure_history: dict[str, list[datetime]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RequestContext:
    request_id: str
    user_id: str
    session_id: str
    carousel_type: str
    limit: int
    non_personalized_work_ids: frozenset[str] = frozenset()
    device: str = "UNKNOWN"
    now: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class Candidate:
    item: Item
    sources: set[CandidateSource] = field(default_factory=set)
    source_score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    rank_score: float = 0.0
    exposure_fairness: float = 0.0
    final_score: float = 0.0
    is_exploration: bool = False
    propensity: float | None = None


@dataclass(frozen=True, slots=True)
class Policy:
    experiment_id: str
    arm: RankerArm
    xquad_lambda: float
    epsilon: float
    model_version: str
    feature_version: str
    policy_version: str


@dataclass(frozen=True, slots=True)
class Recommendation:
    candidate: Candidate
    position: int
    reason_code: str


@dataclass(frozen=True, slots=True)
class RecommendationResult:
    request_id: str
    carousel_id: str
    recommendations: tuple[Recommendation, ...]
    policy: Policy
    diagnostics: dict[str, Any]
