from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class TrainingState:
    last_ranker_training_at: datetime
    last_candidate_training_at: datetime


@dataclass(frozen=True, slots=True)
class TrainingSignals:
    new_examples: int
    feature_psi: float = 0.0
    ndcg_drop: float = 0.0


@dataclass(frozen=True, slots=True)
class RetrainingDecision:
    train_ranker: bool
    train_candidates: bool
    reasons: tuple[str, ...]


class RetrainingPolicy:
    """배치 오케스트레이터가 호출하는 초기 재학습 정책."""

    def __init__(
        self,
        ranker_interval: timedelta = timedelta(days=7),
        candidate_interval: timedelta = timedelta(days=1),
        min_new_examples: int = 10_000,
        psi_threshold: float = 0.2,
        ndcg_drop_threshold: float = 0.03,
    ) -> None:
        self.ranker_interval = ranker_interval
        self.candidate_interval = candidate_interval
        self.min_new_examples = min_new_examples
        self.psi_threshold = psi_threshold
        self.ndcg_drop_threshold = ndcg_drop_threshold

    def decide(
        self, now: datetime, state: TrainingState, signals: TrainingSignals
    ) -> RetrainingDecision:
        reasons: list[str] = []
        scheduled_ranker = now - state.last_ranker_training_at >= self.ranker_interval
        drifted = (
            signals.feature_psi >= self.psi_threshold
            or signals.ndcg_drop >= self.ndcg_drop_threshold
        )
        enough_data = signals.new_examples >= self.min_new_examples
        train_ranker = enough_data and (scheduled_ranker or drifted)
        if train_ranker:
            reasons.append("ranker_schedule" if scheduled_ranker else "ranker_drift")

        train_candidates = now - state.last_candidate_training_at >= self.candidate_interval
        if train_candidates:
            reasons.append("candidate_schedule")
        return RetrainingDecision(train_ranker, train_candidates, tuple(reasons))
