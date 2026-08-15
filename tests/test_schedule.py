from datetime import UTC, datetime, timedelta

from recommendation_service.training.schedule import (
    RetrainingPolicy,
    TrainingSignals,
    TrainingState,
)


def test_weekly_ranker_and_daily_candidate_schedule() -> None:
    now = datetime.now(UTC)
    decision = RetrainingPolicy().decide(
        now,
        TrainingState(now - timedelta(days=8), now - timedelta(days=2)),
        TrainingSignals(new_examples=20_000),
    )
    assert decision.train_ranker
    assert decision.train_candidates


def test_drift_triggers_early_ranker_training() -> None:
    now = datetime.now(UTC)
    decision = RetrainingPolicy().decide(
        now,
        TrainingState(now - timedelta(days=2), now),
        TrainingSignals(new_examples=20_000, feature_psi=0.25),
    )
    assert decision.train_ranker
    assert "ranker_drift" in decision.reasons
