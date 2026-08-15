from __future__ import annotations

from recommendation_service.training.arm_c import stable_bucket
from recommendation_service.training.calibration import (
    BetaCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    calibrator_from_dict,
    select_calibrator,
)


def test_all_calibrators_round_trip_and_bound_probabilities() -> None:
    calibrators = (
        PlattCalibrator(1.2, -0.1),
        BetaCalibrator(0.9, -1.1, 0.2),
        IsotonicCalibrator((0.3, 0.7, 1.0), (0.1, 0.6, 0.9)),
    )
    for calibrator in calibrators:
        restored = calibrator_from_dict(calibrator.to_dict())
        assert 0 < restored.predict(0.4) < 1


def test_calibrator_selection_and_stable_embedding_bucket() -> None:
    selected = select_calibrator([0.1, 0.2, 0.8, 0.9], [0.0, 0.0, 1.0, 1.0])
    assert selected.predict(0.8) > selected.predict(0.2)
    assert stable_bucket("work-123", 1000) == stable_bucket("work-123", 1000)
    assert 0 <= stable_bucket("work-123", 1000) < 1000

