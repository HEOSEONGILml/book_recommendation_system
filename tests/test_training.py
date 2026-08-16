from datetime import UTC, datetime, timedelta
from pathlib import Path

from recommendation_service.core.ranking import FEATURE_NAMES, LinearModelRanker
from recommendation_service.training.baseline import train_arm_a
from recommendation_service.training.dataset import TrainingExample, time_split
from recommendation_service.training.evaluation import binary_metrics
from recommendation_service.training.registry import write_arm_a_artifact


def test_arm_a_training_and_artifact(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    examples = []
    for index in range(100):
        affinity = index / 100
        features = {name: 0.0 for name in FEATURE_NAMES}
        features.update(bias=1.0, genre_affinity=affinity, source_score=affinity, item_similarity=affinity)
        examples.append(TrainingExample(
            now + timedelta(minutes=index), f"u{index}", f"w{index}", "FOR_YOU",
            features, float(index > 50),
        ))
    train, validation = time_split(examples)
    model = train_arm_a(train)
    metrics = binary_metrics(validation, model.predict)
    artifact = write_arm_a_artifact(tmp_path, "arm-a-test", model, metrics)
    assert metrics["count"] == 20
    assert LinearModelRanker(artifact).artifact["arm"] == "A"
